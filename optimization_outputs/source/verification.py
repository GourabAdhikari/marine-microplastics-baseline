"""Automated invariants and independently recomputed metrics."""
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from experiment import SCENARIO, BASE_NAME, OPT_NAME, REFERENCE, select, dump, sha


def protect(exp):
    before=json.loads((exp.out/'protected_hashes_before.json').read_text())
    def digest(p):
        return 'symlink:'+os.readlink(p) if p.is_symlink() else sha(p)
    actual={str(p.relative_to(exp.root)) for p in exp.root.rglob('*') if (p.is_file() or p.is_symlink()) and 'optimization_outputs' not in p.relative_to(exp.root).parts}
    changed=[p for p,h in before.items() if not (exp.root/p).exists() and not (exp.root/p).is_symlink() or digest(exp.root/p)!=h]
    exp.check('all protected paths byte-identical / symlinks identical', not changed)
    exp.check('no new or removed paths outside optimization_outputs',actual==set(before))
    dump(exp.out/'protected_hashes_after.json',{p:digest(exp.root/p) for p in before})
    (exp.out/'git_status_after.txt').write_text(subprocess.check_output(['git','status','--porcelain'],text=True,env=os.environ|{'GIT_OPTIONAL_LOCKS':'0'}))
    return len(before)


def run_tests(exp):
    exp.validate_data()
    nums=exp.ns['NUMERIC']; cats=exp.ns['CATEGORICAL']
    # Verify actual original preprocessor on every inner and outer training partition.
    partitions=[('outer'+str(f),a,b) for f,(a,b) in enumerate(exp.outer,1)]
    partitions += [(f'inner{f}/{j}',a,b) for f,folds in exp.inner.items() for j,(a,b) in enumerate(folds,1)]
    for label,a,b in partitions:
        train=exp.X.iloc[a]; test=exp.X.iloc[b]
        prep=exp.ns['preprocessor']().fit(train)
        num=prep.named_transformers_['num']
        imp=num.named_steps['simpleimputer']; scaler=num.named_steps['standardscaler']
        median=train[nums].median().to_numpy()
        filled=train[nums].fillna(dict(zip(nums,median))).to_numpy()
        exp.check(label+' training-only imputer statistics',np.allclose(imp.statistics_,median))
        exp.check(label+' training-only scaler mean/variance',np.allclose(scaler.mean_,filled.mean(axis=0)) and np.allclose(scaler.var_,filled.var(axis=0)))
        encoder=prep.named_transformers_['cat']
        exp.check(label+' training-only encoder vocab',np.array_equal(encoder.categories_[0],np.sort(train[cats[0]].unique())))
        altered=test.iloc[:2].copy()
        for col in nums: altered[col]=[train[col].min()-1000,train[col].max()+1000]
        altered[cats[0]]='__unseen_family__'
        clip=exp.ns['clip_to_range'](train)
        clipped=clip(altered)
        exp.check(label+' training-only lower/upper clipping',np.allclose(clipped[nums].iloc[0],train[nums].min()) and np.allclose(clipped[nums].iloc[1],train[nums].max()))
        transformed=prep.transform(clipped)
        exp.check(label+' unseen family ignored',np.all(transformed[:,len(nums):]==0))
        # Mutating validation data cannot change training-fitted state.
        exp.check(label+' validation transform never refits statistics',np.array_equal(scaler.mean_,filled.mean(axis=0)))
    # Rule unit tests exercise safety, fallback, R2 tie band and exact macro tie ID order.
    def choose(rows):
        return select(pd.DataFrame(rows,columns=['candidate','R2','study_macro_MAE']))[0]
    exp.check('selector rejects unsafe higher R2',choose([('C0_original',.4,1),('C1_shallow',.8,1.051)])=='C0_original')
    exp.check('selector rejects lower R2 even with better MAE',choose([('C0_original',.4,1),('C1_shallow',.39,.8)])=='C0_original')
    exp.check('selector prefers lower MAE within 0.005 including C0',choose([('C0_original',.4,1),('C1_shallow',.404,1.01)])=='C0_original')
    exp.check('selector uses ID order for exact macro tie',choose([('C0_original',.4,1),('C1_shallow',.5,.9),('C2_conservative',.502,.9)])=='C1_shallow')
    exp.check('selector accepts safe clear improvement',choose([('C0_original',.4,1),('C1_shallow',.5,1.04)])=='C1_shallow')
    exp.check('selector respects inclusive 5% safety limit',choose([('C0_original',.4,1),('C1_shallow',.5,1.05)])=='C1_shallow')
    before_fits=exp.new_fits
    for f,(tr,te) in enumerate(exp.outer,1):
        perturbed=exp.y.copy();perturbed[te]=np.arange(len(te))*1234.+9999.
        # Same selector and cache path receives the sliced training labels, not full y.
        m,selected=exp.inner_search(f,perturbed[tr])
        expected=exp.selections.loc[exp.selections.outer==f].iloc[0]
        exp.check(f'outer {f} label perturbation leaves selection and reason identical', selected['candidate']==expected.candidate and selected['reason']==expected.reason)
        original=exp.inner_metrics[exp.inner_metrics.outer==f].reset_index(drop=True)
        exp.check(f'outer {f} label perturbation leaves all inner metrics identical',np.allclose(m.select_dtypes('number'),original.select_dtypes('number'),rtol=1e-13,atol=1e-13))
    exp.check('outer-label perturbation reused caches, no additional fits',exp.new_fits==before_fits)
    # Exactly two essential repeat fits, both on the same fixed training slice and same held-out rows.
    tr,te=exp.inner[1][0]
    a,b=tr[:1200],te[:200]
    p=exp.cached_fit('verification',0,1,'C5_row_and_feature_subsample',a,b,exp.y[a])
    q=exp.cached_fit('verification',0,2,'C5_row_and_feature_subsample',a,b,exp.y[a])
    exp.check('deterministic fixed-slice repeat including subsampling',np.array_equal(p,q))
    out=pd.read_csv(exp.out/'optimized_oof_predictions.csv',float_precision='round_trip')
    exp.check('optimized exact aligned unique IDs',len(out)==16988 and out.water_row.is_unique and out.source_row.is_unique and np.array_equal(out.water_row,exp.baseline.water_row) and np.array_equal(out.source_row,exp.baseline.source_row))
    exp.check('optimized same target/study/fold and original predictions',all(np.array_equal(out[c],exp.baseline[c]) for c in ['target_log10','study','fold']) and np.array_equal(out.original_prediction_log10,exp.baseline.prediction_log10))
    exp.check('all optimized and original predictions finite',np.isfinite(out[['original_prediction_log10','optimized_prediction_log10']]).all().all())
    exp.check('scenario labels complete and frozen',out.scenario.eq(SCENARIO).all() and exp.inner_metrics.scenario.eq(SCENARIO).all() and exp.selections.scenario.eq(SCENARIO).all())
    summary=json.loads((exp.out/'optimization_summary.json').read_text())
    for name,col in [(BASE_NAME,'original_prediction_log10'),(OPT_NAME,'optimized_prediction_log10')]:
        y=out.target_log10.to_numpy();p=out[col].to_numpy();e=p-y
        individual=[(g[col]-g.target_log10).abs().mean() for _,g in out.groupby('study')]
        square=[((g[col]-g.target_log10)**2).mean() for _,g in out.groupby('study')]
        independent={'R2':r2_score(y,p),'MAE':mean_absolute_error(y,p),'RMSE':np.sqrt(mean_squared_error(y,p)),
                     'study_macro_MAE':np.mean(individual),'study_macro_MSE':np.mean(square),'median_study_MAE':np.median(individual),
                     'p90_study_MAE':np.quantile(individual,.9),'worst_study_MAE':max(individual),'within_1_log':np.mean(abs(e)<=1),
                     'mean_fold_R2':np.mean([r2_score(g.target_log10,g[col]) for _,g in out.groupby('fold')])}
        for k,v in independent.items():
            exp.check(name+' independently recomputed '+k,np.isclose(v,summary[name][k],rtol=1e-13,atol=1e-13))
        for grouping,file in [('fold','paired_per_fold_metrics.csv'),('study','paired_per_study_metrics.csv')]:
            saved=pd.read_csv(exp.out/file).set_index(grouping)
            for key,g in out.groupby(grouping):
                exp.check(name+f' {grouping} {key} saved MAE and MSE-derived RMSE',np.isclose(saved.loc[key,name+'__MAE'],mean_absolute_error(g.target_log10,g[col])) and np.isclose(saved.loc[key,name+'__RMSE'],np.sqrt(mean_squared_error(g.target_log10,g[col]))))
    exp.check('exact reported delta from reference',summary['delta_R2']==summary[OPT_NAME]['R2']-REFERENCE)
    events=[json.loads(l) for l in (exp.out/'fit_log.jsonl').read_text().splitlines()]
    for stage,expected in [('inner',120),('outer',5),('verification',2)]:
        exp.check(stage+' bounded fit counts and successful completion',sum(e['stage']==stage and e['event']=='started' for e in events)==expected and sum(e['stage']==stage and e['event']=='completed' for e in events)==expected)
    protected=protect(exp)
    result={'scenario':SCENARIO,'passed':len(exp.checks),'checks':exp.checks,'protected_paths_verified':protected,'all_protected_unchanged':True,'new_fits_this_process':exp.new_fits,'cache_hits_this_process':exp.cache_hits}
    dump(exp.out/'reproduction_and_integrity_checks.json',result)
    (exp.out/'tests.txt').write_text('\n'.join('PASS '+s for s in exp.checks)+f'\n{len(exp.checks)} assertions passed.\n')
    print(f'{len(exp.checks)} assertions passed; {protected} protected paths unchanged.')
    return result

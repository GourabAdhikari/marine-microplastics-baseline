"""Bounded training/test audit. Run from repo root using the existing venv.
Default: reuse complete audit caches or fit each locked S2 outer model once (cap 5).
--tests-only: verify saved predictions, metrics and protected files; never fit.
No historical experiment, candidate selection or external scoring is rerun.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import warnings

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'final_research_audit'
SCENARIO='historical_hypothetical_unit_scenario'


def sha(path):
    if path.is_symlink():return 'symlink:'+os.readlink(path)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def dump(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(obj,indent=2,default=lambda v:v.item() if hasattr(v,'item') else str(v)));temp.replace(path)


def fingerprint(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str).encode()).hexdigest()


def protected():
    return {str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if (p.is_file() or p.is_symlink()) and p.relative_to(ROOT).parts[0]!='final_research_audit'}


def markdown(df):
    def fmt(x):return f'{x:.9f}' if isinstance(x,float) else str(x)
    return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tests-only',action='store_true');args=parser.parse_args()
    os.chdir(ROOT);sys.dont_write_bytecode=True;os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['GIT_OPTIONAL_LOCKS']='0'
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
    for key,sub in [('TMPDIR','tmp'),('XDG_CACHE_HOME','cache'),('MPLCONFIGDIR','matplotlib'),('IPYTHONDIR','ipython')]:
        p=OUT/sub;p.mkdir(exist_ok=True);os.environ[key]=str(p)
    before_path=OUT/'protected_before.json'
    if not before_path.exists():dump(before_path,protected())
    import numpy as np
    import pandas as pd
    import joblib
    from lightgbm import LGBMRegressor
    from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
    from sklearn.model_selection import GroupKFold
    from unittest.mock import patch
    checks=[]
    def check(name,ok):
        if not ok:raise AssertionError(name)
        checks.append(name)
    check('repository identity',ROOT.name=='marine-microplastics-baseline')
    def csv(path):return pd.read_csv(path,float_precision='round_trip')
    before=json.loads(before_path.read_text())
    # Validate historical source/OOF/config identities before fitting.
    status=json.loads((ROOT/'study_robustness_outputs/EXECUTION_STATUS.json').read_text())
    for p,h in status['artifact_sha256'].items():check('Stage 4 artifact hash '+p,sha(ROOT/'study_robustness_outputs'/p)==h)
    for p,h in json.loads((ROOT/'reproduction_outputs/input_hashes.json').read_text()).items():check('original file hash '+p,sha(ROOT/p)==h)
    env={'python':sys.version,'packages':{p:metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','lightgbm','joblib']},'threads':4,'seed':0}
    oldenv=json.loads((ROOT/'optimization_outputs/checkpoint_identity.json').read_text())['versions']
    check('core runtime versions exact',all(env['packages'][k]==v for k,v in oldenv.items()))
    dump(OUT/'environment.json',env)
    source=ROOT/'optimization_outputs/source/experiment.py'
    spec=importlib.util.spec_from_file_location('original_readonly_helpers',source);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    ns=old.original_namespace(ROOT);df=ns['df'];features=ns['FEATURES'];nums=ns['NUMERIC']
    base=csv(ROOT/'reproduction_outputs/out_of_fold_predictions.csv');oof=csv(ROOT/'study_robustness_outputs/all_procedure_oof_predictions.csv')
    registry=json.loads((ROOT/'optimization_outputs/candidate_registry.json').read_text())['configs']
    lock=json.loads((ROOT/'study_robustness_outputs/locked_selections.json').read_text())
    selected={v['outer']:v['candidate'] for v in lock['payload']['selections'] if v['rule']=='S2'}
    splits=[(np.array(a),np.array(b)) for a,b in json.loads((ROOT/'optimization_outputs/frozen_splits.json').read_text())['outer']]
    check('exact 16988 rows and 72 studies',len(df)==len(oof)==16988 and df.study.nunique()==72)
    check('row IDs unique and aligned',oof.water_row.is_unique and oof.source_row.is_unique and np.array_equal(oof.water_row,df.index) and np.array_equal(oof.source_row,base.source_row))
    check('target, study and scenario identity',np.array_equal(df.y,oof.target_log10) and np.array_equal(df.study,oof.study) and oof.scenario.eq(SCENARIO).all())
    mask=(ns['data'].medium=='Ocean water') & (ns['data'].unit=='pieces/m3')
    check('original source-row filtering and raw shape',ns['raw'].shape==(29076,35) and np.array_equal(ns['data'].index[mask],base.source_row))
    check('exact frozen physical-unit hypotheses and log target',ns['C']==.001 and np.array_equal(df.y,np.log10(df.conc+.001)) and np.array_equal(df.conc,ns['data'].loc[mask].reset_index(drop=True).conc*ns['data'].loc[mask].reset_index(drop=True).doi.map(ns['UNIT_FIXES']).fillna(1)))
    check('original GroupKFold memberships reproduced',all(np.array_equal(a,c) and np.array_equal(b,d) for (a,b),(c,d) in zip(splits,GroupKFold(5).split(df,groups=df.study))))
    check('original predictor whitelist',features==['lat','lon','year','month_sin','month_cos','log_mesh','log_depth','method_family'])
    check('test sizes and coverage once',[len(b) for a,b in splits]==[6135,2714,2712,2714,2713] and np.array_equal(np.sort(np.concatenate([b for a,b in splits])),np.arange(len(df))))
    paths=['data/noaa_microplastics.csv','microplastics.py','microplastics.ipynb','optimization_outputs/source/experiment.py','optimization_outputs/frozen_splits.json','optimization_outputs/candidate_registry.json','study_robustness_outputs/locked_selections.json','study_robustness_outputs/all_procedure_oof_predictions.csv']
    identity={'inputs':{p:sha(ROOT/p) for p in paths},'environment':env,'scenario':SCENARIO,'audit_code_sha256':sha(Path(__file__))}
    dump(OUT/'input_identity.json',identity)
    # Previous checkpoints contain only predictions, not serialized fitted estimators.
    model_inventory=[str(p.relative_to(ROOT)) for folder in ['reproduction_outputs','optimization_outputs','blending_outputs','study_robustness_outputs','independent_validation_outputs'] for p in (ROOT/folder).glob('**/*') if p.is_file() and p.suffix in ['.joblib','.pkl','.pickle','.model','.ubj'] and not any(part in ['venv','uv_cache','cache'] for part in p.relative_to(ROOT).parts)]
    dump(OUT/'prior_fitted_model_inventory.json',{'matching_serialized_models_found':model_inventory,'previous_npz_content':'test row IDs and predictions only; no fitted booster or transformer','refit_policy':'At most one exact S2 fit per outer fold; held-out predictions must match before training scores are accepted.'})
    check('no unexpected prior fitted models needing inspection',not model_inventory)
    cache=OUT/'checkpoints';cache.mkdir(exist_ok=True);log=OUT/'fit_log.jsonl'
    if not log.exists():log.write_text('')
    def events():return [json.loads(l) for l in log.read_text().splitlines()]
    def event(e):
        with log.open('a') as f:f.write(json.dumps(e)+'\n')
    authorization=None;original_fit=LGBMRegressor.fit
    def guarded(estimator,X,y,*a,**kw):
        check('fit explicitly authorized for one locked fold',authorization is not None and not args.tests_only)
        check('hard five-new-fit cap',sum(e['event']=='fit_started' for e in events())<5)
        f,cid,key,tr=authorization
        check('fit uses exact training labels and parameters',np.array_equal(y,df.y.iloc[tr]) and estimator.get_params()==registry[cid])
        check('no repeated fit identity',not any(e.get('key')==key and e['event']=='fit_started' for e in events()))
        event({'event':'fit_started','fold':f,'candidate':cid,'key':key,'utc':datetime.now(timezone.utc).isoformat(),'training_rows':len(tr),'scenario':SCENARIO})
        return original_fit(estimator,X,y,*a,**kw)
    rows=[];new_fits=0
    with patch.object(LGBMRegressor,'fit',guarded):
        for f,(tr,te) in enumerate(splits,1):
            cid=selected[f]
            check(f'fold {f} exact locked rows and disjoint studies',np.array_equal(te,np.flatnonzero(oof.fold==f)) and np.array_equal(tr,np.flatnonzero(oof.fold!=f)) and not set(df.study.iloc[tr])&set(df.study.iloc[te]))
            key=fingerprint({k:v for k,v in identity.items() if k!='audit_code_sha256'}|{'fold':f,'candidate':cid,'parameters':registry[cid],'train_ids':tr.tolist(),'test_ids':te.tolist()})
            predfile=cache/f'fold_{f}.npz';modelfile=cache/f'fold_{f}.joblib';metafile=cache/f'fold_{f}.json'
            if predfile.exists() and metafile.exists() and modelfile.exists():
                meta=json.loads(metafile.read_text());check(f'fold {f} exact cache identity and hashes',meta['key']==key and meta['predictions_sha256']==sha(predfile) and meta['model_sha256']==sha(modelfile))
                with np.load(predfile) as z:
                    check(f'fold {f} cached row IDs',np.array_equal(z['train_ids'],tr) and np.array_equal(z['test_ids'],te))
                    train_pred=z['train_prediction'];test_pred=z['test_prediction']
                bundle=joblib.load(modelfile)
            else:
                check('tests-only prohibits missing cache refit',not args.tests_only)
                check('completed audit prohibits any new fit',(OUT/'completed.json').exists() is False)
                authorization=(f,cid,key,tr)
                with warnings.catch_warnings(record=True) as warnings_seen:
                    warnings.simplefilter('always');predictor=ns['sklearn_model'](LGBMRegressor(**registry[cid]))(df.iloc[tr])
                    train_pred=predictor(df.iloc[tr]);test_pred=predictor(df.iloc[te])
                authorization=None;new_fits+=1
                closure=dict(zip(predictor.__code__.co_freevars,[c.cell_contents for c in predictor.__closure__]))
                bundle={'pipeline':closure['pipe'],'numeric_min':df.iloc[tr][nums].min(),'numeric_max':df.iloc[tr][nums].max(),'features':features,'numeric':nums,'parameters':registry[cid],'scenario':SCENARIO}
                joblib.dump(bundle,modelfile)
                with predfile.open('wb') as h:np.savez_compressed(h,train_ids=tr,test_ids=te,train_prediction=train_pred,test_prediction=test_pred)
                dump(metafile,{'key':key,'fold':f,'candidate':cid,'parameters':registry[cid],'predictions_sha256':sha(predfile),'model_sha256':sha(modelfile),'warnings':[str(w.message) for w in warnings_seen]})
                event({'event':'fit_completed','fold':f,'key':key})
            check(f'fold {f} all train/test predictions finite',np.isfinite(train_pred).all() and np.isfinite(test_pred).all())
            # Exact equality, not only rounded metric agreement, is required to attribute training metrics to the locked S2 model.
            check(f'fold {f} bitwise held-out agreement with Stage 4',np.array_equal(test_pred,oof.S2.iloc[te]))
            prep=bundle['pipeline'].steps[0][1];num=prep.named_transformers_['num']
            check(f'fold {f} imputer uses training median only',np.allclose(num.named_steps['simpleimputer'].statistics_,df.iloc[tr][nums].median()))
            filled=df.iloc[tr][nums].fillna(df.iloc[tr][nums].median()).to_numpy()
            check(f'fold {f} scaler uses training mean/variance only',np.allclose(num.named_steps['standardscaler'].mean_,filled.mean(axis=0)) and np.allclose(num.named_steps['standardscaler'].var_,filled.var(axis=0)))
            check(f'fold {f} serialized model reproduces seen predictions',np.array_equal(bundle['pipeline'].predict(df.iloc[tr][features]),train_pred))
            row={'fold':f,'candidate':cid,'train_rows':len(tr),'test_rows':len(te),'train_studies':df.study.iloc[tr].nunique(),'test_studies':df.study.iloc[te].nunique(),'scenario':SCENARIO,'max_test_prediction_difference':float(np.max(abs(test_pred-oof.S2.iloc[te].to_numpy())))}
            for split,ids,p in [('train',tr,train_pred),('test',te,test_pred)]:
                y=df.y.iloc[ids].to_numpy();err=p-y
                row.update({split+'_R2':float(r2_score(y,p)),split+'_MAE':float(mean_absolute_error(y,p)),split+'_RMSE':float(np.sqrt(mean_squared_error(y,p)))})
                check(f'fold {f} {split} independently recomputed R2',np.isclose(row[split+'_R2'],1-np.sum(err**2)/np.sum((y-y.mean())**2),rtol=1e-14,atol=1e-14))
            row['R2_train_minus_test']=row['train_R2']-row['test_R2'];rows.append(row)
    folds=pd.DataFrame(rows);folds.to_csv(OUT/'fold_training_testing_metrics.csv',index=False)
    pooled=float(r2_score(df.y,oof.S2));check('immutable S2 pooled held-out R2',pooled==.5256280075581679)
    # Verify all four existing summary tables from identical saved predictions; do not train other procedures.
    saved=csv(ROOT/'study_robustness_outputs/summary_metrics.csv').set_index('procedure');comp=[]
    for procedure in ['original_C0','previously_optimized','S1','S2']:
        y=df.y.to_numpy();p=oof[procedure].to_numpy();err=p-y
        computed={'R2':float(r2_score(y,p)),'MAE':float(abs(err).mean()),'RMSE':float(np.sqrt(np.mean(err**2))),
                  'study_macro_MAE':float(pd.Series(abs(err)).groupby(df.study).mean().mean())}
        for k,v in computed.items():check(procedure+' frozen '+k,np.isclose(v,saved.loc[procedure,k],rtol=1e-13,atol=1e-13))
        comp.append({'procedure':procedure,'scenario':SCENARIO,'rows':16988,'studies':72,**{'heldout_pooled_'+k:v for k,v in computed.items()},
                     'mean_fold_testing_R2':float(saved.loc[procedure,'mean_fold_R2']),
                     'mean_fold_training_R2':float(folds.train_R2.mean()) if procedure=='S2' else 'NOT_AVAILABLE',
                     'mean_fold_training_MAE':float(folds.train_MAE.mean()) if procedure=='S2' else 'NOT_AVAILABLE',
                     'mean_fold_training_RMSE':float(folds.train_RMSE.mean()) if procedure=='S2' else 'NOT_AVAILABLE',
                     'p90_study_MAE':float(saved.loc[procedure,'p90_study_MAE']),'worst_study_MAE':float(saved.loc[procedure,'worst_study_MAE']),
                     'within_1_log_percent':float(saved.loc[procedure,'within_1_log_percent']),
                     'external_R2':'NOT_AVAILABLE','external_status':'NO_EXTERNAL_DATA',
                     'original_random_5fold_R2_displayed':.764 if procedure=='original_C0' else 'NOT_AVAILABLE'})
    comparison=pd.DataFrame(comp);comparison.to_csv(OUT/'METRICS_COMPARISON.csv',index=False)
    notebook=json.loads((ROOT/'microplastics.ipynb').read_text());code='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
    comparison_cell=notebook['cells'][27];text='\n'.join(''.join(x.get('data',{}).get('text/plain',[])) for x in comparison_cell['outputs'])
    check('original near-0.7 value is displayed random OOF',any(line.split()[:2]==['LightGBM','0.764'] for line in text.splitlines()) and 'out_of_fold(model, scheme)' in ''.join(comparison_cell['source']))
    check('original random split definition verified','KFold(5, shuffle=True, random_state=SEED)' in code)
    dump(OUT/'original_notebook_metric_evidence.json',{'notebook':'microplastics.ipynb','zero_based_comparison_cell':27,'source':comparison_cell['source'],'saved_output_text':text,
        'original_resubstitution_training_R2':'NOT_AVAILABLE','verified_near_0_7_value':.764,'precision':'three-decimal notebook display only; unrounded random OOF score unavailable',
        'definition':'Pooled held-out predictions from shuffled row KFold(5, random_state=0), not training predictions; same studies may cross row folds.'})
    check('only five fresh fits total',sum(e['event']=='fit_started' for e in events())==5 and sum(e['event']=='fit_completed' for e in events())==5)
    after=protected();check('all protected file paths/content unchanged',after==before);dump(OUT/'protected_after.json',after)
    summary={'scenario':SCENARIO,'mean_training_R2':float(folds.train_R2.mean()),'mean_testing_R2':float(folds.test_R2.mean()),'pooled_heldout_R2':pooled,
             'mean_per_fold_R2_gap':float(folds.R2_train_minus_test.mean()),'new_fits_total':5,'new_fits_this_run':new_fits,
             'heldout_predictions_bitwise_identical':True,'original_training_R2':'NOT_AVAILABLE','original_random_CV_R2_displayed':.764,
             'tests_passed':len(checks),'protected_paths_verified':len(after),'external_R2':'NOT_AVAILABLE'}
    dump(OUT/'summary.json',summary);(OUT/'tests.txt').write_text('\n'.join('PASS '+c for c in checks)+f'\n{len(checks)} checks passed; new fits this run: {new_fits}.\n')
    dump(OUT/'completed.json',{'cache_only_after_completion':True,'code_sha256':sha(Path(__file__)),'fits_total':5})
    write_reports(folds,comparison,summary,markdown)
    print(json.dumps(summary,indent=2))


def write_reports(folds,comparison,summary,table):
    s=summary
    text=f'''# Stage 4 S2: training versus study-held-out testing

Scenario: `{SCENARIO}`. Target: `log10(concentration + 0.001)`. Historical Barrows/Dehm ×10^6 assumptions are unverified, not physical ground truth.

## Exact models and replication

No fitted prior LightGBM pipeline/booster artifacts were available: the old NPZ checkpoints contain held-out predictions and row IDs, not estimators. Exactly **five new fits** reconstructed S2's locked outer-fold models; no original model, inner grid or alternate parameter was trained. Folds 1–2 use C5_row_and_feature_subsample; folds 3–5 use C4_feature_subsample. Full parameters and original train-only preprocessing are preserved in checkpoint metadata. Seed 0, four threads, same core library versions.

**Every reconstructed held-out prediction matches the frozen Stage 4 S2 prediction bit-for-bit (maximum difference 0)**. Training scores therefore refer to the reconstructed locked S2 fold models, not a newly selected variant. Complete fitted pipelines with training clipping ranges and train/test predictions are saved locally for reuse. No cache in any prior output directory was changed.

## Counts and per-fold R²

{table(folds[['fold','candidate','train_rows','test_rows','train_studies','test_studies','train_R2','test_R2','R2_train_minus_test']])}

## MAE and RMSE (log10 units)

{table(folds[['fold','train_MAE','test_MAE','train_RMSE','test_RMSE']])}

- **Mean training R² across five folds: {s['mean_training_R2']:.12f}.**
- **Mean testing R² across five folds: {s['mean_testing_R2']:.12f}.**
- **Pooled study-held-out R² across all 16,988 rows: {s['pooled_heldout_R2']:.16f}.**
- Mean of the five training-minus-testing R² gaps: **{s['mean_per_fold_R2_gap']:.12f}**.

The training average is unweighted over the five training fits. Training sets overlap heavily; there is no unique once-per-row training prediction and no pooled training R² is reported. The held-out pool contains each eligible row exactly once across the five disjoint test partitions. There are 72 DOI/accession-linked studies; test sizes remain 6,135 / 2,714 / 2,712 / 2,714 / 2,713.

## Generalization gap and interpretation

Training scores are resubstitution on samples already used to fit the estimator and preprocessing, not validation. The large positive fold-wise gaps and negative testing R² in folds 1, 2 and 4 show much weaker performance when whole studies are withheld. They are consistent with overfitting/study-specific pattern learning and limited cross-study transfer. They cannot attribute all loss to model complexity: train/test populations differ in protocols, geography, target range and variance. Each R² uses its own set's mean/variance, so a gap is descriptive, not a calibrated estimate of pure algorithmic overfitting.

Positive pooled R² alongside negative mean fold R² is not a contradiction: pooled R² includes between-study target variation, whereas each fold's R² uses that fold's own denominator. The largest 6,135-row study alone occupies fold 1. Do not subtract mean training R² from pooled held-out R² and call that the mean generalization gap.

## Original notebook: was approximately 0.7 a training score?

**Original LightGBM training/resubstitution R²: NOT AVAILABLE / not verified.** No such saved metric was identified, and the original model was not retrained.

The identifiable saved LightGBM value is **0.764 (rounded to three decimals)** in original notebook cell 27 (zero-based), model comparison. Its exact definition is **pooled out-of-fold R² under shuffled row-wise KFold(5, random_state=0)**, on the same log10 target; the scoring cell calls `out_of_fold` and scores withheld rows, not the training rows. Samples from a study can appear on both sides of this random split, so it is easier than whole-study-held-out testing. The displayed 0.473 beside it is grouped held-out R². Thus **0.764 must not be called original training R² or compared as if it were the same split as S2's grouped result**. The unrounded random-CV value is not available in the saved output. Source/output evidence: `original_notebook_metric_evidence.json`; Python counterparts are `microplastics.py:302`, `:468-486`.

## Verification and rerun

{summary['tests_passed']} focused checks passed; {summary['protected_paths_verified']} pre-existing paths remain byte-identical. Tests cover immutable artifacts, exact rows/groups/targets/folds, no study overlap, exact held-out predictions, train-only imputer/scaler, saved-model prediction equivalence, independent R² calculation, all four reference metrics and the hard five-fit bound. Actual new fits in the full audit: **5**. `--tests-only` cannot fit.

```sh
PYTHONDONTWRITEBYTECODE=1 reproduction_outputs/venv/bin/python final_research_audit/analyze.py
PYTHONDONTWRITEBYTECODE=1 reproduction_outputs/venv/bin/python final_research_audit/analyze.py --tests-only
```

Default reruns reuse verified local caches; the completed marker prohibits additional fits. All new files are under `final_research_audit/`. No downloads, searches, target edits, chronological tests or independent external evaluations were run.
'''
    (OUT/'TRAINING_VS_TESTING.md').write_text(text)
    simple=comparison[['procedure','heldout_pooled_R2','heldout_pooled_MAE','heldout_pooled_RMSE','heldout_pooled_study_macro_MAE','mean_fold_testing_R2','p90_study_MAE','worst_study_MAE','within_1_log_percent']]
    assessment=f'''# Final research assessment

## Verdict

**Qualified readiness for a transparent exploratory methods/reproducibility manuscript, not for a validated marine-concentration prediction or generalization claim.** The project is auditable and can report useful negative and positive findings, but a publishable scientific argument must foreground unresolved labels, study confounding and repeated benchmark inspection. This is not an acceptance guarantee or proof of novelty. The modest same-family selector comparisons alone do not establish a new architecture or broad predictive superiority.

The defensible contribution is a reproducible study-aware benchmark and an analysis of reliability trade-offs under an explicitly hypothetical historical unit scenario. Stronger environmental or deployment claims are **not ready**. Make the scenario caveat prominent in title/abstract/methods/results, correct any original notebook wording that presents unit hypotheses as established fact in the manuscript (the protected source itself is unchanged), and publish complete folds, failures and provenance limitations.

## Four preserved held-out procedures

Scenario: `{SCENARIO}`; identical 16,988 rows and 72 studies. These are existing grouped cross-validation results, not four independent datasets.

{table(simple)}

S2 improves pooled R² by 0.052997168 over original C0 and 0.022246723 over the prior optimized procedure. Study-macro MAE is best for S1, not S2; S2's macro gain over optimized is small (~0.00206 log10). Both S1 and S2 pass the prior frozen descriptive screen but neither dominates every endpoint. Compared with original, S2 has worse worst-study MAE (5.165887 versus 4.943935) and lower within-one-log coverage (62.83% versus 63.94%). Compared with optimized, S2 still worsens 26 of 72 studies, including 10 by >0.10 log10 MAE; versus original it worsens 33, including 17 by that threshold (`study_robustness_outputs/study_harm_counts.csv`). This is not universal study-level robustness.

## Seen-data fit, held-out CV, and independent testing are distinct

1. **Training:** S2's mean fold resubstitution R² is {s['mean_training_R2']:.9f}. These predictions reuse training observations and fitted preprocessing, so this is optimistic goodness-of-fit, not validation.
2. **Historical grouped CV:** S2 pooled R² is {s['pooled_heldout_R2']:.16f}; mean fold R² is {s['mean_testing_R2']:.9f}. Fold-wise gaps and MAE/RMSE are in `TRAINING_VS_TESTING.md`. All four procedures have negative test R² on folds 1 and 4; optimized/S1/S2 also have negative fold 2. All four mean fold R² values remain negative. Positive pooled R² partly reflects between-study concentration variation and cannot mean reliable prediction for every new study.
3. **Independent external testing:** **NOT AVAILABLE for all four procedures.** Stage 5 status is NO_EXTERNAL_DATA, provenance remains unresolved, and no all-development research model or external evaluation exists. Do not relabel a re-scored historical fold, a training score or the 0.764 random-row OOF result as independent external validation.

The large seen/unseen gaps support concern about overfitting or recognition of study-specific conditions. Distribution shifts, different fold variance denominators and hidden measurement protocols also contribute; the gaps do not quantify the contribution of each mechanism.

## Integrity strengths and selection-bias limits

- Exact grouping through DOI/accession connected components and fixed outer study partitions; no study crosses an outer train/test boundary. Each test row is scored once. Median imputation, scaling, one-hot encoding and clipping use training data only; identifiers/outcomes/fold IDs are not predictors.
- Prior optimization selected configurations using inner study-held-out metrics; Stage 4 S1/S2 selected among the same eight configurations using only saved inner summaries with auditable locked rules. This final audit performs no new selection and reproduces S2's saved test predictions exactly.
- However, the outer results guided multiple rounds of research interpretation and candidate consideration. Nested selection within each procedure does **not** undo this between-experiment selection bias. S2 was named the final research candidate after repeated inspection of these same five folds. Improvements are exploratory; no valid independent significance or generalization claim follows from their rank.
- The 6,135-row study is 36.11% of the data and entirely determines fold 1. The Stage 4 decomposition shows only ~15.18% of S2's gain over optimized comes from that largest study, so its gain is not solely due to that study. Nonetheless pooled weighting and sparse environmental/method coverage constrain interpretation; macro/tail metrics must accompany pooled R².

## Measurement quality and scientific provenance

The NOAA export mixes media/units; the fixed benchmark filters ocean water in pieces/m³. Stage 5 documented 53 eligible rows missing depth, 4,958 numerical zeros, 643 rows sharing a measurement signature and 2,378 sharing date/location/method. Repeated signatures may be real replicates, not proven duplicate errors; no rows were removed. Zero interpretation, detection limits, blanks, recovery, filtered volume and synthetic-polymer confirmation are not adequately represented in the export.

**Barrows (1,389 rows) and Dehm (81 rows) ×10^6 multipliers remain unverified hypotheses.** SI conversion of particles/L to particles/m³ is ×1000; an extra factor of 1000 requires independently documented erroneous prior division and row-level source mapping. Neither predictive score improvement nor a plausible study mean establishes that. Primary methods were not accessible in the bounded Stage 5 audit. Barrows' grab-sample mesh-field value may describe a membrane pore, not actual net mesh; synthetic-only comparability and Dehm row-specific traceability remain unresolved. The 33 bongo-net observations are also flagged as unverified, not deleted or rescaled.

Thus the target is a **historical hypothetical numerical scenario**, not certified physical ground truth or verified true synthetic microplastic abundance. Apparent model skill may reflect differences in study protocol, particle detection/size range and units as well as environmental variation. Feature associations are not causal effects and cannot be claimed as maps of true marine plastic burden.

## Defensible claims and prerequisites for stronger claims

**Defensible now:** computational reproduction of stated scores and transformations; study-disjoint CV performance under this scenario; exact seen-data/held-out gaps; conditional exploratory trade-offs among fixed LightGBM configurations/selectors; unresolved scientific-provenance and external-validation limitations. Report every negative fold and harmed study and avoid describing R² as percent accuracy.

**Not defensible now:** independently superior new-study generalization; verified real-world concentration forecasts; certified unit corrections; causal geography/protocol effects; global monitoring/deployment readiness; a claim that all studies improve; or external R² of any value.

**Next scientific actions (recommendations only, not executed):** obtain primary Barrows/Dehm articles, supplements, original observations and NCEI ingest mappings; audit counts/volumes, synthetic confirmation, filter/mesh context, blanks and nondetects; resolve duplicate/campaign provenance; obtain genuinely new comparable studies and follow the locked Stage 5 protocol before outcomes are examined. If evidence changes the target definition, preregister a separate scientific scenario rather than rewriting this benchmark. Then one frozen S2 full-development model can be assessed on a truly independent set with study-cluster uncertainty, not another score-driven historical-fold iteration.

## Audit artifacts and limits

`METRICS_COMPARISON.csv` explicitly marks unmeasured training metrics for original/optimized/S1 and every external R² as NOT_AVAILABLE. The original notebook's identifiable 0.764 is rounded random-row five-fold held-out R², not a verified training score. No original model was retrained. All new calculations use the established environment and exact original pipeline; five new S2 fits were necessary because old checkpoint files saved predictions only.

**Actual new fits: 5; no tuning, searches, downloads, external datasets or follow-on phase.** Original files and all previous results remain unchanged. This audit ends here.
'''
    (OUT/'FINAL_RESEARCH_ASSESSMENT.md').write_text(assessment)


if __name__=='__main__':
    import fcntl
    with (OUT/'audit_process.lock').open('a') as process_lock:
        fcntl.flock(process_lock.fileno(),fcntl.LOCK_EX)
        main()

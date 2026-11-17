"""Zero-fit fixed-blend analysis and fail-closed conditional budget assessment.

Imports perform no I/O or fitting. Prediction blending has no target argument.
All writes are restricted by the runner to this phase's output directory.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

SCENARIO = 'historical_hypothetical_unit_scenario'
WEIGHTS = {'B0': 0., 'B25': .25, 'B50': .5, 'B75': .75, 'B100': 1.}
METRICS = ['R2','MAE','RMSE','study_macro_MAE','study_macro_MSE','median_study_MAE','p90_study_MAE','worst_study_MAE','within_1_log']


def read_csv(path):
    return pd.read_csv(path, float_precision='round_trip')


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda x: x.item() if isinstance(x,np.generic) else str(x)))


def sha(path):
    path=Path(path)
    if path.is_symlink(): return 'symlink:'+os.readlink(path)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def identity(payload):
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()


def require(condition, message):
    if not condition: raise ValueError(message)


def vectors(y, pred, groups):
    y=np.asarray(y,dtype=float); pred=np.asarray(pred,dtype=float); groups=np.asarray(groups)
    require(y.ndim==pred.ndim==groups.ndim==1 and len(y)==len(pred)==len(groups)>0,'Metric vectors must be nonempty aligned 1D arrays')
    require(np.isfinite(y).all() and np.isfinite(pred).all(),'Nonfinite targets/predictions: refusing to drop rows')
    require(not pd.isna(groups).any() and (groups.astype(str)!='').all(),'Missing study identifier: refusing to drop groups')
    return y,pred,groups


def blend_predictions(original, optimized):
    """Target-free fixed arithmetic; no learned, study-specific or row-specific weights."""
    a=np.asarray(original,dtype=float); b=np.asarray(optimized,dtype=float)
    require(a.ndim==b.ndim==1 and a.shape==b.shape and len(a)>0,'Blend prediction arrays must align')
    require(np.isfinite(a).all() and np.isfinite(b).all(),'Nonfinite base prediction')
    return {key:(1-w)*a+w*b for key,w in WEIGHTS.items()}


def metrics(y,pred,groups):
    y,pred,groups=vectors(y,pred,groups)
    e=pred-y
    denom=np.sum((y-y.mean())**2)
    require(denom>0,'R2 undefined for constant target; refusing silent omission')
    study=pd.DataFrame({'study':groups,'ae':abs(e),'se':e**2}).groupby('study',sort=True).mean()
    return {'R2':float(1-np.sum(e**2)/denom),'MAE':float(abs(e).mean()),'RMSE':float(np.sqrt(np.mean(e**2))),
            'study_macro_MAE':float(study.ae.mean()),'study_macro_MSE':float(study.se.mean()),
            'median_study_MAE':float(study.ae.median()),'p90_study_MAE':float(study.ae.quantile(.9)),
            'worst_study_MAE':float(study.ae.max()),'within_1_log':float((abs(e)<=1).mean())}


def validate_alignment(original,optimized):
    require(len(original)==len(optimized)==16988,'Expected exactly 16988 rows in both OOF files')
    require(not original.isna().any().any() and not optimized.isna().any().any(),'Missing values in input OOF artifacts')
    for col in ['water_row','source_row']:
        require(original[col].is_unique and optimized[col].is_unique,f'Duplicate {col}')
    for col in ['water_row','source_row','study','doi','accession','fold','target_log10']:
        require(np.array_equal(original[col],optimized[col]),f'OOF mismatch: {col}')
    require(np.array_equal(original.water_row,np.arange(16988)),'Changed modelling row order')
    require(np.array_equal(original.prediction_log10,optimized.original_prediction_log10),'Baseline prediction copy differs')
    require(optimized.scenario.eq(SCENARIO).all(),'Unexpected target/unit scenario')
    require(original.study.nunique()==72,'Expected 72 linked studies')
    require(original.groupby('fold').size().to_dict()==dict(zip(range(1,6),[6135,2714,2712,2714,2713])),'Changed outer fold sizes')
    require(original.groupby('study').fold.nunique().eq(1).all(),'Study crosses outer validation folds')
    vectors(optimized.target_log10,optimized.original_prediction_log10,optimized.study)
    vectors(optimized.target_log10,optimized.optimized_prediction_log10,optimized.study)


def gate(main,optimized,integrity_ok=True):
    improvements={'pooled_MAE':main['MAE']<optimized['MAE'],
                  'p90_study_MAE':main['p90_study_MAE']<optimized['p90_study_MAE'],
                  'within_1_log':main['within_1_log']>optimized['within_1_log']}
    conditions={'R2_loss_no_more_than_0.015':main['R2']>=optimized['R2']-.015,
                'study_macro_MAE_does_not_increase':main['study_macro_MAE']<=optimized['study_macro_MAE'],
                'at_least_two_weaknesses_improve':sum(improvements.values())>=2,
                'worst_study_MAE_increase_at_most_0.10':main['worst_study_MAE']<=optimized['worst_study_MAE']+.10,
                'no_integrity_issue':bool(integrity_ok)}
    return {'scenario':SCENARIO,'candidate':'B50','comparison':'B100','passed':all(conditions.values()),
            'conditions':{k:bool(v) for k,v in conditions.items()},'weakness_improvements':{k:bool(v) for k,v in improvements.items()},
            'deltas_B50_minus_B100':{k:float(main[k]-optimized[k]) for k in METRICS},
            'failed_conditions':[k for k,v in conditions.items() if not v],
            'interpretation':'Cost-control screen on previously inspected outer outcomes; not independent confirmation.'}


def enforce_budget(estimated_new_fits,cap=100):
    require(isinstance(estimated_new_fits,int) and estimated_new_fits>=0,'Invalid fit estimate')
    if estimated_new_fits>cap: raise RuntimeError(f'Budget blocks Stage C: {estimated_new_fits} > {cap}; no fitting authorized')
    return True


def differences(frame,keys,columns):
    """Paired differences on identical slices, no filtering of hard observations."""
    result=frame.copy()
    for ref in ['B0','B100']:
        reference=frame[frame.blend==ref]
        for metric in columns:
            if keys:
                index=pd.MultiIndex.from_frame(result[keys]) if len(keys)>1 else result[keys[0]]
                lookup=reference.set_index(keys)[metric]
                base=lookup.reindex(index).to_numpy()
            else: base=reference[metric].iloc[0]
            result[f'delta_vs_{ref}__{metric}']=result[metric].to_numpy()-base
    require(not result.isna().any().any(),'Paired metric alignment generated missing values')
    return result


class Analysis:
    def __init__(self,out):
        self.out=Path(out).resolve();self.root=self.out.parent
        require(self.root.name=='marine-microplastics-baseline','Wrong repository')
        self.checks=[]

    def check(self,name,condition):
        require(condition,name);self.checks.append(name)

    def load(self):
        r,o=self.root/'reproduction_outputs',self.root/'optimization_outputs'
        self.plan=json.loads((self.out/'predeclared_plan.json').read_text())
        self.check('predeclared registry exactly five fixed weights and B50 main',self.plan['weights_optimized']==WEIGHTS and self.plan['main_candidate']=='B50')
        # Revalidate manifests instead of trusting filenames alone.
        for file,key in [(r/'input_hashes.json',None),(o/'input_artifacts.json','relative_paths_sha256'),(o/'artifact_checksums.json',None)]:
            manifest=json.loads(file.read_text());manifest=manifest[key] if key else manifest
            self.check('prior manifest verified: '+str(file.relative_to(self.root)),all(sha(self.root/p)==h for p,h in manifest.items()))
        paths=['reproduction_outputs/out_of_fold_predictions.csv','reproduction_outputs/summary.json','reproduction_outputs/fold_metrics.csv',
               'reproduction_outputs/model_parameters.json','reproduction_outputs/unit_adjustments.json','reproduction_outputs/input_hashes.json',
               'optimization_outputs/optimized_oof_predictions.csv','optimization_outputs/optimization_summary.json','optimization_outputs/frozen_splits.json',
               'optimization_outputs/candidate_registry.json','optimization_outputs/checkpoint_identity.json','optimization_outputs/paired_per_study_metrics.csv',
               'optimization_outputs/outer_selections.csv','optimization_outputs/reproduction_and_integrity_checks.json','optimization_outputs/fit_log.jsonl',
               'optimization_outputs/source/experiment.py','microplastics.py','microplastics.ipynb','data/noaa_microplastics.csv','uv.lock',
               'optimization_outputs/LightGBM_0503_Leakage_Safe_Blending_and_Robustness_Codex_Prompt.md']
        self.input_hashes={p:sha(self.root/p) for p in paths}
        dump(self.out/'input_artifacts.json',self.input_hashes)
        self.original=read_csv(r/'out_of_fold_predictions.csv');self.d=read_csv(o/'optimized_oof_predictions.csv')
        validate_alignment(self.original,self.d)
        self.check('exact OOF alignment, finite targets/predictions and scenario',True)
        self.reference=json.loads((o/'optimization_summary.json').read_text())
        self.splits=json.loads((o/'frozen_splits.json').read_text())
        for f,(tr,te) in enumerate(self.splits['outer'],1):
            self.check(f'outer {f} frozen memberships match saved OOF',np.array_equal(te,np.flatnonzero(self.d.fold==f)) and np.array_equal(tr,np.flatnonzero(self.d.fold!=f)))
        # Invoke ONLY the old read-only loader, not Experiment.load/tune/refit or any prior runner.
        spec=importlib.util.spec_from_file_location('protected_original_helpers',o/'source/experiment.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.ns=module.original_namespace(self.root)
        df=self.ns['df']
        self.check('original target/group/family exactly reconstructed without fitting',np.array_equal(df.y,self.d.target_log10) and np.array_equal(df.study,self.d.study) and np.array_equal(df.method_family,self.d.method_family))
        raw=self.ns['data'];mask=(raw.medium=='Ocean water')&(raw.unit=='pieces/m3')
        self.check('original source-row selection exact',np.array_equal(raw.index[mask],self.d.source_row))
        adjustments=json.loads((r/'unit_adjustments.json').read_text())
        self.check('historical multipliers and counts frozen',self.ns['UNIT_FIXES']=={v['doi']:v['multiplier'] for v in adjustments} and [int((df.doi==v['doi']).sum()) for v in adjustments]==[1389,81] and self.ns['C']==.001)
        self.check('study sizes agree with complete evaluation data',np.array_equal(self.d.study_size,self.d.study.map(self.d.study.value_counts())))
        prior=read_csv(o/'paired_per_study_metrics.csv')
        self.prior_harmed=prior.loc[prior.delta_MAE>.10,'study'].tolist()
        self.check('14 prior harmed studies retained solely for diagnostics',len(self.prior_harmed)==14 and set(self.prior_harmed)<=set(self.d.study))
        prior_identity=json.loads((o/'checkpoint_identity.json').read_text())
        self.check('prior optimized code/splits/config hashes valid',sha(o/'source/experiment.py')==prior_identity['code'] and sha(o/'frozen_splits.json')==prior_identity['splits'] and sha(o/'candidate_registry.json')==prior_identity['registry'])
        return self.d.head()

    def evaluate(self):
        d=self.d.copy()
        for name,p in blend_predictions(d.original_prediction_log10,d.optimized_prediction_log10).items():d[name]=p
        self.d=d
        d.to_csv(self.out/'aligned_oof_predictions.csv',index=False)
        pooled=[]
        for name,w in WEIGHTS.items():
            rec={'blend':name,'weight_optimized':w,'rows':len(d),'studies':d.study.nunique(),'scenario':SCENARIO,**metrics(d.target_log10,d[name],d.study)}
            rec['mean_fold_R2']=float(np.mean([metrics(g.target_log10,g[name],g.study)['R2'] for _,g in d.groupby('fold')]))
            pooled.append(rec)
        self.pooled=differences(pd.DataFrame(pooled),[],METRICS+['mean_fold_R2'])
        self.pooled.to_csv(self.out/'fixed_blend_metrics.csv',index=False)
        audit=[]
        for name,ref in [('B0','original_reproduced_fixed'),('B100','nested_optimized_procedure')]:
            actual=self.pooled.set_index('blend').loc[name]
            for k,v in self.reference[ref].items():
                delta=float(actual[k]-v)
                self.check(f'reference {name} {k} strict tolerance',np.isclose(actual[k],v,rtol=1e-12,atol=1e-12))
                audit.append({'blend':name,'metric':k,'recomputed':actual[k],'reference':v,'difference':delta})
        # Original reproduction includes additional raw-backtransform diagnostics: verify, never optimize on them.
        rep=json.loads((self.root/'reproduction_outputs/summary.json').read_text())
        mapping={'R2':'R2','MAE':'MAE','RMSE':'RMSE','within 1 log':'within_1_log','study MAE':'study_macro_MAE'}
        for k,v in rep['pooled'].items():
            if k in mapping: actual=float(self.pooled.set_index('blend').loc['B0',mapping[k]])
            else:
                err=np.clip(10**d.B0.to_numpy()-.001,0,None)-(10**d.target_log10.to_numpy()-.001)
                actual=float(abs(err).mean() if k=='raw MAE' else np.sqrt(np.mean(err**2)))
            self.check('original reproduction additional '+k,np.isclose(actual,v,rtol=1e-12,atol=1e-12))
            audit.append({'blend':'B0','metric':k,'recomputed':actual,'reference':v,'difference':actual-v})
        pd.DataFrame(audit).to_csv(self.out/'reference_metric_audit.csv',index=False)
        rows=[]
        for f,g in d.groupby('fold'):
            for name in WEIGHTS:rows.append({'fold':f,'blend':name,'rows':len(g),'studies':g.study.nunique(),'scenario':SCENARIO,**metrics(g.target_log10,g[name],g.study)})
        self.fold=differences(pd.DataFrame(rows),['fold'],METRICS)
        self.fold.to_csv(self.out/'paired_fold_metrics.csv',index=False)
        rows=[]
        for study,g in d.groupby('study'):
            for name in WEIGHTS:
                e=g[name].to_numpy()-g.target_log10.to_numpy()
                rows.append({'study':study,'fold':int(g.fold.iloc[0]),'rows':len(g),'blend':name,'scenario':SCENARIO,
                             'prior_harmed_diagnostic_only':study in self.prior_harmed,'MAE':float(abs(e).mean()),'MSE':float(np.mean(e**2)),
                             'RMSE':float(np.sqrt(np.mean(e**2))),'within_1_log':float((abs(e)<=1).mean())})
        self.study=differences(pd.DataFrame(rows),['study'],['MAE','MSE','RMSE','within_1_log'])
        self.study.to_csv(self.out/'paired_study_metrics.csv',index=False)
        rows=[]
        for group in ['method_family','study_size_bin']:
            for value,g in d.groupby(group):
                for name in WEIGHTS:rows.append({'grouping':group,'group':value,'blend':name,'rows':len(g),'studies':g.study.nunique(),'scenario':SCENARIO,**metrics(g.target_log10,g[name],g.study)})
        self.subgroups=differences(pd.DataFrame(rows),['grouping','group'],METRICS)
        self.subgroups.to_csv(self.out/'method_and_study_size_metrics.csv',index=False)
        self.study[self.study.prior_harmed_diagnostic_only].to_csv(self.out/'prior_14_harmed_studies_diagnostic.csv',index=False)
        self.fold[self.fold.fold.isin([2,5])].to_csv(self.out/'folds_2_and_5_diagnostic.csv',index=False)
        self.check('all-case study and fold outputs complete',len(self.study)==72*5 and len(self.fold)==5*5 and self.study.groupby('blend').rows.sum().eq(16988).all())
        return self.pooled

    def screen(self):
        t=self.pooled.set_index('blend')
        self.gate=gate(t.loc['B50'],t.loc['B100'])
        dump(self.out/'stage_b_gate.json',self.gate)
        return self.gate

    def assess_stage_c(self):
        plan=self.plan['stage_c']; cap=plan['hard_fit_cap']
        if not self.gate['passed']:
            self.stage_c={'scenario':SCENARIO,'status':'skipped_gate_failed','ran':False,'new_fits_started':0,'cap':cap,
                          'reason':'Stage B failed: '+', '.join(self.gate['failed_conditions']),
                          'partition_checks':'Not needed: mandatory stop at Stage A; no new training partitions used.'}
        else:
            # Exact original 3-way inner splits plus 3-way inner-inner selection over 8 fixed candidates.
            # This audits feasibility only. No fitting, no use of labels, and no altered folds to save fits.
            audits=[]; planned=[]; groups=self.d.study.to_numpy()
            for outer,(tr,te) in enumerate(self.splits['outer'],1):
                for inner,(a,b) in enumerate(self.splits['inner'][str(outer)],1):
                    a=np.asarray(a);b=np.asarray(b)
                    self.check(f'Stage C planned {outer}/{inner}: training scope and study disjointness',set(a)|set(b)==set(tr) and not set(groups[a])&set(groups[b]) and not (set(a)|set(b))&set(te))
                    if len(set(groups[a]))<3:
                        planned.append({'outer':outer,'inner':inner,'fallback':'C0 due to fewer than 3 groups'})
                        continue
                    for nested,(p,q) in enumerate(GroupKFold(3).split(a,groups=groups[a]),1):
                        train,val=a[p],a[q]
                        self.check(f'Stage C planned {outer}/{inner}/{nested}: nested groups disjoint',not set(groups[train])&set(groups[val]) and len(set(groups[train]))>1 and not (set(train)|set(val))&set(b))
                        audits.append({'outer':outer,'inner':inner,'inner_inner':nested,'train_rows':len(train),'validation_rows':len(val),
                                       'train_studies':len(set(groups[train])),'validation_studies':len(set(groups[val])),'scenario':SCENARIO})
                        planned.append({'outer':outer,'inner':inner,'inner_inner':nested,'train_rows':train.tolist(),'validation_rows':val.tolist(),
                                        'train_study_ids':sorted(set(groups[train])),'validation_study_ids':sorted(set(groups[val]))})
            inner_inner=len(audits)*plan['candidates']
            # Conservative no-cache upper bound: 360 candidate + 30 cross-fit base + 10 outer base fits.
            cross_fits=2*5*3;outer_refits=2*5
            bound=inner_inner+cross_fits+outer_refits
            pd.DataFrame(audits).to_csv(self.out/'stage_c_planned_partition_counts.csv',index=False)
            dump(self.out/'stage_c_planned_splits.json',planned)
            self.check('expected complete 45 planned inner-inner partitions',len(audits)==45)
            try: enforce_budget(bound,cap)
            except RuntimeError as e:
                self.stage_c={'scenario':SCENARIO,'status':'skipped_budget_exceeded','ran':False,'new_fits_started':0,'cap':cap,
                              'upper_bound_new_fits':bound,'components':{'inner_inner_candidates':inner_inner,'inner_crossfit_base_refits':cross_fits,'outer_base_refits':outer_refits},
                              'even_reusing_all_compatible_prior_base_predictions_inner_inner_fits':inner_inner,
                              'reason':str(e),'partition_checks':'Passed without fitting; no group-count fallback needed.',
                              'incompatible_reuse_warning':'Prior chosen configurations used inner validation outcomes. They cannot provide selection-independent optimized predictions for those same rows; inner-inner selection is essential.'}
            else: raise RuntimeError('Unexpected in-budget Stage C design: stop for implementation review, never silently fit')
        dump(self.out/'stage_c_status.json',self.stage_c)
        # No model starts are logged. Decision event is explicitly not a fit.
        (self.out/'fit_log.jsonl').write_text(json.dumps({'event':'stage_c_not_run','status':self.stage_c['status'],'new_fits_started':0,'scenario':SCENARIO})+'\n')
        return self.stage_c

    def checkpoint_identity(self):
        payload={'input_hashes':self.input_hashes,'source':{str(p.relative_to(self.out)):sha(p) for p in sorted((self.out/'source').glob('*.py'))},
                 'test_source':{str(p.relative_to(self.out)):sha(p) for p in sorted((self.out/'tests').glob('*.py'))},
                 'plan_sha256':sha(self.out/'predeclared_plan.json'),'environment':json.loads((self.out/'environment.json').read_text()),
                 'folds_sha256':sha(self.root/'optimization_outputs/frozen_splits.json'),'scenario':SCENARIO,
                 'cache_policy':'No model checkpoints or training reused; zero-fit arithmetic recomputed on each run. Prior artifacts trusted only after exact hash verification.'}
        payload['identity_sha256']=identity(payload)
        dump(self.out/'checkpoints/analysis_identity.json',payload)
        return payload

    def verify_protected(self):
        path=self.out/'integrity_manifest.json';m=json.loads(path.read_text());before=m['before']
        paths={str(p.relative_to(self.root)) for p in self.root.rglob('*') if (p.is_file() or p.is_symlink()) and p.relative_to(self.root).parts[0]!='blending_outputs'}
        self.check('no new or removed paths outside blending_outputs',paths==set(before))
        after={p:sha(self.root/p) for p in before}
        self.check('all protected files unchanged including both prior output directories',after==before)
        m.update(after_verified=True,protected_path_count=len(before),after_sha256_manifest=after)
        dump(path,m)
        return len(before)

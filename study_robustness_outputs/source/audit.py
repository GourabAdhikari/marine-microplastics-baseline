"""Read-only provenance audit, checkpoint reconstruction and pre-fit selection lock."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from common import SCENARIO,sha,dump,csv,require,fingerprint,load_module,metrics
from selection_rules import FIELDS,RULES,select_all


class Context:
    def __init__(self,out):
        self.out=Path(out).resolve();self.root=self.out.parent
        require(self.root.name=='marine-microplastics-baseline','Wrong repository')
        self.checks=[]

    def check(self,label,ok):
        require(ok,label);self.checks.append(label)

    def load(self):
        r=self.root/'reproduction_outputs';o=self.root/'optimization_outputs'
        paths=['microplastics.py','microplastics.ipynb','data/noaa_microplastics.csv','uv.lock',
               'reproduction_outputs/out_of_fold_predictions.csv','reproduction_outputs/summary.json','reproduction_outputs/model_parameters.json','reproduction_outputs/unit_adjustments.json',
               'optimization_outputs/optimized_oof_predictions.csv','optimization_outputs/optimization_summary.json','optimization_outputs/inner_metrics_by_outer_fold.csv',
               'optimization_outputs/candidate_registry.json','optimization_outputs/outer_selections.csv','optimization_outputs/frozen_splits.json',
               'optimization_outputs/checkpoint_identity.json','optimization_outputs/fit_log.jsonl','optimization_outputs/source/experiment.py',
               'blending_outputs/LightGBM_0503_Study_Robustness_Stage4_Codex_Prompt.md']
        missing=[p for p in paths if not (self.root/p).is_file()]
        require(not missing,'Essential artifacts absent; restore exact prior files, do not refit: '+str(missing))
        self.inputs={p:sha(self.root/p) for p in paths};dump(self.out/'input_artifacts.json',self.inputs)
        for p,key in [(r/'input_hashes.json',None),(o/'input_artifacts.json','relative_paths_sha256'),(o/'artifact_checksums.json',None)]:
            m=json.loads(p.read_text());m=m[key] if key else m
            self.check('prior manifest '+str(p.relative_to(self.root)),all(sha(self.root/q)==h for q,h in m.items()))
        self.base=csv(r/'out_of_fold_predictions.csv');self.opt=csv(o/'optimized_oof_predictions.csv')
        self.check('16988 rows without missing/duplicate OOF IDs',len(self.base)==len(self.opt)==16988 and self.base.water_row.is_unique and self.opt.water_row.is_unique and self.base.source_row.is_unique and self.opt.source_row.is_unique and not self.base.isna().any().any() and not self.opt.isna().any().any())
        for col in ['water_row','source_row','study','doi','accession','fold','target_log10']:
            self.check('OOF identical '+col,np.array_equal(self.base[col],self.opt[col]))
        self.check('original prediction copy and scenario exact',np.array_equal(self.base.prediction_log10,self.opt.original_prediction_log10) and self.opt.scenario.eq(SCENARIO).all())
        self.check('contiguous row order and 72 studies',np.array_equal(self.base.water_row,np.arange(16988)) and self.base.study.nunique()==72)
        self.registry_document=json.loads((o/'candidate_registry.json').read_text());self.registry=self.registry_document['configs']
        self.params0=json.loads((r/'model_parameters.json').read_text())
        self.check('C0 and frozen eight models exact',len(self.registry)==8 and self.registry['C0_original']==self.params0 and self.registry_document['scenario']==SCENARIO)
        self.previous_selections=csv(o/'outer_selections.csv')
        self.splits=json.loads((o/'frozen_splits.json').read_text())
        self.outer=[(np.array(a),np.array(b)) for a,b in self.splits['outer']]
        self.inner={int(k):[(np.array(a),np.array(b)) for a,b in v] for k,v in self.splits['inner'].items()}
        self.saved_inner=csv(o/'inner_metrics_by_outer_fold.csv')
        require(set(FIELDS)<=set(self.saved_inner.columns),'S1/S2 not executable: mandatory inner fields missing; restore correctly keyed inner artifacts. No inner fits permitted.')
        self.check('complete finite inner summaries',len(self.saved_inner)==40 and self.saved_inner[FIELDS].notna().all().all() and np.isfinite(self.saved_inner[FIELDS[3:]].to_numpy()).all())
        self.prior_identity=json.loads((o/'checkpoint_identity.json').read_text())
        self.environment=json.loads((self.out/'environment.json').read_text())
        self.check('core versions match optimized cache identity',all(self.environment['packages'][p]==v for p,v in self.prior_identity['versions'].items()))
        self.check('original optimized code/config/split identity verified',sha(o/'source/experiment.py')==self.prior_identity['code'] and sha(o/'frozen_splits.json')==self.prior_identity['splits'] and sha(o/'candidate_registry.json')==self.prior_identity['registry'])
        self.old=load_module('readonly_original_helpers',o/'source/experiment.py')
        self.ns=self.old.original_namespace(self.root);self.df=self.ns['df'];self.X=self.df[self.ns['FEATURES']].copy();self.y=self.df.y.to_numpy();self.groups=self.df.study.to_numpy()
        self.check('original preprocessing produces exact frozen targets, groups and methods',np.array_equal(self.y,self.base.target_log10) and np.array_equal(self.groups,self.base.study) and np.array_equal(self.df.method_family,self.opt.method_family))
        self.check('exact feature whitelist, no identifiers or outcomes',list(self.X.columns)==['lat','lon','year','month_sin','month_cos','log_mesh','log_depth','method_family'])
        raw=self.ns['data'];mask=(raw.medium=='Ocean water')&(raw.unit=='pieces/m3')
        self.check('source-row selection preserved',np.array_equal(raw.index[mask],self.base.source_row))
        unit=json.loads((r/'unit_adjustments.json').read_text())
        self.check('unit assumptions and offset frozen',self.ns['C']==.001 and self.ns['UNIT_FIXES']=={v['doi']:v['multiplier'] for v in unit} and [int((self.df.doi==v['doi']).sum()) for v in unit]==[1389,81])
        self.check('outer test sizes exact',[len(b) for a,b in self.outer]==[6135,2714,2712,2714,2713])
        for f,(tr,te) in enumerate(self.outer,1):
            self.check(f'outer {f} exact membership and no study leakage',np.array_equal(te,np.flatnonzero(self.base.fold==f)) and np.array_equal(tr,np.flatnonzero(self.base.fold!=f)) and not set(self.groups[tr])&set(self.groups[te]))
            reconstructed=[(tr[a],tr[b]) for a,b in GroupKFold(3).split(self.X.iloc[tr],groups=self.groups[tr])]
            self.check(f'inner {f} original memberships',all(np.array_equal(a,c) and np.array_equal(b,d) for (a,b),(c,d) in zip(self.inner[f],reconstructed)))
            for j,(a,b) in enumerate(self.inner[f],1):
                self.check(f'inner {f}/{j} training-only study partitions',set(a)|set(b)==set(tr) and not set(a)&set(b) and not set(self.groups[a])&set(self.groups[b]) and len(set(self.groups[a]))>1)
        return self.saved_inner[FIELDS]

    def prior_key(self,stage,f,j,cid,tr,te):
        return fingerprint(self.prior_identity|{'stage':stage,'outer':f,'inner':j,'candidate':cid,
                           'train_ids':tr.tolist(),'test_ids':te.tolist(),'train_y_sha':hashlib.sha256(self.y[tr].tobytes()).hexdigest()})

    def verify_inner_cache(self):
        """Re-score saved predictions, never refit: verify all rows/studies in all 40 summaries."""
        o=self.root/'optimization_outputs'
        events=[json.loads(l) for l in (o/'fit_log.jsonl').read_text().splitlines()]
        completed={e['key']:e for e in events if e['event']=='completed'}
        self.cache_hashes={};audit=[];max_delta=0.
        for f,(tr,te) in enumerate(self.outer,1):
            self.check(f'inner outer-{f} every training row validated once',np.array_equal(np.sort(np.concatenate([b for a,b in self.inner[f]])),tr))
            for cid in self.registry:
                pred=pd.Series(np.nan,index=tr,dtype=float)
                for j,(a,b) in enumerate(self.inner[f],1):
                    key=self.prior_key('inner',f,j,cid,a,b);path=o/'checkpoints'/f'{key}.npz'
                    require(path.is_file() and key in completed,f'Missing compatible inner checkpoint {path}; no inner grid rerun allowed')
                    e=completed[key]
                    require(e['stage']=='inner' and e['outer']==f and e['inner']==j and e['candidate']==cid,'Checkpoint log scope mismatch')
                    with np.load(path) as z:
                        require(np.array_equal(z['test_ids'],b) and np.isfinite(z['pred']).all() and z['pred'].shape==b.shape,'Inner checkpoint coverage/finite failure')
                        pred.loc[b]=z['pred']
                    self.cache_hashes[str(path.relative_to(self.root))]=sha(path)
                require(np.isfinite(pred).all(),'Incomplete inner OOF, refusing to drop rows')
                rec=metrics(self.y[tr],pred.to_numpy(),self.groups[tr])
                match=self.saved_inner[(self.saved_inner.outer==f)&(self.saved_inner.candidate==cid)]
                require(len(match)==1,'Missing/duplicate inner candidate summary')
                saved=match.iloc[0]
                delta=max(abs(rec[k]-float(saved[k])) for k in rec);max_delta=max(max_delta,delta)
                self.check(f'inner metrics exact within 1e-12: {f}/{cid}',all(np.isclose(rec[k],saved[k],rtol=1e-12,atol=1e-12) for k in rec))
                audit.append({'outer':f,'candidate':cid,'evaluated_rows':len(tr),'evaluated_studies':len(set(self.groups[tr])),
                              'max_absolute_metric_difference':delta,'scenario':SCENARIO,**rec})
        # Verify exact optimized outer-cache identity and prediction equality, without fitting.
        for f,(tr,te) in enumerate(self.outer,1):
            cid=self.previous_selections.loc[self.previous_selections.outer==f,'candidate'].iloc[0]
            key=self.prior_key('outer',f,0,cid,tr,te);path=o/'checkpoints'/f'{key}.npz'
            require(key in completed and path.is_file(),'Optimized outer checkpoint missing')
            with np.load(path) as z:
                self.check(f'optimized reuse verified fold {f}',np.array_equal(z['test_ids'],te) and np.array_equal(z['pred'],self.opt.optimized_prediction_log10.iloc[te]) and self.opt.selected_candidate.iloc[te].eq(cid).all())
            self.cache_hashes[str(path.relative_to(self.root))]=sha(path)
        pd.DataFrame(audit).to_csv(self.out/'inner_cache_audit.csv',index=False)
        dump(self.out/'verified_prior_checkpoint_hashes.json',self.cache_hashes)
        self.inner_audit={'inner_checkpoints_verified':120,'optimized_outer_checkpoints_verified':5,'new_fits':0,'max_metric_difference':max_delta,
                          'provenance':'Exact prior key rebuilt from code/data/splits/configs/versions and training-only labels; inner validation covers every outer-training row once; all study groups counted.'}
        dump(self.out/'inner_audit_summary.json',self.inner_audit)
        return self.inner_audit

    def lock(self):
        chosen,eligibility=select_all(self.saved_inner[FIELDS].copy())
        payload={'version':1,'scenario':SCENARIO,'rules':RULES,'selections':chosen,'eligibility':eligibility.to_dict(orient='records'),
                 'source_hashes':{p:sha(self.out/'source'/p) for p in ['selection_rules.py','common.py']},
                 'inner_table_sha256':sha(self.root/'optimization_outputs/inner_metrics_by_outer_fold.csv'),
                 'candidate_registry_sha256':sha(self.root/'optimization_outputs/candidate_registry.json'),
                 'splits_sha256':sha(self.root/'optimization_outputs/frozen_splits.json'),
                 'verified_inner_cache_hashes_sha256':sha(self.out/'verified_prior_checkpoint_hashes.json'),
                 'outer_labels_or_outer_predictions_used_for_selection':False,
                 'zero_baseline_error_policy':'No division: multiply bounds; zero C0 error permits only zero corresponding candidate error. Nonfinite or negative error metrics fail safely.',
                 'screening_after_evaluation_only':{'R2_min':.5033812847921568-.015,'macro_MAE_max':'previous optimized','p90_MAE_max':'previous optimized','worst_MAE_max':'previous optimized'}}
        path=self.out/'locked_selections.json'
        if path.exists():
            locked=json.loads(path.read_text());self.check('selection lock immutable on resume',locked['payload']==payload and locked['payload_sha256']==fingerprint(payload))
        else:
            require(not any(json.loads(l).get('event')=='fit_started' for l in (self.out/'model_fit_log.jsonl').read_text().splitlines()),'Cannot create pre-fit lock after fitting')
            locked={'created_utc':datetime.now(timezone.utc).isoformat(),'payload_sha256':fingerprint(payload),'payload':payload}
            dump(path,locked);dump(self.out/'locked_selections_v1.json',locked)
            dump(self.out/'lock_sha256.json',{'sha256':sha(path),'created_utc':locked['created_utc']})
        self.locked=locked;self.selections=pd.DataFrame(chosen)
        eligibility.to_csv(self.out/'selector_eligibility.csv',index=False)
        self.selections.to_csv(self.out/'selected_candidates.csv',index=False)
        self.check('locked JSON and versioned copy unchanged',sha(path)==json.loads((self.out/'lock_sha256.json').read_text())['sha256'] and path.read_bytes()==(self.out/'locked_selections_v1.json').read_bytes())
        return self.selections[['rule','outer','candidate','eligible_count','contender_count']]

    def verify_protected(self):
        before=json.loads((self.out/'protected_hashes_before.json').read_text())
        paths={str(p.relative_to(self.root)) for p in self.root.rglob('*') if (p.is_file() or p.is_symlink()) and p.relative_to(self.root).parts[0]!='study_robustness_outputs'}
        self.check('no paths added/removed outside study_robustness_outputs',paths==set(before))
        after={p:sha(self.root/p) for p in before}
        self.check('all original and prior-phase files byte-identical',after==before)
        dump(self.out/'protected_hashes_after.json',after)
        dump(self.out/'integrity_result.json',{'protected_paths':len(before),'unchanged':True,'before_manifest_sha256':sha(self.out/'protected_hashes_before.json'),'after_manifest_sha256':sha(self.out/'protected_hashes_after.json')})
        return len(before)

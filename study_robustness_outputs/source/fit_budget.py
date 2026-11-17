"""Hard fit-call guard, complete checkpoint identity, and frozen outer refits."""
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import json
import time
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from common import SCENARIO,ORIGINAL,OPTIMIZED,dump,sha,fingerprint,require


def ensure_available(started,cap=10):
    if started>=cap:raise RuntimeError(f'Hard fit cap reached: {started}/{cap}. No model call authorized.')
    return True


class FitBudget:
    """Intercept every LGBMRegressor.fit call, including unintended calls outside refit code."""
    def __init__(self,out):
        self.out=Path(out);self.path=self.out/'model_fit_log.jsonl';self.authorization=None;self.allow_new=True;self.calls_this_process=0

    def events(self):return [json.loads(l) for l in self.path.read_text().splitlines()]
    def append(self,obj):
        with self.path.open('a') as f:f.write(json.dumps(obj,default=str)+'\n')

    def __enter__(self):
        original=LGBMRegressor.fit;budget=self
        def guarded(estimator,X,y,*args,**kwargs):
            require(budget.allow_new and budget.authorization is not None,'Unauthorized fit: no inner fits, tests or post-evaluation refits allowed')
            events=budget.events();ensure_available(sum(e['event']=='fit_started' for e in events))
            meta,expected_y=budget.authorization
            require(np.array_equal(np.asarray(y),expected_y),'Fit labels differ from authorized outer-training labels')
            require(estimator.get_params()==meta['parameters'],'Fit parameters differ from locked configuration')
            require(len(X)==len(expected_y)==meta['train_rows'],'Fit training size mismatch')
            require(not any(e.get('key')==meta['key'] and e['event']=='fit_started' for e in events),'This fit identity already started; refusing needless repeat')
            budget.calls_this_process+=1
            budget.append(meta|{'event':'fit_started','timestamp_utc':datetime.now(timezone.utc).isoformat(),'time':time.time()})
            start=time.time()
            try:result=original(estimator,X,y,*args,**kwargs)
            except Exception as e:
                budget.append(meta|{'event':'fit_failed','error':repr(e)});raise
            budget.append(meta|{'event':'fit_completed','duration_seconds':time.time()-start})
            return result
        self.patch=patch.object(LGBMRegressor,'fit',guarded);self.patch.start();return self

    def __exit__(self,*exc):self.patch.stop()

    @contextmanager
    def authorize(self,metadata,train_y):
        require(self.authorization is None,'Nested fit authorization forbidden')
        self.authorization=(metadata,np.asarray(train_y).copy())
        try:yield
        finally:self.authorization=None


def array_hash(values):return hashlib.sha256(np.asarray(values).tobytes()).hexdigest()


def cache_identity(ctx,fold,cid):
    tr,te=ctx.outer[fold-1]
    core=['common.py','audit.py','fit_budget.py','selection_rules.py']
    return {'scenario':SCENARIO,'fold':fold,'candidate':cid,'parameters':ctx.registry[cid],
            'core_source_sha256':{p:sha(ctx.out/'source'/p) for p in core},'original_source_sha256':ctx.inputs['microplastics.py'],
            'dataset_sha256':ctx.inputs['data/noaa_microplastics.csv'],'frozen_splits_sha256':ctx.inputs['optimization_outputs/frozen_splits.json'],
            'candidate_registry_sha256':ctx.inputs['optimization_outputs/candidate_registry.json'],
            'lock_sha256':sha(ctx.out/'locked_selections.json'),'environment':ctx.environment,
            'training_row_ids':tr.tolist(),'training_source_row_ids':ctx.base.source_row.iloc[tr].tolist(),
            'test_row_ids':te.tolist(),'test_source_row_ids':ctx.base.source_row.iloc[te].tolist(),
            'training_labels_sha256':array_hash(ctx.y[tr]),'training_features_sha256':array_hash(pd.util.hash_pandas_object(ctx.X.iloc[tr],index=True).to_numpy()),
            'test_features_sha256':array_hash(pd.util.hash_pandas_object(ctx.X.iloc[te],index=True).to_numpy()),
            'training_studies_sha256':fingerprint(ctx.groups[tr].tolist())}


def read_checkpoint(path,metadata_path,payload,test_ids):
    meta=json.loads(Path(metadata_path).read_text())
    require(meta['identity']==payload and meta['identity_sha256']==fingerprint(payload),'Checkpoint identity mismatch; no incompatible reuse')
    require(sha(path)==meta['npz_sha256'],'Checkpoint content hash mismatch')
    with np.load(path) as z:
        require(np.array_equal(z['test_ids'],test_ids) and z['pred'].shape==test_ids.shape and np.isfinite(z['pred']).all(),'Invalid checkpoint prediction coverage')
        return z['pred'].copy()


def refit_locked(ctx,budget):
    lock_hash=json.loads((ctx.out/'lock_sha256.json').read_text())['sha256']
    require(sha(ctx.out/'locked_selections.json')==lock_hash,'Selections changed after pre-fit lock')
    choices=ctx.locked['payload']['selections'];pending=set()
    for item in choices:
        f=item['outer'];cid=item['candidate'];prev=ctx.previous_selections.loc[ctx.previous_selections.outer==f,'candidate'].iloc[0]
        if cid!='C0_original' and cid!=prev:pending.add((f,cid))
    require(len(pending)<=10,'Pre-fit upper bound exceeds ten fits')
    dump(ctx.out/'fit_plan.json',{'unique_nonreference_fold_configurations':[{'fold':f,'candidate':c} for f,c in sorted(pending)],'upper_bound_new_fits':len(pending),'cap':10,'scenario':SCENARIO,'locked_selections_sha256':lock_hash})
    out=ctx.opt[['water_row','source_row','study','doi','accession','fold','target_log10','method_family','study_size','study_size_bin','scenario']].copy()
    out[ORIGINAL]=ctx.base.prediction_log10;out[OPTIMIZED]=ctx.opt.optimized_prediction_log10
    out['S1']=np.nan;out['S2']=np.nan
    routes=[];warnings_log=[]
    closed=(ctx.out/'stage_b_complete.json').exists()
    for item in choices:
        rule=item['rule'];f=item['outer'];cid=item['candidate'];tr,te=ctx.outer[f-1]
        prev=ctx.previous_selections.loc[ctx.previous_selections.outer==f,'candidate'].iloc[0]
        if cid=='C0_original':
            pred=ctx.base.prediction_log10.iloc[te].to_numpy();route='original_frozen_OOF';key=ctx.inputs['reproduction_outputs/out_of_fold_predictions.csv']
        elif cid==prev:
            pred=ctx.opt.optimized_prediction_log10.iloc[te].to_numpy();route='optimized_frozen_OOF';key=ctx.inputs['optimization_outputs/optimized_oof_predictions.csv']
        else:
            payload=cache_identity(ctx,f,cid);key=fingerprint(payload)
            path=ctx.out/'checkpoints'/f'{key}.npz';meta_path=path.with_suffix('.json')
            if path.exists() and meta_path.exists():
                pred=read_checkpoint(path,meta_path,payload,te);route='stage4_checkpoint'
            else:
                require(not closed,'Stage B already completed; no new runs after outer metrics. Restore the matching Stage 4 checkpoint.')
                train=ctx.X.iloc[tr].copy();train['y']=ctx.y[tr]
                metadata={'key':key,'outer':f,'first_requesting_selector':rule,'candidate':cid,'parameters':ctx.registry[cid],
                          'train_rows':len(tr),'test_rows':len(te),'training_labels_sha256':payload['training_labels_sha256'],
                          'training_input_fingerprint':fingerprint(payload),'lock_sha256':lock_hash,'scenario':SCENARIO}
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    with budget.authorize(metadata,ctx.y[tr]):
                        predictor=ctx.ns['sklearn_model'](LGBMRegressor(**ctx.registry[cid]))(train)
                    # Predictor receives features only. No outer target is passed into fit or predict.
                    pred=predictor(ctx.X.iloc[te])
                warnings_log.extend({'fold':f,'candidate':cid,'category':w.category.__name__,'message':str(w.message)} for w in caught)
                require(np.isfinite(pred).all() and len(pred)==len(te),'Nonfinite or incomplete outer predictions')
                # Inspect actual fitted original pipeline state, without refitting it.
                closure=dict(zip(predictor.__code__.co_freevars,[c.cell_contents for c in predictor.__closure__]))
                pipe=closure['pipe'];num=pipe.steps[0][1].named_transformers_['num']
                prep_audit={'train_rows':tr.tolist(),'numeric_columns':ctx.ns['NUMERIC'],
                            'imputer_medians':num.named_steps['simpleimputer'].statistics_.tolist(),
                            'scaler_mean':num.named_steps['standardscaler'].mean_.tolist(),'scaler_var':num.named_steps['standardscaler'].var_.tolist(),
                            'category_values':[v.tolist() for v in pipe.steps[0][1].named_transformers_['cat'].categories_],
                            'numeric_min':train[ctx.ns['NUMERIC']].min().tolist(),'numeric_max':train[ctx.ns['NUMERIC']].max().tolist()}
                tmp=path.with_suffix('.tmp')
                with tmp.open('wb') as h:np.savez_compressed(h,pred=pred,test_ids=te)
                tmp.replace(path)
                dump(meta_path,{'identity':payload,'identity_sha256':key,'npz_sha256':sha(path),'preprocessor_audit':prep_audit,'warnings':[str(w.message) for w in caught]})
                route='fresh_fit'
        require(np.isfinite(pred).all() and len(pred)==len(te),'Bad reused predictions')
        out.loc[te,rule]=pred
        out.loc[te,rule+'_candidate']=cid
        routes.append({'rule':rule,'fold':f,'candidate':cid,'route':route,'key':key,'rows':len(te),'scenario':SCENARIO})
        print(f'{rule} fold {f}: {cid} [{route}]',flush=True)
    require(out[['S1','S2']].notna().all().all(),'Incomplete outer prediction coverage')
    out.to_csv(ctx.out/'all_procedure_oof_predictions.csv',index=False)
    pd.DataFrame(routes).to_csv(ctx.out/'prediction_reuse_audit.csv',index=False)
    # Preserve first-run route accounting independently from checkpoint-only notebook reruns.
    if not (ctx.out/'initial_prediction_reuse_audit.csv').exists():pd.DataFrame(routes).to_csv(ctx.out/'initial_prediction_reuse_audit.csv',index=False)
    if not (ctx.out/'fit_warnings.json').exists():dump(ctx.out/'fit_warnings.json',warnings_log)
    budget.allow_new=False
    if not closed:dump(ctx.out/'stage_b_complete.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),'lock_sha256':lock_hash,'oof_sha256':sha(ctx.out/'all_procedure_oof_predictions.csv'),'no_more_fitting_allowed':True})
    else:require(sha(ctx.out/'all_procedure_oof_predictions.csv')==json.loads((ctx.out/'stage_b_complete.json').read_text())['oof_sha256'],'Checkpoint rerun changed frozen outer predictions')
    ctx.oof=out;ctx.routes=pd.DataFrame(routes)
    return out

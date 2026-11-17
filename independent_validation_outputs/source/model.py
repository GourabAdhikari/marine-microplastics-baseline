"""Optional 24+1 full-development training. Default gate is NOT_TRAINED.

No fit on import; no prior outer checkpoint reused; external labels are never loaded.
"""
import hashlib
import importlib.metadata as package_metadata
import fcntl
import json
import sys
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from validation_core import *
from protocol import verify_lock
from external import development_review
from metrics import scores


class FrozenResearchModel:
    def __init__(self,pipe,low,high,metadata):self.pipe=pipe;self.low=low;self.high=high;self.metadata=metadata
    def predict(self,X):
        require(list(X.columns)==FEATURES,'Frozen model feature schema mismatch')
        clipped=X.assign(**{c:X[c].clip(self.low[c],self.high[c]) for c in FEATURES[:-1]})
        return self.pipe.predict(clipped)


def training_gate(out):
    reasons=[]
    expected=json.loads((out/'environment.json').read_text())
    for package,version in expected['packages'].items():
        if package_metadata.version(package)!=version:reasons.append('Runtime package changed: '+package)
    try:lock=verify_lock(out)
    except (ValueError,FileNotFoundError) as e:reasons.append(str(e));lock=None
    receiptpath=out/'external_preparation_receipt.json'
    receipt=json.loads(receiptpath.read_text()) if receiptpath.exists() else {}
    if receipt.get('status')!='ELIGIBLE_UNSCORED':reasons.append('No identified metadata-eligible independent set with resolved target comparability')
    if receipt:
        path=out.parent/receipt.get('path','')
        if not path.is_file() or sha(path)!=receipt.get('source_sha256'):reasons.append('Registered external file changed/missing')
        side=Path(str(path)+'.manifest.json')
        if not side.is_file() or sha(side)!=receipt.get('manifest_sha256'):reasons.append('External source manifest changed/missing')
        for path,h in receipt.get('review_evidence_hashes',{}).items():
            if not (out.parent/path).is_file() or sha(out.parent/path)!=h:reasons.append('External review evidence changed/missing')
        if not (out/'protocol_lock.json').exists() or receipt.get('protocol_sha256')!=sha(out/'protocol_lock.json'):reasons.append('Preparation receipt protocol mismatch')
    ok,devreasons=development_review(out)
    if not ok:reasons.extend(devreasons)
    if receipt and (not (out/'development_provenance_review.json').exists() or sha(out/'development_provenance_review.json')!=receipt.get('development_review_sha256')):reasons.append('Development evidence review changed after metadata preparation')
    if (out/'label_access_ledger.jsonl').exists():reasons.append('External outcomes already inspected; no more model selection')
    return not reasons,reasons,receipt


def fit_cap(started):
    require(started<25,'Hard ceiling: at most 25 newly started LightGBM fits')


def fit_frozen(out):
    # Serialize training requests so concurrent processes cannot bypass the global 25-start cap.
    with (out/'training_process.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        try:return _fit_frozen_locked(out)
        finally:fcntl.flock(lock.fileno(),fcntl.LOCK_UN)


def _fit_frozen_locked(out):
    ok,reasons,receipt=training_gate(out)
    if not ok:
        result={'status':'NOT_TRAINED','model_label':MODEL_LABEL,'reasons':reasons,'new_fits_started':0,'hard_fit_cap':25}
        dump(out/'full_data_model_status.json',result);return result
    require(not (out/'external_predictions.csv').exists(),'No training after external results')
    root=out.parent;identity=json.loads((out/'baseline_identity.json').read_text());ns=development(root);df=ns['df'];X=df[FEATURES];y=df.y.to_numpy()
    registry=json.loads((root/'optimization_outputs/candidate_registry.json').read_text())['configs']
    sys.path.append(str(root/'study_robustness_outputs/source'))
    selectors=load_module('stage5_final_S2',root/'study_robustness_outputs/source/selection_rules.py')
    plan=json.loads((out/'research_training_plan.json').read_text())
    require(sha(root/'study_robustness_outputs/source/selection_rules.py')==plan['selection_source_sha256'],'S2 source changed')
    require(sha(root/'optimization_outputs/candidate_registry.json')==plan['candidate_registry_sha256'],'Registry changed')
    folds=list(GroupKFold(3).split(X,groups=df.study));require(len(df)==16988 and df.study.nunique()==72,'Development scope changed')
    for tr,te in folds:require(not set(df.study.iloc[tr])&set(df.study.iloc[te]),'Training inner-group leakage')
    recipe={'label':MODEL_LABEL,'source':{p.name:sha(p) for p in (out/'source').glob('*.py')},'data_sha256':identity['input_sha256']['data/noaa_microplastics.csv'],
            'registry':registry,'folds':[[a.tolist(),b.tolist()] for a,b in folds],'environment':json.loads((out/'environment.json').read_text()),
            'protocol_sha256':sha(out/'protocol_lock.json'),'training_labels_sha256':hashlib.sha256(y.tobytes()).hexdigest(),
            'external_campaign_metadata_sha256':receipt['metadata_sha256'],'external_labels_used':False,'development_review_sha256':receipt['development_review_sha256']}
    recipekey=digest(recipe);checkpoint=out/'checkpoints/full_data_training';checkpoint.mkdir(parents=True,exist_ok=True)
    fitlog=out/'fit_log.jsonl'
    def events():return [json.loads(l) for l in fitlog.read_text().splitlines()]
    def append(record):
        with fitlog.open('a') as f:f.write(json.dumps(record,default=str)+'\n')
    def start(key,stage,cid,rows):
        history=events();fit_cap(sum(e['event']=='fit_started' for e in history))
        require(not any(e.get('key')==key and e['event']=='fit_started' for e in history),'Incomplete/already-started fit cannot be repeated; restore completed checkpoint or stop')
        append({'event':'fit_started','key':key,'recipe':recipekey,'stage':stage,'candidate':cid,'train_rows':rows,'utc':now(),'label':MODEL_LABEL})
    def fitted_predictor(tr,cid):
        pipe=make_pipeline(ns['preprocessor'](),LGBMRegressor(**registry[cid])).fit(X.iloc[tr],y[tr])
        return FrozenResearchModel(pipe,X.iloc[tr][FEATURES[:-1]].min().to_dict(),X.iloc[tr][FEATURES[:-1]].max().to_dict(),{'label':MODEL_LABEL,'recipe':recipekey,'candidate':cid})
    dump(out/'full_data_training_recipe.json',recipe)
    table=[]
    for cid in registry:
        pred=np.full(len(df),np.nan)
        for j,(tr,te) in enumerate(folds,1):
            key=digest({'recipe':recipekey,'candidate':cid,'inner':j});path=checkpoint/f'{key}.npz';meta=path.with_suffix('.json')
            if path.exists() and meta.exists():
                m=json.loads(meta.read_text());require(m['key']==key and m['sha256']==sha(path),'Inner checkpoint identity mismatch')
                with np.load(path) as z:require(np.array_equal(z['row_ids'],te),'Inner checkpoint row mismatch');p=z['prediction']
            else:
                start(key,'inner',cid,len(tr))
                try:
                    model=fitted_predictor(tr,cid);p=model.predict(X.iloc[te]);require(np.isfinite(p).all(),'Nonfinite training-only predictions')
                    tmp=path.with_suffix('.tmp')
                    with tmp.open('wb') as f:np.savez_compressed(f,prediction=p,row_ids=te)
                    tmp.replace(path);dump(meta,{'key':key,'sha256':sha(path)})
                    append({'event':'fit_completed','key':key,'utc':now()})
                except Exception as e:append({'event':'fit_failed','key':key,'error':repr(e),'utc':now()});raise
            pred[te]=p
        m=scores(y,pred,df.study);table.append({'outer':1,'candidate':cid,'scenario':SCENARIO,**{k:m[k] for k in selectors.FIELDS[3:]}})
    inner=pd.DataFrame(table);chosen,eligibility=selectors.select_fold(inner[selectors.FIELDS],'S2');cid=chosen['candidate']
    inner.to_csv(out/'full_data_inner_metrics.csv',index=False);eligibility.to_csv(out/'full_data_selector_eligibility.csv',index=False)
    folder=out/'full_data_research_model';folder.mkdir(exist_ok=True);artifact=folder/'pipeline.joblib';meta=folder/'metadata.json'
    key=digest({'recipe':recipekey,'full_data':True,'candidate':cid})
    if artifact.exists() and meta.exists():
        metadata=json.loads(meta.read_text());require(metadata['key']==key and metadata['artifact_sha256']==sha(artifact),'Full model identity mismatch')
    else:
        start(key,'full_data',cid,len(df))
        try:
            model=fitted_predictor(np.arange(len(df)),cid);tmp=artifact.with_suffix('.tmp');joblib.dump(model,tmp);tmp.replace(artifact)
            metadata={'key':key,'label':MODEL_LABEL,'candidate':cid,'selection':chosen,'artifact_sha256':sha(artifact),'recipe_sha256':recipekey,
                      'protocol_sha256':sha(out/'protocol_lock.json'),'schema_sha256':sha(out/'external_schema.json'),'training_rows':16988,'training_studies':72,
                      'external_metadata_sha256':receipt['metadata_sha256'],'source_hashes':recipe['source'],'trained_utc':now(),'no_independent_training_score':True}
            dump(meta,metadata);append({'event':'fit_completed','key':key,'utc':now()})
        except Exception as e:append({'event':'fit_failed','key':key,'error':repr(e),'utc':now()});raise
    result={'status':'TRAINED_RESEARCH_ONLY','model_label':MODEL_LABEL,'candidate':cid,'new_fits_started':sum(e['event']=='fit_started' for e in events()),'hard_fit_cap':25,'artifact_sha256':sha(artifact)}
    dump(out/'full_data_model_status.json',result);return result

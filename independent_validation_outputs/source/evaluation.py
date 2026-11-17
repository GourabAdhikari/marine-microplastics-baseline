"""One-time external outcome evaluation, guarded by lock/provenance/model receipts."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from validation_core import *
from protocol import verify_lock
from external import prepare_external,features,linked_external_groups
from model import training_gate
from metrics import scores,cluster_bootstrap


def evaluate_external(out,path=None):
    verify_lock(out)
    if not path or not Path(path).is_file():return prepare_external(out,path)
    require(not (out/'label_access_ledger.jsonl').exists(),'External outcomes already accessed; this dataset/run cannot be reused for tuning or repeated scoring')
    receiptpath=out/'external_preparation_receipt.json'
    require(receiptpath.exists(),'Prepare external metadata after the lock, before evaluating outcomes')
    receipt=json.loads(receiptpath.read_text());p=Path(path).resolve()
    require(receipt.get('path')==str(p.relative_to(out.parent)) and sha(p)==receipt.get('source_sha256'),'Prepared external source identity changed')
    ok,reasons,_=training_gate(out)
    if not ok:
        state={'status':'UNVERIFIED_TARGET_PROVENANCE','reasons':reasons,'targets_read':False,'independent_R2':None}
        dump(out/'external_status.json',state);return state
    folder=out/'full_data_research_model';artifact=folder/'pipeline.joblib';mp=folder/'metadata.json'
    if not artifact.exists() or not mp.exists():
        state={'status':'ELIGIBLE_UNSCORED','reason':'Frozen full-development research model does not exist','targets_read':False,'independent_R2':None}
        dump(out/'external_status.json',state);return state
    metadata=json.loads(mp.read_text())
    require(metadata['label']==MODEL_LABEL and metadata['artifact_sha256']==sha(artifact),'Model label/artifact identity mismatch')
    require(metadata['protocol_sha256']==sha(out/'protocol_lock.json') and metadata['schema_sha256']==sha(out/'external_schema.json'),'Model protocol/schema mismatch')
    require(metadata['external_metadata_sha256']==receipt['metadata_sha256'],'Model belongs to a different prespecified campaign')
    for name,h in metadata['source_hashes'].items():require(sha(out/'source'/name)==h,'Training/evaluation source code changed since freezing model')
    require(receipt['outcome_header_present'],'No concentration outcome column; cannot evaluate')
    require(not (out/'external_predictions.csv').exists() and not (out/'EXTERNAL_TEST_REPORT.md').exists(),'Immutable external outputs already exist')
    event={'event':'external_outcome_access_started','utc':now(),'source_sha256':sha(p),'metadata_sha256':receipt['metadata_sha256'],
           'source_identities':receipt['source_identities'],'doi_identities':receipt['doi_identities'],'campaign_identities':receipt['campaign_identities'],'protocol_sha256':sha(out/'protocol_lock.json')}
    # Exclusive reservation BEFORE deserializing model or reading concentration. Failures consume the test.
    ledger=out/'label_access_ledger.jsonl'
    with ledger.open('x') as f:f.write(json.dumps(event)+'\n')
    try:
        model=joblib.load(artifact)  # Only locally generated, hash-verified trusted artifact; never load external pickle.
        d=csv(p,dtype={'source_row_id':str,'study_id':str,'doi':str,'accession':str,'campaign_id':str})
        yraw=pd.to_numeric(d.concentration,errors='coerce').to_numpy()
        require(np.isfinite(yraw).all() and (yraw>=0).all(),'Invalid/missing/negative external outcomes: evaluation aborted, no row deletion or imputation')
        identity=json.loads((out/'baseline_identity.json').read_text());X=features(d,identity['method_family_map']);groups=linked_external_groups(d)
        pred=model.predict(X);require(np.isfinite(pred).all() and len(pred)==len(d),'Incomplete/nonfinite external predictions')
        y=np.log10(yraw+.001);m=scores(y,pred,groups);uncertainty=cluster_bootstrap(y,pred,groups)
        result=d.copy();result['evaluation_study_group']=groups;result['target_log10']=y;result['prediction_log10']=pred;result['error_log10']=pred-y
        result['model_label']=MODEL_LABEL
        audit=csv(out/'external_eligibility_audit.csv',keep_default_na=False)
        require(np.array_equal(audit.source_row_id.astype(str),d.source_row_id.astype(str)),'Eligibility row identities changed')
        result['support_flags']=audit['flags'];result.to_csv(out/'external_predictions.csv',index=False,mode='x')
        studies=pd.DataFrame([{'study':s,'rows':len(g),**scores(g.target_log10,g.prediction_log10,g.evaluation_study_group)} for s,g in result.groupby('evaluation_study_group')])
        studies.to_csv(out/'external_per_study_metrics.csv',index=False,mode='x')
        dump(out/'external_uncertainty.json',uncertainty)
        state={'status':'INDEPENDENT_EVALUATION_COMPLETED','observations':len(d),'studies':len(set(groups)),'metrics':m,'uncertainty':uncertainty,'targets_read':True,
               'model_label':MODEL_LABEL,'scenario_warning':'Historical development-label assumptions remain the immutable training scenario; signed source/comparability reviews are evidence prerequisites, not model certification.',
               'comparison_to_historical':'Unpaired different-population descriptive comparison only; S2 historical R2 0.5256280075581679 is not a second independent test.'}
        dump(out/'external_status.json',state)
        with (out/'EXTERNAL_TEST_REPORT.md').open('x') as f:f.write('# One-time independent external evaluation\n\n'+json.dumps(state,indent=2)+'\n\nNo tuning or further outcome access is authorized. Whole-study bootstrap only; small-study-count uncertainty warnings apply.\n')
        with ledger.open('a') as f:f.write(json.dumps({'event':'evaluation_completed','utc':now(),'rows':len(d)})+'\n')
        return state
    except Exception as e:
        with ledger.open('a') as f:f.write(json.dumps({'event':'evaluation_failed_after_access','utc':now(),'error':repr(e)})+'\n')
        dump(out/'external_status.json',{'status':'SCHEMA_INCOMPATIBLE','reason':str(e),'targets_read':True,'evaluation_completed':False,'independent_R2':None,'retry_allowed':False})
        raise

"""Metadata-only external preparation and fail-closed provenance/overlap gates."""
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from validation_core import *
from protocol import REQUIRED,MANIFEST_FIELDS,verify_lock


def features(meta,method_map):
    """Pure predictor engineering: no concentration/outcome column is read."""
    date=pd.to_datetime(meta.date,format='%Y-%m-%d',errors='raise');month=date.dt.month
    method=meta.method.map(normalized_method)
    return pd.DataFrame({'lat':pd.to_numeric(meta.lat),'lon':pd.to_numeric(meta.lon),'year':date.dt.year,
                         'month_sin':np.sin(2*np.pi*month/12),'month_cos':np.cos(2*np.pi*month/12),
                         'log_mesh':np.log10(pd.to_numeric(meta.mesh_mm)),'log_depth':np.log1p(pd.to_numeric(meta.depth_m)),
                         'method_family':method.map(method_map).fillna('bulk water')},index=meta.index)[FEATURES]


def linked_external_groups(meta):
    parent=list(range(len(meta)))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for col,normalizer in [('study_id',str),('doi',normalize_doi),('accession',normalize_accession),('campaign_id',lambda x:str(x).strip().lower())]:
        seen={}
        for i,value in enumerate(meta[col]):
            v=normalizer(value)
            if not v or v.lower() in ['nan','none','null']:continue
            if v in seen:parent[root(i)]=root(seen[v])
            else:seen[v]=i
    return np.array([f'external_component_{root(i):06d}' for i in range(len(meta))])


def inspect_metadata(meta,development_index,support,method_map,manifests):
    """Outcome-invariant eligibility. No file I/O and no learned model access."""
    require(set(REQUIRED)<=set(meta.columns),'Missing required metadata columns')
    dev=development_index;dev_dois=set(dev.doi_normalized);dev_access=set(dev.accession_normalized.astype(str));dev_ids=set(dev.source_observation_id.astype(str))
    dev_fp=set(dev.observation_fingerprint);known=set(support['known_methods']);records=[]
    bydate={date:g for date,g in dev.groupby('date')}
    duplicated_source=meta.source_row_id.astype(str).duplicated(keep=False)
    seen_fingerprints={};manifestmap={str(m.get('study_id')):m for m in manifests}
    for position,(idx,r) in enumerate(meta.iterrows()):
        reasons=[];flags=[];m=manifestmap.get(str(r.study_id),{})
        def reject(reason):reasons.append(reason)
        for col in REQUIRED:
            if col not in ['depth_m','accession'] and (pd.isna(r[col]) or not str(r[col]).strip()):reject('missing_'+col)
        doi=normalize_doi(r.doi);accession=normalize_accession(r.accession);method=normalized_method(r.method)
        if not doi.startswith('10.') or '/' not in doi:reject('invalid_DOI')
        if doi in dev_dois or set(map(normalize_doi,m.get('ancestor_dois',[])))&dev_dois:reject('DOI_overlap')
        if (accession and accession in dev_access) or set(map(normalize_accession,m.get('ancestor_accessions',[])))&dev_access:reject('accession_overlap')
        if str(r.source_row_id) in dev_ids or set(map(str,m.get('ancestor_source_ids',[])))&dev_ids:reject('source_observation_overlap')
        # Development lacks campaign inventory: explicitly declared aliases to known studies/accessions block.
        aliases=set(str(x).strip().lower() for x in [r.campaign_id,*m.get('campaign_aliases',[])])
        if aliases & (dev_dois|dev_access|dev_ids):reject('campaign_alias_overlap')
        if duplicated_source.iloc[position]:reject('duplicate_external_source_identity')
        if r.medium!='Ocean water':reject('incompatible_medium')
        if r.unit!='pieces/m3':reject('unit_mismatch_no_auto_conversion')
        if r.target_type!='synthetic_particle_count':reject('target_type_mismatch')
        if r.scenario!=EXTERNAL_SCENARIO:reject('scenario_mismatch')
        if method not in known:reject('unknown_or_missing_sampling_method')
        family=method_map.get(method,'bulk water')
        if (family=='surface net' and r.mesh_kind!='net_mesh') or (family!='surface net' and r.mesh_kind!='membrane_pore_size'):reject('mesh_context_mismatch')
        numeric={k:pd.to_numeric(pd.Series([r[k]]),errors='coerce').iloc[0] for k in ['lat','lon','mesh_mm','depth_m']}
        if not np.isfinite(numeric['lat']) or not -90<=numeric['lat']<=90:reject('invalid_latitude')
        if not np.isfinite(numeric['lon']) or not -180<=numeric['lon']<=180:reject('invalid_longitude')
        if not np.isfinite(numeric['mesh_mm']) or numeric['mesh_mm']<=0:reject('invalid_mesh')
        if pd.isna(r.depth_m):flags.append('missing_depth_training_median')
        elif not np.isfinite(numeric['depth_m']) or numeric['depth_m']<0:reject('invalid_depth')
        try:date=pd.to_datetime(r.date,format='%Y-%m-%d',errors='raise').strftime('%Y-%m-%d')
        except (ValueError,TypeError):date=None;reject('invalid_date')
        fp=''
        if date and np.isfinite(numeric['lat']) and np.isfinite(numeric['lon']):
            fp=observation_fingerprint(date,numeric['lat'],numeric['lon'],method)
            if fp in dev_fp:reject('exact_measurement_metadata_overlap')
            g=bydate.get(date)
            if g is not None:
                g=g[g.method==method]
                if len(g):
                    distance=BallTree(np.radians(g[['lat','lon']]),metric='haversine').query(np.radians([[numeric['lat'],numeric['lon']]]),k=1)[0][0,0]*6371.
                    if distance<=1.:reject('near_duplicate_same_day_method_within_1km')
            if fp in seen_fingerprints:
                reject('duplicate_external_measurement_metadata');records[seen_fingerprints[fp]]['reasons'].append('duplicate_external_measurement_metadata')
            else:seen_fingerprints[fp]=position
        if not m:reject('missing_study_manifest')
        else:
            for field in ['provenance_verified','independence_reviewed','method_context_signed']:
                if m.get(field) is not True:reject('unverified_'+field)
            if m.get('previously_used_for_tuning') is not False:reject('previous_outcome_use_unknown_or_true')
            for field in ['reviewer','reviewed_utc','review_signature','citation','source_url','volume_calculation_reference','synthetic_verification_reference','zero_and_detection_limit_reference','method_context_reference']:
                if not str(m.get(field,'')).strip():reject('missing_'+field)
            if m.get('original_unit')!='pieces/m3':reject('unverified_canonical_source_unit')
        # Support checks use only supplied predictors and development min/max.
        if date and not any(x.startswith(('invalid_','missing_')) for x in reasons):
            try:
                x=features(meta.loc[[idx]],method_map)
                for col in FEATURES[:-1]:
                    value=x[col].iloc[0]
                    if np.isfinite(value) and (value<support['numeric_min'][col] or value>support['numeric_max'][col]):flags.append('OOD_'+col+'_will_be_clipped')
            except (ValueError,TypeError):reject('feature_transformation_failure')
        records.append({'source_row_id':str(r.source_row_id),'study_id':str(r.study_id),'doi_normalized':doi,'observation_fingerprint':fp,'reasons':reasons,'flags':flags})
    # Quarantine whole study, never delete outcome-selected rows.
    bad={r['study_id'] for r in records if r['reasons']}
    for r in records:
        if r['study_id'] in bad and not r['reasons']:r['reasons'].append('study_quarantined_due_to_other_row')
        r['eligible_metadata']=not r['reasons'];r['reasons']=';'.join(sorted(set(r['reasons'])));r['flags']=';'.join(sorted(set(r['flags'])))
    return pd.DataFrame(records)


def local_evidence(root,relative,expected):
    try:
        path=(root/str(relative)).resolve();path.relative_to(root.resolve())
        return path.is_file() and bool(expected) and sha(path)==expected
    except (ValueError,OSError):return False


def development_review(out):
    """Review evidence is required; unresolved local audit is not silently upgraded."""
    path=out/'development_provenance_review.json'
    if not path.exists():return False,['development_unit_and_synthetic_provenance_unresolved']
    m=json.loads(path.read_text());root=out.parent;reasons=[]
    for k in ['unit_traceability_resolved','synthetic_target_compatible','external_target_comparable','historical_values_unchanged']:
        if m.get(k) is not True:reasons.append('development_review_'+k)
    for k in ['reviewer','reviewed_utc','review_signature']:
        if not str(m.get(k,'')).strip():reasons.append('development_missing_'+k)
    if m.get('raw_data_sha256')!=sha(root/'data/noaa_microplastics.csv') or m.get('scenario')!=SCENARIO:reasons.append('development_review_identity_mismatch')
    if not local_evidence(root,m.get('evidence_path'),m.get('evidence_sha256')):reasons.append('development_evidence_not_inspectable')
    # Require source measurement and explicit unit conversion for every historically adjusted row.
    if not local_evidence(root,m.get('row_mapping_path'),m.get('row_mapping_sha256')):reasons.append('missing_auditable_row_mapping')
    else:
        try:
            mapping=csv(root/m['row_mapping_path']);ns=development(root);df=ns['df'];ids=csv(out/'development_identity_index.csv')
            adjusted=df.doi.isin(ns['UNIT_FIXES']);expected=ids.loc[adjusted,'source_row'].to_numpy()
            required={'source_row','source_concentration','source_unit','conversion_factor','evidence_reference'}
            require(required<=set(mapping),'Incomplete row provenance schema')
            require(len(mapping)==1470 and mapping.source_row.is_unique and set(mapping.source_row)==set(expected),'Wrong source-row mapping coverage')
            mapping=mapping.set_index('source_row').loc[expected]
            validfactor=np.where(mapping.source_unit=='pieces/L',1000.,np.where(mapping.source_unit=='pieces/m3',1.,np.nan))
            require(np.isfinite(validfactor).all() and np.array_equal(mapping.conversion_factor,validfactor),'Conversion factors not derivable from supported volume units')
            require(np.isfinite(mapping.source_concentration).all() and (mapping.source_concentration>=0).all(),'Invalid original measurements')
            require(mapping.evidence_reference.fillna('').str.strip().ne('').all(),'Missing row-specific evidence')
            require(np.allclose(mapping.source_concentration*validfactor,df.loc[adjusted,'conc'],rtol=1e-10,atol=1e-12),'Source measurements do not support frozen historical labels')
        except Exception as e:reasons.append('row_provenance_verification_failed: '+str(e))
    return not reasons,reasons


def prepare_external(out,path=None):
    lock=verify_lock(out)
    if not path or not Path(path).is_file():
        state={'status':'NO_EXTERNAL_DATA','path':str(path) if path else None,'studies':0,'observations':0,'targets_read':False,'reason':'No explicitly supplied external observation file exists. Historical/derived OOF artifacts are not independent data.'}
        dump(out/'external_status.json',state);return state
    path=Path(path).resolve();require(path.is_relative_to(out.parent.resolve()),'External file must be staged inside this repository; no writes outside stage directory')
    header=csv(path,nrows=0).columns.tolist();unknown=set(header)-set(REQUIRED)-{'concentration'};missing=set(REQUIRED)-set(header)
    if missing or unknown:
        state={'status':'SCHEMA_INCOMPATIBLE','missing_columns':sorted(missing),'unknown_columns':sorted(unknown),'targets_read':False,'outcome_header_present':'concentration' in header}
        dump(out/'external_status.json',state);return state
    meta=csv(path,usecols=REQUIRED,dtype={'source_row_id':str,'study_id':str,'doi':str,'accession':str,'campaign_id':str})
    require(len(meta)>0,'External file has no observations')
    side=Path(str(path)+'.manifest.json')
    manifest=json.loads(side.read_text()) if side.exists() else {'studies':[]};ms=manifest.get('studies',[])
    require(isinstance(ms,list) and len({m.get('study_id') for m in ms})==len(ms),'Missing/duplicate manifest study IDs')
    identity=json.loads((out/'baseline_identity.json').read_text());support=json.loads((out/'training_support.json').read_text())
    index=csv(out/'development_identity_index.csv',dtype={'source_observation_id':str,'accession_normalized':str})
    audit=inspect_metadata(meta,index,support,identity['method_family_map'],ms)
    for m in ms:
        issues=[]
        if m.get('source_file_sha256')!=sha(path):issues.append('manifest_dataset_hash_mismatch')
        if not local_evidence(out.parent,m.get('source_document_path'),m.get('source_document_sha256')):issues.append('primary_source_document_missing_or_hash_mismatch')
        if issues:
            mask=audit.study_id==str(m.get('study_id'));audit.loc[mask,'eligible_metadata']=False
            audit.loc[mask,'reasons']=audit.loc[mask,'reasons'].map(lambda s:';'.join([x for x in [s,*issues] if x]))
    reasons=';'.join(audit.reasons)
    if any(k in reasons for k in ['overlap','near_duplicate','duplicate_external']):status='OVERLAP_FOUND'
    elif not audit.eligible_metadata.all():status='UNVERIFIED_TARGET_PROVENANCE' if any(k in reasons for k in ['unverified','manifest','source_document','review','reference']) else 'SCHEMA_INCOMPATIBLE'
    else:status='ELIGIBLE_UNSCORED'
    dev_ok,dev_reasons=development_review(out)
    if status=='ELIGIBLE_UNSCORED' and not dev_ok:status='UNVERIFIED_TARGET_PROVENANCE'
    audit.to_csv(out/'external_eligibility_audit.csv',index=False)
    groups=linked_external_groups(meta)
    receipt={'status':status,'path':str(path.relative_to(out.parent)),'source_sha256':sha(path),'manifest_sha256':sha(side) if side.exists() else None,
             'metadata_sha256':digest(meta.fillna('').to_dict(orient='records')),'source_identities':sorted(set(meta.source_row_id.astype(str))),
             'doi_identities':sorted(set(meta.doi.map(normalize_doi))),'campaign_identities':sorted(set(meta.campaign_id.astype(str))),
             'observations':len(meta),'studies':len(set(groups)),'declared_study_ids':meta.study_id.nunique(),
             'eligible_metadata_rows':int(audit.eligible_metadata.sum()),'outcome_header_present':'concentration' in header,'targets_read':False,
             'protocol_sha256':sha(out/'protocol_lock.json'),'protocol_locked_utc':lock['locked_utc'],'prepared_utc':now(),
             'development_provenance_gate':dev_ok,'development_provenance_reasons':dev_reasons,
             'development_review_sha256':sha(out/'development_provenance_review.json') if (out/'development_provenance_review.json').exists() else None,
             'review_evidence_hashes':{m['source_document_path']:m['source_document_sha256'] for m in ms if m.get('source_document_path') and m.get('source_document_sha256')}}
    require(datetime.fromisoformat(lock['locked_utc'])<=datetime.fromisoformat(receipt['prepared_utc']),'Lock must predate dataset preparation')
    # Once any external evaluation has exposed outcomes, no additional fitting/test registration is authorized in this stage.
    if (out/'label_access_ledger.jsonl').exists():receipt.update(status='SCHEMA_INCOMPATIBLE',reason='External outcomes were already accessed; no fresh test reuse authorized')
    dump(out/'external_status.json',receipt);dump(out/'external_preparation_receipt.json',receipt);return receipt

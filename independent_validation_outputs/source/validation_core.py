"""Side-effect-free Stage 5 utilities. No model training occurs on import."""
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
from urllib.parse import unquote
import numpy as np
import pandas as pd

SCENARIO='historical_hypothetical_unit_scenario'
MODEL_LABEL='research_only_historical_hypothetical_unit_scenario'
EXTERNAL_SCENARIO='traceable_synthetic_particle_count_per_m3'
FEATURES=['lat','lon','year','month_sin','month_cos','log_mesh','log_depth','method_family']
STATUS=['NO_EXTERNAL_DATA','UNVERIFIED_TARGET_PROVENANCE','OVERLAP_FOUND','SCHEMA_INCOMPATIBLE','ELIGIBLE_UNSCORED','INDEPENDENT_EVALUATION_COMPLETED']


def now():return datetime.now(timezone.utc).isoformat()
def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    path=Path(path)
    if path.is_symlink():return 'symlink:'+os.readlink(path)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str).encode()).hexdigest()

def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)));tmp.replace(path)

def csv(path,**kwargs):return pd.read_csv(path,float_precision='round_trip',**kwargs)

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def workspace():return Path(__file__).resolve().parents[1]

def normalize_doi(value):
    if pd.isna(value):return ''
    text=unquote(str(value)).strip().lower()
    text=re.sub(r'^https?://(?:dx\.)?doi\.org/','',text)
    text=re.sub(r'^doi\s*:\s*','',text).strip()
    return text.rstrip(' .;')

def normalize_accession(value):
    if pd.isna(value):return ''
    text=str(value).strip().lower()
    text=re.sub(r'^.*(?:nodc:|accession[:/ ]+)','',text)
    if re.fullmatch(r'\d+(?:\.0)?',text):return str(int(float(text)))
    return text

def normalized_method(value):return str(value).strip().lower().replace('-',' ')

def observation_fingerprint(date,lat,lon,method):
    return digest([str(pd.Timestamp(date).date()),round(float(lat),6),round(float(lon),6),normalized_method(method)])


def development(root):
    old=load_module('stage5_original_helpers',root/'optimization_outputs/source/experiment.py')
    ns=old.original_namespace(root)
    return ns


def protected_check(out):
    root=out.parent;before=json.loads((out/'protected_hash_manifest_before.json').read_text())
    paths={str(p.relative_to(root)) for p in root.rglob('*') if (p.is_file() or p.is_symlink()) and p.relative_to(root).parts[0]!='independent_validation_outputs'}
    require(paths==set(before),'Unexpected new/deleted file outside independent_validation_outputs')
    after={p:sha(root/p) for p in before}
    require(after==before,'Protected-file content changed')
    dump(out/'protected_hash_manifest_after.json',after)
    result={'unchanged':True,'protected_paths':len(before),'before_sha256':sha(out/'protected_hash_manifest_before.json'),'after_sha256':sha(out/'protected_hash_manifest_after.json')}
    dump(out/'integrity_result.json',result);return result

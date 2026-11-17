"""Shared, side-effect-free I/O and metric helpers."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

SCENARIO='historical_hypothetical_unit_scenario'
ORIGINAL='original_C0'
OPTIMIZED='previously_optimized'
PROCEDURES=[ORIGINAL,OPTIMIZED,'S1','S2']
METRIC_NAMES=['R2','MAE','RMSE','study_macro_MAE','study_macro_MSE','median_study_MAE','p90_study_MAE','worst_study_MAE','within_1_log']


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    path=Path(path)
    if path.is_symlink():return 'symlink:'+os.readlink(path)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str).encode()).hexdigest()


def dump(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)))
    tmp.replace(path)


def csv(path):return pd.read_csv(path,float_precision='round_trip')


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def metrics(y,p,study):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float);study=np.asarray(study)
    require(y.ndim==p.ndim==study.ndim==1 and len(y)==len(p)==len(study)>0,'Metric arrays must align and be nonempty')
    require(np.isfinite(y).all() and np.isfinite(p).all() and not pd.isna(study).any(),'Nonfinite metric inputs: no row dropping allowed')
    e=p-y;denom=np.sum((y-y.mean())**2)
    require(denom>0,'Undefined aggregate R2: constant target')
    s=pd.DataFrame({'study':study,'ae':abs(e),'se':e**2}).groupby('study').mean()
    return dict(R2=float(1-np.sum(e**2)/denom),MAE=float(abs(e).mean()),RMSE=float(np.sqrt(np.mean(e**2))),
                study_macro_MAE=float(s.ae.mean()),study_macro_MSE=float(s.se.mean()),median_study_MAE=float(s.ae.median()),
                p90_study_MAE=float(s.ae.quantile(.9)),worst_study_MAE=float(s.ae.max()),within_1_log=float((abs(e)<=1).mean()))

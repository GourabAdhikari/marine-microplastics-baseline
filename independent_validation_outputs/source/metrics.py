"""External metrics and study-cluster uncertainty; never fit or select models."""
import numpy as np
import pandas as pd
from validation_core import require


def scores(y,p,groups):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float);groups=np.asarray(groups)
    require(y.ndim==p.ndim==groups.ndim==1 and len(y)==len(p)==len(groups)>0,'Metric arrays must align')
    require(np.isfinite(y).all() and np.isfinite(p).all() and not pd.isna(groups).any(),'No imputation/deletion of outcomes or predictions')
    e=p-y;den=float(np.sum((y-y.mean())**2));s=pd.DataFrame({'study':groups,'ae':abs(e),'se':e**2}).groupby('study').mean()
    return {'R2':None if len(y)<2 or den==0 else float(1-np.sum(e**2)/den),'R2_status':'undefined_constant_or_single_target' if len(y)<2 or den==0 else 'defined',
            'MAE':float(abs(e).mean()),'RMSE':float(np.sqrt(np.mean(e**2))),'study_macro_MAE':float(s.ae.mean()),'study_macro_MSE':float(s.se.mean()),
            'median_study_MAE':float(s.ae.median()),'p90_study_MAE':float(s.ae.quantile(.9)),'worst_study_MAE':float(s.ae.max()),'within_1_log':float((abs(e)<=1).mean())}


def cluster_bootstrap(y,p,groups,replicates=1000,seed=0):
    y=np.asarray(y);p=np.asarray(p);groups=np.asarray(groups);scores(y,p,groups)
    unique=np.unique(groups)
    if len(unique)<5:return {'status':'insufficient_independent_studies','studies':len(unique),'replicates':0,'intervals':None,'reason':'Predeclared minimum five independent study clusters; single-study rows are not independent replicates.'}
    rng=np.random.default_rng(seed);indices={g:np.flatnonzero(groups==g) for g in unique};values={k:[] for k in ['MAE','RMSE','R2','study_macro_MAE']}
    for _ in range(replicates):
        drawn=rng.choice(unique,len(unique),replace=True)
        ids=np.concatenate([indices[g] for g in drawn]);bootgroups=np.concatenate([np.full(len(indices[g]),j) for j,g in enumerate(drawn)])
        m=scores(y[ids],p[ids],bootgroups)
        for k in values:
            if m[k] is not None:values[k].append(m[k])
    intervals={}
    for k,v in values.items():
        valid=len(v)
        intervals[k]={'valid_replicates':valid,'failed_replicates':replicates-valid,
                      '95_percent_interval':np.quantile(v,[.025,.975]).tolist() if valid>=.9*replicates else None,
                      'status':'unstable_insufficient_valid_replicates' if valid<.9*replicates else ('unstable_few_studies' if len(unique)<10 else 'descriptive_cluster_bootstrap')}
    return {'status':'unstable_few_studies' if len(unique)<10 else 'computed','studies':len(unique),'replicates':replicates,'seed':seed,'resampling_unit':'whole_study','intervals':intervals}

"""Bounded nested grouped LightGBM experiment. No work occurs on import."""
import ast
from collections.abc import Callable
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SCENARIO = 'historical_hypothetical_unit_scenario'
REFERENCE = 0.4726308398591973
BASE_NAME = 'original_reproduced_fixed'
OPT_NAME = 'nested_optimized_procedure'


def dump(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)))
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def original_namespace(root):
    """Execute only verbatim loading/feature cells and selected pure helper definitions."""
    src = (root/'microplastics.py').read_text()
    notebook = json.loads((root/'microplastics.ipynb').read_text())
    ns = dict(globals(), DATA=root/'data/noaa_microplastics.csv', SEED=0)
    for i in (3, 10):
        text = ''.join(notebook['cells'][i]['source'])
        assert text.strip() in src
        exec(compile(text, f'microplastics.ipynb:cell{i}', 'exec'), ns)
    wanted = {'NUMERIC', 'CATEGORICAL', 'FEATURES', 'Predictor', 'Model', 'preprocessor', 'clip_to_range', 'sklearn_model'}
    for n in ast.parse(src).body:
        names = {t.id for t in getattr(n, 'targets', []) if isinstance(t, ast.Name)}
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            names.add(n.target.id)
        if getattr(n, 'name', None) in wanted or names & wanted:
            exec(compile(ast.get_source_segment(src, n), 'microplastics.py:extracted', 'exec'), ns)
    return ns


def metric(y, p, studies):
    y, p = np.asarray(y), np.asarray(p)
    e = p-y
    s = pd.DataFrame({'study': np.asarray(studies), 'ae': abs(e), 'se': e**2}).groupby('study').mean()
    denominator = np.sum((y-y.mean())**2)
    return dict(R2=float(1-np.sum(e**2)/denominator) if len(y)>1 and denominator>0 else None,
                MAE=float(abs(e).mean()), RMSE=float(np.sqrt(np.mean(e**2))),
                study_macro_MAE=float(s.ae.mean()), study_macro_MSE=float(s.se.mean()),
                median_study_MAE=float(s.ae.median()), p90_study_MAE=float(s.ae.quantile(.9)),
                worst_study_MAE=float(s.ae.max()), within_1_log=float((abs(e)<=1).mean()))


def select(table):
    """Predeclared safety + improvement gate, then best-R2 tie band, macro MAE, ID."""
    b = table.loc[table.candidate == 'C0_original'].iloc[0]
    admissible = table[(table.candidate == 'C0_original') | ((table.R2 > b.R2) & (table.study_macro_MAE <= 1.05*b.study_macro_MAE))]
    best = admissible.R2.max()
    band = admissible[admissible.R2 >= best-0.005]
    chosen = band.sort_values(['study_macro_MAE','candidate'], kind='stable').iloc[0]
    return chosen.candidate, (f"Safety gate <= {1.05*b.study_macro_MAE:.12g}; nonbaseline must strictly improve C0 R2 {b.R2:.12g}. "
                              f"{len(admissible)} admissible; best R2 {best:.12g}; {len(band)} within 0.005; "
                              f"lowest macro MAE then ID chooses {chosen.candidate}.")


class Experiment:
    def __init__(self, out):
        self.out = Path(out).resolve()
        self.root = self.out.parent
        assert self.root.name == 'marine-microplastics-baseline'
        self.rep = self.root/'reproduction_outputs'
        self.ns = None
        self.checks = []
        self.cache_hits = 0
        self.new_fits = 0

    def check(self, name, condition):
        if not condition:
            raise AssertionError(name)
        self.checks.append(name)

    def load(self):
        names = ['REPORT.md','summary.json','fold_metrics.csv','out_of_fold_predictions.csv','model_parameters.json',
                 'unit_adjustments.json','lightgbm_reproduction.ipynb','environment.json']
        self.inputs = {str((self.rep/n).relative_to(self.root)): sha(self.rep/n) for n in names}
        self.inputs.update({p: sha(self.root/p) for p in ['microplastics.py','microplastics.ipynb','data/noaa_microplastics.csv','uv.lock','optimization.md']})
        dump(self.out/'input_artifacts.json', {'scenario': SCENARIO, 'repository': str(self.root), 'relative_paths_sha256': self.inputs})
        self.baseline = pd.read_csv(self.rep/'out_of_fold_predictions.csv', float_precision='round_trip')
        self.reference_summary = json.loads((self.rep/'summary.json').read_text())
        self.base_params = json.loads((self.rep/'model_parameters.json').read_text())
        self.ns = original_namespace(self.root)
        self.df = self.ns['df']
        self.X = self.df[self.ns['FEATURES']].copy()
        self.y = self.df.y.to_numpy()
        self.groups = self.df.study.to_numpy()
        self.outer = [(np.flatnonzero(self.baseline.fold.to_numpy()!=f), np.flatnonzero(self.baseline.fold.to_numpy()==f)) for f in range(1,6)]
        self.inner = {}
        split_log = []
        for f,(tr,te) in enumerate(self.outer,1):
            self.inner[f] = [(tr[a],tr[b]) for a,b in GroupKFold(3).split(self.X.iloc[tr], groups=self.groups[tr])]
            for j,(a,b) in enumerate(self.inner[f],1):
                split_log.append(dict(outer=f, inner=j, train_rows=len(a), validation_rows=len(b),
                                      train_groups=len(set(self.groups[a])), validation_groups=len(set(self.groups[b])), scenario=SCENARIO))
        dump(self.out/'frozen_splits.json', {'scenario':SCENARIO,'outer':[[a.tolist(),b.tolist()] for a,b in self.outer],
                                           'inner':{f:[[a.tolist(),b.tolist()] for a,b in folds] for f,folds in self.inner.items()}})
        pd.DataFrame(split_log).to_csv(self.out/'inner_split_counts.csv', index=False)
        self.validate_data()
        return pd.DataFrame(split_log)

    def validate_data(self):
        b,df = self.baseline,self.df
        self.check('16988 rows, 72 studies, raw 29076x35', len(df)==16988 and df.study.nunique()==72 and self.ns['raw'].shape==(29076,35))
        self.check('baseline exact contiguous row IDs', np.array_equal(b.water_row,np.arange(len(df))) and b.source_row.is_unique)
        self.check('exact saved study IDs', np.array_equal(b.study,df.study))
        self.check('exact saved labels round-trip', np.array_equal(b.target_log10,self.y))
        mask = (self.ns['data'].medium=='Ocean water') & (self.ns['data'].unit=='pieces/m3')
        self.check('exact original source rows', np.array_equal(self.ns['data'].index[mask],b.source_row))
        self.check('fixed outer sizes', [len(te) for _,te in self.outer]==[6135,2714,2712,2714,2713])
        self.check('saved folds also match original GroupKFold', all(np.array_equal(a,c) and np.array_equal(b,d) for (a,b),(c,d) in zip(self.outer,GroupKFold(5).split(df,groups=df.study))))
        self.check('no target or study-derived predictors', self.ns['FEATURES']==['lat','lon','year','month_sin','month_cos','log_mesh','log_depth','method_family'])
        base = self.ns['data'].loc[mask].reset_index(drop=True)
        audit = json.loads((self.rep/'unit_adjustments.json').read_text())
        mult = {v['doi']:v['multiplier'] for v in audit}
        self.check('historical unit multipliers and offset frozen', mult==self.ns['UNIT_FIXES'] and self.ns['C']==.001 and np.array_equal(df.conc,base.conc*base.doi.map(mult).fillna(1)))
        self.check('historical adjustment row counts', [int((base.doi==v['doi']).sum()) for v in audit]==[1389,81])
        self.check('target log10 exact', np.array_equal(self.y,np.log10(df.conc+.001)))
        # Missing/placeholder identifiers cannot accidentally bridge otherwise unrelated rows.
        missing_tokens = {'','nan','none','null','na','n/a','not available','unknown','not applicable'}
        for c in ['doi','accession']:
            self.check(f'no missing/placeholder {c} joining groups', df[c].notna().all() and not df[c].astype(str).str.strip().str.lower().isin(missing_tokens).any())
        parent=list(range(len(df)))
        def root(i):
            while parent[i]!=i:
                parent[i]=parent[parent[i]]
                i=parent[i]
            return i
        for col in ['doi','accession']:
            for ids in df.groupby(col).indices.values():
                for i in ids[1:]: parent[root(int(i))]=root(int(ids[0]))
        components=pd.Series([root(i) for i in range(len(df))])
        self.check('independent connected-components grouping', components.groupby(df.study).nunique().max()==1 and df.study.groupby(components).nunique().max()==1)
        self.check('outer test coverage once', np.array_equal(np.sort(np.concatenate([b for a,b in self.outer])),np.arange(len(df))))
        for f,(tr,te) in enumerate(self.outer,1):
            self.check(f'outer {f} disjoint rows and studies', not set(tr)&set(te) and not set(self.groups[tr])&set(self.groups[te]))
            for j,(a,b) in enumerate(self.inner[f],1):
                self.check(f'inner {f}/{j} train only, disjoint studies and multiple train groups', set(a)|set(b)==set(tr) and not set(a)&set(b) and not set(self.groups[a])&set(self.groups[b]) and len(set(self.groups[a]))>1)
            self.check(f'inner {f} test coverage once', np.array_equal(np.sort(np.concatenate([b for a,b in self.inner[f]])),tr))
        self.check('baseline saved predictions reproduce reference', metric(self.y,self.baseline.prediction_log10,self.groups)['R2']==REFERENCE)

    def freeze_registry(self):
        deltas = {
            'C0_original': {},
            'C1_shallow': dict(max_depth=4,num_leaves=15,min_child_samples=40),
            'C2_conservative': dict(max_depth=5,num_leaves=20,min_child_samples=70,reg_lambda=5.),
            'C3_strong_regularization': dict(max_depth=5,num_leaves=20,min_child_samples=50,reg_alpha=.5,reg_lambda=15.),
            'C4_feature_subsample': dict(max_depth=5,num_leaves=20,min_child_samples=50,colsample_bytree=.8,reg_lambda=5.),
            'C5_row_and_feature_subsample': dict(max_depth=5,num_leaves=20,min_child_samples=50,colsample_bytree=.8,reg_lambda=5.,subsample=.8,subsample_freq=1),
            'C6_slower': dict(learning_rate=.03,n_estimators=800,max_depth=5,num_leaves=20,min_child_samples=50,reg_lambda=5.),
            'C7_fewer_trees': dict(learning_rate=.05,n_estimators=250,max_depth=5,num_leaves=20,min_child_samples=50,reg_lambda=5.),
        }
        self.registry = {k:self.base_params|v for k,v in deltas.items()}
        valid=set(LGBMRegressor(verbose=-1).get_params())
        self.check('all eight configurations use valid installed parameter names', len(self.registry)==8 and all(set(v)<=valid for v in self.registry.values()))
        payload={'scenario':SCENARIO,'configs':self.registry,'differences_from_baseline':deltas,
                 'selection':'Nonbaseline eligible only if R2 > C0 and macro MAE <= 1.05*C0. Include C0 in admissible set. Within 0.005 of best admissible R2, minimize macro MAE, then lexicographic ID. Exact macro-MAE ties use ID order.',
                 'material_deterioration_reporting':'MAE increase > 0.10 log10 units (descriptive flag, not selection); also report >5% macro deterioration.',
                 'budget':{'inner':120,'outer':5,'verification':2},'seed':0,'threads':4,'early_stopping':False}
        path=self.out/'candidate_registry.json'
        if path.exists():
            self.check('registry unchanged across resume', json.loads(path.read_text())==payload)
        else: dump(path,payload)
        self.versions={p:metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','lightgbm']}
        self.key_base={'code':sha(Path(__file__)),'inputs':self.inputs,'splits':sha(self.out/'frozen_splits.json'),
                       'registry':sha(path),'versions':self.versions,'scenario':SCENARIO}
        dump(self.out/'checkpoint_identity.json',self.key_base)
        (self.out/'checkpoints').mkdir(exist_ok=True)
        return pd.DataFrame([{'candidate':k,**v} for k,v in deltas.items()])

    def fit_predict(self, train_X, train_y, test_X, params):
        # Exact original wrapper: pipeline on training rows, clipping only at prediction.
        train=train_X.copy()
        train['y']=np.asarray(train_y)
        return self.ns['sklearn_model'](LGBMRegressor(**params))(train)(test_X)

    def cached_fit(self, stage, f, j, cid, train_ids, test_ids, train_y):
        key=fingerprint(self.key_base | {'stage':stage,'outer':f,'inner':j,'candidate':cid,
                                        'train_ids':train_ids.tolist(),'test_ids':test_ids.tolist(),
                                        'train_y_sha':hashlib.sha256(np.asarray(train_y).tobytes()).hexdigest()})
        path=self.out/'checkpoints'/f'{key}.npz'
        if path.exists():
            with np.load(path) as z:
                self.check('checkpoint row IDs match '+key[:8],np.array_equal(z['test_ids'],test_ids))
                pred=z['pred'].copy()
            self.cache_hits+=1
            return pred
        log=self.out/'fit_log.jsonl'
        events=[json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
        if stage=='inner' and sum(e['stage']=='inner' and e['event']=='started' for e in events)>=120:
            raise RuntimeError('120-inner-fit bound reached; refusing an extra fit. Use valid checkpoints or report failure.')
        if stage=='outer' and sum(e['stage']=='outer' and e['event']=='started' for e in events)>=5:
            raise RuntimeError('Five outer-refit bound reached.')
        if stage=='verification' and sum(e['stage']=='verification' and e['event']=='started' for e in events)>=2:
            raise RuntimeError('Two verification-fit bound reached.')
        event=dict(stage=stage,outer=f,inner=j,candidate=cid,key=key,train_rows=len(train_ids),test_rows=len(test_ids),scenario=SCENARIO)
        with log.open('a') as h: h.write(json.dumps(event|{'event':'started','time':time.time()})+'\n')
        start=time.time()
        pred=self.fit_predict(self.X.iloc[train_ids],train_y,self.X.iloc[test_ids],self.registry[cid])
        assert np.isfinite(pred).all()
        tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as h: np.savez_compressed(h,pred=pred,test_ids=test_ids)
        tmp.replace(path)
        with log.open('a') as h: h.write(json.dumps(event|{'event':'completed','seconds':time.time()-start})+'\n')
        self.new_fits+=1
        return pred

    def inner_search(self, f, training_y):
        """API receives ONLY outer-training targets; no outer-test target access."""
        tr,_=self.outer[f-1]
        assert len(training_y)==len(tr)
        lookup=pd.Series(np.asarray(training_y),index=tr)
        records=[]
        for cid in self.registry:
            p=pd.Series(np.nan,index=tr,dtype=float)
            for j,(a,b) in enumerate(self.inner[f],1):
                p.loc[b]=self.cached_fit('inner',f,j,cid,a,b,lookup.loc[a].to_numpy())
            assert np.isfinite(p).all()
            records.append(dict(outer=f,candidate=cid,scenario=SCENARIO,**metric(training_y,p.to_numpy(),self.groups[tr])))
        table=pd.DataFrame(records)
        selected,reason=select(table)
        return table,dict(outer=f,candidate=selected,reason=reason,scenario=SCENARIO)

    def tune(self):
        metrics,selections=[],[]
        for f,(tr,_) in enumerate(self.outer,1):
            table,selection=self.inner_search(f,self.y[tr])
            metrics.append(table); selections.append(selection)
            print('Inner selection:',f,selection['candidate'],flush=True)
            pd.concat(metrics).to_csv(self.out/'inner_metrics_by_outer_fold.csv',index=False)
            pd.DataFrame(selections).to_csv(self.out/'outer_selections.csv',index=False)
        self.inner_metrics=pd.concat(metrics,ignore_index=True)
        self.selections=pd.DataFrame(selections)
        return self.selections

    def refit(self):
        p=np.full(len(self.y),np.nan)
        for f,(tr,te) in enumerate(self.outer,1):
            cid=self.selections.loc[self.selections.outer==f,'candidate'].iloc[0]
            p[te]=self.cached_fit('outer',f,0,cid,tr,te,self.y[tr])
            print('Outer refit completed:',f,cid,flush=True)
        out=self.baseline.rename(columns={'prediction_log10':'original_prediction_log10'}).copy()
        out['optimized_prediction_log10']=p
        out['method_family']=self.df.method_family
        counts=self.df.study.value_counts()
        out['study_size']=self.df.study.map(counts)
        out['study_size_bin']=pd.cut(out.study_size,[0,50,200,1000,np.inf],labels=['1-50','51-200','201-1000','1001+']).astype(str)
        out['selected_candidate']=[self.selections.loc[self.selections.outer==f,'candidate'].iloc[0] for f in out.fold]
        out['scenario']=SCENARIO
        out.to_csv(self.out/'optimized_oof_predictions.csv',index=False)
        self.oof=out
        return out

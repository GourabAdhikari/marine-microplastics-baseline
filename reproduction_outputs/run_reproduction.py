"""Build and execute only the original LightGBM grouped-CV path."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reproduction_outputs'
os.chdir(ROOT)
for key, sub in [('JUPYTER_DATA_DIR','jupyter'), ('JUPYTER_RUNTIME_DIR','runtime'), ('IPYTHONDIR','ipython'), ('MPLCONFIGDIR','matplotlib'), ('TMPDIR','tmp'), ('PYTHONPYCACHEPREFIX','pycache')]:
    os.environ[key] = str(OUT / sub)
    (OUT / sub).mkdir(exist_ok=True)
import nbformat
from nbclient import NotebookClient
paths = subprocess.check_output(['git','ls-files'], text=True).splitlines()
hashes = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
(OUT/'input_hashes.json').write_text(json.dumps(hashes, indent=2))
original = json.loads((ROOT/'microplastics.ipynb').read_text())
source = (ROOT/'microplastics.py').read_text()
tree = ast.parse(source)
def node(name):
    for n in tree.body:
        names = [t.id for t in getattr(n,'targets',[]) if isinstance(t,ast.Name)]
        if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name): names.append(n.target.id)
        if getattr(n,'name',None)==name or name in names:
            return ast.get_source_segment(source,n)
    raise KeyError(name)
def cell(i):
    s=''.join(original['cells'][i]['source'])
    # Ensure copied notebook code is also present verbatim in paired Python source.
    assert s.strip() in source
    return s
cells=[]
def md(s): cells.append(nbformat.v4.new_markdown_cell(s))
def code(s): cells.append(nbformat.v4.new_code_cell(s))
md('# Original LightGBM grouped five-fold reproduction\nOnly the original preprocessing and LightGBM grouped validation are executed. No searches or other models. Source cells are copied verbatim; model helper definitions are extracted verbatim by AST from `microplastics.py`. Historical unit assumptions are reproduced, not revalidated. All generated files remain in `reproduction_outputs/`.')
code('''import json, hashlib, platform, sys, importlib.metadata as metadata
from pathlib import Path
from collections.abc import Callable
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score
ROOT = Path.cwd()
OUT = ROOT / 'reproduction_outputs'
DATA = ROOT / 'data/noaa_microplastics.csv'
SEED = 0
pd.set_option('display.precision', 9)
environment = {'python': sys.version, 'platform': platform.platform(), 'packages': {p: metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','lightgbm','nbclient','ipykernel']}}
(OUT/'environment.json').write_text(json.dumps(environment, indent=2))
print(environment)''')
md('## Original loading, historical unit adjustments, features and folds\nThe original split cell also constructs random and chronological indices; neither is fitted or scored.')
for i in [3,10,20]: code(cell(i))
code('\n\n'.join(node(n) for n in ['Predictor','Model','preprocessor','clip_to_range','sklearn_model','BOOSTER_DEFAULTS','metrics','score','out_of_fold']) + '\n\nmodel = sklearn_model(LGBMRegressor(**BOOSTER_DEFAULTS, verbose=-1))\n(OUT/"model_parameters.json").write_text(json.dumps(LGBMRegressor(**BOOSTER_DEFAULTS, verbose=-1).get_params(), indent=2))')
md('## Focused integrity and preprocessing tests')
code('''checks = []
def check(name, condition):
    assert condition, name
    checks.append(name)
check('raw shape matches saved notebook', raw.shape == (29076,35))
check('modelling counts match saved notebook', (len(df),df.accession.nunique(),df.doi.nunique(),df.study.nunique()) == (16988,73,73,72))
check('finite target; numeric features contain no infinity', np.isfinite(df.y).all() and not np.isinf(df[NUMERIC].to_numpy()).any())
base = data[(data.medium == 'Ocean water') & (data.unit == 'pieces/m3')].reset_index(drop=True)
check('historical scaling exact', np.array_equal(df.conc.to_numpy(), (base.conc * base.doi.map(UNIT_FIXES).fillna(1)).to_numpy()))
check('log target exact', np.array_equal(df.y.to_numpy(), np.log10(df.conc.to_numpy()+C)))
# Independent union-find verifies connected components of DOI/accession links.
parent = list(range(len(base)))
def root(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i
for column in ['doi','accession']:
    for indices in base.groupby(column).indices.values():
        for i in indices[1:]: parent[root(int(i))] = root(int(indices[0]))
components = pd.Series([root(i) for i in range(len(base))])
check('study grouping equals independent connected components', components.groupby(df.study).nunique().max()==1 and df.study.groupby(components).nunique().max()==1)
fold_id = np.zeros(len(df),dtype=int)
check('original fold sizes', [len(t) for _,t in SPLITS['grouped']] == [6135,2714,2712,2714,2713])
for f,(tr,te) in enumerate(SPLITS['grouped'],1):
    check(f'fold {f} no row or study leakage', not set(tr)&set(te) and not set(df.study.iloc[tr])&set(df.study.iloc[te]) and len(tr)+len(te)==len(df))
    check(f'fold {f} no DOI/accession leakage', all(not set(df[c].iloc[tr])&set(df[c].iloc[te]) for c in ['doi','accession']))
    check(f'fold {f} unique test assignment', (fold_id[te]==0).all())
    fold_id[te]=f
    prep=preprocessor().fit(df.iloc[tr][FEATURES])
    check(f'fold {f} training-only median imputation', np.allclose(prep.named_transformers_['num'].named_steps['simpleimputer'].statistics_, df.iloc[tr][NUMERIC].median()))
check('every row tested once', (fold_id>0).all())
unit_audit = [{'doi': doi, 'multiplier': factor, 'rows': int((base.doi==doi).sum()), 'raw_mean': float(base.loc[base.doi==doi,'conc'].mean()), 'adjusted_mean': float(df.loc[df.doi==doi,'conc'].mean())} for doi,factor in UNIT_FIXES.items()]
(OUT/'unit_adjustments.json').write_text(json.dumps(unit_audit, indent=2))
print(unit_audit)
print(f'{len(checks)} preprocessing tests passed')''')
md('## Execute the unchanged original out-of-fold helper (five LightGBM fits)')
code('''pred = out_of_fold(model, 'grouped')
check('finite complete predictions', np.isfinite(pred).all() and len(pred)==len(df))
pooled = score(pred)
check('R2 agrees with sklearn', np.isclose(pooled['R2'], r2_score(df.y,pred), atol=1e-14))
fold_metrics = pd.DataFrame([{'fold': f, 'train_rows':len(tr), 'test_rows':len(te), 'train_studies':df.study.iloc[tr].nunique(), 'test_studies':df.study.iloc[te].nunique(), **score(pred.iloc[te])} for f,(tr,te) in enumerate(SPLITS['grouped'],1)])
fold_metrics.to_csv(OUT/'fold_metrics.csv',index=False)
pd.DataFrame({'water_row':df.index, 'source_row':data.index[(data.medium=='Ocean water') & (data.unit=='pieces/m3')], 'study':df.study, 'doi':df.doi, 'accession':df.accession, 'fold':fold_id, 'target_log10':df.y, 'prediction_log10':pred}).to_csv(OUT/'out_of_fold_predictions.csv',index=False)
summary = {'raw_rows':len(raw), 'raw_columns':len(raw.columns), 'modelling_rows':len(df), 'studies':df.study.nunique(), 'accessions':df.accession.nunique(), 'publications':df.doi.nunique(), 'pooled':pooled, 'fold_mean_R2':float(fold_metrics.R2.mean()), 'reported_R2':0.473, 'difference_from_reported':float(pooled['R2']-0.473), 'matches_reported_to_3_decimals':round(float(pooled['R2']),3)==0.473}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=float))
print(json.dumps(summary,indent=2,default=float))
display(fold_metrics)
''')
code('''before = json.loads((OUT/'input_hashes.json').read_text())
check('all original tracked files unchanged', all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in before.items()))
(OUT/'tests.json').write_text(json.dumps({'passed':len(checks),'checks':checks},indent=2))
print(f'{len(checks)} focused tests passed; source and dataset unchanged.')''')
nb=nbformat.v4.new_notebook(cells=cells)
kdir=OUT/'jupyter/kernels/reproduction'
kdir.mkdir(parents=True,exist_ok=True)
(kdir/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Reproduction isolated environment','language':'python'}))
nb.metadata['kernelspec']={'name':'reproduction','display_name':'Reproduction isolated environment','language':'python'}
path=OUT/'lightgbm_reproduction.ipynb'
nbformat.write(nb,path)
try:
    NotebookClient(nb,timeout=600,kernel_name='reproduction',resources={'metadata':{'path':str(ROOT)}}).execute()
finally:
    nbformat.write(nb,path)
print('Executed',path)

"""Execute the frozen notebook or its tests. Run from the repository root."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tests',action='store_true')
    args=parser.parse_args()
    out=Path(__file__).resolve().parents[1]
    root=out.parent
    os.chdir(root)
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    os.environ['GIT_OPTIONAL_LOCKS']='0'
    for key,sub in [('JUPYTER_DATA_DIR','jupyter'),('JUPYTER_RUNTIME_DIR','runtime'),('IPYTHONDIR','ipython'),('MPLCONFIGDIR','matplotlib'),('TMPDIR','tmp'),('XDG_CACHE_HOME','cache')]:
        path=out/sub;path.mkdir(exist_ok=True)
        os.environ[key]=str(path)
    # Single-process sequential fits; each LightGBM learner keeps the original four-thread setting.
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[key]='4'
    if args.tests:
        import pandas as pd
        from experiment import Experiment
        from verification import run_tests
        from analysis_report import report
        exp=Experiment(out);exp.load();exp.freeze_registry()
        exp.inner_metrics=pd.read_csv(out/'inner_metrics_by_outer_fold.csv',float_precision='round_trip')
        exp.selections=pd.read_csv(out/'outer_selections.csv')
        run_tests(exp);report(exp)
        return
    import nbformat
    from nbclient import NotebookClient
    cells=[]
    def section(title,source):
        cells.append(nbformat.v4.new_markdown_cell('## '+title))
        cells.append(nbformat.v4.new_code_cell(source))
    cells.append(nbformat.v4.new_markdown_cell('# Bounded nested LightGBM optimization\n\n**historical_hypothetical_unit_scenario** throughout. Unit multipliers are unresolved assumptions. Previously inspected outer folds make all results exploratory, not independent prospective validation. Re-execution resumes exactly this experiment from hashed checkpoints; no additional search.'))
    section('1. Setup and read-only verification',"""import sys, json
from pathlib import Path
from IPython.display import display, SVG, Markdown
ROOT=Path.cwd()
OUT=ROOT/'optimization_outputs'
sys.path.insert(0,str(OUT/'source'))
from experiment import Experiment, SCENARIO
from analysis_report import analyze, plots, report
from verification import run_tests
exp=Experiment(OUT)
print('Repository:',ROOT,'\\nScenario:',SCENARIO)
print((OUT/'git_status_before.txt').read_text())
print((OUT/'environment.json').read_text())""")
    section('2. Baseline artifacts and protected hashes',"""hashes=json.loads((OUT/'protected_hashes_before.json').read_text())
print('Protected paths:',len(hashes))
print('Dataset SHA256:',hashes['data/noaa_microplastics.csv'])
print((ROOT/'reproduction_outputs/summary.json').read_text())
# All baseline files are read-only; no original notebook training is executed.""")
    section('3. Dataset, linked study groups and frozen fold assertions',"""split_counts=exp.load()
display(split_counts)
print('Data checks passed:',len(exp.checks))
print((OUT/'input_artifacts.json').read_text())""")
    section('4. Frozen registry and precise selection rule (before tuned outer outcomes)',"""display(exp.freeze_registry())
print(json.loads((OUT/'candidate_registry.json').read_text())['selection'])
print('120 inner fits maximum, 5 outer refits, 2 deterministic verification fits. No early stopping.')""")
    section('5. Inner grouped tuning on outer-training observations only',"""display(exp.tune())
display(exp.inner_metrics)
# Finish all five selections before any outer selected-model scoring.""")
    section('6. Selected outer refits and aligned predictions',"""oof=exp.refit()
display(oof.head())
print('OOF rows:',len(oof),'finite predictions:',oof.optimized_prediction_log10.notna().all())""")
    section('7. Paired metrics and per-study analysis',"""display(analyze(exp))
import pandas as pd
display(pd.read_csv(OUT/'paired_per_fold_metrics.csv'))
display(pd.read_csv(OUT/'paired_per_study_metrics.csv').sort_values('delta_MAE',ascending=False))""")
    section('8. Diagnostic tables and plots',"""display(pd.read_csv(OUT/'performance_by_method_and_study_size.csv'))
print((OUT/'largest_study_influence.json').read_text())
for path in plots(exp):
    display(SVG(filename=str(path)))""")
    section('9. Report generation',"""summary=report(exp)
print('Original R2:',summary['original_reproduced_fixed']['R2'])
print('Optimized R2:',summary['nested_optimized_procedure']['R2'])
print('Delta:',summary['delta_R2'])
print('Report written; final integrity section will be updated after tests.')""")
    section('10. Automated tests, final integrity verification and finalized report',"""checks=run_tests(exp)
summary=report(exp)
print('New fits this kernel:',exp.new_fits,'cache hits:',exp.cache_hits)
print('Tests passed:',checks['passed'])
display(Markdown((OUT/'OPTIMIZATION_REPORT.md').read_text()))
print('Bounded experiment complete. No subsequent phase started.')""")
    nb=nbformat.v4.new_notebook(cells=cells)
    kd=out/'jupyter/kernels/optimization';kd.mkdir(parents=True,exist_ok=True)
    (kd/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Frozen LightGBM optimization','language':'python'}))
    nb.metadata['kernelspec']={'name':'optimization','display_name':'Frozen LightGBM optimization','language':'python'}
    path=out/'LightGBM_Optimization_EXECUTED.ipynb'
    nbformat.write(nb,path)
    try:
        NotebookClient(nb,timeout=1800,kernel_name='optimization',resources={'metadata':{'path':str(root)}}).execute()
    finally:
        nbformat.write(nb,path)
        statuses=[{'cell_index':i,'execution_count':c.execution_count,'executed':c.execution_count is not None,'errors':[o.get('evalue') for o in c.get('outputs',[]) if o.output_type=='error'],'output_types':[o.output_type for o in c.get('outputs',[])]} for i,c in enumerate(nb.cells) if c.cell_type=='code']
        (out/'notebook_execution_status.json').write_text(json.dumps(statuses,indent=2))
    print('Executed:',path)
    print((out/'optimization_summary.json').read_text())


if __name__=='__main__':
    main()

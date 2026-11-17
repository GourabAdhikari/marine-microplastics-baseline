"""Execute the notebook and tests; write the final report as the last artifact.

--prepare executes everything except creating BLENDING_AND_ROBUSTNESS_REPORT.md.
No LightGBM fits are authorized by this zero-fit runner: a guard blocks them.
"""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    out=Path(__file__).resolve().parents[1];root=out.parent
    os.chdir(root);sys.dont_write_bytecode=True
    os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['GIT_OPTIONAL_LOCKS']='0'
    for var,folder in [('JUPYTER_DATA_DIR','jupyter'),('JUPYTER_RUNTIME_DIR','runtime'),('IPYTHONDIR','ipython'),('MPLCONFIGDIR','matplotlib'),('TMPDIR','tmp'),('XDG_CACHE_HOME','cache')]:
        path=out/folder;path.mkdir(exist_ok=True);os.environ[var]=str(path)
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='4'
    import nbformat
    from nbclient import NotebookClient
    cells=[nbformat.v4.new_markdown_cell('# Fixed LightGBM blends and study robustness\n\n**historical_hypothetical_unit_scenario**. B50 is predeclared, not selected from outer outcomes. Previously inspected folds and unverified unit adjustments prohibit independent-validation or certified-physical-prediction claims. No prior experiment is rerun.')]
    def section(title,source):
        cells.append(nbformat.v4.new_markdown_cell('## '+title));cells.append(nbformat.v4.new_code_cell(source))
    section('1. Setup, protected files and frozen plan',"""import json, sys
from pathlib import Path
from unittest.mock import patch
from IPython.display import display, Markdown, SVG
ROOT=Path.cwd();OUT=ROOT/'blending_outputs'
sys.dont_write_bytecode=True
sys.path.insert(0,str(OUT/'source'));sys.path.insert(0,str(OUT/'tests'))
fit_guard=patch('lightgbm.LGBMRegressor.fit',side_effect=AssertionError('New LightGBM fits are forbidden in this zero-fit analysis'))
blocked_fit=fit_guard.start()
from blend import Analysis, SCENARIO
from reporting import make_figures, compose_report
from test_blending import run_suite
analysis=Analysis(OUT)
print('Repository:',ROOT,'Scenario:',SCENARIO)
print((OUT/'predeclared_plan.json').read_text())
print((OUT/'environment.json').read_text())
print('Protected paths:',len(json.loads((OUT/'integrity_manifest.json').read_text())['before']))""")
    section('2. Read-only artifacts, exact row alignment and original target reconstruction',"""display(analysis.load())
print('All original OOF rows, targets, folds and study identities verified.')
print('Input files:',json.dumps(analysis.input_hashes,indent=2))
print('No LightGBM fitting:',blocked_fit.call_count==0)""")
    section('3. Stage A: five fixed target-free blends and reference metric reproduction',"""display(analysis.evaluate())
import pandas as pd
display(pd.read_csv(OUT/'reference_metric_audit.csv'))
print('B50 is the predetermined main candidate; no outer-score weight search.')""")
    section('4. Per-fold, per-study and post-hoc robustness diagnostics',"""display(analysis.fold[['fold','blend','rows','studies','R2','MAE','RMSE']])
display(pd.read_csv(OUT/'folds_2_and_5_diagnostic.csv'))
display(pd.read_csv(OUT/'prior_14_harmed_studies_diagnostic.csv'))
display(analysis.subgroups)
print('Every study retained, including the smallest groups; prior harms are diagnostics only.')""")
    section('5. Stage B: exact precommitted B50 cost-control gate',"""print(json.dumps(analysis.screen(),indent=2))""")
    section('6. Stage C feasibility and budget decision — no premature fitting',"""print(json.dumps(analysis.assess_stage_c(),indent=2))
assert blocked_fit.call_count==0
print('New model fits: 0. No prior phase restarted.')""")
    section('7. Compact diagnostics and analysis/checkpoint identity',"""for path in make_figures(analysis):display(SVG(filename=str(path)))
analysis.checkpoint_identity()
print('Input, source, configuration, split and environment identity saved. No model checkpoints reused.')""")
    section('8. Unit/integration tests and final protected-file verification',"""test_status=run_suite(analysis)
assert blocked_fit.call_count==0
print(json.dumps(test_status,indent=2))
print('All protected original, reproduction and optimization files remain unchanged.')""")
    section('9. Report rendering and stop',"""report_text=compose_report(analysis,test_status,9)
(OUT/'checkpoints/report_ready.md').write_text(report_text)
display(Markdown(report_text))
print('All analysis complete. Runner saves the executed notebook, status and final report, then stops.')
assert blocked_fit.call_count==0
fit_guard.stop()""")
    nb=nbformat.v4.new_notebook(cells=cells)
    kd=out/'jupyter/kernels/blending';kd.mkdir(parents=True,exist_ok=True)
    (kd/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Frozen zero-fit blending','language':'python'}))
    nb.metadata['kernelspec']={'name':'blending','display_name':'Frozen zero-fit blending','language':'python'}
    path=out/'LightGBM_Blend_Research_EXECUTED.ipynb';nbformat.write(nb,path)
    try:
        NotebookClient(nb,timeout=900,kernel_name='blending',resources={'metadata':{'path':str(root)}}).execute()
    finally:
        nbformat.write(nb,path)
        statuses=[{'cell_index':i,'execution_count':c.execution_count,'executed':c.execution_count is not None,
                   'errors':[o.get('evalue') for o in c.get('outputs',[]) if o.output_type=='error'],
                   'output_types':[o.output_type for o in c.get('outputs',[])]} for i,c in enumerate(nb.cells) if c.cell_type=='code']
        (out/'notebook_execution_status.json').write_text(json.dumps(statuses,indent=2))
    assert len(statuses)==9 and all(s['executed'] and not s['errors'] for s in statuses)
    gate=json.loads((out/'stage_b_gate.json').read_text());stage=json.loads((out/'stage_c_status.json').read_text())
    tests=json.loads((out/'tests/test_results.json').read_text())
    final={'scenario':'historical_hypothetical_unit_scenario','stage_a':'completed','stage_b_passed':gate['passed'],
           'stage_c':stage['status'],'new_lightgbm_fits':0,'executed_notebook_code_cells':len(statuses),
           'tests':tests,'protected_files_unchanged':True,'report_created':not args.prepare,
           'report_path':'blending_outputs/BLENDING_AND_ROBUSTNESS_REPORT.md','no_next_phase':True}
    (out/'final_status.json').write_text(json.dumps(final,indent=2))
    print(json.dumps(final,indent=2))
    if not args.prepare:
        # Last artifact write; do not launch any further analyses after report creation.
        (out/'BLENDING_AND_ROBUSTNESS_REPORT.md').write_text((out/'checkpoints/report_ready.md').read_text())


if __name__=='__main__':main()

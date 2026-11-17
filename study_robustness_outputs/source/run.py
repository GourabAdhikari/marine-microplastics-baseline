"""Build/execute Stage 4 in a fresh kernel; final report is the last artifact.

--prepare runs the whole experiment and tests but defers the final report.
Completed Stage B cannot start new fits on rerun; it only reads matching caches.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    out=Path(__file__).resolve().parents[1];root=out.parent;os.chdir(root)
    sys.dont_write_bytecode=True;os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['GIT_OPTIONAL_LOCKS']='0'
    for key,folder in [('JUPYTER_DATA_DIR','jupyter'),('JUPYTER_RUNTIME_DIR','runtime'),('IPYTHONDIR','ipython'),('MPLCONFIGDIR','matplotlib'),('TMPDIR','tmp'),('XDG_CACHE_HOME','cache')]:
        p=out/folder;p.mkdir(exist_ok=True);os.environ[key]=str(p)
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='4'
    import nbformat
    from nbclient import NotebookClient
    cells=[nbformat.v4.new_markdown_cell('# Study-robust selection among the original eight LightGBM settings\n\n**historical_hypothetical_unit_scenario**. Historical units remain unverified. All five outer outcomes were already inspected; this is exploratory, not independent validation. S1/S2 are specified by the prompt, not learned from blending outcomes. Maximum ten new outer fits; no inner fitting or architecture searches.')]
    def section(title,source):cells.extend([nbformat.v4.new_markdown_cell('## '+title),nbformat.v4.new_code_cell(source)])
    section('1. Setup, environment and global fit-call guard',"""import json,sys
from pathlib import Path
from IPython.display import display,Markdown,SVG
ROOT=Path.cwd();OUT=ROOT/'study_robustness_outputs'
sys.dont_write_bytecode=True
sys.path.insert(0,str(OUT/'source'));sys.path.insert(0,str(OUT/'tests'))
from audit import Context
from fit_budget import FitBudget,refit_locked
from evaluation import evaluate
from reporting import figures,compose,fit_accounting
from test_stage4 import run_tests
ctx=Context(OUT);budget=FitBudget(OUT);budget.__enter__()
print('Repository:',ROOT)
print((OUT/'environment.json').read_text())
print('Protected paths:',len(json.loads((OUT/'protected_hashes_before.json').read_text())))""")
    section('2. Read-only source, data, reference, configuration and split audit',"""display(ctx.load())
print('Identical row IDs, labels, scenario and folds verified. No historical fits rerun.')
print('New fits this kernel:',budget.calls_this_process)""")
    section('3. Reconstruct saved inner OOF metrics and verify all training studies',"""print(json.dumps(ctx.verify_inner_cache(),indent=2))
print('New fits this kernel:',budget.calls_this_process)
assert budget.calls_this_process==0""")
    section('4. Freeze S1 and S2 choices before any new outer prediction',"""display(ctx.lock())
print('Pre-fit timestamp:',ctx.locked['created_utc'])
print('Locked selection payload hash:',ctx.locked['payload_sha256'])
import pandas as pd
display(pd.read_csv(OUT/'selector_eligibility.csv'))
assert budget.calls_this_process==0""")
    section('5. Exact locked outer refits and verified reference/checkpoint reuse',"""display(refit_locked(ctx,budget).head())
print(json.dumps(fit_accounting(ctx),indent=2))
print('Fit guard now closes new calls. Outer metrics cannot trigger any further training.')""")
    section('6. Four-procedure paired evaluation and frozen descriptive screen',"""display(evaluate(ctx))
print((OUT/'screening.json').read_text())
display(pd.read_csv(OUT/'paired_fold_metrics.csv')[['fold','procedure','rows','studies','R2','MAE','RMSE']])""")
    section('7. All-study harms, method/size summaries and largest-study influence',"""display(pd.read_csv(OUT/'study_harm_counts.csv'))
display(pd.read_csv(OUT/'paired_study_metrics.csv'))
display(pd.read_csv(OUT/'method_and_study_size_metrics.csv'))
print((OUT/'largest_study_influence.json').read_text())
display(SVG(filename=str(figures(ctx))))""")
    section('8. Tests and final protected-file verification — no new fits',"""test_status=run_tests(ctx,budget)
print(json.dumps(test_status,indent=2))
print('New fit calls this kernel:',budget.calls_this_process)
print('Original source/data and all three previous output directories unchanged.')""")
    section('9. Report rendering and stop',"""report_text=compose(ctx,test_status,9)
(OUT/'checkpoints/report_ready.md').write_text(report_text)
display(Markdown(report_text))
print('No further model runs permitted. Runner saves notebook/status and writes final report last.')
budget.__exit__(None,None,None)""")
    nb=nbformat.v4.new_notebook(cells=cells)
    kd=out/'jupyter/kernels/study_robustness';kd.mkdir(parents=True,exist_ok=True)
    (kd/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Bounded study robustness','language':'python'}))
    nb.metadata['kernelspec']={'name':'study_robustness','display_name':'Bounded study robustness','language':'python'}
    path=out/'Study_Robustness_EXECUTED.ipynb';nbformat.write(nb,path)
    start=datetime.now(timezone.utc).isoformat();success=False;error=None
    try:
        NotebookClient(nb,timeout=1200,kernel_name='study_robustness',resources={'metadata':{'path':str(root)}}).execute()
        success=True
    except Exception as e:
        error=repr(e);raise
    finally:
        nbformat.write(nb,path)
        statuses=[{'cell_index':i,'execution_count':c.execution_count,'executed':c.execution_count is not None,
                   'errors':[o.get('evalue') for o in c.get('outputs',[]) if o.output_type=='error'],
                   'output_types':[o.output_type for o in c.get('outputs',[])]} for i,c in enumerate(nb.cells) if c.cell_type=='code']
        attempt={'start_utc':start,'finished_utc':datetime.now(timezone.utc).isoformat(),'success':success,'error':error,'prepare':args.prepare}
        with (out/'execution_attempts.jsonl').open('a') as h:h.write(json.dumps(attempt)+'\n')
        (out/'EXECUTION_STATUS.json').write_text(json.dumps({'success':success,'error':error,'cells':statuses},indent=2))
    assert len(statuses)==9 and all(c['executed'] and not c['errors'] for c in statuses)
    from common import sha
    events=[json.loads(l) for l in (out/'model_fit_log.jsonl').read_text().splitlines()]
    tests=json.loads((out/'tests/test_results.json').read_text());integrity=json.loads((out/'integrity_result.json').read_text())
    stage4=[p for folder in ['source','tests'] for p in (out/folder).glob('*.py')]
    artifacts=['locked_selections.json','selector_eligibility.csv','all_procedure_oof_predictions.csv','summary_metrics.csv','paired_fold_metrics.csv','paired_study_metrics.csv','method_and_study_size_metrics.csv','model_fit_log.jsonl','environment.json','Study_Robustness_EXECUTED.ipynb']
    status={'success':True,'scenario':'historical_hypothetical_unit_scenario','cells':statuses,'executed_code_cells':len(statuses),'tests':tests,
            'new_fits_started':sum(e['event']=='fit_started' for e in events),'new_fits_completed':sum(e['event']=='fit_completed' for e in events),
            'failed_fits':sum(e['event']=='fit_failed' for e in events),'hard_fit_cap':10,'protected_files_unchanged':integrity['unchanged'],
            'protected_paths_verified':integrity['protected_paths'],'source_sha256':{str(p.relative_to(out)):sha(p) for p in stage4},
            'artifact_sha256':{p:sha(out/p) for p in artifacts},'screening':json.loads((out/'screening.json').read_text()),
            'report_created':not args.prepare,'report_path':'study_robustness_outputs/STUDY_ROBUSTNESS_REPORT.md','stop':True}
    (out/'EXECUTION_STATUS.json').write_text(json.dumps(status,indent=2))
    print(json.dumps({k:v for k,v in status.items() if k not in ['cells','source_sha256','artifact_sha256']},indent=2))
    if not args.prepare:
        # Final artifact write. No experiment, test or fitting is launched after this report.
        (out/'STUDY_ROBUSTNESS_REPORT.md').write_text((out/'checkpoints/report_ready.md').read_text())


if __name__=='__main__':main()

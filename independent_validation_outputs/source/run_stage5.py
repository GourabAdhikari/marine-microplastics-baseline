"""Fresh-kernel Stage 5 readiness workflow. Final report is written last."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    from cli import setup
    out=setup();root=out.parent;os.chdir(root)
    import nbformat
    from nbclient import NotebookClient
    cells=[nbformat.v4.new_markdown_cell('# Stage 5: independent-study validation readiness\n\nNo new historical tuning. Immutable original R² 0.4726308398591973 and exploratory S2 R² 0.5256280075581679. Unit hypotheses remain unresolved; no external outcome is fabricated. Default read-only readiness, NOT_TRAINED and NO_EXTERNAL_DATA are valid outcomes.')]
    def section(title,code):cells.extend([nbformat.v4.new_markdown_cell('## '+title),nbformat.v4.new_code_cell(code)])
    section('1. Setup and explicit CLI runner',"""import sys,json,subprocess
from pathlib import Path
from IPython.display import display,Markdown
ROOT=Path.cwd();OUT=ROOT/'independent_validation_outputs'
sys.dont_write_bytecode=True
sys.path.insert(0,str(OUT/'source'))
from protocol import freeze_protocol,verify_lock
from reporting import inventory,command_docs,compose
from validation_core import protected_check
command_docs(OUT)
cli_results=[]
def cli(*args):
    run=subprocess.run([sys.executable,str(OUT/'source/cli.py'),*args],cwd=ROOT,capture_output=True,text=True)
    name=args[0].lstrip('-').replace('-','_')
    (OUT/(name+'_command.log')).write_text(run.stdout+'\\nSTDERR:\\n'+run.stderr)
    cli_results.append({'command':list(args),'returncode':run.returncode,'log':name+'_command.log'})
    print(run.stdout)
    if run.returncode:raise RuntimeError(run.stderr or run.stdout)
print((OUT/'environment.json').read_text())""")
    section('2. Bounded external inventory and benchmark audit — zero fits',"""print(json.dumps(inventory(OUT),indent=2))
cli('--audit')
print((OUT/'baseline_identity.json').read_text())""")
    section('3. Scientific provenance audit and source-access limits',"""display(Markdown((OUT/'PROVENANCE_AUDIT.md').read_text()))
print((OUT/'provenance_status.json').read_text())
print((OUT/'evidence/source_lookup_log.json').read_text())
print('No further public-source lookups. No row-specific correction authorized.')""")
    section('4. Freeze the external protocol before any outcomes or final fitting',"""lock=freeze_protocol(OUT)
print(json.dumps(lock,indent=2))
display(Markdown((OUT/'EXTERNAL_VALIDATION_PROTOCOL.md').read_text()))
assert verify_lock(OUT)==lock""")
    section('5. Execute metadata-preparation CLI with no external observations',"""cli('--prepare-external')
status=json.loads((OUT/'external_status.json').read_text())
assert status['status']=='NO_EXTERNAL_DATA' and not status['targets_read']
print('No external concentration column loaded; no eligibility CSV fabricated.')""")
    section('6. Execute optional research-model command and verify the authorization blocker',"""cli('--fit-frozen-research-model')
model_status=json.loads((OUT/'full_data_model_status.json').read_text())
assert model_status['status']=='NOT_TRAINED'
assert not (OUT/'full_data_research_model').exists()
assert not (OUT/'fit_log.jsonl').read_text().strip()
print('0 new LightGBM fits. Exact future 24+1 plan exists but is not authorized.')""")
    section('7. Execute evaluation CLI with no external data — never invent a score',"""cli('--evaluate-external')
assert json.loads((OUT/'external_status.json').read_text())['status']=='NO_EXTERNAL_DATA'
assert not (OUT/'external_predictions.csv').exists()
assert not (OUT/'EXTERNAL_TEST_REPORT.md').exists()
print('Actual independent held-out R²: NOT AVAILABLE')""")
    section('8. Offline tests through the same CLI',"""cli('--tests')
test_status=json.loads((OUT/'tests/test_results.json').read_text())
print(json.dumps(test_status,indent=2))
assert test_status['failures']==0 and test_status['errors']==0
assert test_status['actual_model_fits_in_tests']==0""")
    section('9. Final lock/protected-file verification, report rendering and stop',"""print(json.dumps(protected_check(OUT),indent=2))
assert verify_lock(OUT)==lock
(OUT/'cli_execution_status.json').write_text(json.dumps(cli_results,indent=2))
report=compose(OUT,9)
(OUT/'checkpoints/report_ready.md').write_text(report)
display(Markdown(report))
print('Readiness work complete. Runner writes final notebook/status then STAGE5_REPORT.md last.')""")
    nb=nbformat.v4.new_notebook(cells=cells)
    kd=out/'jupyter/kernels/stage5';kd.mkdir(parents=True,exist_ok=True)
    (kd/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Stage 5 read-only readiness','language':'python'}))
    nb.metadata['kernelspec']={'name':'stage5','display_name':'Stage 5 read-only readiness','language':'python'}
    path=out/'Stage5_Validation_Readiness_EXECUTED.ipynb';nbformat.write(nb,path)
    success=False;error=None;start=datetime.now(timezone.utc).isoformat()
    try:
        NotebookClient(nb,timeout=1200,kernel_name='stage5',resources={'metadata':{'path':str(root)}}).execute();success=True
    except Exception as e:error=repr(e);raise
    finally:
        nbformat.write(nb,path)
        statuses=[{'cell_index':i,'execution_count':c.execution_count,'executed':c.execution_count is not None,
                   'errors':[o.get('evalue') for o in c.get('outputs',[]) if o.output_type=='error'],'output_types':[o.output_type for o in c.get('outputs',[])]} for i,c in enumerate(nb.cells) if c.cell_type=='code']
        (out/'notebook_execution_status.json').write_text(json.dumps({'success':success,'error':error,'cells':statuses},indent=2))
        with (out/'execution_attempts.jsonl').open('a') as f:f.write(json.dumps({'started_utc':start,'success':success,'error':error,'prepare':args.prepare})+'\n')
    assert len(statuses)==9 and all(c['executed'] and not c['errors'] for c in statuses)
    from validation_core import sha
    result={'status':'READINESS_COMPLETED','notebook_code_cells':len(statuses),'tests':json.loads((out/'tests/test_results.json').read_text()),
            'external_status':json.loads((out/'external_status.json').read_text()),'model_status':json.loads((out/'full_data_model_status.json').read_text()),
            'new_lightgbm_fits':0,'independent_R2':None,'independent_R2_status':'NOT AVAILABLE','protocol_lock_sha256':sha(out/'protocol_lock.json'),
            'protocol_content_sha256':json.loads((out/'protocol_lock.json').read_text())['content_sha256'],
            'integrity':json.loads((out/'integrity_result.json').read_text()),'report_created':not args.prepare,
            'source_sha256':{str(p.relative_to(out)):sha(p) for folder in ['source','tests'] for p in (out/folder).glob('*.py')}}
    (out/'FINAL_STATUS.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2))
    if not args.prepare:
        # Last artifact write; no new test, fit or scientific phase follows this report.
        (out/'STAGE5_REPORT.md').write_text((out/'checkpoints/report_ready.md').read_text())


if __name__=='__main__':main()

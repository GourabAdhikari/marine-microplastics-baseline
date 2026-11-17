"""Explicit Stage 5 commands. Default is read-only audit, never fitting."""
import argparse
import json
import os
from pathlib import Path
import sys


def setup():
    out=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
    os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['GIT_OPTIONAL_LOCKS']='0'
    for var,folder in [('JUPYTER_DATA_DIR','jupyter'),('JUPYTER_RUNTIME_DIR','runtime'),('IPYTHONDIR','ipython'),('MPLCONFIGDIR','matplotlib'),('TMPDIR','tmp'),('XDG_CACHE_HOME','cache')]:
        p=out/folder;p.mkdir(exist_ok=True);os.environ[var]=str(p)
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='4'
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group()
    group.add_argument('--audit',action='store_true');group.add_argument('--prepare-external',nargs='?',const='',default=None,metavar='PATH')
    group.add_argument('--fit-frozen-research-model',action='store_true');group.add_argument('--evaluate-external',nargs='?',const='',default=None,metavar='PATH');group.add_argument('--tests',action='store_true')
    args=parser.parse_args();out=setup()
    if args.fit_frozen_research_model:
        from model import fit_frozen
        result=fit_frozen(out)
    elif args.prepare_external is not None:
        from external import prepare_external
        result=prepare_external(out,args.prepare_external or None)
    elif args.evaluate_external is not None:
        from evaluation import evaluate_external
        result=evaluate_external(out,args.evaluate_external or None)
    elif args.tests:
        sys.path.insert(0,str(out/'tests'));from test_readiness import run_tests
        result=run_tests(out)
    else:
        from unittest.mock import patch
        from benchmark import audit_benchmark,provenance_audit
        with patch('lightgbm.LGBMRegressor.fit',side_effect=AssertionError('Audit may not fit models')):
            identity=audit_benchmark(out);provenance=provenance_audit(out)
        result={'benchmark_verified':True,'historical_references':identity['historical_references'],'provenance':provenance,'new_fits':0}
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

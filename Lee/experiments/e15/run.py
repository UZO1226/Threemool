"""Run the frozen five-seed controller comparison and summarize on completion."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
p=argparse.ArgumentParser()
p.add_argument('--data',type=Path,required=True)
p.add_argument('--surrogate',type=Path,required=True)
p.add_argument('--calibration',type=Path,required=True)
p.add_argument('--upstream',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--parallel',type=int,default=5)
a=p.parse_args()
cal=json.loads(a.calibration.read_text())
assert cal['splits_evaluated']==['train']
environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
    NUMBA_NUM_THREADS='1',DARROW_FAST_DYNAMICS='1',GIT_CONFIG_COUNT='2',
    GIT_CONFIG_KEY_0='safe.directory',GIT_CONFIG_VALUE_0=ROOT.as_posix(),
    GIT_CONFIG_KEY_1='safe.directory',GIT_CONFIG_VALUE_1=a.upstream.resolve().as_posix())
methods=['ppolag','ppolag_fcsrl','ppolag_fcsrl_curriculum']
jobs=[f'{method}:{seed}' for method in methods for seed in range(5)]
cmd=[sys.executable,'-u',str(ROOT/'Lee/experiments/e07/run_dw.py'),
    '--data',str(a.data.resolve()),'--surrogate',str(a.surrogate.resolve()),
    '--upstream',str(a.upstream.resolve()),'--out',str(a.out.resolve()),
    '--limit',str(cal['cost_limit']),'--steps','400000','--evaluations','5',
    '--parallel',str(a.parallel),'--jobs',*jobs]
subprocess.run(cmd,env=environment,check=True)
subprocess.run([sys.executable,str(HERE/'report.py'),'--runs',str(a.out.resolve()),
                '--out',str(HERE)],env=environment,check=True)

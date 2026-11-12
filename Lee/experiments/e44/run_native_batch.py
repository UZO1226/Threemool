"""Evaluate already completed final policies on both frozen simulators."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
p.add_argument('--freeze',type=Path,required=True);p.add_argument('--runs',type=Path,nargs='+',required=True)
p.add_argument('--environments',nargs='+',choices=['old','new'],default=['old','new'])
p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
jobs=[]
for run in a.runs:
    config=json.loads((run/'effective_config.json').read_text())
    assert json.loads((run/'progress.json').read_text())['phase']=='complete'
    label=f"{config['experiment_metadata']['environment']}_{config['method']}_s{config['seed']}"
    for environment in a.environments:jobs.append({'run':str(run.resolve()),'environment':environment,'label':f'{label}_eval_{environment}'})
assert len({j['label'] for j in jobs})==len(jobs)
hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
manifest={'jobs':jobs,'source_hashes':hashes,'freeze_sha256':hashlib.sha256(a.freeze.read_bytes()).hexdigest(),
    'code_commit':subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    'test_evaluated':False}
(a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',PYTHONIOENCODING='utf-8')
completed=[];process=None
try:
    for job in jobs:
        for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,'Evaluation source changed'
        assert hashlib.sha256(a.freeze.read_bytes()).hexdigest()==manifest['freeze_sha256']
        with (a.out/f"{job['label']}.log").open('w',encoding='utf-8') as log:
            command=[sys.executable,'-u',str(HERE/'native_eval.py'),'--data',str(a.data.resolve()),
                '--freeze',str(a.freeze.resolve()),'--run',job['run'],'--environment',job['environment'],
                '--out',str((a.out/f"{job['label']}.json").resolve())]
            process=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT.parent)
            (a.out/'status.json').write_text(json.dumps({'phase':'running','active':job,'pid':process.pid,'complete':completed},indent=2))
            print('START',job['label'],process.pid,flush=True)
            code=process.wait()
        if code:raise RuntimeError(f"{job['label']} exited {code}, inspect retained log")
        completed.append(job['label']);print('COMPLETE',job['label'],flush=True)
    (a.out/'status.json').write_text(json.dumps({'phase':'complete','complete':completed},indent=2))
except BaseException:
    if process is not None and process.poll() is None:
        process.terminate()
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:process.kill();process.wait()
    (a.out/'status.json').write_text(json.dumps({'phase':'failed','complete':completed},indent=2));raise

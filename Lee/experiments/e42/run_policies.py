"""Run named seed jobs with immutable source checks and retained process logs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False))
    for attempt in range(40):
        try:temp.replace(path);return
        except PermissionError:
            if attempt==39:raise
            time.sleep(.05)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
    p.add_argument('--freeze',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--trainer',type=Path,default=Path(__file__).with_name('train_policy.py'))
    p.add_argument('--jobs',nargs='+',required=True);p.add_argument('--parallel',type=int,default=5)
    p.add_argument('--steps',type=int,default=400000);p.add_argument('--evaluations',type=int,default=5)
    a=p.parse_args();assert a.parallel>0 and len(a.jobs)==len(set(a.jobs))
    a.out.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for key in a.jobs:
        environment,method,seed=key.split(':');assert environment in ('old','new')
        assert method.replace('_','').isalnum();assert 0<=int(seed)<5
        jobs.append({'environment':environment,'method':method,'seed':int(seed),'key':key})
    sources=sorted(set([*Path(__file__).parent.glob('*.py'),*(ROOT/'Lee/code/darrow').glob('*.py'),
        *(ROOT/'Lee/experiments/e38').glob('*.py'),
        *a.trainer.parent.glob('*.py')]))
    hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    manifest={'code_commit':subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'freeze_sha256':hashlib.sha256(a.freeze.read_bytes()).hexdigest(),'source_hashes':hashes,
        'jobs':jobs,'steps_requested':a.steps,'actual_steps_per_job':math.ceil(a.steps/6080)*6080,
        'evaluations':a.evaluations,'parallel':a.parallel,'test_evaluated':False,'new_policy_training':True}
    write(a.out/'manifest.json',manifest)
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        NUMBA_NUM_THREADS='1',PYTHONIOENCODING='utf-8',DARROW_FAST_DYNAMICS='1')
    pending=jobs.copy();active=[];complete=[];start=time.monotonic()
    def status(phase):
        rows=[]
        for proc,log,job,folder in active:
            try:progress=json.loads((folder/'progress.json').read_text())
            except (OSError,json.JSONDecodeError):progress={'phase':'starting'}
            rows.append({'key':job['key'],'pid':proc.pid,**progress})
        write(a.out/'status.json',{'phase':phase,'complete':complete,'running':rows,
            'pending':[j['key'] for j in pending],'elapsed_seconds':time.monotonic()-start})
    try:
        while pending or active:
            while pending and len(active)<a.parallel:
                for path,digest in hashes.items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,f'Source changed: {path}'
                assert hashlib.sha256(a.freeze.read_bytes()).hexdigest()==manifest['freeze_sha256']
                job=pending.pop(0);label=f"{job['environment']}_{job['method']}"
                folder=a.out/label/f"seed_{job['seed']}"
                log=(a.out/f"{label}_s{job['seed']}.log").open('w',encoding='utf-8')
                cmd=[sys.executable,'-u',str(a.trainer.resolve()),'--data',str(a.data.resolve()),
                    '--freeze',str(a.freeze.resolve()),'--out',str(folder.resolve()),
                    '--environment',job['environment'],'--method',job['method'],'--seed',str(job['seed']),
                    '--steps',str(a.steps),'--evaluations',str(a.evaluations)]
                proc=subprocess.Popen(cmd,env=environment,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT.parent)
                active.append((proc,log,job,folder));print('START',job['key'],proc.pid,flush=True)
            status('running');time.sleep(5)
            for item in active.copy():
                proc,log,job,folder=item;code=proc.poll()
                if code is None:continue
                active.remove(item);log.close()
                if code:raise RuntimeError(f"{job['key']} exited {code}, inspect retained log")
                metrics=json.loads((folder/'metrics.json').read_text())
                assert metrics['steps']==manifest['actual_steps_per_job']
                complete.append(job['key']);print('COMPLETE',job['key'],flush=True)
        status('complete')
    except BaseException:
        for proc,log,job,folder in active:
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            log.close()
        status('failed');raise


if __name__=='__main__':main()

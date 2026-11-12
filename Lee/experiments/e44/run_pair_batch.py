"""Run explicit completed seed pairs without choosing among duplicate runs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
p.add_argument('--freeze',type=Path,required=True);p.add_argument('--baseline-root',type=Path,required=True)
p.add_argument('--candidate-roots',type=Path,nargs='+',required=True);p.add_argument('--seeds',type=int,nargs='+',required=True)
p.add_argument('--out',type=Path,required=True);p.add_argument('--parallel',type=int,default=1)
p.add_argument('--kind',choices=['agent','capacity','environment','diagnostic'],default='agent')
p.add_argument('--environment',choices=['old','new'],default='new');a=p.parse_args()
assert a.parallel>0 and len(a.seeds)==len(set(a.seeds)) and all(0<=s<5 for s in a.seeds)
jobs=[]
for seed in a.seeds:
    baseline=a.baseline_root/f'seed_{seed}'
    candidates=[root/f'seed_{seed}' for root in a.candidate_roots if (root/f'seed_{seed}'/'metrics.json').exists()]
    assert len(candidates)==1,f'Expected one completed candidate for seed{seed}, found{len(candidates)}'
    candidate=candidates[0]
    for folder in (baseline,candidate):
        assert json.loads((folder/'progress.json').read_text())['phase']=='complete'
        assert json.loads((folder/'metrics.json').read_text())['steps']==401280
    jobs.append({'seed':seed,'baseline':str(baseline.resolve()),'candidate':str(candidate.resolve())})
a.out.mkdir(parents=True,exist_ok=False)
sources=[HERE/'evaluate_pair.py',HERE/'air_match.py',Path(__file__)]
hashes={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
manifest={'jobs':jobs,'source_hashes':hashes,'kind':a.kind,'environment':a.environment,
    'freeze_sha256':hashlib.sha256(a.freeze.read_bytes()).hexdigest(),'test_evaluated':False}
(a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',PYTHONIOENCODING='utf-8')
pending=jobs.copy();active=[];complete=[]
def status(phase):
    (a.out/'status.json').write_text(json.dumps({'phase':phase,'complete':complete,
        'running':[{'seed':job['seed'],'pid':proc.pid} for proc,log,job in active],
        'pending':[job['seed'] for job in pending]},indent=2))
try:
    while pending or active:
        while pending and len(active)<a.parallel:
            for name,digest in hashes.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,'Matching source changed'
            assert hashlib.sha256(a.freeze.read_bytes()).hexdigest()==manifest['freeze_sha256']
            job=pending.pop(0);folder=a.out/f"seed_{job['seed']}"
            command=[sys.executable,'-u',str(HERE/'evaluate_pair.py'),'--data',str(a.data.resolve()),'--freeze',str(a.freeze.resolve()),
                '--baseline',job['baseline'],'--candidate',job['candidate'],'--out',str(folder.resolve()),
                '--kind',a.kind,'--environment',a.environment]
            log=(a.out/f"seed_{job['seed']}.log").open('w',encoding='utf-8')
            proc=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT.parent)
            active.append((proc,log,job));print('START',job['seed'],proc.pid,flush=True)
        status('running');time.sleep(5)
        for item in active.copy():
            proc,log,job=item;code=proc.poll()
            if code is None:continue
            active.remove(item);log.close()
            if code:raise RuntimeError(f"Seed{job['seed']} exited{code}, inspect retained log")
            assert (a.out/f"seed_{job['seed']}"/'summary.json').exists()
            complete.append(job['seed']);print('COMPLETE',job['seed'],flush=True)
    status('complete')
except BaseException:
    for proc,log,job in active:
        if proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        log.close()
    status('failed');raise

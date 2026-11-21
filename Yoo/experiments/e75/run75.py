"""Run a frozen finite diagnostic, with atomic status and source manifests."""
import argparse
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import platform
import sklearn
import plan75 as P


def main(a):
    preflight_path=a.out.parent/'preflight.json'
    preflight=json.loads(preflight_path.read_text())
    assert all(preflight[k] is True for k in ('action_replay','candidate_layout','threshold_cost','feasible_rank'))
    assert all(preflight[s]['baseline_aggregate_parity'] is True for s in ('dev','val'))
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'logs').mkdir();(a.out/'jobs').mkdir()
    sources=list(P.HERE.glob('*.py'))+[preflight_path,P.HERE/'REFERENCES_AND_DESIGN.md',P.BASE,
             P.HERE.parent/'e74/env74.py',P.HERE.parent/'e68/env68.py',P.HERE.parent/'e68/baselines.py',
             P.HERE.parent/'e30/darrow_env.py',P.E.PREPARED/'data.pkl',P.E.PREPARED/'surrogate.pkl']
    hashes={str(p):P.digest(p) for p in sources}
    pending=[]
    for split,mode,algorithm,seed in itertools.product(('dev','val'),('air','cost'),('cem','random'),(0,1)):
        name=f'{split}_{mode}_{algorithm}_{seed}'
        cmd=[sys.executable,'-u',str(P.HERE/'plan75.py'),'--split',split,'--mode',mode,'--algorithm',algorithm,'--seed',str(seed),'--out',str(a.out/'jobs'/name)]
        pending.append((name,cmd))
    P.write_json(a.out/'manifest.json',dict(started_at=datetime.now(timezone.utc).isoformat(),hashes=hashes,
                 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P.HERE,text=True).strip(),commands=pending,
                 design='REFERENCES_AND_DESIGN.md',python=sys.executable,python_version=platform.python_version(),
                 numpy_version=P.np.__version__,sklearn_version=sklearn.__version__,device='CPU',threads_per_job=1,
                 holdout_evaluated=False,test_evaluated=False))
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE',CUDA_VISIBLE_DEVICES='')
    active={};done=[]
    def status(phase):
        P.write_json(a.out/'status.json',dict(phase=phase,running=list(active),pending=len(pending),done=done,updated_at=datetime.now(timezone.utc).isoformat()))
    while pending or active:
        while pending and len(active)<a.jobs:
            name,cmd=pending.pop(0)
            with (a.out/'logs'/f'{name}.log').open('w') as log:
                proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            active[name]=(proc,time.monotonic())
        status('running')
        for name,(proc,start) in list(active.items()):
            if proc.poll() is not None:
                done.append(dict(name=name,exit=proc.returncode,seconds=time.monotonic()-start));del active[name]
        time.sleep(5)
    if any(r['exit'] for r in done):
        status('failed');raise RuntimeError('one or more jobs failed')
    if any(P.digest(p)!=h for p,h in hashes.items()):
        status('source_changed');raise RuntimeError('source changed')
    status('reporting')
    try: subprocess.run([sys.executable,str(P.HERE/'report75.py'),'--root',str(a.out)],check=True,env=env)
    except Exception:
        status('report_failed');raise
    status('complete');print('COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--jobs',type=int,default=8)
    main(p.parse_args())

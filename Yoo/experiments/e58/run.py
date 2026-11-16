"""검사를 통과한 e58 세 조건15개를 실행하고 마지막 결과를 집계한다."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import severity as s
import check
base=s.base


def run(a):
    verified=json.loads(a.check.read_text())
    assert verified['passed'] and all(base.sha(p)==v for p,v in verified['hashes'].items())
    p=json.loads((s.HERE/'protocol.json').read_text())
    a.out.mkdir(parents=True,exist_ok=False)
    hashes=check.hashes();hashes[str(a.check)]=base.sha(a.check)
    manifest=dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.ROOT,text=True).strip(),
                  hashes=hashes,protocol=p,pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),
                  python=sys.executable,torch=base.torch.__version__,sklearn=base.sklearn.__version__,
                  max_parallel=a.jobs,commands=[],test_evaluated=False,
                  gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True))
    base.save(a.out/'manifest.json',manifest)
    pending=[(m,seed) for seed in p['seeds'] for m in p['methods']]
    active=[];complete=[]
    def status(phase):
        base.save(a.out/'status.json',dict(phase=phase,pending=pending,completed=complete,
                  running=[dict(method=m,seed=seed,pid=proc.pid) for proc,log,m,seed in active],
                  updated_at=datetime.now(timezone.utc).isoformat()))
    def stop(sig,frame):
        raise InterruptedError(sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        while pending or active:
            while pending and len(active)<a.jobs:
                m,seed=pending.pop(0)
                dest=a.out/'runs'/m/f'seed_{seed}';dest.parent.mkdir(parents=True,exist_ok=True)
                command=[sys.executable,'-u',str(s.HERE/'train_severity.py'),'--method',m,'--seed',str(seed),
                         '--prepared',p['prepared'],'--out',str(dest),'--manifest',str(a.out/'manifest.json')]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',
                         OMP_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
                log=(a.out/f'{m}_s{seed}.log').open('w')
                proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active.append((proc,log,m,seed));manifest['commands'].append(command)
                base.save(a.out/'manifest.json',manifest)
                print(time.strftime('%H:%M:%S'),'START',m,seed,flush=True)
            status('training');time.sleep(5)
            for item in active.copy():
                proc,log,m,seed=item
                code=proc.poll()
                if code is None:continue
                active.remove(item);log.close();complete.append(dict(method=m,seed=seed,exit_code=code))
                if code:raise RuntimeError((m,seed,code))
                print(time.strftime('%H:%M:%S'),'END',m,seed,flush=True)
        status('reporting')
        subprocess.run([sys.executable,str(s.HERE/'report.py'),'--root',str(a.out)],check=True)
        assert all(base.sha(path)==value for path,value in hashes.items())
        manifest['finished_at']=datetime.now(timezone.utc).isoformat();base.save(a.out/'manifest.json',manifest)
        status('complete');print('COMPLETE',flush=True)
    except BaseException:
        for proc,log,m,seed in active:
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
            log.close()
        status('failed');raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--jobs',type=int,default=3)
    a=p.parse_args();assert 1<=a.jobs<=3;run(a)

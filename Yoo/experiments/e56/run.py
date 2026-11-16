"""사전 검사를 통과한 prior-off 10개 학습과 최종 집계를 실행한다."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'e30'))
from common import save,sha


def run(a):
    p=json.loads((HERE/'protocol.json').read_text());check=json.loads(a.preflight.read_text())
    assert check['passed'] and all(sha(path)==h for path,h in check['hashes'].items())
    a.out.mkdir(parents=True,exist_ok=False)
    hashes=dict(check['hashes'])
    for f in [Path(__file__),HERE/'report.py',a.preflight]:hashes[str(f)]=sha(f)
    manifest=dict(started_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE.parents[2],text=True).strip(),
        python=sys.executable,sklearn=__import__('sklearn').__version__,
        torch=__import__('torch').__version__,numpy=__import__('numpy').__version__,
        upstream_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd='/home/jio/fcsrl_src/FCSRL',text=True).strip(),
        protocol=p,hashes=hashes,commands=[],parallel_jobs=a.jobs)
    save(a.out/'manifest.json',manifest)
    pending=[(m,s) for s in p['seeds'] for m in p['methods']];active=[];done=[]
    env=dict(os.environ,OMP_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def status(phase):
        save(a.out/'status.json',dict(phase=phase,completed=done,pending=pending,
            running=[dict(method=m,seed=s,pid=proc.pid) for proc,log,m,s in active],updated_at=datetime.now(timezone.utc).isoformat()))
    def stop(sig,frame):raise InterruptedError(sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        while pending or active:
            while pending and len(active)<a.jobs:
                m,s=pending.pop(0)
                cmd=[sys.executable,'-u',str(HERE/'train.py'),'--method',m,'--prior-mode','off','--seed',str(s),
                     '--updates',str(p['updates']),'--prepared',p['prepared'],'--manifest',str(a.out/'manifest.json'),
                     '--out',str(a.out/'runs'/m/f'seed_{s}')]
                log=(a.out/f'{m}_s{s}.log').open('w')
                proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active.append((proc,log,m,s));manifest['commands'].append(cmd);save(a.out/'manifest.json',manifest)
                print(time.strftime('%H:%M:%S'),'START',m,s,flush=True)
            status('training');time.sleep(5)
            for item in active.copy():
                proc,log,m,s=item;code=proc.poll()
                if code is None:continue
                active.remove(item);log.close();done.append(dict(method=m,seed=s,exit_code=code))
                print(time.strftime('%H:%M:%S'),'END',m,s,code,flush=True)
                if code:raise RuntimeError(f'{m} seed{s} exited {code}')
        status('reporting')
        subprocess.run([sys.executable,str(HERE/'report.py'),'--root',str(a.out)],env=env,check=True)
        assert all(sha(path)==h for path,h in hashes.items())
        manifest['finished_at']=datetime.now(timezone.utc).isoformat();save(a.out/'manifest.json',manifest)
        status('complete');print('COMPLETE',flush=True)
    except BaseException:
        for proc,log,m,s in active:
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
            log.close()
        status('failed');raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--preflight',type=Path,required=True);p.add_argument('--jobs',type=int,default=3)
    a=p.parse_args();assert 1<=a.jobs<=6
    run(a)

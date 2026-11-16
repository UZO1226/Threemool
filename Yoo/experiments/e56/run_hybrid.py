"""진행 중 3개를 보존하고 대기 7개만 새 backend로 넘긴다."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import train as base

HERE=Path(__file__).resolve().parent


def process_done(pid):
    try:
        s=Path(f'/proc/{pid}/stat').read_text()
        return s[s.rfind(')')+2:].split()[0]=='Z'
    except FileNotFoundError:return True


def main(a):
    old=a.legacy
    pause=json.loads((old.parent/'migration_dispatch_pause.json').read_text())
    check=json.loads(a.check.read_text());assert check['passed']
    assert all(base.sha(p)==v for p,v in check['hashes'].items())
    original=json.loads((old/'manifest.json').read_text())
    assert all(base.sha(p)==v for p,v in original['hashes'].items())
    pending=[tuple(x) for x in pause['original_status']['pending']]
    legacy=pause['original_status']['running']
    expected={(m,s) for m in ['ppo','fcsrl'] for s in range(5)}
    assert set(pending)|{(x['method'],x['seed']) for x in legacy}==expected
    assert len(pending)+len(legacy)==10
    a.out.mkdir(parents=True,exist_ok=False)
    for x in legacy:
        dest=a.out/'runs'/x['method']/f"seed_{x['seed']}"
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.symlink_to(old/'runs'/x['method']/f"seed_{x['seed']}",target_is_directory=True)
    sources=[Path(__file__),HERE/'hybrid.py',HERE/'train_hybrid.py',HERE/'report.py',a.check]
    for m in list(sys.modules.values()):
        f=getattr(m,'__file__',None)
        if f and str(Path(f).resolve()).startswith(str(base.ROOT/'Yoo')) and f.endswith('.py'):sources.append(Path(f).resolve())
    hashes={str(p):base.sha(p) for p in sources}
    hashes.update(original['hashes'])
    hashes.update(check['hashes'])
    hashes[str(old/'manifest.json')]=base.sha(old/'manifest.json')
    p=json.loads((HERE/'protocol.json').read_text())
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'],text=True)
    manifest=dict(started_at=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),
        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base.ROOT,text=True).strip(),
        hashes=hashes,protocol=p,legacy_root=str(old),legacy=legacy,migrated_jobs=pending,
        python=sys.executable,sklearn=base.sklearn.__version__,torch=base.torch.__version__,gpus=gpu,
        parallel_jobs=a.jobs,commands=[],backend_note='Mixed backend exploratory results; exact full-trajectory CPU/GPU equivalence is not claimed')
    base.save(a.out/'manifest.json',manifest)
    active=[];complete=[];retired=False
    def current_legacy():
        live=[]
        for x in legacy:
            prog=old/'runs'/x['method']/f"seed_{x['seed']}"/'progress.json'
            z=json.loads(prog.read_text()) if prog.exists() else {}
            if process_done(x['pid']):
                assert z.get('phase')=='complete',('legacy failed',x,z)
                key=(x['method'],x['seed'])
                if not any((q['method'],q['seed'])==key for q in complete):
                    complete.append(dict(method=x['method'],seed=x['seed'],exit_code=0,backend='legacy_cpu4'))
            else:live.append(x)
        return live
    def status(phase):
        base.save(a.out/'status.json',dict(phase=phase,completed=complete,pending=pending,
            running=[dict(method=m,seed=s,pid=proc.pid,backend='gpu1_update' if m=='fcsrl' else 'cpu1') for proc,log,m,s in active],
            legacy_running=current_legacy(),updated_at=datetime.now(timezone.utc).isoformat()))
    def stop(sig,frame):raise InterruptedError(sig)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        while pending or active or current_legacy() or not retired:
            live=current_legacy()
            if not live and not retired:
                pid=pause['supervisor_pid']
                try:
                    cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
                    assert b'e56/run.py' in cmd and str(old).encode() in cmd
                    # 하위 학습이 모두 끝난 정지 상태의 배차 프로세스만 종료한다.
                    os.kill(pid,signal.SIGKILL)
                except FileNotFoundError:pass
                base.save(old/'status_before_migration_retirement.json',json.loads((old/'status.json').read_text()))
                base.save(old/'status.json',dict(phase='superseded',completed=complete.copy(),pending_transferred=pause['original_status']['pending'],
                    running=[],successor=str(a.out),reason='Existing jobs finished; only pending jobs moved to hybrid runner'))
                retired=True
            while pending and len(active)+len(live)<a.jobs:
                m,s=pending.pop(0)
                cmd=[sys.executable,'-u',str(HERE/'train_hybrid.py'),'--method',m,'--seed',str(s),'--prior-mode','off',
                    '--prepared',p['prepared'],'--out',str(a.out/'runs'/m/f'seed_{s}'),'--manifest',str(a.out/'manifest.json')]
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='1' if m=='fcsrl' else '',CUBLAS_WORKSPACE_CONFIG=':4096:8',
                    OMP_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
                log=(a.out/f'{m}_s{s}.log').open('w')
                proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active.append((proc,log,m,s));manifest['commands'].append(cmd);base.save(a.out/'manifest.json',manifest)
                print(time.strftime('%H:%M:%S'),'START',m,s,'gpu1_update' if m=='fcsrl' else 'cpu1',flush=True)
            status('training');time.sleep(5)
            for item in active.copy():
                proc,log,m,s=item;code=proc.poll()
                if code is None:continue
                active.remove(item);log.close();complete.append(dict(method=m,seed=s,exit_code=code,backend='gpu1_update' if m=='fcsrl' else 'cpu1'))
                if code:raise RuntimeError((m,s,code))
                print(time.strftime('%H:%M:%S'),'END',m,s,flush=True)
        status('reporting')
        subprocess.run([sys.executable,str(HERE/'report.py'),'--root',str(a.out)],check=True)
        report=json.loads((a.out/'summary.json').read_text())
        report.update(backend_migration=complete,prior_only_causal_confirmation=False,
            backend_limitation='FCSRL uses GPU updates for seeds1-4, seed0 retains CPU4; PPO seeds0-1 CPU4 and seeds2-4 CPU1; baseline e50 CPU4. Promising results require matched-backend confirmation before adoption.')
        base.save(a.out/'summary.json',report)
        with (a.out/'results.md').open('a') as f:
            f.write('\n## 장치 전환 해석 제한\n\n기존 3개는 CPU4로 완료했고 대기 7개만 CPU1 수집/FCSRL GPU1 갱신으로 전환했다. 기준선 e50은 CPU4다. 미니배치 CPU 난수열과 수식을 유지했으나 전체 궤적의 동일성을 보장하지 않는다. 탐색 선별이 좋아도 prior만의 인과 효과 확정이나 공식 채택으로 해석하지 않으며 같은 backend의 확인이 필요하다. 장치별 이력은 summary.json에 있다.\n')
        assert all(base.sha(p)==v for p,v in hashes.items())
        manifest['finished_at']=datetime.now(timezone.utc).isoformat();base.save(a.out/'manifest.json',manifest)
        status('complete');print('COMPLETE',flush=True)
    except BaseException:
        for proc,log,m,s in active:
            if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
            log.close()
        status('failed');raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--legacy',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--check',type=Path,required=True)
    p.add_argument('--jobs',type=int,default=3);a=p.parse_args();assert 1<=a.jobs<=6
    main(a)

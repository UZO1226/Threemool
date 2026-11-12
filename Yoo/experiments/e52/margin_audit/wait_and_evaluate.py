"""기존 e53 여섯 실행의 완료를 기다려 고정 정책 평가와 비교만 수행한다."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
SOURCE=Path('/home/jio/fcsrl_src/results/e53/main')


def save(p,x):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False));tmp.replace(p)


def main(a):
    a.out.mkdir(parents=True,exist_ok=True)
    assert not (a.out/'pipeline.json').exists(),'새 출력 경로를 사용한다'
    jobs=[SOURCE/'runs'/m/'d0.45'/f'seed_{s}' for m in ('ppo','fcsrl') for s in range(3)]
    start=time.monotonic()
    status=dict(phase='waiting_for_training',started_at=datetime.now(timezone.utc).isoformat(),
                pid=os.getpid(),weight_updates=0,test_evaluated=False)
    try:
        while True:
            if (SOURCE/'status.json').exists():
                s=json.loads((SOURCE/'status.json').read_text())
                assert s.get('phase')!='failed',s
            completed=[p for p in jobs if (p/'metrics.json').exists() and json.loads((p/'progress.json').read_text()).get('phase')=='complete']
            status.update(completed=len(completed),total=6,elapsed_seconds=time.monotonic()-start)
            save(a.out/'pipeline.json',status)
            if len(completed)==6:break
            if time.monotonic()-start>a.timeout_seconds:raise TimeoutError('학습 완료 대기 시간 초과; 기존 학습은 중지하지 않음')
            print('WAIT',len(completed),'/6',flush=True)
            time.sleep(30)
        env=dict(os.environ,OMP_NUM_THREADS='1',OMP_WAIT_POLICY='PASSIVE',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        status['phase']='evaluating_fixed_policies';save(a.out/'pipeline.json',status)
        subprocess.run([sys.executable,'-u',str(HERE.parent/'audit/replay.py'),'--experiment','e53','--out',str(a.out/'e53')],env=env,check=True)
        status['phase']='comparing';save(a.out/'pipeline.json',status)
        subprocess.run([sys.executable,str(HERE/'compare.py'),'--e53',str(a.out/'e53'),
                        '--e52','/home/jio/fcsrl_src/results/e50_e52_audit_20261112/e52',
                        '--calibration',str(a.out/'calibration'),'--out',str(HERE)],env=env,check=True)
        status.update(phase='complete',elapsed_seconds=time.monotonic()-start,finished_at=datetime.now(timezone.utc).isoformat())
        save(a.out/'pipeline.json',status)
    except BaseException as exc:
        status.update(phase='failed',error=repr(exc));save(a.out/'pipeline.json',status);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--timeout-seconds',type=int,default=7200)
    main(p.parse_args())

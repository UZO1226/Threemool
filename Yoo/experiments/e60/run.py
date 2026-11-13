"""e60 학습 10개 (ppo, fcsrl x seed 0~4) 를 동시 3개로 실행하고 완료 뒤 집계한다."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_shielded as t60  # noqa: E402
base = t60.base


def run(a):
    checks = json.loads(a.check.read_text())
    assert checks['passed'] and all(base.sha(p) == v for p, v in checks['hashes'].items())
    protocol = json.loads((HERE / 'protocol.json').read_text())
    a.out.mkdir(parents=True, exist_ok=False)
    manifest = dict(code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip(), hashes=checks['hashes'],
                    protocol=protocol, started_at=datetime.now(timezone.utc).isoformat(), python=sys.executable, max_parallel=a.jobs,
                    test_evaluated=False, commands=[],
                    gpus=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader'], text=True))
    base.save(a.out / 'manifest.json', manifest)
    pending = [(m, s) for s in protocol['seeds'] for m in protocol['methods']]
    if a.only:
        keep = {(x.split(':')[0], int(x.split(':')[1])) for x in a.only.split(',')}
        pending = [j for j in pending if j in keep]
        manifest['only'] = sorted(map(list, keep))
        base.save(a.out / 'manifest.json', manifest)
    active, complete = [], []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1', OMP_WAIT_POLICY='PASSIVE',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def status(phase):
        base.save(a.out / 'status.json', dict(phase=phase, pending=pending, running=[j for _, _, j in active], completed=complete,
                                              updated_at=datetime.now(timezone.utc).isoformat()))
    while pending or active:
        while pending and len(active) < a.jobs:
            m, s = job = pending.pop(0)
            dest = a.out / 'runs' / m / f'seed_{s}'
            dest.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, '-u', str(HERE / 'train_shielded.py'), '--method', m, '--seed', str(s), '--train-shield', protocol['train_shield'],
                   '--out', str(dest), '--manifest', str(a.out / 'manifest.json')]
            log = (a.out / f'{m}_s{s}.log').open('w')
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            active.append((proc, time.monotonic(), job))
            manifest['commands'].append(cmd)
            base.save(a.out / 'manifest.json', manifest)
            status('running')
        for item in list(active):
            proc, start, job = item
            if proc.poll() is not None:
                active.remove(item)
                complete.append(dict(job=job, exit=proc.returncode, seconds=time.monotonic() - start))
                status('running')
        time.sleep(10)
    failed = [c for c in complete if c['exit'] != 0]
    status('failed' if failed else 'complete')
    assert all(base.sha(p) == v for p, v in checks['hashes'].items())
    if not failed and not a.only:
        subprocess.run([sys.executable, str(HERE / 'report.py'), '--root', str(a.out), '--out', str(HERE)], check=True)
    print('COMPLETE' if not failed else f'FAILED {failed}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--check', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=3)
    p.add_argument('--only', default=None, help='method:seed 목록. 일부 작업만 실행하고 집계는 하지 않는다')
    run(p.parse_args())

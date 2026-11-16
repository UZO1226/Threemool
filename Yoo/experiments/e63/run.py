"""e63: 학습 20개 (조건 2 x 방법 2 x seed 5) 를 동시 a.jobs 개로 실행하고, 이어서 교차 평가 20개와 집계를 한다.

실행 중에는 checks.json 의 hashes 에 있는 파일을 고치지 않는다.
"""
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
import check  # noqa: E402
base = check.base


def pool(jobs, n, env, status):
    """jobs: [(이름, 명령, 로그)]. 끝난 작업의 종료 코드 목록을 돌려준다."""
    pending, active, done = list(jobs), [], []
    while pending or active:
        while pending and len(active) < n:
            name, cmd, log = pending.pop(0)
            active.append((name, time.monotonic(), subprocess.Popen(cmd, env=env, stdout=log.open('w'), stderr=subprocess.STDOUT, start_new_session=True)))
            status(pending, active, done)
        for item in list(active):
            name, t0, proc = item
            if proc.poll() is not None:
                active.remove(item)
                done.append(dict(job=name, exit=proc.returncode, seconds=time.monotonic() - t0))
                status(pending, active, done)
        time.sleep(10)
    return done


def run(a):
    checks = json.loads(a.check.read_text())
    assert checks['passed'] and all(base.sha(p) == v for p, v in checks['hashes'].items())
    protocol = json.loads((HERE / 'protocol.json').read_text())
    a.out.mkdir(parents=True, exist_ok=False)
    manifest = dict(code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip(), hashes=checks['hashes'],
                    protocol=protocol, prepared=str(a.prepared), started_at=datetime.now(timezone.utc).isoformat(), python=sys.executable,
                    max_parallel=a.jobs, test_evaluated=False,
                    gpus=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader'], text=True))
    base.save(a.out / 'manifest.json', manifest)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1', OMP_WAIT_POLICY='PASSIVE',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def status(phase):
        def write(pending, active, done):
            base.save(a.out / 'status.json', dict(phase=phase, pending=[p[0] for p in pending], running=[x[0] for x in active], completed=done,
                                                  updated_at=datetime.now(timezone.utc).isoformat()))
        return write
    train_jobs = []
    for s in protocol['seeds']:
        for cond in ('alloc', 'total'):
            for m in protocol['methods']:
                script = HERE / 'train_alloc.py' if cond == 'alloc' else HERE.parent / 'e60/train_shielded.py'
                dest = a.out / 'runs' / cond / m / f'seed_{s}'
                dest.parent.mkdir(parents=True, exist_ok=True)
                cmd = [sys.executable, '-u', str(script), '--method', m, '--seed', str(s), '--train-shield', protocol['train_shield'],
                       '--prepared', str(a.prepared), '--out', str(dest), '--manifest', str(a.out / 'manifest.json')]
                train_jobs.append(([cond, m, s], cmd, a.out / f'train_{cond}_{m}_s{s}.log'))
    done = pool(train_jobs, a.jobs, env, status('training'))
    failed = [d for d in done if d['exit'] != 0]
    if failed:
        status('failed_training')([], [], done)
        print('FAILED', failed, flush=True)
        return
    cross_jobs = []
    for cond, m, s in [j[0] for j in train_jobs]:
        cmd = [sys.executable, '-u', str(HERE / 'eval_cross.py'), '--condition', cond, '--method', m, '--seed', str(s),
               '--run-dir', str(a.out / 'runs' / cond / m / f'seed_{s}'), '--prepared', str(a.prepared), '--out', str(a.out / 'cross' / cond / m / f'seed_{s}')]
        cross_jobs.append(([cond, m, s], cmd, a.out / f'cross_{cond}_{m}_s{s}.log'))
    done2 = pool(cross_jobs, a.jobs, env, status('cross_eval'))
    failed = [d for d in done2 if d['exit'] != 0]
    assert all(base.sha(p) == v for p, v in checks['hashes'].items())
    status('failed_cross' if failed else 'complete')([], [], done + done2)
    if not failed:
        subprocess.run([sys.executable, str(HERE / 'report.py'), '--root', str(a.out), '--out', str(HERE)], check=True)
    print('COMPLETE' if not failed else f'FAILED {failed}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--check', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=5)
    run(p.parse_args())

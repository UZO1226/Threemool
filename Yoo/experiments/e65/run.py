"""e65: seed 5 ~ 9 학습 20개 (동시 a.train_jobs), 이어서 상한 평가 40개 (동시 a.eval_jobs) 와 집계.

학습은 e63 과 같은 스크립트를 그대로 부른다. 실행 중 checks.json 해시 목록의 파일은 고치지 않는다.
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
EXP = HERE.parent
sys.path.insert(0, str(HERE))
import check  # noqa: E402
base = check.base
PREPARED = Path('/home/jio/fcsrl_src/results/e63/prepared')


def pool(jobs, n, env, out, phase, done_all):
    pending, active, done = list(jobs), [], []

    def status():
        base.save(out / 'status.json', dict(phase=phase, pending=len(pending), running=[x[0] for x in active], completed=done_all + done,
                                            updated_at=datetime.now(timezone.utc).isoformat()))
    while pending or active:
        while pending and len(active) < n:
            name, cmd, log = pending.pop(0)
            active.append((name, time.monotonic(), subprocess.Popen(cmd, env=env, stdout=log.open('w'), stderr=subprocess.STDOUT, start_new_session=True)))
            status()
        for item in list(active):
            name, t0, proc = item
            if proc.poll() is not None:
                active.remove(item)
                done.append(dict(job=name, exit=proc.returncode, seconds=time.monotonic() - t0))
                status()
        time.sleep(10)
    return done


def main(a):
    checks = json.loads(a.check.read_text())
    assert checks['passed'] and all(base.sha(p) == v for p, v in checks['hashes'].items())
    protocol = json.loads((HERE / 'protocol.json').read_text())
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'logs').mkdir()
    manifest = dict(code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip(), hashes=checks['hashes'],
                    protocol=protocol, prepared=str(PREPARED), started_at=datetime.now(timezone.utc).isoformat(), test_evaluated=False,
                    gpus=subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader'], text=True))
    base.save(a.out / 'manifest.json', manifest)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1', OMP_WAIT_POLICY='PASSIVE',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    train = []
    for s in protocol['seeds']:
        for cond in protocol['conditions']:
            for m in protocol['methods']:
                script = EXP / 'e63/train_alloc.py' if cond == 'alloc' else EXP / 'e60/train_shielded.py'
                dest = a.out / 'runs' / cond / m / f'seed_{s}'
                dest.parent.mkdir(parents=True, exist_ok=True)
                cmd = [sys.executable, '-u', str(script), '--method', m, '--seed', str(s), '--train-shield', 'park_e10', '--prepared', str(PREPARED),
                       '--out', str(dest), '--manifest', str(a.out / 'manifest.json')]
                train.append((f'train/{cond}/{m}/{s}', cmd, a.out / 'logs' / f'train_{cond}_{m}_s{s}.log'))
    done = pool(train, a.train_jobs, env, a.out, 'training', [])
    if any(d['exit'] for d in done):
        print('FAILED training', [d for d in done if d['exit']], flush=True)
        return
    evals = []
    for model in ('corrected', 'common'):
        for cond in protocol['conditions']:
            for m in protocol['methods']:
                for s in protocol['seeds']:
                    cmd = [sys.executable, '-u', str(EXP / 'e64/evaluate64.py'), '--policy', 'policy', '--condition', cond, '--method', m, '--seed', str(s),
                           '--model', model, '--cap', '1', '--mode', 'daily', '--shield', 'park_e19', '--runs-root', str(a.out / 'runs'),
                           '--out', str(a.out / 'eval' / f'{model}_cap1_daily' / f'{cond}_{m}_s{s}')]
                    evals.append((f'eval/{model}/{cond}/{m}/{s}', cmd, a.out / 'logs' / f'eval_{model}_{cond}_{m}_s{s}.log'))
    (a.out / 'eval').mkdir()
    for model in ('corrected', 'common'):
        (a.out / 'eval' / f'{model}_cap1_daily').mkdir()
    done2 = pool(evals, a.eval_jobs, dict(env, CUDA_VISIBLE_DEVICES=''), a.out, 'evaluating', done)
    failed = [d for d in done2 if d['exit']]
    assert all(base.sha(p) == v for p, v in checks['hashes'].items())
    base.save(a.out / 'status.json', dict(phase='failed' if failed else 'complete', completed=done + done2, updated_at=datetime.now(timezone.utc).isoformat()))
    if not failed:
        subprocess.run([sys.executable, str(HERE / 'report.py'), '--root', str(a.out), '--out', str(HERE)], check=True)
    print('COMPLETE' if not failed else f'FAILED {failed}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--check', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--train-jobs', type=int, default=5)
    p.add_argument('--eval-jobs', type=int, default=7)
    main(p.parse_args())

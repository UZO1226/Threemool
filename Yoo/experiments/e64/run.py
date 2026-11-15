"""e64 평가 88개를 동시 a.jobs 개로 실행하고 집계한다. 실행 중 checks.json 해시 목록의 파일은 고치지 않는다."""
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
import evaluate64 as ev  # noqa: E402
base = ev.base
CONFIGS = [('corrected', 1, 'daily'), ('common', 1, 'daily'), ('corrected', 0, 'continuous'), ('corrected', 1, 'continuous')]


def jobs(out):
    js = []
    for model, cap, mode in CONFIGS:
        tag = f'{model}_cap{cap}_{mode}'
        common = ['--model', model, '--cap', str(cap), '--mode', mode]
        for cond in ('total', 'alloc'):
            for m in ('ppo', 'fcsrl'):
                for s in range(5):
                    name = f'{cond}_{m}_s{s}'
                    js.append((tag, name, ['--policy', 'policy', '--condition', cond, '--method', m, '--seed', str(s), '--shield', 'park_e19'] + common))
        js.append((tag, 'record', ['--policy', 'record', '--shield', 'none'] + common))
        js.append((tag, 'zero', ['--policy', 'zero', '--shield', 'park_e19'] + common))
    return [(tag, name, [sys.executable, '-u', str(HERE / 'evaluate64.py')] + args + ['--out', str(out / 'runs' / tag / name)]) for tag, name, args in js]


def main(a):
    checks = json.loads(a.check.read_text())
    assert checks['passed'] and all(base.sha(p) == v for p, v in checks['hashes'].items())
    a.out.mkdir(parents=True, exist_ok=False)
    todo = jobs(a.out)
    manifest = dict(code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip(), hashes=checks['hashes'],
                    protocol=json.loads((HERE / 'protocol.json').read_text()), n_jobs=len(todo), started_at=datetime.now(timezone.utc).isoformat(),
                    test_evaluated=False, commands=[c for _, _, c in todo])
    base.save(a.out / 'manifest.json', manifest)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OMP_WAIT_POLICY='PASSIVE', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    (a.out / 'logs').mkdir()
    pending, active, done = list(todo), [], []

    def status(phase):
        base.save(a.out / 'status.json', dict(phase=phase, pending=len(pending), running=[f'{t}/{n}' for t, n, _, _ in active], completed=done,
                                              updated_at=datetime.now(timezone.utc).isoformat()))
    while pending or active:
        while pending and len(active) < a.jobs:
            tag, name, cmd = pending.pop(0)
            (a.out / 'runs' / tag).mkdir(parents=True, exist_ok=True)
            proc = subprocess.Popen(cmd, env=env, stdout=(a.out / 'logs' / f'{tag}_{name}.log').open('w'), stderr=subprocess.STDOUT, start_new_session=True)
            active.append((tag, name, time.monotonic(), proc))
            status('running')
        for item in list(active):
            tag, name, t0, proc = item
            if proc.poll() is not None:
                active.remove(item)
                done.append(dict(job=f'{tag}/{name}', exit=proc.returncode, seconds=time.monotonic() - t0))
                status('running')
        time.sleep(5)
    failed = [d for d in done if d['exit'] != 0]
    assert all(base.sha(p) == v for p, v in checks['hashes'].items())
    status('failed' if failed else 'complete')
    if not failed:
        subprocess.run([sys.executable, str(HERE / 'report.py'), '--root', str(a.out), '--out', str(HERE)], check=True)
    print('COMPLETE' if not failed else f'FAILED {failed}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--check', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=7)
    main(p.parse_args())

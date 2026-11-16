"""e66 test 최종 평가 126개를 한 번만 실행한다.

시작할 때 잠금 파일 TEST_USED.json 을 만든다. 이미 있으면 실행을 거부한다. 중간에 실패해도 잠금은 지우지 않으며,
실패한 작업만 같은 설정으로 다시 돌리는 경우에도 그 사실과 이유를 README 에 남긴다.
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
import evaluate66 as e66  # noqa: E402
base = check.base
CONFIGS = [('corrected', 'daily'), ('common', 'daily'), ('corrected', 'continuous')]


def jobs(out):
    js = []
    for model, mode in CONFIGS:
        tag = f'{model}_cap1_{mode}'
        common = ['--split', 'test', '--model', model, '--cap', '1', '--mode', mode]
        for cond in ('total', 'alloc'):
            for m in ('ppo', 'fcsrl'):
                for s in range(10):
                    root = str(check.ev.E63 if s < 5 else check.E65)
                    js.append((tag, f'{cond}_{m}_s{s}', ['--policy', 'policy', '--condition', cond, '--method', m, '--seed', str(s), '--shield', 'park_e19',
                                                        '--runs-root', root] + common))
        js.append((tag, 'record', ['--policy', 'record', '--shield', 'none'] + common))
        js.append((tag, 'zero', ['--policy', 'zero', '--shield', 'park_e19'] + common))
    return [(tag, name, [sys.executable, '-u', str(HERE / 'evaluate66.py')] + args + ['--out', str(out / 'runs' / tag / name)]) for tag, name, args in js]


def main(a):
    checks = json.loads(a.check.read_text())
    assert checks['passed'] and all(base.sha(p) == v for p, v in checks['hashes'].items())
    assert not e66.LOCK.exists(), 'test 는 이미 사용됐다. 다시 실행하지 않는다'
    e66.LOCK.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()
    base.save(e66.LOCK, dict(used_at=datetime.now(timezone.utc).isoformat(), code_commit=commit, out=str(a.out), check=str(a.check)))
    a.out.mkdir(parents=True, exist_ok=False)
    todo = jobs(a.out)
    base.save(a.out / 'manifest.json', dict(code_commit=commit, hashes=checks['hashes'], protocol=json.loads((HERE / 'protocol.json').read_text()),
                                           n_jobs=len(todo), started_at=datetime.now(timezone.utc).isoformat(), commands=[c for _, _, c in todo]))
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

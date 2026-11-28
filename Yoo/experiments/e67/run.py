"""e67 평가 66개를 동시 a.jobs 개로 실행하고 집계한다."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def main(a):
    assert json.loads(a.check.read_text())['passed']
    protocol = json.loads((HERE / 'protocol.json').read_text())
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'logs').mkdir()
    jobs = []
    for model in ('corrected', 'common'):
        for w in protocol['fixed_w']:
            jobs.append((f'{model}/zero_w{w}', ['--base', 'zero', '--w', str(w), '--model', model]))
            for s in range(10):
                jobs.append((f'{model}/ppo_total_s{s}_w{w}', ['--base', 'ppo_total', '--seed', str(s), '--w', str(w), '--model', model]))
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()
    (a.out / 'manifest.json').write_text(json.dumps(dict(code_commit=commit, n_jobs=len(jobs), jobs=[j[0] for j in jobs]), indent=1))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OMP_WAIT_POLICY='PASSIVE', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    pending, active, done = list(jobs), [], []
    while pending or active:
        while pending and len(active) < a.jobs:
            name, args = pending.pop(0)
            cmd = [sys.executable, '-u', str(HERE / 'evaluate67.py')] + args + ['--out', str(a.out / 'runs' / name)]
            (a.out / 'runs' / name).parent.mkdir(parents=True, exist_ok=True)
            active.append((name, subprocess.Popen(cmd, env=env, stdout=(a.out / 'logs' / (name.replace('/', '_') + '.log')).open('w'), stderr=subprocess.STDOUT)))
        for item in list(active):
            if item[1].poll() is not None:
                active.remove(item)
                done.append(dict(job=item[0], exit=item[1].returncode))
        (a.out / 'status.json').write_text(json.dumps(dict(pending=len(pending), running=[x[0] for x in active], done=done), indent=1))
        time.sleep(5)
    failed = [d for d in done if d['exit']]
    (a.out / 'status.json').write_text(json.dumps(dict(phase='failed' if failed else 'complete', done=done), indent=1))
    if not failed:
        subprocess.run([sys.executable, str(HERE / 'report.py'), '--root', str(a.out), '--out', str(HERE)], check=True)
    print('COMPLETE' if not failed else f'FAILED {failed}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--check', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=8)
    main(p.parse_args())

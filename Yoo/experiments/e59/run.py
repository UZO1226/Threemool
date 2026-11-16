"""e59 평가 전부를 병렬로 실행한다. 검사를 통과한 뒤에만 돈다."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import shield_eval as se

base = se.base


def jobs(protocol):
    out = []
    for shield in protocol['shields']:
        for policy in protocol['policies']:
            for seed in protocol['seeds']:
                out.append((policy, seed, shield))
        for policy in protocol['fixed']:
            out.append((policy, 0, shield))
    return out


def run(a):
    checks = json.loads(a.check.read_text())
    assert checks['passed'] and all(base.sha(p) == v for p, v in checks['hashes'].items())
    protocol = json.loads((se.HERE / 'protocol.json').read_text())
    a.out.mkdir(parents=True, exist_ok=False)
    manifest = dict(code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=se.HERE, text=True).strip(),
                    hashes=checks['hashes'], protocol=protocol, started_at=datetime.now(timezone.utc).isoformat(),
                    python=sys.executable, max_parallel=a.jobs, test_evaluated=False, commands=[])
    base.save(a.out / 'manifest.json', manifest)
    pending, active, complete = jobs(protocol), [], []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OMP_WAIT_POLICY='PASSIVE', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

    def status(phase):
        base.save(a.out / 'status.json', dict(phase=phase, pending=len(pending), running=[j for _, _, j in active], completed=complete,
                                              updated_at=datetime.now(timezone.utc).isoformat()))
    while pending or active:
        while pending and len(active) < a.jobs:
            policy, seed, shield = job = pending.pop(0)
            dest = a.out / 'runs' / shield / f'{policy}_s{seed}'
            dest.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, '-u', str(se.HERE / 'shield_eval.py'), '--policy', policy, '--seed', str(seed), '--shield', shield, '--out', str(dest)]
            log = (a.out / 'logs' / f'{shield}_{policy}_s{seed}.log')
            log.parent.mkdir(exist_ok=True)
            proc = subprocess.Popen(cmd, env=env, stdout=log.open('w'), stderr=subprocess.STDOUT, start_new_session=True)
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
        time.sleep(5)
    failed = [c for c in complete if c['exit'] != 0]
    status('failed' if failed else 'complete')
    assert all(base.sha(p) == v for p, v in checks['hashes'].items())
    if not failed:
        subprocess.run([sys.executable, str(se.HERE / 'report.py'), '--root', str(a.out), '--out', str(se.HERE)], check=True)
    print('COMPLETE' if not failed else f'FAILED {failed}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--check', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=6)
    run(p.parse_args())

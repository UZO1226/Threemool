"""분리 제약 3조건 x 2 seed를 실행한다. 실패 시 중단하고 결과 표를 자동 저장한다."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from dual import HERE, METHODS, save
from common import sha


def main(a):
    assert json.loads((a.out.parent / 'verification.json').read_text())['passed']
    assert json.loads((a.out.parent / 'feasibility_check.json').read_text())['initial_policy_feasible']
    assert json.loads((a.prepared.parent / 'environment_diagnostic.json').read_text())['local_monotonicity_all_members']
    a.out.mkdir(parents=True, exist_ok=False)
    repo = HERE.parents[2]
    upstream = Path('/home/jio/fcsrl_src/FCSRL')
    upstream_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=upstream, text=True).strip()
    assert upstream_commit == 'd882b91054d52c7a63a18a21f7db67bbbed82b63'
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=upstream)
    code = [*HERE.glob('*.py'), * (repo / 'Yoo/code/wwtp').glob('*.py'),
            HERE.parent / 'e30/common.py', HERE.parent / 'e30/darrow_env.py', HERE.parent / 'e29/ablation.py', HERE.parent / 'e25/preserve.py', HERE.parent / 'e23/behavior.py']
    manifest = {'started_at': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
         'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
         'upstream_commit': upstream_commit, 'python': sys.executable, 'parallel_jobs': a.jobs,
         'methods': METHODS, 'seeds': list(range(a.seeds)), 'updates': a.updates,
         'steps_per_run': a.updates * 6080, 'constraint_limits': [6.0,6.0], 'evaluation_split': 'val', 'selection': 'last',
         'source_sha256': {str(p.relative_to(repo)): sha(p) for p in code},
         'input_sha256': {p.name: sha(p) for p in a.prepared.iterdir() if p.is_file()}, 'commands': []}
    pending = [(m, s) for s in range(a.seeds) for m in METHODS]
    active, complete = [], []
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_WAIT_POLICY='PASSIVE')
    def status(phase):
        save(a.out / 'status.json', {'phase': phase, 'completed': complete, 'pending': pending,
             'running': [{'method': m, 'seed': s, 'pid': p.pid} for p, f, m, s in active],
             'updated_at': datetime.now(timezone.utc).isoformat()})
    def stop(signum, frame):
        raise InterruptedError(signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while pending or active:
            while pending and len(active) < a.jobs:
                m, s = pending.pop(0)
                cmd = [sys.executable, '-u', str(HERE / 'train.py'), '--method', m, '--seed', str(s),
                       '--updates', str(a.updates), '--prepared', str(a.prepared),
                       '--out', str(a.out / 'runs' / m / f'seed_{s}')]
                log = (a.out / f'{m}_s{s}.log').open('w')
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=environment,
                                        start_new_session=True)
                active.append((proc, log, m, s))
                manifest['commands'].append(cmd)
                save(a.out / 'manifest.json', manifest)
                print('START', m, s, proc.pid, flush=True)
            status('training')
            time.sleep(5)
            for item in active.copy():
                p, log, m, s = item
                code = p.poll()
                if code is None:
                    continue
                log.close()
                active.remove(item)
                complete.append({'method': m, 'seed': s, 'exit_code': code})
                print('END', m, s, code, flush=True)
                if code:
                    raise RuntimeError(f'{m} seed {s} failed, see log')
        status('summarizing')
        subprocess.run([sys.executable, str(HERE / 'report.py'), '--root', str(a.out),
                        '--prepared', str(a.prepared)], env=environment, check=True)
        manifest['finished_at'] = datetime.now(timezone.utc).isoformat()
        save(a.out / 'manifest.json', manifest)
        status('complete')
    except BaseException:
        for proc, log, _, _ in active:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            log.close()
        status('failed')
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--seeds', type=int, default=2)
    p.add_argument('--updates', type=int, default=66)
    a = p.parse_args()
    assert a.jobs > 0 and a.seeds > 0 and a.updates > 0
    main(a)

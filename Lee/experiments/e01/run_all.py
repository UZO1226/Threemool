"""12개 실행(비용 모드 2 x 방법 2 x seed 3)을 병렬로 돌리고 진행률을 기록한다.

Yoo/experiments/e17/run_pilot.py 와 같은 역할이다. 프로세스 종료 처리는 POSIX
프로세스 그룹 대신 Windows 에서도 동작하는 방식으로 바꿨다.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MODES = ('mean', 'binary')
METHODS = ('ppo', 'fcsrl')
SEEDS = (0, 1, 2)


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--updates', type=int, default=10)
    p.add_argument('--parallel', type=int, default=6)
    p.add_argument('--modes', default=','.join(MODES), help='쉼표로 구분한 비용 모드')
    p.add_argument('--cost-limit', type=float, default=None,
                   help='하루 비용 한도. 생략하면 모드의 기본값')
    p.add_argument('--upstream', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)

    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=a.upstream, text=True).strip()
    assert commit == 'd882b91054d52c7a63a18a21f7db67bbbed82b63', 'Unexpected upstream FCSRL commit'
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                       cwd=a.upstream, text=True), 'Upstream FCSRL must be unmodified'
    metadata = json.loads((a.prepared / 'metadata.json').read_text())
    for name, digest in metadata['files'].items():
        assert hashlib.sha256((a.prepared / name).read_bytes()).hexdigest() == digest, f'{name} changed'

    modes = tuple(x.strip() for x in a.modes.split(',') if x.strip())
    assert all(m in MODES for m in modes), f'Unknown cost mode in {a.modes}'
    jobs = [(mode, method, seed) for seed in SEEDS for mode in modes for method in METHODS]
    steps_per_run = a.updates * 5760
    sources = sorted([*HERE.glob('*.py'), *(ROOT / 'Lee/code/bsm').glob('*.py')])
    save(a.out / 'manifest.json', {
        'started_at': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'upstream_commit': commit, 'python': sys.executable, 'jobs': jobs,
        'parallel_jobs': a.parallel, 'steps_per_run': steps_per_run, 'updates_per_run': a.updates,
        'cost_limit_override': a.cost_limit,
        'source_hashes': {str(s.relative_to(ROOT)): hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
        'prepared_metadata_sha256': hashlib.sha256((a.prepared / 'metadata.json').read_bytes()).hexdigest()})

    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    pending, active, completed = jobs.copy(), [], []
    bar = tqdm(total=len(jobs) * steps_per_run, desc='e01 training transitions', mininterval=10)

    def run_dir(mode, method, seed):
        return a.out / mode / method / f'seed_{seed}'

    def status(phase):
        steps = len(completed) * steps_per_run
        rows = []
        for proc, _, mode, method, seed in active:
            path = run_dir(mode, method, seed) / 'progress.json'
            try:
                progress = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                progress = {'phase': 'starting'}
            steps += progress.get('steps', 0)
            rows.append({'mode': mode, 'method': method, 'seed': seed, 'pid': proc.pid, **progress})
        bar.n = min(steps, bar.total)
        bar.refresh()
        save(a.out / 'status.json', {'phase': phase, 'completed': completed, 'running': rows,
                                     'pending': pending, 'steps': steps, 'total_steps': len(jobs) * steps_per_run,
                                     'updated_at': datetime.now(timezone.utc).isoformat()})

    try:
        while active or pending:
            while pending and len(active) < a.parallel:
                mode, method, seed = pending.pop(0)
                cmd = [sys.executable, '-u', str(HERE / 'train_e01.py'), '--mode', mode, '--method', method,
                       '--seed', str(seed), '--updates', str(a.updates),
                       '--prepared', str(a.prepared.resolve()), '--out', str(run_dir(mode, method, seed).resolve())]
                if a.cost_limit is not None:
                    cmd += ['--cost-limit', str(a.cost_limit)]
                log = (a.out / f'{mode}_{method}_s{seed}.log').open('w')
                proc = subprocess.Popen(cmd, env=environment, stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT))
                active.append((proc, log, mode, method, seed))
                print('START', mode, method, seed, proc.pid, flush=True)
            status('running')
            time.sleep(5)
            for item in active.copy():
                proc, log, mode, method, seed = item
                code = proc.poll()
                if code is None:
                    continue
                active.remove(item)
                log.close()
                if code:
                    raise RuntimeError(f'{mode} {method} seed {seed} exited {code}')
                completed.append({'mode': mode, 'method': method, 'seed': seed, 'exit_code': code})
                print('END', mode, method, seed, flush=True)
        save(a.out / 'results.json', [json.loads((run_dir(*job) / 'metrics.json').read_text()) for job in jobs])
        status('complete')
        print('ALL COMPLETE', flush=True)
    except BaseException:
        for proc, log, *_ in active:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            log.close()
        status('failed')
        raise
    finally:
        bar.close()


if __name__ == '__main__':
    main()

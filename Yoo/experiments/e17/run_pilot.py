"""4조건과 seed 2개의 제한된 BSM 학습을 실행하고 진행률을 기록한다."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def save(path, x):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(x, indent=2, allow_nan=False))
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    upstream = Path('/home/jio/fcsrl_src/FCSRL')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=upstream, text=True).strip()
    assert commit == 'd882b91054d52c7a63a18a21f7db67bbbed82b63'
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=upstream, text=True)
    metadata = json.loads((a.prepared / 'metadata.json').read_text())
    for name, sha in metadata['files'].items():
        assert hashlib.sha256((a.prepared / name).read_bytes()).hexdigest() == sha
    jobs = [(m, s) for s in (0, 1) for m in ('ppo', 'ppo_tail', 'fcsrl', 'fcsrl_tail')]
    pending, active, completed = jobs.copy(), [], []
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    source_files = [*HERE.glob('*.py'), *(ROOT/'Yoo/experiments/e16').glob('*.py'), *(ROOT/'Yoo/code/wwtp').glob('*.py')]
    manifest = {'started_at': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'upstream_commit': commit, 'python': sys.executable, 'jobs': jobs, 'parallel_jobs': 4,
        'steps_per_run': 23040, 'updates_per_run': 4,
        'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        'prepared_metadata_sha256': hashlib.sha256((a.prepared/'metadata.json').read_bytes()).hexdigest()}
    save(a.out/'manifest.json', manifest)
    bar = tqdm(total=len(jobs) * 23040, desc='e17 training transitions', mininterval=5)
    def status(phase):
        steps = len(completed) * 23040
        rows = []
        for proc, _, m, s in active:
            path = a.out / m / f'seed_{s}' / 'progress.json'
            progress = json.loads(path.read_text()) if path.exists() else {'phase': 'starting'}
            steps += progress.get('steps', 0)
            rows.append({'method': m, 'seed': s, 'pid': proc.pid, **progress})
        bar.n = steps
        bar.refresh()
        save(a.out/'status.json', {'phase': phase, 'completed': completed, 'running': rows,
            'pending': pending, 'steps': steps, 'total_steps': len(jobs)*23040,
            'updated_at': datetime.now(timezone.utc).isoformat()})
    def stop(signum, frame):
        raise InterruptedError(f'signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while active or pending:
            while pending and len(active) < 4:
                m, s = pending.pop(0)
                cmd = [sys.executable, '-u', str(HERE/'train_pilot.py'), '--method', m, '--seed', str(s),
                    '--prepared', str(a.prepared.resolve()), '--out', str((a.out/m/f'seed_{s}').resolve())]
                log = (a.out/f'{m}_s{s}.log').open('w')
                proc = subprocess.Popen(cmd, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active.append((proc, log, m, s))
                print('START', m, s, proc.pid, flush=True)
            status('running')
            time.sleep(5)
            for item in active.copy():
                proc, log, m, s = item
                code = proc.poll()
                if code is None:
                    continue
                active.remove(item)
                log.close()
                if code:
                    raise RuntimeError(f'{m} seed {s} exited {code}')
                completed.append({'method': m, 'seed': s, 'exit_code': code})
                print('END', m, s, flush=True)
        save(a.out/'results.json', [json.loads((a.out/m/f'seed_{s}'/'metrics.json').read_text()) for m, s in jobs])
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
    finally:
        bar.close()


if __name__ == '__main__':
    main()

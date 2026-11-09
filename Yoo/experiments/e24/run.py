"""Kaggle PPO/FCSRL/회복 예측 확장을 seed 5개로 비교한다. e23 같은 조건 4개를 재사용한다."""
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CODE = ROOT / 'Yoo/code/wwtp'


def save(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--reuse', type=Path, required=True)
    parser.add_argument('--gbm', type=Path, required=True)
    parser.add_argument('--fcsrl-root', type=Path, default=Path('/home/jio/fcsrl_src/FCSRL'))
    a = parser.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise ValueError(f'Refusing to overwrite {a.out}')
    expected_upstream = 'd882b91054d52c7a63a18a21f7db67bbbed82b63'
    upstream = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=a.fcsrl_root, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=a.fcsrl_root, text=True)
    if upstream != expected_upstream or dirty:
        raise RuntimeError('The pinned upstream FCSRL source must be unchanged')
    a.out.mkdir(parents=True)
    jobs = [(method, seed) for seed in range(5) for method in ('ppo', 'fcsrl', 'fcsrl_recovery')]
    pending, active, completed, reused_files = [], [], [], []
    for method, seed in jobs:
        if seed < 2 and method in ('ppo', 'fcsrl'):
            source = (a.reuse / method / f'seed_{seed}').resolve()
            m = json.loads((source / 'metrics.json').read_text())
            assert m['steps'] == 202048 and m['method'] == method and m['seed'] == seed
            target = a.out / 'runs' / method / f'seed_{seed}'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source, target_is_directory=True)
            reused_files += [source / 'metrics.json', source / 'cross_evaluation.json', *sorted((source / 'model_final').glob('*'))]
            completed.append({'method': method, 'seed': seed, 'exit_code': 0, 'reused': True})
        else:
            pending.append((method, seed))
    environment = dict(os.environ, PYTHONPATH=str(a.fcsrl_root), OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    manifest = {'started_at': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'upstream_commit': upstream, 'python': sys.executable, 'parallel_jobs': 4,
        'steps_requested': 200000, 'steps_actual_per_run': 202048, 'seeds': [0, 1, 2, 3, 4],
        'methods': ['ppo', 'fcsrl', 'fcsrl_recovery'], 'evaluation_split': 'val',
        'inputs': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.data, a.model, a.gbm, a.prepared / 'prior.pt', a.prepared / 'normalizer.json', a.prepared / 'preparation.json')},
        'reused_inputs': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in reused_files},
        'source': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted([*CODE.glob('*.py'), *HERE.glob('*.py'), HERE.parent / 'e23/behavior.py', HERE.parent / 'e23/review.py'])}, 'commands': []}

    def status(phase):
        save(a.out / 'status.json', {'phase': phase, 'pid': os.getpid(), 'completed': completed,
            'running': [{'method': m, 'seed': s, 'pid': p.pid} for p, _, m, s in active],
            'pending': pending, 'updated_at': datetime.now(timezone.utc).isoformat()})

    def stop(signum, frame):
        raise InterruptedError(f'Received signal {signum}')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while pending or active:
            while pending and len(active) < 4:
                method, seed = pending.pop(0)
                command = [sys.executable, '-u', str(HERE / 'train.py'), '--method', method,
                    '--seed', str(seed), '--data', str(a.data.resolve()), '--model', str(a.model.resolve()),
                    '--out', str((a.out / 'runs' / method / f'seed_{seed}').resolve()),
                    '--prepared', str(a.prepared.resolve()), '--gbm', str(a.gbm.resolve())]
                log = (a.out / f'{method}_s{seed}.log').open('w')
                proc = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                active.append((proc, log, method, seed))
                manifest['commands'].append(command)
                save(a.out / 'manifest.json', manifest)
                print('START', method, seed, proc.pid, flush=True)
            status('running')
            time.sleep(2)
            for item in active.copy():
                proc, log, method, seed = item
                code = proc.poll()
                if code is None:
                    continue
                log.close()
                active.remove(item)
                completed.append({'method': method, 'seed': seed, 'exit_code': code})
                print('END', method, seed, code, flush=True)
                if code:
                    raise RuntimeError(f'{method} seed {seed} failed ({code})')
        results = [json.loads((a.out / 'runs' / m / f'seed_{s}' / 'metrics.json').read_text()) for m, s in jobs]
        save(a.out / 'results.json', results)
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
    main()

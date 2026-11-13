"""Run completed history-policy pairs with immutable comparison sources."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'freeze', 'baseline-root', 'candidate-root', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--parallel', type=int, default=5)
    args = parser.parse_args()
    assert args.parallel > 0 and len(args.seeds) == len(set(args.seeds))
    assert all(0 <= seed < 5 for seed in args.seeds)
    jobs = []
    for seed in args.seeds:
        baseline, candidate = [root / f'seed_{seed}' for root in (args.baseline_root, args.candidate_root)]
        for folder in (baseline, candidate):
            assert json.loads((folder / 'progress.json').read_text())['phase'] == 'complete'
            assert json.loads((folder / 'metrics.json').read_text())['steps'] == 401280
        jobs.append({'seed': seed, 'baseline': str(baseline.resolve()), 'candidate': str(candidate.resolve())})
    args.out.mkdir(parents=True, exist_ok=False)
    paths = [*HERE.glob('*.py'), ROOT / 'Lee/experiments/e44/air_match.py', ROOT / 'Lee/experiments/e44/report.py']
    hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    manifest = {'jobs': jobs, 'source_hashes': hashes, 'kind': 'agent', 'environment': 'new',
        'freeze_sha256': hashlib.sha256(args.freeze.read_bytes()).hexdigest(), 'test_evaluated': False}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       NUMBA_NUM_THREADS='1', PYTHONIOENCODING='utf-8')
    pending, active, complete = jobs.copy(), [], []
    def status(phase):
        (args.out / 'status.json').write_text(json.dumps({'phase': phase, 'complete': complete,
            'running': [{'seed': job['seed'], 'pid': proc.pid} for proc, log, job in active],
            'pending': [job['seed'] for job in pending]}, indent=2))
    try:
        while pending or active:
            while pending and len(active) < args.parallel:
                for name, expected in hashes.items():
                    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, 'Comparison source changed'
                assert hashlib.sha256(args.freeze.read_bytes()).hexdigest() == manifest['freeze_sha256']
                job = pending.pop(0)
                log = (args.out / f"seed_{job['seed']}.log").open('w', encoding='utf-8')
                command = [sys.executable, '-u', str(HERE / 'compare.py'), '--data', str(args.data.resolve()),
                    '--freeze', str(args.freeze.resolve()), '--baseline', job['baseline'], '--candidate', job['candidate'],
                    '--out', str((args.out / f"seed_{job['seed']}").resolve())]
                proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment, cwd=ROOT.parent)
                active.append((proc, log, job))
                print('START', job['seed'], proc.pid, flush=True)
            status('running')
            time.sleep(5)
            for entry in active.copy():
                proc, log, job = entry
                code = proc.poll()
                if code is None:
                    continue
                active.remove(entry)
                log.close()
                if code:
                    raise RuntimeError(f"Seed{job['seed']} exited{code}; retain its partial outputs")
                assert (args.out / f"seed_{job['seed']}" / 'summary.json').exists()
                complete.append(job['seed'])
                print('COMPLETE', job['seed'], flush=True)
        status('complete')
    except BaseException:
        for proc, log, job in active:
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


if __name__ == '__main__':
    main()

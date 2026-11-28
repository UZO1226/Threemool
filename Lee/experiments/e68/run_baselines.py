"""Train and authenticate a fresh fixed seed block with historical trainers."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SEEDS = [5, 6, 7, 8, 9]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    for attempt in range(100):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 99:
                raise
            time.sleep(.05)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    workspace, out = args.workspace.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    data = workspace / 'kaggle_data/darrow_15min.pkl'
    freeze = ROOT / 'Lee/experiments/e42/environment_freeze.json'
    folders = [HERE, ROOT / 'Lee/code/darrow', *[ROOT / f'Lee/experiments/{name}' for name in ('e38', 'e42', 'e55')]]
    sources = {str(path.resolve()): sha(path) for folder in folders for path in folder.glob('*.py')}
    out.mkdir(parents=True)
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        NUMBA_NUM_THREADS='1', DARROW_FAST_DYNAMICS='1', PYTHONIOENCODING='utf-8')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    write(out / 'study_manifest.json', {'code_commit': commit, 'policy_seeds': SEEDS,
        'actual_steps_per_policy': 401280, 'independent_original_and_capture_training': True,
        'source_sha256': sources, 'data_sha256': sha(data), 'freeze_sha256': sha(freeze),
        'test_evaluated': False, 'candidate_evaluation': False, 'parallel_trainers': 2})
    completed = []
    started = time.monotonic()
    def phase(name, jobs, parallel=2):
        active, pending = [], jobs.copy()
        try:
            while active or pending:
                assert all(sha(Path(path)) == digest for path, digest in sources.items()), 'Pinned source changed'
                assert sha(freeze) == json.loads((out / 'study_manifest.json').read_text())['freeze_sha256']
                while pending and len(active) < parallel:
                    label, command = pending.pop(0)
                    log = (out / f'{label}.log').open('w', encoding='utf-8')
                    process = subprocess.Popen([sys.executable, '-u', *map(str, command)], cwd=workspace,
                        env=environment, stdout=log, stderr=subprocess.STDOUT)
                    active.append((label, process, log))
                    print('START', label, process.pid, flush=True)
                for label, process, log in active.copy():
                    code = process.poll()
                    if code is None:
                        continue
                    active.remove((label, process, log))
                    log.close()
                    if code:
                        raise RuntimeError(f'{label} exited{code}; inspect preserved log')
                    completed.append(label)
                    print('COMPLETE', label, flush=True)
                write(out / 'study_status.json', {'phase': name, 'completed': completed,
                    'running': [{'label': label, 'pid': process.pid} for label, process, _ in active],
                    'pending': [label for label, _ in pending], 'elapsed_seconds': time.monotonic() - started})
                if active:
                    time.sleep(2)
        except BaseException:
            for _, process, log in active:
                process.terminate()
                process.wait()
                log.close()
            write(out / 'failure.json', {'phase': name, 'terminal': True, 'completed': completed})
            raise
    for mode, trainer in [('original', ROOT / 'Lee/experiments/e42/train_policy.py'),
                          ('captured', ROOT / 'Lee/experiments/e55/train_stream.py')]:
        folder = out / mode
        folder.mkdir()
        write(folder / 'manifest.json', {'code_commit': commit, 'source_hashes': sources,
            'freeze_sha256': sha(freeze), 'jobs': [{'environment': 'new', 'method': 'ppolag_fcsrl', 'seed': s} for s in SEEDS],
            'steps_requested': 400000, 'actual_steps_per_job': math.ceil(400000 / 6080) * 6080,
            'evaluations': 5, 'test_evaluated': False, 'new_policy_training': True,
            'scope': 'Fresh within-Lee block; no shared-protocol modification.'})
        jobs = [(f'{mode}_s{seed}', [trainer, '--data', data, '--freeze', freeze,
            '--out', folder / f'new_ppolag_fcsrl/seed_{seed}', '--environment', 'new',
            '--method', 'ppolag_fcsrl', '--seed', seed, '--steps', 400000, '--evaluations', 5]) for seed in SEEDS]
        phase(mode + '_training', jobs)
    phase('exact_capture_equivalence', [(f'equivalence_s{seed}', [ROOT / 'Lee/experiments/e55/verify_base.py',
          '--run', out / f'captured/new_ppolag_fcsrl/seed_{seed}',
          '--baseline', out / f'original/new_ppolag_fcsrl/seed_{seed}', '--freeze', freeze,
          '--out', out / f'captured/new_ppolag_fcsrl/seed_{seed}/base_equivalence.json']) for seed in SEEDS])
    records = []
    for seed in SEEDS:
        run = out / f'captured/new_ppolag_fcsrl/seed_{seed}'
        check = json.loads((run / 'base_equivalence.json').read_text())
        assert check['passed'] and check['policy_seed'] == seed and check['steps'] == 401280
        progress = json.loads((run / 'progress.json').read_text())
        assert progress['phase'] == 'complete' and progress['steps'] == 401280
        records.append({'policy_seed': seed, 'base_equivalence_sha256': sha(run / 'base_equivalence.json'),
                        'checkpoint_hashes': check['checkpoint_hashes']})
    write(out / 'baseline_training_summary.json', {'passed': True, 'policy_seeds': SEEDS,
        'records': records, 'test_evaluated': False, 'candidate_evaluation': False,
        'scope': 'Fresh original/captured policy parity only; e63 risk models and matched comparators not yet built.'})
    write(out / 'study_status.json', {'phase': 'complete', 'completed': completed,
        'running': [], 'pending': [], 'elapsed_seconds': time.monotonic() - started})
    print('FRESH_BASELINE_TRAINING_COMPLETE', flush=True)


if __name__ == '__main__':
    main()

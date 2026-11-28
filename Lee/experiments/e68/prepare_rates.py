"""Fit unchanged capture-provenance and horizon4 models after fresh parity."""
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
    for name in ('workspace', 'baselines', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    workspace, baselines, out = args.workspace.resolve(), args.baselines.resolve(), args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    training = json.loads((baselines / 'baseline_training_summary.json').read_text())
    status = json.loads((baselines / 'study_status.json').read_text())
    assert training['passed'] and training['policy_seeds'] == SEEDS
    assert status['phase'] == 'complete' and not status['running'] and not status['pending']
    assert not training['test_evaluated'] and not training['candidate_evaluation']
    original_manifest = json.loads((baselines / 'study_manifest.json').read_text())
    assert all(sha(Path(path)) == digest for path, digest in original_manifest['source_sha256'].items())
    for seed, record in zip(SEEDS, training['records']):
        assert record['policy_seed'] == seed
        run = baselines / f'captured/new_ppolag_fcsrl/seed_{seed}'
        assert sha(run / 'base_equivalence.json') == record['base_equivalence_sha256']
        for name, expected in record['checkpoint_hashes'].items():
            assert sha(run / 'model_final' / name) == expected
    folders = [HERE, *[ROOT / f'Lee/experiments/{name}' for name in ('e44', 'e55', 'e56')]]
    sources = {str(path.resolve()): sha(path) for folder in folders for path in folder.glob('*.py')}
    sources.update(original_manifest['source_sha256'])
    out.mkdir(parents=True)
    write(out / 'study_manifest.json', {'source_sha256': sources, 'policy_seeds': SEEDS,
        'baseline_training_summary_sha256': sha(baselines / 'baseline_training_summary.json'),
        'test_evaluated': False, 'candidate_evaluation': False, 'base_training_steps': 401280,
        'horizon': 4, 'extra_environment_training_steps': 0,
        'legacy_one_step_models': 'Same original e55 provenance preparation, not selected or used by h4 controller.'})
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        NUMBA_NUM_THREADS='1', DARROW_FAST_DYNAMICS='1', PYTHONIOENCODING='utf-8')
    completed = []
    def phase(name, jobs):
        active, pending = [], jobs.copy()
        try:
            while active or pending:
                assert all(sha(Path(path)) == expected for path, expected in sources.items())
                while pending and len(active) < 2:
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
                    'pending': [label for label, _ in pending]})
                if active:
                    time.sleep(1)
        except BaseException:
            for _, process, log in active:
                process.terminate()
                process.wait()
                log.close()
            write(out / 'failure.json', {'phase': name, 'terminal': True, 'completed': completed})
            raise
    data = workspace / 'kaggle_data/darrow_15min.pkl'
    freeze = ROOT / 'Lee/experiments/e42/environment_freeze.json'
    phase('legacy_capture_provenance', [(f'capture_fit_s{seed}', [ROOT / 'Lee/experiments/e55/fit_models.py',
          '--run', baselines / f'captured/new_ppolag_fcsrl/seed_{seed}', '--data', data,
          '--out', baselines / f'captured/new_ppolag_fcsrl/seed_{seed}/correction']) for seed in SEEDS])
    phase('fixed_h4_rate_models', [(f'rate_fit_s{seed}', [ROOT / 'Lee/experiments/e56/fit_rates.py',
          '--run', baselines / f'captured/new_ppolag_fcsrl/seed_{seed}', '--data', data, '--freeze', freeze,
          '--baseline', baselines / f'original/new_ppolag_fcsrl/seed_{seed}', '--horizon', 4,
          '--out', out / f'models/h4/seed_{seed}']) for seed in SEEDS])
    sys.path.insert(0, str(ROOT / 'Lee/experiments/e56'))
    from rate_io import load_base, load_rates
    records = []
    for seed in SEEDS:
        run = baselines / f'captured/new_ppolag_fcsrl/seed_{seed}'
        _, config, _, hashes = load_base(run, freeze, baselines / f'original/new_ppolag_fcsrl/seed_{seed}')
        _, fit = load_rates(out / f'models/h4/seed_{seed}', config, 4, False)
        assert fit['base_checkpoint_hashes'] == hashes
        records.append({'policy_seed': seed, 'model_sha256': fit['model_sha256'],
                        'base_checkpoint_hashes': hashes, 'checks': fit['checks']})
    write(out / 'rate_preparation_summary.json', {'passed': True, 'policy_seeds': SEEDS, 'records': records,
        'test_evaluated': False, 'candidate_evaluation': False, 'extra_environment_training_steps': 0,
        'scope': 'Fresh risk weights and parity only; no matched e63 or candidate quality result yet.'})
    write(out / 'study_status.json', {'phase': 'complete', 'completed': completed, 'running': [], 'pending': []})
    print('FRESH_RATE_PREPARATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()

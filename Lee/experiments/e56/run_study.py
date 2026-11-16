"""Run both fixed horizons on five existing policy streams, retaining all stages."""
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


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    for attempt in range(40):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(.05)


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'freeze', 'captures', 'baselines', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    smoke = json.loads((HERE / 'smoke_pipeline.json').read_text())
    assert smoke['passed'] and smoke['smoke_only']
    for name, digest in smoke['runtime_sources'].items():
        assert hashlib.sha256((HERE / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == digest
    args.out.mkdir(parents=True, exist_ok=False)
    sources = sorted({*HERE.glob('*.py'), *(ROOT / 'Lee/code/darrow').glob('*.py'),
                      *(ROOT / 'Lee/experiments/e55').glob('*.py'),
                      *(ROOT / 'Lee/experiments/e42').glob('*.py'),
                      *(ROOT / 'Lee/experiments/e44').glob('*.py'),
                      *(ROOT / 'Lee/experiments/e38').glob('*.py')})
    hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    manifest = {'policy_seeds': list(range(5)), 'regressor_seeds': [5600, 5601],
                'horizons': [4, 1], 'primary_horizon': 4, 'control_horizon': 1,
                'base_policy_steps_per_seed': 401280, 'new_ppo_training': False,
                'extra_environment_training_steps': 0, 'source_sha256': hashes,
                'fit_updates': list(range(1, 53)), 'held_updates': list(range(53, 67)),
                'test_evaluated': False, 'captures': str(args.captures.resolve()),
                'code_commit': subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}',
                                                       'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    write(args.out / 'study_manifest.json', manifest)
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       NUMBA_NUM_THREADS='1', PYTHONIOENCODING='utf-8', DARROW_FAST_DYNAMICS='1')
    completed = []
    start = time.monotonic()
    def phase(name, jobs, parallel=5):
        pending, active = jobs.copy(), []
        try:
            while pending or active:
                for path, digest in hashes.items():
                    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, f'Source changed: {path}'
                while pending and len(active) < parallel:
                    label, command = pending.pop(0)
                    log = (args.out / f'{label}.log').open('w', encoding='utf-8')
                    process = subprocess.Popen([sys.executable, '-u', *map(str, command)], cwd=ROOT.parent,
                                               env=environment, stdout=log, stderr=subprocess.STDOUT)
                    active.append((label, process, log))
                    print('START', name, label, process.pid, flush=True)
                for item in active.copy():
                    label, process, log = item
                    code = process.poll()
                    if code is None:
                        continue
                    active.remove(item)
                    log.close()
                    if code:
                        raise RuntimeError(f'{label} exited{code}; inspect preserved log')
                    completed.append(label)
                    print('COMPLETE', name, label, flush=True)
                write(args.out / 'study_status.json', {'phase': name, 'completed': completed,
                      'running': [{'label': label, 'pid': process.pid} for label, process, _ in active],
                      'pending': [label for label, _ in pending], 'elapsed_seconds': time.monotonic() - start})
                if active:
                    time.sleep(5)
        except BaseException:
            for _, process, log in active:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                log.close()
            write(args.out / 'study_status.json', {'phase': 'failed', 'failed_stage': name,
                  'completed': completed, 'elapsed_seconds': time.monotonic() - start})
            raise
    fits, held, comparisons, actions = [], [], [], []
    for horizon in (4, 1):
        for seed in range(5):
            run = args.captures / f'seed_{seed}'
            baseline = args.baselines / f'seed_{seed}'
            model = args.out / f'models/h{horizon}/seed_{seed}'
            comparison = args.out / f'comparisons/h{horizon}/seed_{seed}'
            shared = ['--run', run.resolve(), '--baseline', baseline.resolve(),
                      '--freeze', args.freeze.resolve(), '--data', args.data.resolve(), '--horizon', horizon]
            label = f'h{horizon}_s{seed}'
            fits.append((f'fit_{label}', [HERE / 'fit_rates.py', *shared, '--out', model.resolve()]))
            held.append((f'held_{label}', [HERE / 'audit_rates.py', *shared, '--correction', model.resolve(),
                                          '--phase', 'held', '--out', (model / 'held_audit.json').resolve()]))
            comparisons.append((f'compare_{label}', [HERE / 'compare_rates.py', '--horizon', horizon,
                '--data', args.data.resolve(), '--freeze', args.freeze.resolve(), '--baseline', baseline.resolve(),
                '--candidate', run.resolve(), '--correction', model.resolve(), '--out', comparison.resolve()]))
            actions.append((f'actions_{label}', [HERE / 'audit_rates.py', *shared, '--correction', model.resolve(),
                '--phase', 'actions', '--comparison', comparison.resolve(), '--out', (comparison / 'action_audit.json').resolve()]))
    phase('fit_models', fits)
    phase('held_model_audits', held)
    phase('matched_air_comparisons', comparisons)
    phase('independent_action_audits', actions)
    reports = []
    for horizon in (4, 1):
        reports.append((f'report_h{horizon}', [ROOT / 'Lee/experiments/e44/report.py', '--pairs',
            *[(args.out / f'comparisons/h{horizon}/seed_{seed}/summary.json').resolve() for seed in range(5)],
            '--out', HERE / f'h{horizon}_comparison.json']))
    phase('five_seed_reports', reports, 1)
    phase('horizon_control_contrast', [('contrast', [HERE / 'horizon_contrast.py', '--study', args.out.resolve(),
                                                  '--out', HERE / 'horizon_contrast.json'])], 1)
    write(args.out / 'study_status.json', {'phase': 'complete', 'completed': completed,
          'running': [], 'pending': [], 'elapsed_seconds': time.monotonic() - start})
    print('E56_STUDY_COMPLETE', flush=True)


if __name__ == '__main__':
    main()

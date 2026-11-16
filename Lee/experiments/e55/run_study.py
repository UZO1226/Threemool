"""Run all fixed policy seeds, retain logs, then fit and compare the correction."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
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
    for name in ('data', 'freeze', 'baselines', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    checks = json.loads((HERE / 'smoke_pipeline.json').read_text())
    assert checks['passed'] and checks['smoke_only']
    for name, digest in checks['runtime_sources'].items():
        assert hashlib.sha256((HERE / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == digest
    sources = sorted({*HERE.glob('*.py'), *(ROOT / 'Lee/code/darrow').glob('*.py'),
                      *(ROOT / 'Lee/experiments/e42').glob('*.py'),
                      *(ROOT / 'Lee/experiments/e44').glob('*.py'),
                      *(ROOT / 'Lee/experiments/e38').glob('*.py')})
    hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    manifest = {'policy_seeds': list(range(5)), 'steps_requested': 400000,
                'actual_steps_per_policy': 401280, 'classifier_seeds': [5500, 5501],
                'extra_environment_training_steps': 0, 'source_sha256': hashes,
                'test_evaluated': False, 'model_fit_updates': 52, 'diagnostic_held_updates': 14,
                'runtime': {'python': platform.python_version(), 'device': 'cpu',
                            'threads_per_process': 1,
                            'packages': {name: importlib.metadata.version(name) for name in
                                         ('numpy', 'pandas', 'scikit-learn', 'torch', 'numba')}},
                'code_commit': subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}',
                                                       'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    write(args.out / 'study_manifest.json', manifest)
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       NUMBA_NUM_THREADS='1', PYTHONIOENCODING='utf-8', DARROW_FAST_DYNAMICS='1')
    completed = []
    started = time.monotonic()
    def run_phase(name, commands, parallel):
        pending = commands.copy()
        active = []
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
                        raise RuntimeError(f'{label} exited{code}; preserve and inspect its log')
                    completed.append(label)
                    print('COMPLETE', name, label, flush=True)
                write(args.out / 'study_status.json', {'phase': name, 'completed': completed,
                      'running': [{'label': label, 'pid': process.pid} for label, process, _ in active],
                      'pending': [label for label, _ in pending], 'elapsed_seconds': time.monotonic() - started})
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
                  'completed': completed, 'elapsed_seconds': time.monotonic() - started})
            raise
    policy_root = args.out / 'policies'
    train = [ROOT / 'Lee/experiments/e42/run_policies.py', '--data', args.data.resolve(),
             '--freeze', args.freeze.resolve(), '--out', policy_root.resolve(),
             '--trainer', HERE / 'train_stream.py', '--parallel', '5', '--jobs',
             *[f'new:ppolag_fcsrl:{seed}' for seed in range(5)]]
    run_phase('capture_training', [('capture_training', train)], 1)
    runs = [policy_root / 'new_ppolag_fcsrl' / f'seed_{seed}' for seed in range(5)]
    base_checks = [(f'base_check_s{seed}', [HERE / 'verify_base.py', '--run', run.resolve(),
                    '--freeze', args.freeze.resolve(), '--baseline', (args.baselines / f'seed_{seed}').resolve(),
                    '--out', (run / 'base_equivalence.json').resolve()]) for seed, run in enumerate(runs)]
    run_phase('unchanged_reference_equivalence', base_checks, 5)
    fits = [(f'fit_s{seed}', [HERE / 'fit_models.py', '--run', run.resolve(), '--data', args.data.resolve(),
                              '--out', (run / 'correction').resolve()]) for seed, run in enumerate(runs)]
    run_phase('cost_model_fit', fits, 5)
    model_audits = [(f'model_audit_s{seed}', [HERE / 'audit_cost_models.py', '--run', run.resolve(),
                    '--data', args.data.resolve(), '--correction', (run / 'correction').resolve(),
                    '--out', (run / 'correction' / 'audit.json').resolve()]) for seed, run in enumerate(runs)]
    run_phase('held_stream_model_audit', model_audits, 5)
    comparisons = []
    audits = []
    for seed, run in enumerate(runs):
        shared = ['--data', args.data.resolve(), '--freeze', args.freeze.resolve(),
                  '--baseline', (args.baselines / f'seed_{seed}').resolve(),
                  '--candidate', run.resolve(), '--correction', (run / 'correction').resolve()]
        comparison = args.out / 'comparisons' / f'seed_{seed}'
        comparisons.append((f'compare_s{seed}', [HERE / 'compare.py', *shared, '--out', comparison.resolve()]))
        audits.append((f'audit_s{seed}', [HERE / 'audit_correction.py', *shared, '--comparison', comparison.resolve(),
                                         '--out', (comparison / 'action_audit.json').resolve()]))
    run_phase('matched_air_comparison', comparisons, 5)
    run_phase('independent_action_audit', audits, 5)
    report = [ROOT / 'Lee/experiments/e44/report.py', '--pairs',
              *[(args.out / 'comparisons' / f'seed_{seed}' / 'summary.json').resolve() for seed in range(5)],
              '--out', (HERE / 'comparison.json').resolve()]
    run_phase('five_seed_report', [('report', report)], 1)
    write(args.out / 'study_status.json', {'phase': 'complete', 'completed': completed,
          'running': [], 'pending': [], 'elapsed_seconds': time.monotonic() - started})
    print('E55_STUDY_COMPLETE', flush=True)


if __name__ == '__main__':
    main()

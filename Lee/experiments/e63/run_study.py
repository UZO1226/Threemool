"""Guard the native smoke, then run all fixed risk-only comparison stages."""
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'freeze', 'policies', 'baselines', 'previous', 'smoke', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    smoke = json.loads((args.smoke / 'summary.json').read_text())
    audit = json.loads((args.smoke / 'action_audit.json').read_text())
    assert smoke['pipeline_smoke_only'] and smoke['matching'] is None
    assert smoke['training_steps'] == 401280 and not smoke['test_evaluated']
    assert smoke['candidate_method'] == 'ppolag_fcsrl_risk_only_h4' and smoke['penalty'] == 0
    assert audit['passed'] and audit['flat_risk_null_full_native_exact']
    assert audit['comparison_summary_sha256'] == digest(args.smoke / 'summary.json')
    assert audit['replays']['native']['action_rows'] == 152000
    assert audit['replays']['native']['independent_sklearn_actions_exact']
    assert audit['replays']['native']['trajectory_replay_exact']
    assert audit['replays']['native']['executed_air_accounting_exact']
    for name, expected in smoke['comparison_sources_sha256'].items():
        assert digest(HERE / name) == expected
    folders = [HERE, ROOT / 'Lee/code/darrow', *[ROOT / ('Lee/experiments/' + name)
                for name in ('e38', 'e42', 'e44', 'e55', 'e56')]]
    sources = sorted({path for folder in folders for path in folder.glob('*.py')})
    hashes = {str(path.resolve()): digest(path) for path in sources}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'final_reports').mkdir()
    write(args.out / 'study_manifest.json', {'policy_seeds': list(range(5)), 'penalty': 0, 'horizon': 4,
        'base_policy_steps_per_seed': 401280, 'new_policy_training': False, 'new_model_fitting': False,
        'extra_environment_training_steps': 0, 'test_evaluated': False, 'source_sha256': hashes,
        'smoke_summary_sha256': digest(args.smoke / 'summary.json'), 'smoke_audit_sha256': digest(args.smoke / 'action_audit.json'),
        'code_commit': subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}',
                                               'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       NUMBA_NUM_THREADS='1', PYTHONIOENCODING='utf-8', DARROW_FAST_DYNAMICS='1')
    completed = []
    start = time.monotonic()

    def phase(name, jobs, parallel=5):
        pending, active = jobs.copy(), []
        try:
            while pending or active:
                for path, expected in hashes.items():
                    assert digest(Path(path)) == expected, f'Source changed: {path}'
                while pending and len(active) < parallel:
                    label, command = pending.pop(0)
                    log = (args.out / (label + '.log')).open('w', encoding='utf-8')
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

    comparisons, audits = [], []
    for seed in range(5):
        comparison = args.out / f'comparisons/seed_{seed}'
        shared = ['--data', args.data.resolve(), '--freeze', args.freeze.resolve(),
                  '--baseline', (args.baselines / f'seed_{seed}').resolve(),
                  '--candidate', (args.policies / f'seed_{seed}').resolve(),
                  '--correction', (args.previous / f'models/h4/seed_{seed}').resolve()]
        comparisons.append((f'compare_s{seed}', [HERE / 'compare_trust.py', *shared, '--out', comparison.resolve()]))
        audits.append((f'audit_s{seed}', [HERE / 'audit_trust.py', *shared, '--comparison', comparison.resolve(),
                                        '--out', (comparison / 'action_audit.json').resolve()]))
    phase('matched_air_comparisons', comparisons)
    phase('independent_native_matched_audits', audits)
    phase('five_seed_report', [('report', [ROOT / 'Lee/experiments/e44/report.py', '--pairs',
        *[(args.out / f'comparisons/seed_{seed}/summary.json').resolve() for seed in range(5)],
        '--out', (args.out / 'final_reports/comparison.json').resolve()])], 1)
    phase('secondary_penalty_contrast', [('contrast', [HERE / 'penalty_contrast.py', '--study', args.out.resolve(),
        '--previous', args.previous.resolve(), '--out', (args.out / 'final_reports/penalty_contrast.json').resolve()])], 1)
    write(args.out / 'study_status.json', {'phase': 'complete', 'completed': completed,
        'running': [], 'pending': [], 'elapsed_seconds': time.monotonic() - start})
    print('E63_STUDY_COMPLETE', flush=True)


if __name__ == '__main__':
    main()

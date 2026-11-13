"""Run all five fixed actuator-filter pairs and independent matched-feedback audits."""
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
    for name in ('data', 'freeze', 'policies', 'baselines', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    check = json.loads((HERE / 'pipeline_check.json').read_text())
    assert check['passed'] and check['smoke_only']
    for name, expected in check['runtime_sources'].items():
        assert hashlib.sha256((HERE / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() == expected
    assert hashlib.sha256((ROOT / 'Lee/experiments/e44/air_match.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest() == check['shared_matcher_source_sha256']
    args.out.mkdir(parents=True, exist_ok=False)
    sources = sorted({*HERE.glob('*.py'), *(ROOT / 'Lee/code/darrow').glob('*.py'),
                      *(ROOT / 'Lee/experiments/e55').glob('*.py'),
                      *(ROOT / 'Lee/experiments/e42').glob('*.py'),
                      *(ROOT / 'Lee/experiments/e44').glob('*.py'),
                      *(ROOT / 'Lee/experiments/e38').glob('*.py')})
    hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    manifest = {'policy_seeds': list(range(5)), 'filter_gain': .5, 'new_policy_training': False,
                'new_supervised_fitting': False, 'base_policy_steps_per_seed': 401280,
                'extra_environment_training_steps': 0, 'test_evaluated': False,
                'source_sha256': hashes, 'original_policies': str(args.policies.resolve()),
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
    comparisons, audits = [], []
    for seed in range(5):
        policy = args.policies / f'seed_{seed}'
        baseline = args.baselines / f'seed_{seed}'
        out = args.out / f'comparisons/seed_{seed}'
        comparisons.append((f'compare_s{seed}', [HERE / 'compare_filter.py', '--data', args.data.resolve(),
            '--freeze', args.freeze.resolve(), '--baseline', baseline.resolve(), '--candidate', policy.resolve(),
            '--out', out.resolve()]))
        audits.append((f'audit_s{seed}', [HERE / 'audit_filter.py', '--run', policy.resolve(),
            '--baseline', baseline.resolve(), '--freeze', args.freeze.resolve(), '--data', args.data.resolve(),
            '--comparison', out.resolve(), '--out', (out / 'action_audit.json').resolve()]))
    phase('matched_air_comparisons', comparisons)
    phase('independent_native_and_matched_audits', audits)
    phase('five_seed_report', [('report', [ROOT / 'Lee/experiments/e44/report.py', '--pairs',
          *[(args.out / f'comparisons/seed_{seed}/summary.json').resolve() for seed in range(5)],
          '--out', HERE / 'comparison.json'])], 1)
    write(args.out / 'study_status.json', {'phase': 'complete', 'completed': completed,
          'running': [], 'pending': [], 'elapsed_seconds': time.monotonic() - start})
    print('E57_STUDY_COMPLETE', flush=True)


if __name__ == '__main__':
    main()

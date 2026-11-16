"""Guard the native smoke, then compare and audit every fixed policy seed."""
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    results = workspace / 'results'
    out = results / 'e65_study'
    if out.exists():
        raise FileExistsError(out)
    smoke = results / 'e65_native_smoke'
    pair = json.loads((smoke / 'summary.json').read_text())
    assert pair['pipeline_smoke_only'] and pair['checkpoint_replay_verified']
    assert pair['candidate_matched'] is None and not pair['test_evaluated']
    out.mkdir()
    folders = [HERE, *[ROOT / f'Lee/experiments/{name}' for name in ('e38', 'e42', 'e44', 'e55', 'e56', 'e63')],
               ROOT / 'Lee/code/darrow']
    hashes = {str(path): digest(path) for folder in folders for path in folder.glob('*.py')}
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               NUMBA_NUM_THREADS='1', PYTHONIOENCODING='utf-8', DARROW_FAST_DYNAMICS='1')
    manifest = {'source_sha256': hashes, 'policy_seeds': list(range(5)), 'test_evaluated': False,
                'base_training_steps': 401280, 'new_model_fitting': False,
                'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'smoke_summary_sha256': digest(smoke / 'summary.json')}
    write(out / 'study_manifest.json', manifest)
    completed = []
    start = time.monotonic()
    def phase(name, jobs, parallel=5):
        pending, active = jobs.copy(), []
        try:
            while pending or active:
                assert all(digest(Path(path)) == value for path, value in hashes.items()), 'Pinned source changed'
                while pending and len(active) < parallel:
                    label, command = pending.pop(0)
                    log = (out / f'{label}.log').open('w', encoding='utf-8')
                    process = subprocess.Popen([sys.executable, '-u', *map(str, command)], cwd=workspace,
                                               env=env, stdout=log, stderr=subprocess.STDOUT)
                    active.append((label, process, log))
                    print('START', label, process.pid, flush=True)
                for label, process, log in active.copy():
                    code = process.poll()
                    if code is None:
                        continue
                    active.remove((label, process, log))
                    log.close()
                    if code:
                        raise RuntimeError(f'{label} exited{code}; preserved log: {out / (label + ".log")}')
                    completed.append(label)
                    print('COMPLETE', label, flush=True)
                write(out / 'study_status.json', {'phase': name, 'completed': completed,
                     'running': [{'label': label, 'pid': process.pid} for label, process, _ in active],
                     'pending': [label for label, _ in pending], 'elapsed_seconds': time.monotonic() - start})
                if active:
                    time.sleep(1)
        except BaseException:
            for _, process, log in active:
                process.terminate()
                process.wait()
                log.close()
            write(out / 'failure.json', {'phase': name, 'completed': completed, 'terminal': True})
            raise
    common = ['--data', workspace / 'kaggle_data/darrow_15min.pkl',
              '--freeze', ROOT / 'Lee/experiments/e42/environment_freeze.json']
    def artifacts(seed):
        return ['--candidate', results / f'e55_study/policies/new_ppolag_fcsrl/seed_{seed}',
                '--baseline', results / f'e42_baselines/new_ppolag_fcsrl/seed_{seed}',
                '--correction', results / f'e56_study/models/h4/seed_{seed}']
    phase('smoke_audit', [('smoke_audit', [HERE / 'audit.py', *common, *artifacts(0),
          '--comparison', smoke, '--out', smoke / 'action_audit.json', '--null-check'])], 1)
    audit = json.loads((smoke / 'action_audit.json').read_text())
    assert audit['passed'] and audit['flat_risk_null_full_native_exact']
    assert audit['replays']['native']['action_rows'] == 152000
    phase('comparisons', [(f'compare_s{seed}', [HERE / 'compare.py', *common, *artifacts(seed),
          '--certificate', ROOT / 'Lee/experiments/e64/baseline_authentication.json',
          '--preflight', results / 'e65_preflight_v2/preflight.json', '--previous', results / 'e63_study',
          '--out', out / f'comparisons/seed_{seed}']) for seed in range(5)])
    phase('independent_audits', [(f'audit_s{seed}', [HERE / 'audit.py', *common, *artifacts(seed),
          '--comparison', out / f'comparisons/seed_{seed}',
          '--out', out / f'comparisons/seed_{seed}/action_audit.json']) for seed in range(5)])
    sys.path.insert(0, str(ROOT / 'Lee/experiments/e44'))
    from report import summarize
    pairs = [json.loads((out / f'comparisons/seed_{seed}/summary.json').read_text()) for seed in range(5)]
    write(out / 'comparison.json', summarize(pairs))
    write(out / 'study_status.json', {'phase': 'complete', 'completed': completed, 'running': [],
                                    'pending': [], 'elapsed_seconds': time.monotonic() - start})
    print('GRID_STUDY_COMPLETE', flush=True)


if __name__ == '__main__':
    main()

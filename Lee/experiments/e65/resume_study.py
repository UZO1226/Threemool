"""Recover a terminal file-lock interruption without changing any air query."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(path):
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


def recover(workspace, seed):
    previous = workspace / 'results/e65_study'
    original = previous / f'comparisons/seed_{seed}'
    output = workspace / f'results/e65_recovery/comparisons/seed_{seed}'
    assert not output.exists()
    manifest = json.loads((previous / 'study_manifest.json').read_text())
    assert json.loads((previous / 'failure.json').read_text())['terminal']
    assert all(digest(Path(path)) == value for path, value in manifest['source_sha256'].items())
    assert not (original / 'summary.json').exists()
    spec = importlib.util.spec_from_file_location('e65_recovery_comparison', HERE / 'compare.py')
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    baseline = json.loads((original / 'baseline_native.json').read_text())
    native = json.loads((original / 'candidate_native.json').read_text())
    text = (previous / f'compare_s{seed}.log').read_text()
    lines = re.findall(r'^AIR_QUERY (\d+) ([-+\deE.]+) ([-+\deE.]+)$', text, re.MULTILINE)
    cache = {0.: native}
    cache_sources = {}
    for index, (saved_seed, offset, volume) in enumerate(lines, 1):
        assert int(saved_seed) == seed
        path = original / f'query_{index:02d}.json'
        value = json.loads(path.read_text())
        assert value['summary']['air_m3'] == float(volume)
        cache[float(offset)] = value
        cache_sources[str(offset)] = {'path': str(path), 'sha256': digest(path)}
    score = compare.score_policy
    reused, fresh = [], []
    def cached_score(data, dynamics, policy, offset=0):
        if isinstance(policy, compare.TrustPolicy):
            reused.append({'kind': 'authenticated_baseline', 'offset': offset})
            return baseline['summary'], baseline['rows']
        if offset in cache:
            reused.append({'kind': 'previous_native_or_air_query', 'offset': offset})
            return cache[offset]['summary'], cache[offset]['rows']
        fresh.append(offset)
        return score(data, dynamics, policy, offset)
    arguments = [str(HERE / 'compare.py'), '--data', str(workspace / 'kaggle_data/darrow_15min.pkl'),
        '--freeze', str(ROOT / 'Lee/experiments/e42/environment_freeze.json'),
        '--candidate', str(workspace / f'results/e55_study/policies/new_ppolag_fcsrl/seed_{seed}'),
        '--baseline', str(workspace / f'results/e42_baselines/new_ppolag_fcsrl/seed_{seed}'),
        '--correction', str(workspace / f'results/e56_study/models/h4/seed_{seed}'),
        '--certificate', str(ROOT / 'Lee/experiments/e64/baseline_authentication.json'),
        '--preflight', str(workspace / 'results/e65_preflight_v2/preflight.json'),
        '--previous', str(workspace / 'results/e63_study'), '--out', str(output)]
    with patch.object(compare, 'score_policy', cached_score), patch.object(sys, 'argv', arguments):
        compare.main()
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['checkpoint_replay_verified']
    provenance = {'reason': 'Terminal Windows status-file replacement permission error, not a scientific failure.',
        'original_manifest_sha256': digest(previous / 'study_manifest.json'),
        'original_failure_sha256': digest(previous / 'failure.json'),
        'original_log_sha256': digest(previous / f'compare_s{seed}.log'),
        'original_baseline_sha256': digest(original / 'baseline_native.json'),
        'original_native_sha256': digest(original / 'candidate_native.json'),
        'previous_query_sources': cache_sources, 'reused_calls': reused, 'fresh_offsets': fresh,
        'native_serialization_replay': 'Reused original native file, which original pinned compare.py writes only after two exact independently reloaded trajectories.',
        'new_independent_full_trajectory_audits_required': True, 'source_sha256': digest(Path(__file__))}
    write(output / 'recovery_provenance.json', provenance)
    summary['recovery_provenance_sha256'] = digest(output / 'recovery_provenance.json')
    write(output / 'summary.json', summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--recover-seed', type=int, choices=(2, 4))
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if args.recover_seed is not None:
        recover(workspace, args.recover_seed)
        return
    old = workspace / 'results/e65_study'
    out = workspace / 'results/e65_recovery'
    if out.exists():
        raise FileExistsError(out)
    assert json.loads((old / 'failure.json').read_text())['terminal']
    old_manifest = json.loads((old / 'study_manifest.json').read_text())
    assert all(digest(Path(path)) == value for path, value in old_manifest['source_sha256'].items())
    (out / 'comparisons').mkdir(parents=True)
    for seed in (0, 1, 3):
        source = old / f'comparisons/seed_{seed}'
        pair = json.loads((source / 'summary.json').read_text())
        assert pair['candidate_matched'] is not None and pair['checkpoint_replay_verified']
        shutil.copytree(source, out / f'comparisons/seed_{seed}')
    hashes = dict(old_manifest['source_sha256'])
    hashes[str(Path(__file__).resolve())] = digest(Path(__file__))
    write(out / 'study_manifest.json', {'source_sha256': hashes, 'test_evaluated': False,
        'policy_seeds': list(range(5)), 'previous_terminal_manifest_sha256': digest(old / 'study_manifest.json'),
        'scope': 'Identical policy, environment, target air and matcher; recover completed queries and audit every trajectory.'})
    environment = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        NUMBA_NUM_THREADS='1', DARROW_FAST_DYNAMICS='1', PYTHONIOENCODING='utf-8')
    completed = []
    def phase(name, jobs):
        active = []
        try:
            for label, command in jobs:
                log = (out / f'{label}.log').open('w', encoding='utf-8')
                process = subprocess.Popen([sys.executable, '-u', *map(str, command)], cwd=workspace,
                                            env=environment, stdout=log, stderr=subprocess.STDOUT)
                active.append((label, process, log))
                print('START', label, process.pid, flush=True)
            while active:
                assert all(digest(Path(path)) == value for path, value in hashes.items())
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
                    'running': [{'label': label, 'pid': process.pid} for label, process, _ in active], 'pending': []})
                if active:
                    time.sleep(1)
        except BaseException:
            for _, process, log in active:
                process.terminate()
                process.wait()
                log.close()
            write(out / 'failure.json', {'phase': name, 'terminal': True, 'completed': completed})
            raise
    phase('recover_comparisons', [(f'compare_s{seed}', [Path(__file__), '--workspace', workspace,
          '--recover-seed', seed]) for seed in (2, 4)])
    phase('independent_audits', [(f'audit_s{seed}', [HERE / 'audit.py',
          '--data', workspace / 'kaggle_data/darrow_15min.pkl',
          '--freeze', ROOT / 'Lee/experiments/e42/environment_freeze.json',
          '--candidate', workspace / f'results/e55_study/policies/new_ppolag_fcsrl/seed_{seed}',
          '--baseline', workspace / f'results/e42_baselines/new_ppolag_fcsrl/seed_{seed}',
          '--correction', workspace / f'results/e56_study/models/h4/seed_{seed}',
          '--comparison', out / f'comparisons/seed_{seed}',
          '--out', out / f'comparisons/seed_{seed}/action_audit.json']) for seed in range(5)])
    sys.path.insert(0, str(ROOT / 'Lee/experiments/e44'))
    from report import summarize
    pairs = [json.loads((out / f'comparisons/seed_{seed}/summary.json').read_text()) for seed in range(5)]
    write(out / 'comparison.json', summarize(pairs))
    write(out / 'study_status.json', {'phase': 'complete', 'completed': completed, 'running': [], 'pending': []})
    print('RECOVERED_GRID_STUDY_COMPLETE', flush=True)


if __name__ == '__main__':
    main()

"""Preserve completed fixed-filter comparisons and unchanged base weights only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e44'))
from report import summarize


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('results', 'study', 'policies', 'freeze', 'out', 'index'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.index.exists():
        raise FileExistsError('Refusing to overwrite archive or index')
    assert json.loads((args.study / 'study_status.json').read_text())['phase'] == 'complete'
    results = args.results.resolve()
    files, pairs, included = {}, [], []
    for seed in range(5):
        comparison = args.study / f'comparisons/seed_{seed}'
        pair = json.loads((comparison / 'summary.json').read_text())
        audit = json.loads((comparison / 'action_audit.json').read_text())
        assert pair['policy_seed'] == audit['policy_seed'] == seed
        assert pair['candidate_method'] == 'ppolag_fcsrl_air_filter' and pair['filter_gain'] == .5
        assert pair['training_steps'] == 401280 and not pair['smoke_only'] and not pair['test_evaluated']
        assert pair['checkpoint_replay_verified'] and all(pair['checks'].values())
        assert audit['passed'] and audit['native_and_matched_replayed'] and not audit['smoke_only']
        assert pair['extra_environment_training_steps'] == 0
        assert sha(comparison / 'filter_config.json') == pair['filter_config_sha256']
        base = args.policies / f'seed_{seed}'
        assert pair['candidate_checkpoint_hashes'] == pair['baseline_checkpoint_hashes']
        for name, digest in pair['candidate_checkpoint_hashes'].items():
            path = base / 'model_final' / name
            assert sha(path) == digest
            files[path.resolve().relative_to(results).as_posix()] = path
        for name in ('effective_config.json', 'metrics.json', 'progress.json', 'train_log.json', 'base_equivalence.json'):
            path = base / name
            files[path.resolve().relative_to(results).as_posix()] = path
        included.append({'policy_seed': seed, 'filter_gain': .5, 'base_policy_steps': 401280,
                         'new_policy_training': False, 'filter_config_sha256': pair['filter_config_sha256']})
        pairs.append(pair)
    report = ROOT / 'Lee/experiments/e57/comparison.json'
    assert json.loads(report.read_text()) == summarize(pairs)
    for path in args.study.rglob('*'):
        if path.is_file() and path.suffix in ('.json', '.log'):
            assert 'transitions' not in path.relative_to(args.study).parts
            files[path.resolve().relative_to(results).as_posix()] = path
    for name in ('manifest.json', 'status.json'):
        path = args.policies.parent / name
        files[path.resolve().relative_to(results).as_posix()] = path
    prefix = args.study.resolve().relative_to(results).as_posix()
    files[f'{prefix}/final_reports/comparison.json'] = report
    files['metadata/environment_freeze.json'] = args.freeze
    files['metadata/FCSRL_LICENSE'] = ROOT / 'Lee/code/darrow/FCSRL_LICENSE'
    assert not any(name.endswith(('.npz', '.npy', '.csv')) or '/transitions/' in name for name in files)
    manifest = {'new_policy_training': False, 'source_data_included': False,
                'residual_banks_included': False, 'pit_banks_included': False,
                'captured_training_arrays_included': False, 'included_controllers': included,
                'scope': 'All five completed fixed-filter configurations, generated evaluations/queries/audits/logs, and unchanged final base checkpoints. No source observations or captured arrays.',
                'files': {name: {'bytes': path.stat().st_size, 'sha256': sha(path)}
                          for name, path in sorted(files.items())}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
        archive.writestr('artifact_manifest.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(args.out) as archive:
        assert archive.testzip() is None
        for name, metadata in manifest['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == metadata['sha256']
    manifest['archive'] = {'name': args.out.name, 'bytes': args.out.stat().st_size, 'sha256': sha(args.out)}
    args.index.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest['archive']), flush=True)


if __name__ == '__main__':
    main()

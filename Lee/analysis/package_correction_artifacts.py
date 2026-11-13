"""Archive completed e55 controllers and generated reports, excluding captures."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('study', 'freeze', 'out', 'index'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.index.exists():
        raise FileExistsError('Refusing to overwrite an existing artifact or index')
    status = json.loads((args.study / 'study_status.json').read_text())
    assert status['phase'] == 'complete'
    files = {}
    runs = []
    for seed in range(5):
        run = args.study / 'policies/new_ppolag_fcsrl' / f'seed_{seed}'
        config = json.loads((run / 'effective_config.json').read_text())
        metrics = json.loads((run / 'metrics.json').read_text())
        equivalent = json.loads((run / 'base_equivalence.json').read_text())
        fit = json.loads((run / 'correction/fit_summary.json').read_text())
        audit = json.loads((run / 'correction/audit.json').read_text())
        pair = json.loads((args.study / 'comparisons' / f'seed_{seed}/summary.json').read_text())
        actions = json.loads((args.study / 'comparisons' / f'seed_{seed}/action_audit.json').read_text())
        assert config['seed'] == fit['policy_seed'] == pair['policy_seed'] == seed
        assert metrics['steps'] == fit['base_training_steps'] == pair['training_steps'] == 401280
        assert not config['experiment_metadata']['test_evaluated']
        assert equivalent['passed'] and fit['checks']['passed'] and audit['passed'] and actions['passed']
        assert not fit['smoke_only'] and not pair['smoke_only'] and not pair['test_evaluated']
        assert fit['extra_environment_steps'] == 0 and fit['model_sha256'] == sha(run / 'correction/risk_models.pkl')
        for path in run.rglob('*'):
            if not path.is_file() or 'transitions' in path.relative_to(run).parts:
                continue
            if path.suffix in ('.pt', '.json') or path == run / 'correction/risk_models.pkl':
                files[path.relative_to(args.study).as_posix()] = path
        assert f'policies/new_ppolag_fcsrl/seed_{seed}/correction/risk_models.pkl' in files
        runs.append({'policy_seed': seed, 'training_steps': 401280, 'extra_environment_steps': 0,
                     'risk_models_sha256': fit['model_sha256'],
                     'base_checkpoint_hashes': equivalent['checkpoint_hashes']})
    for path in args.study.rglob('*'):
        if not path.is_file() or path.suffix not in ('.json', '.log'):
            continue
        if 'transitions' in path.relative_to(args.study).parts:
            continue
        files[path.relative_to(args.study).as_posix()] = path
    files['metadata/environment_freeze.json'] = args.freeze
    files['metadata/FCSRL_LICENSE'] = Path(__file__).resolve().parents[1] / 'code/darrow/FCSRL_LICENSE'
    assert not any(name.endswith('.npz') or '/transitions/' in name for name in files)
    manifest = {'new_policy_training': True, 'included_completed_runs': runs,
                'source_data_included': False, 'captured_training_arrays_included': False,
                'residual_banks_included': False, 'pit_banks_included': False,
                'scope': 'All five unchanged-base checkpoints, fitted correction models, full logs and generated comparisons. No source plant data or captured training arrays.',
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

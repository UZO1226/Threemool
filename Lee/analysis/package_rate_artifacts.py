"""Archive completed rate controllers and necessary base weights without captures."""
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
    for name in ('results', 'study', 'captures', 'freeze', 'out', 'index'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.index.exists():
        raise FileExistsError('Refusing to overwrite artifact or index')
    assert json.loads((args.study / 'study_status.json').read_text())['phase'] == 'complete'
    results = args.results.resolve()
    files = {}
    included = []
    for horizon in (1, 4):
        pairs = []
        for seed in range(5):
            model = args.study / f'models/h{horizon}/seed_{seed}'
            comparison = args.study / f'comparisons/h{horizon}/seed_{seed}'
            fit = json.loads((model / 'fit_summary.json').read_text())
            held = json.loads((model / 'held_audit.json').read_text())
            pair = json.loads((comparison / 'summary.json').read_text())
            actions = json.loads((comparison / 'action_audit.json').read_text())
            assert fit['policy_seed'] == pair['policy_seed'] == seed
            assert fit['horizon'] == pair['correction_horizon'] == horizon
            assert fit['base_training_steps'] == pair['training_steps'] == 401280
            assert fit['fit_transitions'] == 316160 and fit['held_transitions'] == 85120
            assert not fit['smoke_only'] and not pair['smoke_only']
            assert not fit['test_evaluated'] and not pair['test_evaluated']
            assert fit['checks']['passed'] and held['passed'] and actions['passed']
            assert fit['extra_environment_training_steps'] == 0
            assert sha(model / 'risk_models.pkl') == fit['model_sha256']
            base = args.captures / f'seed_{seed}'
            for name, digest in fit['base_checkpoint_hashes'].items():
                path = base / 'model_final' / name
                assert sha(path) == digest
                files[path.resolve().relative_to(results).as_posix()] = path
            assert fit['base_checkpoint_hashes'] == pair['baseline_checkpoint_hashes'] == pair['candidate_checkpoint_hashes']
            for name in ('effective_config.json', 'metrics.json', 'progress.json', 'train_log.json',
                         'base_equivalence.json', 'correction/fit_summary.json'):
                path = base / name
                files[path.resolve().relative_to(results).as_posix()] = path
            pairs.append(pair)
            included.append({'policy_seed': seed, 'horizon': horizon,
                             'base_policy_steps': 401280, 'extra_environment_training_steps': 0,
                             'risk_models_sha256': fit['model_sha256']})
        generated = summarize(pairs)
        report = ROOT / f'Lee/experiments/e56/h{horizon}_comparison.json'
        assert json.loads(report.read_text()) == generated
        prefix = args.study.resolve().relative_to(results).as_posix()
        files[f'{prefix}/final_reports/h{horizon}_comparison.json'] = report
    for path in args.study.rglob('*'):
        if not path.is_file():
            continue
        if path.suffix in ('.json', '.log') or path.name == 'risk_models.pkl':
            assert 'transitions' not in path.relative_to(args.study).parts
            files[path.resolve().relative_to(results).as_posix()] = path
    for name in ('manifest.json', 'status.json'):
        path = args.captures.parent / name
        files[path.resolve().relative_to(results).as_posix()] = path
    prefix = args.study.resolve().relative_to(results).as_posix()
    files[f'{prefix}/final_reports/horizon_contrast.json'] = ROOT / 'Lee/experiments/e56/horizon_contrast.json'
    files['metadata/environment_freeze.json'] = args.freeze
    files['metadata/FCSRL_LICENSE'] = ROOT / 'Lee/code/darrow/FCSRL_LICENSE'
    assert not any(name.endswith(('.npz', '.npy', '.csv')) or '/transitions/' in name for name in files)
    manifest = {'new_policy_training': False, 'source_data_included': False,
                'captured_training_arrays_included': False, 'residual_banks_included': False,
                'pit_banks_included': False, 'policy_seeds': list(range(5)),
                'primary_horizon': 4, 'control_horizon': 1, 'included_controllers': included,
                'scope': 'Both full regression horizons, all fitted models/logs/generated evaluations, and the five unchanged final base checkpoints. Captures and original source data are excluded.',
                'files': {name: {'bytes': path.stat().st_size, 'sha256': sha(path)}
                          for name, path in sorted(files.items())}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
        archive.writestr('artifact_manifest.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(args.out) as archive:
        assert archive.testzip() is None
        for name, entry in manifest['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == entry['sha256']
    manifest['archive'] = {'name': args.out.name, 'bytes': args.out.stat().st_size, 'sha256': sha(args.out)}
    args.index.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest['archive']), flush=True)


if __name__ == '__main__':
    main()

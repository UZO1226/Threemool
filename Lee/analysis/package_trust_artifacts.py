"""Preserve a completed five-seed risk-only study without source observations."""
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
    for name in ('results', 'study', 'policies', 'previous', 'smoke', 'freeze', 'out', 'index'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.index.exists():
        raise FileExistsError('Refusing to overwrite archive or index')
    assert json.loads((args.study / 'study_status.json').read_text())['phase'] == 'complete'
    results = args.results.resolve()
    files, pairs, included = {}, [], []

    def add(path):
        assert path.is_file()
        relative = path.resolve().relative_to(results).as_posix()
        assert '/transitions/' not in relative and path.suffix not in ('.npz', '.npy', '.csv')
        files[relative] = path

    for seed in range(5):
        comparison = args.study / f'comparisons/seed_{seed}'
        pair = json.loads((comparison / 'summary.json').read_text())
        audit = json.loads((comparison / 'action_audit.json').read_text())
        model = args.previous / f'models/h4/seed_{seed}'
        fit = json.loads((model / 'fit_summary.json').read_text())
        previous_pair = args.previous / f'comparisons/h4/seed_{seed}/summary.json'
        old = json.loads(previous_pair.read_text())
        assert pair['policy_seed'] == audit['policy_seed'] == fit['policy_seed'] == seed
        assert pair['candidate_method'] == 'ppolag_fcsrl_risk_only_h4'
        assert pair['penalty'] == 0 and pair['correction_horizon'] == fit['horizon'] == 4
        assert pair['training_steps'] == fit['base_training_steps'] == 401280
        assert not pair['smoke_only'] and not pair['test_evaluated'] and not pair['pipeline_smoke_only']
        assert pair['checkpoint_replay_verified'] and all(pair['checks'].values())
        assert audit['passed'] and not audit['test_evaluated']
        assert audit['comparison_summary_sha256'] == sha(comparison / 'summary.json')
        assert set(audit['replays']) == {'native', 'matched'}
        for replay in audit['replays'].values():
            assert replay['action_rows'] == 152000
            assert replay['independent_sklearn_actions_exact'] and replay['trajectory_replay_exact']
            assert replay['executed_air_accounting_exact']
        assert pair['extra_environment_training_steps'] == 0 and not pair['new_model_fitting']
        assert sha(comparison / 'controller_config.json') == pair['controller_config_sha256']
        assert sha(model / 'risk_models.pkl') == fit['model_sha256'] == pair['correction_model_sha256']
        for key in ('candidate_checkpoint_hashes', 'baseline_checkpoint_hashes', 'correction_model_sha256',
                    'freeze_canonical_sha256', 'baseline_native'):
            assert pair[key] == old[key]
        assert pair['candidate_checkpoint_hashes'] == pair['baseline_checkpoint_hashes'] == fit['base_checkpoint_hashes']
        base = args.policies / f'seed_{seed}'
        for name, expected in pair['candidate_checkpoint_hashes'].items():
            path = base / 'model_final' / name
            assert sha(path) == expected
            add(path)
        for name in ('effective_config.json', 'metrics.json', 'progress.json', 'train_log.json',
                     'base_equivalence.json', 'correction/fit_summary.json'):
            add(base / name)
        for path in model.iterdir():
            if path.is_file() and (path.suffix == '.json' or path.name == 'risk_models.pkl'):
                add(path)
        add(previous_pair)
        add(previous_pair.parent / 'action_audit.json')
        pairs.append(pair)
        included.append({'policy_seed': seed, 'horizon': 4, 'penalty': 0, 'base_policy_steps': 401280,
                         'new_policy_training': False, 'model_sha256': fit['model_sha256'],
                         'controller_config_sha256': pair['controller_config_sha256']})
    assert json.loads((args.study / 'final_reports/comparison.json').read_text()) == summarize(pairs)
    secondary = json.loads((args.study / 'final_reports/penalty_contrast.json').read_text())
    assert secondary['same_base_rate_models_and_reference_air_verified'] and secondary['policy_seeds'] == list(range(5))
    for folder in (args.study, args.smoke):
        for path in folder.rglob('*'):
            if path.is_file() and path.suffix in ('.json', '.log'):
                add(path)
    for name in ('manifest.json', 'status.json'):
        add(args.policies.parent / name)
    for name in ('e63_smoke_compare.log', 'e63_smoke_audit.log', 'e63_study_run.log'):
        add(args.results / name)
    files['metadata/environment_freeze.json'] = args.freeze
    files['metadata/FCSRL_LICENSE'] = ROOT / 'Lee/code/darrow/FCSRL_LICENSE'
    manifest = {'new_policy_training': False, 'new_model_fitting': False, 'source_data_included': False,
                'residual_banks_included': False, 'pit_banks_included': False,
                'captured_training_arrays_included': False, 'included_controllers': included,
                'scope': 'Complete five-seed comparisons, queries, native/matched audits, fixed configs, necessary unchanged base/rate weights and provenance. No observation arrays.',
                'files': {name: {'bytes': path.stat().st_size, 'sha256': sha(path)} for name, path in sorted(files.items())}}
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

"""Preserve all authenticated actual fresh baselines without captured observations."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    for name in ('workspace', 'out', 'index'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists() and not args.index.exists(), 'Existing output is immutable'
    workspace = args.workspace.resolve()
    root = workspace / 'Threemool'
    results = workspace / 'results'
    seeds = list(range(5, 10))
    training = results / 'e68_fresh_baselines'
    rates = results / 'e68_fresh_rates'
    references = results / 'e68_fresh_references'
    cert = read(references / 'baseline_authentication.json')
    train_summary = read(training / 'baseline_training_summary.json')
    rate_summary = read(rates / 'rate_preparation_summary.json')
    assert cert['authentication_passed'] and cert['all_original_fcsrl_air_matches']
    assert train_summary['passed'] and rate_summary['passed']
    files = {}

    def add(path):
        relative = path.resolve().relative_to(workspace).as_posix()
        assert path.is_file() and path.suffix in ('.json', '.log', '.pkl', '.pt')
        assert '/transitions/' not in relative and 'pit_bank' not in relative
        assert 'residual_bank' not in relative
        files[relative] = path

    for folder in (training, rates, references):
        status = read(folder / 'study_status.json')
        assert status['phase'] == 'complete' and not status['running'] and not status['pending']
        manifest = read(folder / 'study_manifest.json')
        assert manifest['policy_seeds'] == seeds and not manifest['test_evaluated']
        for path, expected in manifest['source_sha256'].items():
            assert sha(Path(path)) == expected, path
        for path in folder.glob('*'):
            if path.is_file() and path.suffix in ('.json', '.log'):
                add(path)
    for summary in (cert, train_summary, rate_summary):
        assert summary['policy_seeds'] == seeds and not summary['test_evaluated']
        assert [row['policy_seed'] for row in summary['records']] == seeds
    for seed in seeds:
        record = next(row for row in cert['records'] if row['policy_seed'] == seed)
        base = training / f'captured/new_ppolag_fcsrl/seed_{seed}'
        original = training / f'original/new_ppolag_fcsrl/seed_{seed}'
        parity = read(base / 'base_equivalence.json')
        assert parity['passed'] and parity['policy_seed'] == seed and parity['steps'] == 401280
        assert parity['all_final_checkpoint_files_exact'] and parity['source_config_and_budget_guards']
        assert parity['stored_native_metrics_exact'] and not parity['test_evaluated']
        train_record = next(row for row in train_summary['records'] if row['policy_seed'] == seed)
        assert sha(base / 'base_equivalence.json') == train_record['base_equivalence_sha256']
        assert parity['checkpoint_hashes'] == record['base_checkpoint_hashes'] == train_record['checkpoint_hashes']
        for folder in (base, original):
            config = read(folder / 'effective_config.json')
            assert config['seed'] == seed and config['actual_steps'] == 401280
            for name, expected in record['base_checkpoint_hashes'].items():
                path = folder / 'model_final' / name
                assert sha(path) == expected
                add(path)
            for name in ('effective_config.json', 'metrics.json', 'progress.json', 'train_log.json'):
                add(folder / name)
        add(base / 'base_equivalence.json')
        for path in (base / 'correction').rglob('*'):
            if path.is_file() and path.suffix in ('.json', '.pkl'):
                add(path)
        model = rates / f'models/h4/seed_{seed}'
        fit = read(model / 'fit_summary.json')
        assert fit['policy_seed'] == seed and fit['base_training_steps'] == 401280
        assert fit['checks']['passed'] and not fit['test_evaluated'] and not fit['smoke_only']
        assert fit['base_checkpoint_hashes'] == record['base_checkpoint_hashes']
        assert sha(model / 'risk_models.pkl') == fit['model_sha256'] == record['rate_model_sha256']
        add(model / 'risk_models.pkl')
        add(model / 'fit_summary.json')
        folder = references / f'seed_{seed}'
        pair, audit = read(folder / 'summary.json'), read(folder / 'action_audit.json')
        assert audit['passed'] and audit['policy_seed'] == pair['policy_seed'] == seed
        assert not audit['test_evaluated'] and not pair['test_evaluated']
        assert sha(folder / 'summary.json') == audit['summary_sha256'] == record['summary_sha256']
        assert sha(folder / 'action_audit.json') == record['action_audit_sha256']
        assert sha(folder / 'controller_config.json') == record['controller_config_sha256']
        assert sha(folder / 'e63_matched.json') == record['matched_file_sha256']
        assert pair['e63_matched'] == record['primary_reference_metrics']
        assert pair['matching']['offset'] == record['fixed_e63_offset_after_wrapper']
        assert pair['matching']['matched_within_one_percent']
        assert set(audit['replays']) == {'native', 'matched'}
        for replay in audit['replays'].values():
            assert replay['action_rows'] == 152000 and replay['trajectory_replay_exact']
            assert replay['independent_sklearn_actions_exact'] and replay['executed_air_accounting_exact']
        for path in folder.rglob('*'):
            if path.is_file() and path.suffix in ('.json', '.log'):
                add(path)
    add(root / 'Lee/experiments/e42/environment_freeze.json')
    files['metadata/FCSRL_LICENSE'] = root / 'Lee/code/darrow/FCSRL_LICENSE'
    manifest = {'trial': 'e68', 'policy_seeds': seeds, 'source_data_included': False,
        'residual_banks_included': False, 'pit_banks_included': False,
        'captured_training_arrays_included': False, 'test_evaluated': False,
        'candidate_evaluation': False, 'actual_steps_per_policy': 401280,
        'scope': 'Five actual policy seeds, each with exact original/captured training parity. Final NN and h4 weights, preparation models, complete logs and authenticated matched e63 references. Twins do not double seed count. Frozen world/data restored separately.',
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
    args.index.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(manifest['archive']), flush=True)


if __name__ == '__main__':
    main()

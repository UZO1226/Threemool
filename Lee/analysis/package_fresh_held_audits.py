"""Archive independent held-stream checks separately from candidate outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    for name in ('workspace', 'archive', 'index'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.archive.exists() and not args.index.exists()
    workspace = args.workspace.resolve()
    root = workspace / 'Threemool'
    results = workspace / 'results'
    certificate_path = root / 'Lee/experiments/e68/baseline_authentication.json'
    certificate = read(certificate_path)
    assert certificate['authentication_passed'] and certificate['policy_seeds'] == list(range(5, 10))
    source_manifest = read(results / 'e68_fresh_rates/study_manifest.json')
    for name, expected in source_manifest['source_sha256'].items():
        assert sha(Path(name)) == expected
    files = {certificate_path.relative_to(workspace).as_posix(): certificate_path}
    rows = []
    for seed in range(5, 10):
        path = results / f'e68_fresh_held_audits/seed_{seed}.json'
        audit = read(path)
        assert audit['passed'] and audit['policy_seed'] == seed and audit['horizon'] == 4
        assert audit['phase'] == 'held' and not audit['test_evaluated'] and not audit['smoke_only']
        held = audit['held']
        assert held['held_metrics_reconstructed'] and held['held_rows'] == 85120
        model_folder = results / f'e68_fresh_rates/models/h4/seed_{seed}'
        fit_path = model_folder / 'fit_summary.json'
        fit = read(fit_path)
        reference = next(record for record in certificate['records'] if record['policy_seed'] == seed)
        assert sha(model_folder / 'risk_models.pkl') == fit['model_sha256'] == reference['rate_model_sha256']
        assert fit['fit_updates'] == list(range(1, 53)) and fit['held_updates'] == list(range(53, 67))
        for label in ('nh4', 'do'):
            actual = held['labels'][label]
            stored = fit['label_diagnostics'][label]
            for field, stored_field in (('rate_mse', 'mse'), ('target_mean', 'target_mean'), ('prediction_mean', 'prediction_mean')):
                np.testing.assert_allclose(actual[field], stored['held'][stored_field], atol=1e-12, rtol=0)
            for control in ('current_state_persistence', 'fit_mean_constant'):
                np.testing.assert_allclose(actual['controls'][control]['rate_mse_difference'],
                    stored['held']['mse'] - stored[control]['mse'], atol=1e-12, rtol=0)
            rows.append({'policy_seed': seed, 'label': label, 'held_training_dates': held['held_training_dates'], **actual})
        for item in (path, fit_path):
            files[item.relative_to(workspace).as_posix()] = item
    baseline_index = root / 'Lee/experiments/e68/artifacts.json'
    files[baseline_index.relative_to(workspace).as_posix()] = baseline_index
    manifest = {'trial': 'e68_fresh_held', 'policy_seeds': list(range(5, 10)), 'passed': True,
        'source_data_included': False, 'captured_training_arrays_included': False,
        'residual_banks_included': False, 'pit_banks_included': False,
        'test_evaluated': False, 'new_environment_queries': 0, 'new_model_fitting': False,
        'audit_sources_sha256': source_manifest['source_sha256'],
        'scope': 'Independent original-sklearn held logged-action errors and date-conditional contrasts. Held updates are excluded from rate fitting, but not from base-policy training. This is not counterfactual candidate quality or fresh observational confirmation.',
        'rows': rows, 'files': {name: {'bytes': path.stat().st_size, 'sha256': sha(path)} for name, path in sorted(files.items())}}
    with zipfile.ZipFile(args.archive, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
        archive.writestr('artifact_manifest.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(args.archive) as archive:
        assert archive.testzip() is None
        for name, item in manifest['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == item['sha256']
    manifest['archive'] = {'name': args.archive.name, 'bytes': args.archive.stat().st_size, 'sha256': sha(args.archive)}
    args.index.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(manifest['archive']), flush=True)


if __name__ == '__main__':
    main()

"""Summarize full e55 parity and held-stream diagnostics without candidate selection."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    for name in ('study', 'baselines', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    manifest = read(args.study / 'study_manifest.json')
    assert manifest['policy_seeds'] == list(range(5))
    assert manifest['actual_steps_per_policy'] == 401280 and manifest['extra_environment_training_steps'] == 0
    assert not manifest['test_evaluated']
    root = Path(__file__).resolve().parents[2]
    for name, expected in manifest['source_sha256'].items():
        normalized = name.replace('\\', '/')
        path = root / normalized.split('/Threemool/', 1)[1] if '/Threemool/' in normalized else Path(name)
        raw = path.read_bytes()
        lf = raw.replace(b'\r\n', b'\n')
        assert expected in {hashlib.sha256(value).hexdigest() for value in (raw, lf, lf.replace(b'\n', b'\r\n'))}
    rows = []
    for seed in range(5):
        run = args.study / 'policies/new_ppolag_fcsrl' / f'seed_{seed}'
        reference = args.baselines / f'seed_{seed}'
        parity = read(run / 'base_equivalence.json')
        fit = read(run / 'correction/fit_summary.json')
        audit = read(run / 'correction/audit.json')
        assert parity['passed'] and parity['all_final_checkpoint_files_exact']
        assert parity['policy_seed'] == fit['policy_seed'] == audit['policy_seed'] == seed
        assert not parity['smoke_only'] and not fit['smoke_only'] and not audit['smoke_only']
        assert fit['checks']['passed'] and audit['passed']
        assert fit['fit_transitions'] == 316160 and fit['held_transitions'] == 85120
        assert fit['extra_environment_steps'] == 0 and not fit['test_evaluated']
        assert len(fit['capture_file_sha256']) == 66
        assert read(run / 'train_log.json') == read(reference / 'train_log.json')
        assert len(read(run / 'train_log.json')) == 66
        assert read(run / 'metrics.json')['val_final'] == read(reference / 'metrics.json')['val_final']
        for name, digest in parity['checkpoint_hashes'].items():
            assert hashlib.sha256((run / 'model_final' / name).read_bytes()).hexdigest() == digest
            assert hashlib.sha256((reference / 'model_final' / name).read_bytes()).hexdigest() == digest
        assert hashlib.sha256((run / 'correction/risk_models.pkl').read_bytes()).hexdigest() == fit['model_sha256']
        rows.append({'policy_seed': seed, 'training_steps': 401280,
                     'full66_training_records_and5_native_evaluations_exact': True,
                     'checkpoint_files_exact': True, 'checkpoint_hashes': parity['checkpoint_hashes'],
                     'fit_transitions': 316160, 'held_transitions': 85120,
                     'extra_environment_training_steps': 0, 'capture_updates': 66,
                     'inference_checks': fit['checks'], 'held_diagnostics': audit['labels'],
                     'held_training_dates': audit['held_training_dates'],
                     'risk_models_sha256': fit['model_sha256']})
    result = {'passed': True, 'policy_seeds': list(range(5)), 'classifier_seeds': [5500, 5501],
              'training_commit': manifest['code_commit'], 'runtime_sources_verified': True,
              'summary_generator_sha256': hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
              'test_evaluated': False, 'smoke_only': False, 'runtime': manifest['runtime'],
              'policy_training_transitions_total': 5 * 401280,
              'additional_environment_training_transitions': 0,
              'correction_hyperparameters_selected_by_validation': False,
              'seeds': rows,
              'scope': 'Complete unchanged-base training parity and held on-policy classifier diagnostics. Corrected-controller matched-air performance remains a separate five-seed comparison; high predictive AP is not evidence of control improvement.'}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('FULL_CAPTURE_TRAINING_PARITY_PASS', len(rows), flush=True)


if __name__ == '__main__':
    main()

# Adapted from e75/audit_forecast.py at e077ba3, independently adding current maxDO to raw changes.
"""Reconstruct raw predictions, calibration and admission without refitting."""
import argparse
import json
from pathlib import Path
import pickle
import subprocess

import numpy as np

from residual_forecast import ROOT, read_blocks, sha
from rate_model import FastRate
from rate_io import load_base
from policy_io import verify_sources


def main():
    parser = argparse.ArgumentParser()
    for name in ('workspace', 'study', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    workspace = args.workspace.resolve()
    data = workspace / 'kaggle_data/darrow_15min.pkl'
    freeze = ROOT / 'Lee/experiments/e42/environment_freeze.json'
    cert = json.loads((ROOT / 'Lee/experiments/e64/baseline_authentication.json').read_text())
    summary = json.loads((args.study / 'summary.json').read_text())
    assert summary['completed'] and summary['computation_passed'] and summary['policy_seeds'] == list(range(5))
    verify_sources(summary['source_sha256'], ROOT)
    records = []
    for seed in range(5):
        folder = args.study / f'seed_{seed}'
        stored = json.loads((folder / 'fit_summary.json').read_text())
        assert stored == summary['records'][seed] and stored['policy_seed'] == seed
        assert not stored['test_evaluated'] and not stored['controller_evaluation']
        run = workspace / f'results/e55_study/policies/new_ppolag_fcsrl/seed_{seed}'
        reference = workspace / f'results/e42_baselines/new_ppolag_fcsrl/seed_{seed}'
        base, _, _, hashes = load_base(run, freeze, reference)
        del base
        assert hashes == stored['base_checkpoint_hashes'] == cert['records'][seed]['base_checkpoint_hashes']
        blocks, provenance = read_blocks(run, data)
        assert provenance == stored['capture_provenance']
        assert sha(folder / 'do_model.pkl') == stored['model_sha256']
        with (folder / 'do_model.pkl').open('rb') as file:
            model = pickle.load(file)
        fast = FastRate([model])
        predictions, raw_error = {}, 0.
        for name, block in blocks.items():
            original = model.predict(block['x'])
            compiled = fast.raw(block['x'])[:, 0]
            raw_error = max(raw_error, float(np.max(np.abs(original - compiled))))
            assert raw_error <= 1e-12
            predictions[name] = np.maximum(block['x'][:, 2:5].max(-1) + original, 0)
        cal = blocks['calibration']
        adjustment = float(max(0., np.quantile(cal['y'] - predictions['calibration'], .95, method='higher')))
        assert adjustment == stored['upper_margin']
        y, x = blocks['held']['y'], blocks['held']['x']
        point, current = predictions['held'], x[:, 2:5].max(-1)
        upper = point + adjustment
        persistence_margin = float(max(0., np.quantile(cal['y'] - cal['x'][:, 2:5].max(-1), .95, method='higher')))
        assert persistence_margin == stored['held']['persistence_margin']
        probes = blocks['held']['x'][np.linspace(0, len(x) - 1, 10000, dtype=int)]
        low, high = probes.copy(), probes.copy()
        low[:, -1], high[:, -1] = -1, 1
        violation = float(np.maximum(model.predict(low) - model.predict(high), 0).max())
        assert violation <= 1e-12
        event, alarm = y > 2, upper > 2
        counts = {'n': len(y), 'next_do_events': int(event.sum()),
            'false_negatives': int(np.count_nonzero(event & ~alarm)),
            'false_positives': int(np.count_nonzero(~event & alarm)),
            'onset_events': int(np.count_nonzero((current <= 2) & event)),
            'onset_false_negatives': int(np.count_nonzero((current <= 2) & event & ~alarm)),
            'onset_true_positives': int(np.count_nonzero((current <= 2) & event & alarm))}
        metrics = {'mse': float(np.mean((point - y) ** 2)), 'mae': float(np.mean(np.abs(point - y))),
            'mean_bias': float(np.mean(point - y)), 'target_mean': float(y.mean()), 'prediction_mean': float(point.mean()),
            'upper_coverage': float(np.mean(y <= upper)),
            'upper_pinball95': float(np.mean(np.where(y > upper, .95 * (y - upper), .05 * (upper - y))))}
        for key, value in {**counts, **metrics}.items():
            np.testing.assert_allclose(value, stored['held']['forecasts']['model'][key], atol=1e-12, rtol=0)
        days = np.unique(blocks['held']['day'])
        groups = [blocks['held']['day'] == day for day in days]
        n = np.array([mask.sum() for mask in groups])
        ix = np.random.default_rng(7502).integers(0, len(days), (10000, len(days)))
        controls = {'current_state_persistence': current,
                    'fitting_mean_constant': np.full(len(y), blocks['fit']['y'].mean())}
        intervals = {}
        for name, control in controls.items():
            difference = (point - y) ** 2 - (control - y) ** 2
            grouped = np.array([difference[mask].sum() for mask in groups])
            intervals[name] = np.quantile(grouped[ix].sum(1) / n[ix].sum(1), [.025, .975]).tolist()
            np.testing.assert_allclose(intervals[name], stored['held']['mse_contrasts'][name]['paired_training_date95_ci'], atol=1e-12, rtol=0)
        covered = np.array([(y[mask] <= upper[mask]).sum() for mask in groups])
        np.testing.assert_allclose(np.quantile(covered[ix].sum(1) / n[ix].sum(1), [.025, .975]),
            stored['held']['upper_coverage_training_date95_ci'], atol=1e-12, rtol=0)
        persistence_fn = int(np.count_nonzero(event & ~(current + persistence_margin > 2)))
        assert persistence_fn == stored['held']['forecasts']['current_state_persistence']['false_negatives']
        checks = {'provenance_and_partition_passed': True, 'compiled_raw_prediction_passed': raw_error <= 1e-12,
            'monotone_action_passed': violation <= 1e-12,
            'mse_interval_below_persistence': intervals['current_state_persistence'][1] < 0,
            'mse_interval_below_fitting_constant': intervals['fitting_mean_constant'][1] < 0,
            'held_upper_coverage_at_least_95_percent': metrics['upper_coverage'] >= .95,
            'next_do_false_negatives_not_above_persistence': counts['false_negatives'] <= persistence_fn}
        assert checks == stored['admission']['checks'] and all(checks.values()) == stored['admission']['passed']
        records.append({'policy_seed': seed, 'passed': True, 'max_raw_prediction_difference': raw_error,
            'raw_prediction_rows': sum(len(block['y']) for block in blocks.values()),
            'monotone_held_action_probe_rows': len(probes), 'admission': stored['admission'],
            'fit_summary_sha256': sha(folder / 'fit_summary.json'), 'model_sha256': stored['model_sha256']})
        print('INDEPENDENT_RESIDUAL_DO_AUDIT_PASS', seed, 'admitted', stored['admission']['passed'], flush=True)
    assert all(row['admission']['passed'] for row in records) == summary['all_models_admitted']
    result = {'passed': True, 'policy_seeds': list(range(5)), 'records': records,
        'all_models_admitted': summary['all_models_admitted'], 'summary_sha256': sha(args.study / 'summary.json'),
        'audit_source_sha256': sha(Path(__file__)),
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'new_environment_queries': 0, 'new_model_fitting': False, 'test_evaluated': False,
        'scope': 'Independently reconstruct admission, including raw changes and physical forecasts on all401280 rows per seed. A passed audit does not mean an admitted model.'}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

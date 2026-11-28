"""Independently reconstruct conditional calibration and all admission gates."""
import argparse
import json
import pickle
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e76'))
from residual_forecast import read_blocks, sha, FastRate
from policy_io import verify_sources


def main():
    parser = argparse.ArgumentParser()
    for name in ('workspace', 'models', 'study', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    result = json.loads((args.study / 'summary.json').read_text())
    original = json.loads((args.models / 'summary.json').read_text())
    original_audit = json.loads((args.models / 'audit.json').read_text())
    assert original_audit['passed'] and result['completed'] and result['policy_seeds'] == list(range(5))
    assert result['e76_summary_sha256'] == sha(args.models / 'summary.json') == original_audit['summary_sha256']
    assert result['e76_audit_sha256'] == sha(args.models / 'audit.json')
    verify_sources(original['source_sha256'], ROOT)
    verify_sources(result['source_sha256'], ROOT)
    records = []
    for seed, saved in enumerate(result['records']):
        old = original['records'][seed]
        model_path = args.models / f'seed_{seed}/do_model.pkl'
        assert saved['policy_seed'] == old['policy_seed'] == seed
        assert saved == json.loads((args.study / f'seed_{seed}.json').read_text())
        assert sha(model_path) == saved['model_sha256'] == old['model_sha256']
        blocks, provenance = read_blocks(args.workspace / f'results/e55_study/policies/new_ppolag_fcsrl/seed_{seed}', args.workspace / 'kaggle_data/darrow_15min.pkl')
        assert provenance == old['capture_provenance']
        with model_path.open('rb') as handle:
            model = pickle.load(handle)
        raw = {key: model.predict(b['x']) for key, b in blocks.items()}
        fast = FastRate([model])
        for key, block in blocks.items():
            np.testing.assert_allclose(raw[key], fast.raw(block['x'])[:, 0], atol=1e-12, rtol=0)
        point = {key: np.maximum(b['x'][:, 2:5].max(-1) + raw[key], 0) for key, b in blocks.items()}
        cal = blocks['calibration']
        current_cal = cal['x'][:, 2:5].max(-1)
        residual = cal['y'] - point['calibration']
        groups = np.digitize(current_cal, [1., 1.5, 2.])
        global_margin = max(0., float(np.quantile(residual, .975, method='higher')))
        margins = np.array([max(global_margin, float(np.quantile(residual[groups == k], .975, method='higher')))
                            if np.any(groups == k) else global_margin for k in range(4)])
        persistence_margin = max(0., float(np.quantile(cal['y'] - current_cal, .95, method='higher')))
        assert saved['config']['global_margin'] == global_margin
        assert saved['config']['stratum_counts'] == [int(np.count_nonzero(groups == k)) for k in range(4)]
        np.testing.assert_array_equal(saved['config']['stratum_margins'], margins)
        assert saved['config']['persistence_margin'] == persistence_margin
        for label in ('calibration', 'held'):
            block = blocks[label]
            current = block['x'][:, 2:5].max(-1)
            upper = np.maximum(point[label] + margins[np.digitize(current, [1., 1.5, 2.])], current + persistence_margin)
            event, alarm = block['y'] > 2, upper > 2
            onset = event & (current <= 2)
            residual_upper = block['y'] - upper
            metrics = {'n': len(event), 'mse': float(np.mean((point[label] - block['y']) ** 2)),
                'upper_coverage': float(np.mean(block['y'] <= upper)),
                'false_negatives': int(np.count_nonzero(event & ~alarm)),
                'false_positives': int(np.count_nonzero(~event & alarm)),
                'onset_false_negatives': int(np.count_nonzero(onset & ~alarm)),
                'onset_events': int(onset.sum()), 'next_do_events': int(event.sum()),
                'mean_upper_minus_point': float(np.mean(upper - point[label])),
                'upper_pinball95': float(np.mean(np.maximum(.95 * residual_upper, -.05 * residual_upper))),
                'alarm_rows_added_to_persistence': int(np.count_nonzero(alarm & (current + persistence_margin <= 2)))}
            for name, value in metrics.items():
                np.testing.assert_allclose(value, saved['reports'][label][name], rtol=0, atol=1e-12)
        held = blocks['held']
        current = held['x'][:, 2:5].max(-1)
        days, codes = np.unique(held['day'], return_inverse=True)
        counts = np.bincount(codes)
        ix = np.random.default_rng(7502).integers(0, len(days), size=(10000, len(days)))
        contrasts = {}
        for name, control in [('current_state_persistence', current), ('fitting_mean_constant', np.full(len(current), blocks['fit']['y'].mean()))]:
            diff = (point['held'] - held['y']) ** 2 - (control - held['y']) ** 2
            samples = np.bincount(codes, weights=diff)[ix].sum(1) / counts[ix].sum(1)
            interval = np.quantile(samples, [.025, .975])
            np.testing.assert_allclose(interval, saved['mse_contrasts'][name]['paired_training_date95_ci'], atol=1e-12, rtol=0)
            contrasts[name] = interval[1] < 0
        probes = held['x'][np.linspace(0, len(current) - 1, 10000, dtype=int)]
        lo, hi = probes.copy(), probes.copy()
        lo[:, -1], hi[:, -1] = -1, 1
        assert np.all(model.predict(lo) <= model.predict(hi) + 1e-12)
        checks = {'provenance_and_partition_passed': True, 'compiled_raw_prediction_passed': True,
            'monotone_action_passed': True,
            'mse_interval_below_persistence': bool(contrasts['current_state_persistence']),
            'mse_interval_below_fitting_constant': bool(contrasts['fitting_mean_constant']),
            'held_upper_coverage_at_least_95_percent': saved['reports']['held']['upper_coverage'] >= .95,
            'next_do_false_negatives_not_above_persistence': saved['reports']['held']['false_negatives'] <= old['held']['forecasts']['current_state_persistence']['false_negatives']}
        assert checks == saved['admission']['checks'] and all(checks.values()) == saved['admission']['passed']
        records.append({'policy_seed': seed, 'passed': True, 'admitted': all(checks.values()), 'raw_prediction_rows': len(codes) + len(cal['y']) + len(blocks['fit']['y'])})
        print(json.dumps(records[-1]), flush=True)
    output = {'passed': True, 'records': records, 'all_models_admitted': all(r['admitted'] for r in records),
        'summary_sha256': sha(args.study / 'summary.json'), 'audit_source_sha256': sha(Path(__file__)),
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'test_evaluated': False, 'new_environment_queries': 0}
    args.out.write_text(json.dumps(output, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

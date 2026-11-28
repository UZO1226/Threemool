"""Calibrate the preregistered upper rule on all five frozen residual models."""
import argparse
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e76'))
from residual_forecast import read_blocks, sha, FastRate, diagnostics
from policy_io import verify_sources
from upper_forecast import calibrate_upper, upper_prediction


def main():
    parser = argparse.ArgumentParser()
    for name in ('workspace', 'study', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    args.out.mkdir(parents=True)
    summary = json.loads((args.study / 'summary.json').read_text())
    audit = json.loads((args.study / 'audit.json').read_text())
    assert summary['completed'] and summary['computation_passed'] and audit['passed']
    assert summary['policy_seeds'] == list(range(5)) and audit['summary_sha256'] == sha(args.study / 'summary.json')
    verify_sources(summary['source_sha256'], ROOT)
    paths = [Path(__file__), Path(__file__).with_name('upper_forecast.py')]
    sources = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    records = []
    start = time.process_time()
    for seed, old in enumerate(summary['records']):
        assert old['policy_seed'] == seed
        assert all(sha(ROOT / p) == h for p, h in sources.items())
        path = args.study / f'seed_{seed}/do_model.pkl'
        assert sha(path) == old['model_sha256'] == audit['records'][seed]['model_sha256']
        blocks, provenance = read_blocks(args.workspace / f'results/e55_study/policies/new_ppolag_fcsrl/seed_{seed}', args.workspace / 'kaggle_data/darrow_15min.pkl')
        assert provenance == old['capture_provenance']
        with path.open('rb') as handle:
            model = pickle.load(handle)
        fast = FastRate([model])
        predictions = {}
        max_raw_error = 0.
        for label, block in blocks.items():
            raw = model.predict(block['x'])
            max_raw_error = max(max_raw_error, float(np.max(np.abs(raw - fast.raw(block['x'])[:, 0]))))
            predictions[label] = np.maximum(block['x'][:, 2:5].max(-1) + raw, 0)
        assert max_raw_error <= 1e-12
        cal = blocks['calibration']
        config = calibrate_upper(cal['y'], predictions['calibration'], cal['x'][:, 2:5].max(-1))
        assert config['persistence_margin'] == old['held']['persistence_margin']
        reports = {}
        for label in ('calibration', 'held'):
            block = blocks[label]
            current = block['x'][:, 2:5].max(-1)
            point = predictions[label]
            upper = upper_prediction(point, current, config)
            persistence = current + config['persistence_margin']
            assert np.all(upper >= persistence)
            metric = diagnostics(block['y'], point, upper, current)
            metric['mean_upper_minus_point'] = float(np.mean(upper - point))
            metric['alarm_rows_added_to_persistence'] = int(np.count_nonzero((upper > 2) & (persistence <= 2)))
            masks = np.searchsorted(np.array(config['edges']), current, side='right')
            metric['conditional_coverage'] = [float(np.mean(block['y'][masks == k] <= upper[masks == k])) if np.any(masks == k) else None for k in range(4)]
            reports[label] = metric
        held = blocks['held']
        probes = held['x'][np.linspace(0, len(held['x']) - 1, 10000, dtype=int)]
        lo, hi = probes.copy(), probes.copy()
        lo[:, -1], hi[:, -1] = -1, 1
        current = probes[:, 2:5].max(-1)
        lower = upper_prediction(np.maximum(current + model.predict(lo), 0), current, config)
        upper = upper_prediction(np.maximum(current + model.predict(hi), 0), current, config)
        order_error = float(np.maximum(lower - upper, 0).max())
        checks = dict(old['admission']['checks'])
        checks['compiled_raw_prediction_passed'] = max_raw_error <= 1e-12
        checks['monotone_action_passed'] = order_error <= 1e-12
        checks['held_upper_coverage_at_least_95_percent'] = reports['held']['upper_coverage'] >= .95
        checks['next_do_false_negatives_not_above_persistence'] = reports['held']['false_negatives'] <= old['held']['forecasts']['current_state_persistence']['false_negatives']
        assert abs(reports['held']['mse'] - old['held']['forecasts']['model']['mse']) <= 1e-12
        record = {'policy_seed': seed, 'model_sha256': old['model_sha256'], 'config': config,
            'reports': reports, 'mse_contrasts': old['held']['mse_contrasts'],
            'max_raw_compiler_difference': max_raw_error, 'max_upper_action_order_violation': order_error,
            'admission': {'passed': all(checks.values()), 'checks': checks}}
        records.append(record)
        (args.out / f'seed_{seed}.json').write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps({'policy_seed': seed, 'admitted': record['admission']['passed']}), flush=True)
    value = {'completed': True, 'policy_seeds': list(range(5)), 'records': records,
        'all_models_admitted': all(r['admission']['passed'] for r in records),
        'source_sha256': sources, 'e76_summary_sha256': sha(args.study / 'summary.json'),
        'e76_audit_sha256': sha(args.study / 'audit.json'),
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'cpu_seconds': time.process_time() - start, 'new_model_fitting': False,
        'new_environment_queries': 0, 'test_evaluated': False, 'controller_evaluation': False,
        'limitation': 'Persistence miss dominance follows by construction. Calibration and held captures are exposed; point MSE evidence is inherited from audited e76.'}
    (args.out / 'summary.json').write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()

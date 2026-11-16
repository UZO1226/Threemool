"""Calibration-only factorial diagnosis of state feedback and noise ordering."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e58'))
from upper_tail import UpperDynamics, interpolate, q
sys.path.insert(0, str(ROOT / 'Lee/analysis'))
from audit_long_history_events import event_metrics


def shuffled_paths(original, days):
    result = original.copy()
    for member in range(5):
        for draw in range(8):
            for day_index, day in enumerate(days):
                case = (member * 8 + draw) * len(days) + day_index
                seed = int.from_bytes(hashlib.sha256(f'e60shuffle:{day}:{member}:{draw}'.encode()).digest()[:4], 'little')
                order = np.random.default_rng(seed).permutation(95)
                np.testing.assert_array_equal(np.sort(order), np.arange(95))
                result[case] = original[case, order]
                np.testing.assert_array_equal(result[case, np.argsort(order)], original[case])
    return result


def forecast(dyn, batch, noise, horizon):
    true, exo, zones = batch
    n = len(true)
    members = np.tile(np.repeat(np.arange(5), 8 * n), 2)
    state = np.tile(true[:, 0], (2 * 5 * 8, 1))
    exo = np.tile(exo, (2 * 5 * 8, 1, 1))
    zones = zones * 1.
    zones *= np.minimum(1, 9000 / np.maximum(zones.sum(-1, keepdims=True), 1))
    zones = np.tile(zones, (2 * 5 * 8, 1, 1))
    output = []
    for t in range(95):
        if t % horizon == 0:
            state = np.tile(true[:, t], (2 * 5 * 8, 1))
        x = q.features(state, exo[:, t], zones[:, t], zones[:, max(0, t - 1)])
        values = dyn.quantiles(x, members)
        sampled = interpolate(values, noise[:, t])
        state = np.clip(.25 * state + .75 * sampled, 0, dyn.hi)
        assert np.isfinite(state).all()
        output.append(state.copy())
    return np.stack(output, 1).reshape(2, 5, 8, n, 95, 5)


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'e51', 'e58', 'e59', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256(args.data.read_bytes()).hexdigest()
    raw = pd.read_pickle(args.data)
    previous = json.loads(args.e59.read_text())
    assert previous['data_sha256'] == digest and previous['test_evaluated'] is False
    summaries, models, noises = {}, {}, {}
    for name, directory in [('e51', args.e51), ('e58', args.e58)]:
        summary = json.loads((directory / 'summary.json').read_text())
        assert summary['data_sha256'] == digest and summary['test_evaluated'] is False
        assert summary['selected'] == 'pit_b95_a0.25'
        for filename, expected in summary['model_hashes'].items():
            assert hashlib.sha256((directory / filename).read_bytes()).hexdigest() == expected
        summaries[name] = summary
        backbone = q.QuantileDynamics(directory)
        models[name] = backbone if name == 'e51' else UpperDynamics(backbone, q.GBMDynamics(directory / 'upper_do_0.99.pkl'))
    assert summaries['e51']['fit_days'] == summaries['e58']['fit_days']
    assert summaries['e51']['calibration_days'] == summaries['e58']['calibration_days']
    days = summaries['e58']['calibration_days']
    assert len(days) == 30
    assert set(days).isdisjoint(summaries['e58']['fit_days'] + raw['days']['val'] + raw['days']['test'])
    for name, directory in [('e51', args.e51), ('e58', args.e58)]:
        original = q.paths(np.load(directory / 'pit_bank.npz')['pit'], days, 95)
        noises[name] = np.concatenate([original, shuffled_paths(original, days)])
    batch = q.arrays(raw['data'], days)
    sd = np.concatenate([raw['data'].loc[day][q.STATE].to_numpy(float)
                         for day in summaries['e58']['fit_days']]).std(0)
    starts = np.random.default_rng(20261112).integers(0, len(days), size=(10000, 8))
    indices = ((starts[:, :, None] + np.arange(5)) % len(days)).reshape(10000, -1)[:, :len(days)]

    def job(task):
        name, horizon = task
        prediction = forecast(models[name], batch, noises[name], horizon)
        records, daily = {}, {}
        for index, mode in enumerate(('original', 'shuffled')):
            trajectory = prediction[index]
            score = q.summarize(trajectory, batch[0], sd)
            metrics, errors = event_metrics(trajectory, batch[0])
            records[mode] = {'score': score, 'event_metrics': metrics}
            daily[mode] = errors
            if mode == 'original' and horizon == 95:
                stored = summaries[name]['inner'][summaries[name]['selected']]
                for key in ('normalized_mae', 'proper_selection_score', 'nh4_brier', 'do_brier'):
                    np.testing.assert_allclose(score[key], stored[key], rtol=1e-12, atol=1e-14)
            if mode == 'original' and horizon == 1:
                stored = previous['observed_state']['calibration'][name]['event_metrics']
                for label in ('nh4', 'do'):
                    for key in ('brier', 'average_precision', 'roc_auc'):
                        np.testing.assert_allclose(metrics[label][key], stored[label][key], rtol=1e-12, atol=1e-14)
        contrast = {}
        for label in ('nh4', 'do'):
            difference = daily['shuffled'][label] - daily['original'][label]
            contrast[label] = {'mean': float(difference.mean()),
                'circular_five_day_bootstrap95_ci': np.quantile(difference[indices].mean(1), [.025, .975]).tolist()}
        result = {'model': name, 'reset_horizon': horizon, 'conditions': records,
                  'shuffled_minus_original_brier': contrast}
        (args.out / f'{name}_h{horizon}.json').write_text(json.dumps(result, indent=2, allow_nan=False))
        print('COMPLETED', name, horizon, flush=True)
        return result

    manifest = {'data_sha256': digest, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'test_evaluated': False, 'validation_evaluated': False, 'model_refit': False,
                'tuple_permutation_inverse_exact': True, 'days': days}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(job, [(name, horizon) for name in ('e51', 'e58') for horizon in (1, 4, 16, 95)]))
    manifest.update({'results': results, 'complete': True,
                     'reset1_e59_identity_passed': True, 'reset95_inner_identity_passed': True,
                     'scope': 'Calibration-only descriptive diagnostic with factual state resets; not deployment or model adoption.'})
    (args.out / 'summary.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
    print('DIAGNOSIS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()

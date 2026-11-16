"""Independently replay tail forecasts and compare fixed environment references."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from tail_dynamics import q, rollout, paired_paths
from quantile_env import FrozenDynamics, WORKSPACE
from cooperative import recursive, GBMDynamics

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/analysis'))
from audit_long_history_events import event_metrics


def independent_paths(pit, tail, days, block, uniform=False):
    result = []
    for member in range(5):
        for draw in range(8):
            for day in days:
                seed = int.from_bytes(hashlib.sha256(f'e38:{day}:{member}:{draw}'.encode()).digest()[:4], 'little')
                rng = np.random.default_rng(seed)
                if uniform:
                    u, innovation = rng.random((95, 5)), np.zeros((95, 5))
                else:
                    count = int(np.ceil(95 / block))
                    start = rng.integers(0, len(pit[member]) // 95, size=count) * 95
                    start += rng.integers(0, 96 - block, size=count)
                    index = (start[:, None] + np.arange(block)).reshape(-1)[:95]
                    assert np.all(start // 95 == (start + block - 1) // 95)
                    u, innovation = pit[member, index].copy(), tail[member, index].copy()
                u[:, 1] = .5
                result.append(np.c_[u, innovation])
    return np.stack(result)


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'reference-summary', 'runs', 'freeze', 'e51-runs', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    source = json.loads((args.runs / 'summary.json').read_text())
    parent = FrozenDynamics(args.freeze)
    assert source['test_evaluated'] is False and source['weight_fitting'] is False
    for name, digest in source['model_hashes'].items():
        assert hashlib.sha256((args.runs / name).read_bytes()).hexdigest() == digest
        assert hashlib.sha256((parent.path / name).read_bytes()).hexdigest() == digest
    raw = pd.read_pickle(args.data)
    assert hashlib.sha256(args.data.read_bytes()).hexdigest() == source['data_sha256']
    ref = json.loads(args.reference_summary.read_text())
    assert source['fit_days'] == ref['fit_days'] and source['calibration_days'] == ref['calibration_days']
    days = raw['days']['val']
    assert set(source['fit_days'] + source['calibration_days']).isdisjoint(days + raw['days']['test'])
    batch = q.arrays(raw['data'], days)
    truth, exo, zone = batch
    sd = np.concatenate([raw['data'].loc[day][q.STATE].to_numpy(float) for day in ref['fit_days']]).std(0)
    bank = np.load(args.runs / 'paired_bank.npz')
    mode, block, alpha = source['selected'].split('_')
    block, alpha = int(block[1:]), float(alpha[1:])
    noise = independent_paths(bank['pit'], bank['tail'], days, block, uniform=mode == 'uniform')
    np.testing.assert_array_equal(noise, paired_paths(bank['pit'], bank['tail'], days, block, uniform=mode == 'uniform'))
    prediction = rollout(parent.model, batch, noise, alpha)
    score = q.summarize(prediction, truth, sd)
    for metric in ('normalized_mae', 'nh4_brier', 'do_brier'):
        np.testing.assert_allclose(score[metric], source['validation'][source['selected']][metric], rtol=1e-12)
    state, slow = truth[0, 0].copy(), []
    for t in range(95):
        z = zone[:1, t].copy()
        pz = zone[:1, max(0, t - 1)].copy()
        z *= np.minimum(1, 9000 / np.maximum(z.sum(-1, keepdims=True), 1))
        pz *= np.minimum(1, 9000 / np.maximum(pz.sum(-1, keepdims=True), 1))
        x = q.features(state[None], exo[:1, t], z, pz)
        quantiles = np.array([[state[j] + model.models[0][j].predict(x)[0]
                              for j in range(5)] for model in parent.model.models])
        quantiles = np.clip(np.sort(quantiles, axis=0), 0, parent.hi)
        sampled = np.array([np.interp(noise[0, t, j], q.LEVELS, quantiles[:, j]) for j in range(5)])
        state = np.clip(alpha * state + (1 - alpha) * sampled + noise[0, t, 5:], 0, parent.hi)
        slow.append(state.copy())
    np.testing.assert_allclose(slow, prediction[0, 0, 0], rtol=1e-9, atol=1e-9)
    e51_path = args.e51_runs / 'summary.json'
    assert hashlib.sha256(e51_path.read_bytes()).hexdigest() == source['e51_reference_summary_sha256']
    e51 = json.loads(e51_path.read_text())
    assert e51['model_hashes'] == source['model_hashes']
    m51, b51, a51 = e51['selected'].split('_')
    u51 = q.paths(np.load(args.e51_runs / 'pit_bank.npz')['pit'], days, int(b51[1:]), uniform=m51 == 'uniform')
    reference51 = q.rollout(parent.model, batch, u51, float(a51[1:]))
    zero = rollout(parent.model, batch, np.concatenate([u51, np.zeros_like(u51)], -1), float(a51[1:]), use_tail=False)
    np.testing.assert_array_equal(zero, reference51)
    score51 = q.summarize(reference51, truth, sd)
    for metric in ('normalized_mae', 'nh4_brier', 'do_brier'):
        np.testing.assert_allclose(score51[metric], e51['validation'][e51['selected']][metric], rtol=1e-12)
    frozen = q.rollout(parent.model, batch, q.paths(parent.bank, days, 95), .5)
    old = recursive(GBMDynamics(WORKSPACE / parent.manifest['original_environment_path']), batch)[:, None]
    persistence = np.broadcast_to(truth[None, None, :, 0, None, :], (1, 1, len(days), 95, 5))
    prior = json.loads((ROOT / 'Lee/experiments/e40/event_reference_audit.json').read_text())
    result = {'selected': source['selected'], 'test_evaluated': False, 'model_fitted': False,
              'reconstruction_passed': True, 'independent_paired_noise_indices_passed': True,
              'independent_full_day_sklearn_numpy_interp_passed': True,
              'zero_tail_e51_full_validation_exact': True, 'model_hashes_verified': True,
              'scope': 'Exploratory reused validation. No environment adoption or policy improvement claim.',
              'metrics': {}, 'paired_date_brier_differences': {}}
    errors = {}
    for name, trajectory in [('candidate', prediction), ('e40', frozen), ('e51', reference51), ('old', old), ('persistence', persistence)]:
        result['metrics'][name], errors[name] = event_metrics(trajectory, truth)
        if name in ('e40', 'old', 'persistence'):
            for label in ('nh4', 'do'):
                for metric in ('brier', 'average_precision', 'roc_auc'):
                    np.testing.assert_allclose(result['metrics'][name][label][metric], prior['metrics'][name][label][metric], rtol=1e-12)
    rng = np.random.default_rng(20261112)
    starts = rng.integers(0, len(days), size=(10000, 8))
    indices = ((starts[:, :, None] + np.arange(5)) % len(days)).reshape(10000, -1)[:, :len(days)]
    for baseline in ('e40', 'e51', 'old', 'persistence'):
        result['paired_date_brier_differences'][baseline] = {}
        for label in ('nh4', 'do'):
            difference = errors['candidate'][label] - errors[baseline][label]
            result['paired_date_brier_differences'][baseline][label] = {'mean': float(difference.mean()),
                'circular_five_day_bootstrap95_ci': np.quantile(difference[indices].mean(1), [.025, .975]).tolist()}
    mae = np.abs(prediction - truth[None, None, :, 1:]).mean((0, 1, 3))
    daily = {'normalized_mae': (mae / sd).mean(1), 'nh4_mae': mae[:, 0],
        'nh4_bias_pp': (prediction[..., 0] > 4).mean((0, 1, 3)) * 100 - (truth[:, 1:, 0] > 4).mean(1) * 100,
        'do_bias_pp': (prediction[..., 2:].max(-1) > 2).mean((0, 1, 3)) * 100 - (truth[:, 1:, 2:].max(-1) > 2).mean(1) * 100}
    result['date_intervals'] = {key: {'mean': float(value.mean()),
        'circular_five_day_bootstrap95_ci': np.quantile(value[indices].mean(1), [.025, .975]).tolist()}
        for key, value in daily.items()}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result['paired_date_brier_differences']), flush=True)


if __name__ == '__main__':
    main()

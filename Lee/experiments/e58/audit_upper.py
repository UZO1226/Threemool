"""Reconstruct the selected forecast and independently rebuild full-day paths."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from upper_tail import UpperDynamics, interpolate, q, LEVELS


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'reference-summary', 'runs', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    source = json.loads((args.runs / 'summary.json').read_text())
    reference = json.loads(args.reference_summary.read_text())
    assert source['test_evaluated'] is False
    assert source['data_sha256'] == hashlib.sha256(args.data.read_bytes()).hexdigest()
    for name, expected in source['model_hashes'].items():
        assert hashlib.sha256((args.runs / name).read_bytes()).hexdigest() == expected
    raw = pd.read_pickle(args.data)
    assert source['fit_days'] == reference['fit_days']
    assert source['calibration_days'] == reference['calibration_days']
    days = raw['days']['val']
    assert set(source['fit_days'] + source['calibration_days']).isdisjoint(days + raw['days']['test'])
    mode, block, alpha = source['selected'].split('_')
    block, alpha = int(block[1:]), float(alpha[1:])
    bank = np.load(args.runs / 'pit_bank.npz')['pit']
    noise = q.paths(bank, days, block, uniform=mode == 'uniform')
    batch = q.arrays(raw['data'], days)
    dyn = UpperDynamics(q.QuantileDynamics(args.runs), q.GBMDynamics(args.runs / 'upper_do_0.99.pkl'))
    with patch.object(q, 'interpolate', interpolate):
        prediction = q.rollout(dyn, batch, noise, alpha)
    sd = np.concatenate([raw['data'].loc[day][q.STATE].to_numpy(float)
                         for day in source['fit_days']]).std(0)
    score = q.summarize(prediction, batch[0], sd)
    for key in ('normalized_mae', 'proper_selection_score', 'nh4_brier', 'do_brier'):
        np.testing.assert_allclose(score[key], source['validation'][source['selected']][key], rtol=1e-12)
    truth, exo, zones = batch
    maximum = 0.
    # Separate sklearn inference and scalar np.interp, every member, first date/draw.
    for member in range(5):
        state = truth[0, 0].copy()
        slow = []
        case = member * 8 * len(days)
        for t in range(95):
            x = q.features(state[None], exo[:1, t], zones[:1, t], zones[:1, max(0, t - 1)])
            central = np.array([[state[j] + model.models[member][j].predict(x)[0]
                                 for j in range(5)] for model in dyn.backbone.models])
            central = np.clip(np.sort(central, axis=0), 0, dyn.hi)
            endpoint = central[-1].copy()
            for j in (2, 3, 4):
                upper = np.clip(state[j] + dyn.upper.models[member][j].predict(x)[0], 0, dyn.hi[j])
                endpoint[j] = max(endpoint[j], upper)
            sampled = np.array([np.interp(noise[case, t, j], LEVELS,
                                          np.r_[central[:, j], endpoint[j]]) for j in range(5)])
            state = np.clip(alpha * state + (1 - alpha) * sampled, 0, dyn.hi)
            slow.append(state.copy())
        np.testing.assert_allclose(slow, prediction[member, 0, 0], rtol=1e-9, atol=1e-9)
        maximum = max(maximum, float(np.abs(np.asarray(slow) - prediction[member, 0, 0]).max()))
    target = truth[:, 1:]
    mae = np.abs(prediction - target[None, None]).mean((0, 1, 3))
    daily = {'normalized_mae': (mae / sd).mean(1), 'nh4_mae': mae[:, 0],
             'nh4_bias_pp': (prediction[..., 0] > 4).mean((0, 1, 3)) * 100 - (target[..., 0] > 4).mean(1) * 100,
             'do_bias_pp': (prediction[..., 2:].max(-1) > 2).mean((0, 1, 3)) * 100 - (target[..., 2:].max(-1) > 2).mean(1) * 100}
    starts = np.random.default_rng(20261112).integers(0, len(days), size=(10000, 8))
    indices = ((starts[:, :, None] + np.arange(5)) % len(days)).reshape(10000, -1)[:, :len(days)]
    result = {'selected': source['selected'], 'days': days, 'test_evaluated': False,
              'reconstruction_passed': True, 'checkpoint_hashes_verified': True,
              'independent_full_day_sklearn_numpy_interp_passed': True,
              'independent_scope': 'Five members, first validation date and draw, all95 steps; aggregate fast forecast all40 dates/eight draws.',
              'independent_max_difference': maximum,
              'date_uncertainty': {key: {'mean': float(value.mean()),
                  'five_day_block_95_ci': np.quantile(value[indices].mean(1), [.025, .975]).tolist()}
                  for key, value in daily.items()}}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('AUDIT_PASS', maximum, flush=True)


if __name__ == '__main__':
    main()

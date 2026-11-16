"""Refit monotone quantiles with fixed training-only forecast-state exposure."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e42'))
sys.path.insert(0, str(ROOT / 'Lee/experiments/e40'))
from quantile_env import FrozenDynamics
import quantile_dynamics as q
from monotone_bounds import preserve_bounds


def exposure_arrays(parent, data, days):
    batch = q.arrays(data['data'], days)
    truth, exo, zone = batch
    assert len(days) == 158 and truth.shape == (158, 96, 5)
    assert np.isfinite(truth).all() and np.all(zone.sum(-1) <= 9000 + 1e-8)
    noise = np.full((5 * len(days), 95, 5), .5)
    trajectory = q.rollout(parent.model, batch, noise, .5, draws=1)[:, 0]
    before = np.concatenate([np.broadcast_to(truth[None, :, :1], (5, len(days), 1, 5)),
                             trajectory[:, :, :-1]], axis=2)
    previous = np.concatenate([zone[:, :1], zone[:, :-2]], axis=1)
    inputs = np.stack([q.features(state.reshape(-1, 5), exo[:, :-1].reshape(-1, 9),
                      zone[:, :-1].reshape(-1, 5), previous.reshape(-1, 5)) for state in before])
    observed, delta = q.build_arrays(data['data'], days)
    target = observed[:, :5] + delta
    assert np.isfinite(inputs).all()
    np.testing.assert_array_equal(before[:, :, 0], np.broadcast_to(truth[None, :, 0], (5, len(days), 5)))
    return observed, inputs, target, before


def training_rows(observed, exposed, target, rows, augment=True):
    if not augment:
        return observed[rows].copy(), target[rows].copy()
    return np.concatenate([observed[rows], exposed[rows]]), np.tile(target[rows], (2, 1))


def prepare_checks(parent, data, days):
    observed, exposed, target, before = exposure_arrays(parent, data, days)
    rows = np.arange(len(observed))
    zero_x, zero_y = training_rows(observed, exposed[0], target, rows, augment=False)
    np.testing.assert_array_equal(zero_x, observed)
    np.testing.assert_array_equal(zero_y, target)
    # Alter every future concentration of one day; episode initial state stays.
    changed = {'data': data['data'].copy()}
    indices = changed['data'].loc[days[0]].index[1:]
    changed['data'].loc[indices, q.STATE] += 17
    changed_x, changed_exposed, changed_target, _ = exposure_arrays(parent, changed, days)
    np.testing.assert_array_equal(changed_exposed, exposed)
    assert not np.array_equal(changed_target, target)
    assert not np.array_equal(changed_x, observed)
    error = np.abs(before - observed[:, :5].reshape(1, len(days), 95, 5))
    checks = {'zero_augmentation_original_inputs_exact': True, 'zero_augmentation_original_targets_exact': True,
              'first_exposure_state_observed_initial_exact': True,
              'future_concentration_scrambling_leaves_exposure_exact': True,
              'future_scrambling_changes_supervision': True, 'finite_exposure_inputs': True,
              'fit_air_within_cap': True, 'calibration_or_validation_used_for_exposure': False,
              'exposure_mean_absolute_difference_by_state': error.mean((0, 1, 2)).tolist(),
              'exposure_max_absolute_difference_by_state': error.max((0, 1, 2)).tolist()}
    return observed, exposed, target, checks


def fit_exposed(parent, data, days, out):
    x, exposed, target, checks = prepare_checks(parent, data, days)
    (out / 'exposure_check.json').write_text(json.dumps(checks, indent=2, allow_nan=False))
    states = np.concatenate([data['data'].loc[day][q.STATE].to_numpy(float) for day in days])
    hi = states.max(0) * 1.1
    subsets = [(np.random.RandomState(m).choice(len(days), int(.8 * len(days)), replace=False)[:, None]
                * 95 + np.arange(95)).reshape(-1) for m in range(5)]

    def job(member, quantile, no3=None):
        models = []
        for j in range(5):
            if j == 1 and no3 is not None:
                models.append(no3)
                continue
            xx, yy = training_rows(x, exposed[member], target, subsets[member], augment=j != 1)
            constraint = np.zeros(24, dtype=int)
            if j == 0:
                constraint[0] = 1
                constraint[2:5] = -1
                constraint[14:24] = -1
                xx[:, 1] = 0
            elif j >= 2:
                constraint[0] = -1
                constraint[2:5] = 1
                constraint[14:24] = 1
                xx[:, 1] = 0
            else:
                constraint[1] = 1
            model = q.HGB(loss='quantile', quantile=quantile, max_iter=200, learning_rate=.05,
                max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1,
                monotonic_cst=constraint, early_stopping=False, random_state=100 * member + j)
            model.fit(xx, yy[:, j])
            if j != 1:
                assert not any(np.any(tree[0].nodes['feature_idx'][tree[0].nodes['is_leaf'] == 0] == 1)
                               for tree in model._predictors)
            else:
                expected = parent.model.models[2].models[member][1].model.predict(x[:256])
                np.testing.assert_allclose(model.predict(x[:256]), expected, rtol=0, atol=1e-12)
            models.append(q.DirectDeltaRegressor(model, j))
        print('TRAINED', member, quantile, flush=True)
        return models

    with ThreadPoolExecutor(max_workers=4) as pool:
        median = list(pool.map(lambda member: job(member, .5), range(5)))
        with (out / 'quantile_0.5.pkl').open('wb') as stream:
            pickle.dump({'models': median, 'hi': hi, 'bias': np.zeros(5)}, stream)
        for quantile in [.05, .25, .75, .95]:
            models = list(pool.map(lambda member: job(member, quantile, median[member][1]), range(5)))
            with (out / f'quantile_{quantile}.pkl').open('wb') as stream:
                pickle.dump({'models': models, 'hi': hi, 'bias': np.zeros(5)}, stream)
    return x, hi


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--parent-freeze', type=Path, required=True)
    parser.add_argument('--preflight-out', type=Path)
    options, remaining = parser.parse_known_args()
    parent = FrozenDynamics(options.parent_freeze)
    if options.preflight_out is not None:
        if options.preflight_out.exists():
            raise FileExistsError(options.preflight_out)
        inputs = argparse.ArgumentParser()
        inputs.add_argument('--data', type=Path, required=True)
        inputs.add_argument('--reference-summary', type=Path, required=True)
        args = inputs.parse_args(remaining)
        raw = pd.read_pickle(args.data)
        ref = json.loads(args.reference_summary.read_text())
        assert set(ref['fit_days']).isdisjoint(raw['days']['test'] + raw['days']['val'])
        _, _, _, checks = prepare_checks(parent, raw, ref['fit_days'])
        checks['test_evaluated'] = False
        options.preflight_out.write_text(json.dumps(checks, indent=2, allow_nan=False))
        print(json.dumps(checks), flush=True)
        return
    original_fit = q.fit
    q.fit = lambda data, days, out: fit_exposed(parent, data, days, out)
    sys.argv = [sys.argv[0]] + remaining
    try:
        q.main(training_context=preserve_bounds, strict_order=True)
    finally:
        q.fit = original_fit
    out = Path(remaining[remaining.index('--out') + 1])
    summary_path = out / 'summary.json'
    summary = json.loads(summary_path.read_text())
    summary.update({'experiment': 'e50_quantile_exposure', 'augmentation_rounds': 1,
                    'observed_exposed_mixture': [.5, .5], 'exposure_quantile': .5,
                    'exposure_alpha': .5, 'no3_observed_only': True,
                    'parent_freeze_sha256': hashlib.sha256(options.parent_freeze.read_bytes()).hexdigest()})
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

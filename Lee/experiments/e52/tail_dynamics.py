"""Paired PIT and action-independent tail innovations for frozen quantile models."""
import hashlib
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e51'))
from calibrate_blend import prepare, ALPHAS
import quantile_dynamics as q

SAFETY = [0, 2, 3, 4]


def paired_paths(pit, tail, days, block, draws=8, uniform=False):
    assert pit.shape == tail.shape and pit.shape[-1] == 5
    if uniform:
        u = q.paths(pit, days, block, draws=draws, uniform=True)
        return np.concatenate([u, np.zeros_like(u)], axis=-1)
    return q.paths(np.concatenate([pit, tail], axis=-1), days, block, draws=draws)


def rollout(dyn, batch, noise, alpha, draws=8, air_scale=1., use_tail=True, diagnostics=False):
    truth, exo, zones = batch
    n = len(truth)
    assert noise.shape == (5 * draws * n, 95, 10)
    members = np.repeat(np.arange(5), draws * n)
    state = np.tile(truth[:, 0], (5 * draws, 1))
    exo = np.tile(exo, (5 * draws, 1, 1))
    zones = zones * air_scale
    zones *= np.minimum(1, 9000 / np.maximum(zones.sum(-1, keepdims=True), 1))
    zones = np.tile(zones, (5 * draws, 1, 1))
    values, below, above = [], np.zeros(5, dtype=int), np.zeros(5, dtype=int)
    for t in range(95):
        x = q.features(state, exo[:, t], zones[:, t], zones[:, max(0, t - 1)])
        sampled = q.interpolate(dyn.quantiles(x, members), noise[:, t, :5])
        proposed = alpha * state + (1 - alpha) * sampled
        if use_tail:
            proposed = proposed + noise[:, t, 5:]
        below += (proposed < 0).sum(0)
        above += (proposed > dyn.hi).sum(0)
        state = np.clip(proposed, 0, dyn.hi)
        values.append(state.copy())
    trajectory = np.stack(values, 1).reshape(5, draws, n, 95, 5)
    if not diagnostics:
        return trajectory
    count = 5 * draws * n * 95
    return trajectory, {'member_draw_transition_count': count,
        'below_zero_count_by_state': below.tolist(), 'above_hi_count_by_state': above.tolist(),
        'below_zero_fraction_by_state': (below / count).tolist(),
        'above_hi_fraction_by_state': (above / count).tolist()}


def prepare_tails(parent, raw, reference):
    pits, checks = prepare(parent, raw, reference)
    cal = reference['calibration_days']
    x, delta = q.build_arrays(raw['data'], cal)
    current = np.tile(x[:, :5], (5, 1))
    target = np.tile(x[:, :5] + delta, (5, 1))
    values = parent.model.quantiles(np.tile(x, (5, 1)), np.repeat(np.arange(5), len(x)))
    tails, statistics = {}, {}
    for alpha in ALPHAS:
        blended = np.clip(alpha * current[None] + (1 - alpha) * values, 0, parent.hi)
        tail = np.where(target < blended[0], target - blended[0],
                        np.where(target > blended[-1], target - blended[-1], 0.))
        tail[:, 1] = 0
        assert np.isfinite(tail).all()
        outside = (target < blended[0]) | (target > blended[-1])
        np.testing.assert_array_equal(tail[:, SAFETY][~outside[:, SAFETY]], 0)
        assert np.all(tail[:, SAFETY][target[:, SAFETY] < blended[0][:, SAFETY]] <= 0)
        assert np.all(tail[:, SAFETY][target[:, SAFETY] > blended[-1][:, SAFETY]] >= 0)
        u = pits[alpha].reshape(-1, 5)
        recovered = np.clip(alpha * current + (1 - alpha) * q.interpolate(values, u) + tail, 0, parent.hi)
        np.testing.assert_allclose(recovered[:, SAFETY], np.clip(target, 0, parent.hi)[:, SAFETY], rtol=0, atol=1e-8)
        statistics[str(alpha)] = {'nonzero_fraction_by_state': (tail != 0).mean(0).tolist(),
            'mean_absolute_by_state': np.abs(tail).mean(0).tolist(),
            'maximum_absolute_by_state': np.abs(tail).max(0).tolist(),
            'target_above_original_hi_count_by_state': (target > parent.hi).sum(0).tolist()}
        tails[alpha] = tail.reshape(5, len(x), 5)
        np.testing.assert_array_equal(tails[alpha][..., 1], 0)
    checks.update({'tail_signs_and_zero_inside_support_passed': True, 'no3_tail_exact_zero': True,
                   'own_calibration_clipped_safety_target_reconstruction': True,
                   'calibration_reconstruction_is_not_generalization': True,
                   'tail_statistics_by_alpha': statistics})
    return pits, tails, checks


def preflight(parent, raw, reference, e51_path):
    import json
    pits, tails, checks = prepare_tails(parent, raw, reference)
    e51 = json.loads((e51_path / 'summary.json').read_text())
    assert e51['selected'] == 'pit_b95_a0.25' and e51['weight_fitting'] is False
    assert e51['model_hashes'] == json.loads((parent.path / 'summary.json').read_text())['model_hashes']
    np.testing.assert_array_equal(pits[.25], np.load(e51_path / 'pit_bank.npz')['pit'])
    days = reference['calibration_days'][:2]
    for block in (1, 4, 16, 95):
        noise = paired_paths(pits[.25], tails[.25], days, block, draws=1)
        np.testing.assert_array_equal(noise[..., :5], q.paths(pits[.25], days, block, draws=1))
        seed = int.from_bytes(hashlib.sha256(f'e38:{days[0]}:0:0'.encode()).digest()[:4], 'little')
        rng = np.random.default_rng(seed)
        count = int(np.ceil(95 / block))
        start = rng.integers(0, len(pits[.25][0]) // 95, size=count) * 95 + rng.integers(0, 96 - block, size=count)
        indices = (start[:, None] + np.arange(block)).reshape(-1)[:95]
        np.testing.assert_array_equal(noise[0, :, 5:], tails[.25][0, indices])
    batch = q.arrays(raw['data'], days)
    noise = paired_paths(pits[.25], tails[.25], days, 95, draws=1)
    zero = rollout(parent.model, batch, noise, .25, draws=1, use_tail=False)
    baseline = q.rollout(parent.model, batch, noise[..., :5], .25, draws=1)
    np.testing.assert_array_equal(zero, baseline)
    low = rollout(parent.model, batch, noise, .25, draws=1, air_scale=.75)
    high = rollout(parent.model, batch, noise, .25, draws=1, air_scale=1.25)
    nh4_order = float(np.maximum(high[..., 0] - low[..., 0], 0).max())
    do_order = float(np.maximum(low[..., 2:] - high[..., 2:], 0).max())
    assert nh4_order < 1e-8 and do_order < 1e-8
    checks.update({'independent_paired_path_indices_passed': True,
                   'pit_path_identity_all_four_blocks_passed': True,
                   'zero_tail_selected_e51_bank_full_day_exact': True,
                   'nonzero_tail_common_noise_preflight_order': {'nh4_max_violation': nh4_order, 'do_max_violation': do_order},
                   'preflight_days': days, 'preflight_draws': 1})
    return pits, tails, checks

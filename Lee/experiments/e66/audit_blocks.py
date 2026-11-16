"""Independent donor/trajectory reconstruction and the fixed safety gates."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from evaluate_blocks import ROOT, FrozenDynamics, q
from check_blocks import digest

sys.path.insert(0, str(ROOT / 'Lee/analysis'))
from audit_long_history_events import event_metrics
from cooperative import recursive


def independent_noise(bank, query, donors, days, cal, sd, context=True):
    eligible = [(day, start) for day in range(30) for start in range(92)]
    pools = []
    for weather in query:
        row = []
        for time in range(0, 95, 4):
            ranked = sorted(eligible, key=lambda item: (float(np.square(
                (donors[item[0], item[1]] - weather[time]) / sd).sum()), item[0] * 95 + item[1])) if context else eligible
            row.append(sorted(day * 95 + start for day, start in (ranked[:32] if context else ranked)))
        pools.append(row)
    result = []
    for member in range(5):
        for draw in range(8):
            for day, row in zip(days, pools):
                seed = int.from_bytes(hashlib.sha256(f'e66:{day}:{member}:{draw}'.encode()).digest()[:4], 'little')
                rng = np.random.default_rng(seed)
                values = []
                for pool in row:
                    start = pool[int(rng.integers(len(pool)))]
                    values.extend([bank[member, start + step].copy() for step in range(4)])
                result.append(values[:95])
    return np.array(result)


def slow_days(dyn, batch, noise, fast):
    truth, exo, zone = batch
    zone = zone.copy()
    zone *= np.minimum(1, 9000 / np.maximum(zone.sum(-1, keepdims=True), 1))
    maximum = 0.
    for member in range(5):
        state = truth[0, 0].copy()
        values = []
        case = member * 8 * len(truth)
        for time in range(95):
            x = q.features(state[None], exo[:1, time], zone[:1, time], zone[:1, max(0, time - 1)])
            levels = np.array([[state[j] + model.models[member][j].predict(x)[0]
                                for j in range(5)] for model in dyn.models])
            levels = np.clip(np.sort(levels, axis=0), 0, dyn.hi)
            sampled = np.array([np.interp(noise[case, time, j], q.LEVELS, levels[:, j]) for j in range(5)])
            state = np.clip(.5 * state + .5 * sampled, 0, dyn.hi)
            values.append(state.copy())
        np.testing.assert_allclose(values, fast[member, 0, 0], rtol=1e-9, atol=1e-9)
        maximum = max(maximum, float(np.max(np.abs(np.array(values) - fast[member, 0, 0]))))
    return maximum


def main():
    parser = argparse.ArgumentParser()
    for name in ('workspace', 'freeze', 'preflight', 'runs', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    source = json.loads((args.runs / 'summary.json').read_text())
    assert source['primary'] == 'context32_block4' and not source['test_evaluated']
    for name, expected in source['source_sha256'].items():
        assert digest(Path(__file__).parent / name) == expected
    assert digest(args.preflight) == source['preflight_sha256']
    preflight = json.loads(args.preflight.read_text())
    frozen = FrozenDynamics(args.freeze)
    workspace = args.workspace.resolve()
    data_path = workspace / 'kaggle_data/darrow_15min.pkl'
    assert digest(data_path) == source['data_sha256']
    raw = pd.read_pickle(data_path)
    fit, cal, days = source['fit_days'], source['calibration_days'], raw['days']['val']
    assert len(days) == 40 and set(fit + cal).isdisjoint(days + raw['days']['test'])
    batch = q.arrays(raw['data'], days)
    donor_batch = q.arrays(raw['data'], cal)
    fitting_batch = q.arrays(raw['data'], fit)
    sd_context = np.maximum(fitting_batch[1][:, :95, :4].reshape(-1, 4).std(0), 1e-6)
    np.testing.assert_array_equal(sd_context, preflight['fit_context_sd'])
    sd = np.concatenate([raw['data'].loc[day][q.STATE].to_numpy(float) for day in fit]).std(0)
    result = {'test_evaluated': False, 'model_refit': False, 'selection_changed': False,
              'policy_seeds': [], 'model_members': 5, 'metrics': {}, 'independent_max_difference': {},
              'independent_point_gates': {}, 'paired_date_brier_differences': {},
              'scope': 'Exploratory reused validation; no physical plant or policy improvement.'}
    errors = {}
    def record(name, trajectory, expected=None):
        result['metrics'][name], errors[name] = event_metrics(trajectory, batch[0])
        if expected is not None:
            score = q.summarize(trajectory, batch[0], sd)
            for key in ('normalized_mae', 'proper_selection_score', 'nh4_brier', 'do_brier'):
                np.testing.assert_allclose(score[key], expected[key], rtol=1e-12, atol=1e-14)
            return score
    old = json.loads((workspace / 'results/e17_contractive/summary.json').read_text())
    for mode in ('global_block4', 'context32_block4'):
        noise = independent_noise(frozen.bank, batch[1][:, :95, :4], donor_batch[1][:, :95, :4],
                                  days, cal, sd_context, context=mode == 'context32_block4')
        from exogenous_blocks import donor_pools, paths
        mean = np.array(preflight['fit_context_mean'])
        pool = donor_pools(batch[1][:, :95, :4], donor_batch[1][:, :95, :4], days, cal, mean,
                           sd_context, k=32 if mode == 'context32_block4' else None, exclude_self=False)
        np.testing.assert_array_equal(noise, paths(frozen.bank, days, pool))
        trajectory = q.rollout(frozen.model, batch, noise, .5)
        score = record(mode, trajectory, source['validation'][mode])
        result['independent_max_difference'][mode] = slow_days(frozen.model, batch, noise, trajectory)
        low = q.rollout(frozen.model, batch, noise, .5, air_scale=.75)
        high = q.rollout(frozen.model, batch, noise, .5, air_scale=1.25)
        order = {'nh4_max_order_violation': float(np.maximum(high[..., 0] - low[..., 0], 0).max()),
                 'do_max_order_violation': float(np.maximum(low[..., 2:] - high[..., 2:], 0).max())}
        assert max(order.values()) < 1e-8
        assert order == source['validation'][mode]['recursive_order']
        score['recursive_order'] = order
        result['independent_point_gates'][mode] = q.gate(score, old['validation']['old'])
        print('INDEPENDENT_BLOCK_PASS', mode, flush=True)
    assert result['independent_point_gates']['context32_block4'] == source['primary_point_gate']
    record('e40', q.rollout(frozen.model, batch, q.paths(frozen.bank, days, 95), .5), source['frozen_reference'])
    record('persistence', np.broadcast_to(batch[0][None, None, :, 0, None, :], (1, 1, 40, 95, 5)))
    record('old', recursive(q.GBMDynamics(workspace / frozen.manifest['original_environment_path']), batch)[:, None])
    prior = json.loads((ROOT / 'Lee/experiments/e40/event_reference_audit.json').read_text())
    for name in ('e40', 'persistence', 'old'):
        for label in ('nh4', 'do'):
            for key in ('brier', 'average_precision', 'roc_auc'):
                np.testing.assert_allclose(result['metrics'][name][label][key], prior['metrics'][name][label][key], rtol=1e-12)
    starts = np.random.default_rng(20261112).integers(0, 40, size=(10000, 8))
    indices = ((starts[:, :, None] + np.arange(5)) % 40).reshape(10000, 40)
    for candidate in ('context32_block4', 'global_block4'):
        differences = {}
        for baseline in ('e40', 'persistence', 'old', 'global_block4'):
            differences[baseline] = {}
            for label in ('nh4', 'do'):
                delta = errors[candidate][label] - errors[baseline][label]
                differences[baseline][label] = {'mean': float(delta.mean()),
                    'circular_five_day_bootstrap95_ci': np.quantile(delta[indices].mean(1), [.025, .975]).tolist()}
        result['paired_date_brier_differences'][candidate] = differences
    differences = result['paired_date_brier_differences']['context32_block4']
    do = result['metrics']['context32_block4']['do']
    checks = {'original_point_gate': source['primary_point_gate']['passed'],
              'nh4_brier_not_worse_e40': differences['e40']['nh4']['circular_five_day_bootstrap95_ci'][1] <= 0,
              'do_brier_not_worse_e40': differences['e40']['do']['circular_five_day_bootstrap95_ci'][1] <= 0,
              'do_brier_not_worse_persistence': differences['persistence']['do']['circular_five_day_bootstrap95_ci'][1] <= 0,
              'do_ap_above_prevalence': do['average_precision'] > do['prevalence'],
              'do_auc_above_half': do['roc_auc'] > .5,
              'at_least_one_new_do_onset_within_four_steps': do['onset_tolerance_steps']['4']['matched'] >= 1}
    result.update({'adoption': {'passed': all(checks.values()), 'checks': checks},
                   'independent_donors_and_paths_exact': True, 'independent_full_day_passed': True,
                   'independent_full_day_scope': 'Every member, first validation date/draw, all95 steps; all validation paths rebuilt.',
                   'audit_source_sha256': digest(Path(__file__)), 'summary_sha256': digest(args.runs / 'summary.json')})
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('BLOCK_ADOPTION', json.dumps(result['adoption']), flush=True)


if __name__ == '__main__':
    main()

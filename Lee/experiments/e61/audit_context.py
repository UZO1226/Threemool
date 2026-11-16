"""Independent donor/full-day reconstruction and fixed e61 event criteria."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from context_paths import initial_context, donor_pools, paths
from runtime import ROOT, q, load_model, rollout

sys.path.insert(0, str(ROOT / 'Lee/analysis'))
sys.path.insert(0, str(ROOT / 'Lee/experiments/e42'))
from audit_long_history_events import event_metrics
from quantile_env import FrozenDynamics, WORKSPACE
from cooperative import recursive


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_context(batch):
    truth, exo, _ = batch
    return np.array([[row[0, 0], row[0, 2], row[0, 3], row[0, 4], weather[0, 1], weather[0, 0]]
                     for row, weather in zip(truth, exo)])


def independent_paths(bank, batch, donor_batch, fitting_batch, days, cal):
    fitting = raw_context(fitting_batch)
    mean, sd = fitting.mean(0), np.maximum(fitting.std(0), 1e-6)
    queries, donors = raw_context(batch), raw_context(donor_batch)
    pools = []
    for query in queries:
        ranked = sorted(range(len(cal)), key=lambda j: (float(np.square(
            (donors[j] - mean) / sd - (query - mean) / sd).sum()), j))
        pools.append(sorted(ranked[:5]))
    values = []
    for member in range(5):
        for draw in range(8):
            for day, pool in zip(days, pools):
                seed = int.from_bytes(hashlib.sha256(f'e38:{day}:{member}:{draw}'.encode()).digest()[:4], 'little')
                choice = np.random.default_rng(seed).integers(0, 5)
                values.append(bank[member, pool[choice] * 95:(pool[choice] + 1) * 95].copy())
    checked = donor_pools(initial_context(batch), initial_context(donor_batch), days, cal, mean, sd,
                          k=5, exclude_self=False)
    for expected, actual in zip(pools, checked):
        np.testing.assert_array_equal(expected, actual)
    noise = np.stack(values)
    np.testing.assert_array_equal(noise, paths(bank, days, checked))
    return noise


def slow_days(dyn, batch, noise, fast):
    truth, exo, zones = batch
    zones = zones.copy()
    zones *= np.minimum(1, 9000 / np.maximum(zones.sum(-1, keepdims=True), 1))
    backbone = dyn.backbone if hasattr(dyn, 'backbone') else dyn
    maximum = 0.
    for member in range(5):
        state = truth[0, 0].copy()
        values = []
        case = member * 8 * len(truth)
        for t in range(95):
            x = q.features(state[None], exo[:1, t], zones[:1, t], zones[:1, max(0, t - 1)])
            central = np.array([[state[j] + model.models[member][j].predict(x)[0]
                                 for j in range(5)] for model in backbone.models])
            central = np.clip(np.sort(central, axis=0), 0, dyn.hi)
            levels = q.LEVELS
            if hasattr(dyn, 'upper'):
                endpoint = central[-1].copy()
                for j in (2, 3, 4):
                    upper = np.clip(state[j] + dyn.upper.models[member][j].predict(x)[0], 0, dyn.hi[j])
                    endpoint[j] = max(endpoint[j], upper)
                central = np.concatenate([central, endpoint[None]])
                levels = np.r_[levels, .99]
            sampled = np.array([np.interp(noise[case, t, j], levels, central[:, j]) for j in range(5)])
            state = np.clip(.25 * state + .75 * sampled, 0, dyn.hi)
            values.append(state.copy())
        np.testing.assert_allclose(values, fast[member, 0, 0], rtol=1e-9, atol=1e-9)
        maximum = max(maximum, float(np.abs(np.asarray(values) - fast[member, 0, 0]).max()))
    return maximum


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'e51', 'e58', 'freeze', 'runs', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    source = json.loads((args.runs / 'summary.json').read_text())
    assert source['test_evaluated'] is False and source['primary'] == 'e51_context5'
    assert source['both_global_reference_reconstructions_exact']
    assert source['data_sha256'] == digest(args.data)
    for name, expected in source['source_sha256'].items():
        assert digest(Path(__file__).parent / name) == expected
    preflight_path = Path(__file__).parent / 'preflight.json'
    assert digest(preflight_path) == source['preflight_sha256']
    preflight = json.loads(preflight_path.read_text())
    raw = pd.read_pickle(args.data)
    fit, cal, days = source['fit_days'], source['calibration_days'], raw['days']['val']
    assert len(fit) == 158 and len(cal) == 30 and len(days) == 40
    assert set(fit).isdisjoint(cal)
    assert set(fit + cal).isdisjoint(days + raw['days']['test'])
    batch, donor_batch, fitting_batch = [q.arrays(raw['data'], dates) for dates in (days, cal, fit)]
    sd = np.concatenate([raw['data'].loc[day][q.STATE].to_numpy(float) for day in fit]).std(0)
    result = {'test_evaluated': False, 'model_refit': False, 'selection_changed': False,
              'primary': 'e51_context5', 'policy_seeds': [], 'model_members': 5,
              'scope': 'Exploratory reused validation, no plant or policy improvement claim.',
              'independent_full_day_scope': 'Every member, first validation date/draw, all95 steps; all donor paths and aggregate forecasts rebuilt.',
              'metrics': {}, 'independent_max_difference': {}, 'paired_date_brier_differences': {}}
    errors = {}

    def record(name, trajectory, expected=None):
        metrics, daily = event_metrics(trajectory, batch[0])
        result['metrics'][name], errors[name] = metrics, daily
        if expected is not None:
            score = q.summarize(trajectory, batch[0], sd)
            for key in ('normalized_mae', 'proper_selection_score', 'nh4_brier', 'do_brier'):
                np.testing.assert_allclose(score[key], expected[key], rtol=1e-12, atol=1e-14)

    for parent, directory in [('e51', args.e51), ('e58', args.e58)]:
        assert digest(directory / 'summary.json') == source['parent_summary_sha256'][parent]
        summary = json.loads((directory / 'summary.json').read_text())
        for name, expected in summary['model_hashes'].items():
            assert digest(directory / name) == expected
        bank = np.load(directory / 'pit_bank.npz')['pit']
        assert hashlib.sha256(bank.tobytes()).hexdigest() == preflight['bank_array_sha256'][parent]
        dyn = load_model(directory, parent)
        noise = independent_paths(bank, batch, donor_batch, fitting_batch, days, cal)
        trajectory = rollout(dyn, batch, noise)
        record(parent + '_context5', trajectory, source['validation_context5'][parent])
        result['independent_max_difference'][parent] = slow_days(dyn, batch, noise, trajectory)
        print('INDEPENDENT_CONTEXT_PASS', parent, flush=True)
        global_trajectory = rollout(dyn, batch, q.paths(bank, days, 95))
        record(parent + '_global', global_trajectory, source['validation_global'][parent])
    frozen = FrozenDynamics(args.freeze)
    record('e40', q.rollout(frozen.model, batch, q.paths(frozen.bank, days, 95), .5))
    record('old', recursive(q.GBMDynamics(WORKSPACE / frozen.manifest['original_environment_path']), batch)[:, None])
    record('persistence', np.broadcast_to(batch[0][None, None, :, 0, None, :], (1, 1, len(days), 95, 5)))
    prior = json.loads((ROOT / 'Lee/experiments/e40/event_reference_audit.json').read_text())
    for name in ('e40', 'old', 'persistence'):
        for label in ('nh4', 'do'):
            for key in ('brier', 'average_precision', 'roc_auc'):
                np.testing.assert_allclose(result['metrics'][name][label][key], prior['metrics'][name][label][key], rtol=1e-12)
    starts = np.random.default_rng(20261112).integers(0, len(days), size=(10000, 8))
    indices = ((starts[:, :, None] + np.arange(5)) % len(days)).reshape(10000, -1)[:, :len(days)]
    for parent in ('e51', 'e58'):
        name = parent + '_context5'
        result['paired_date_brier_differences'][name] = {}
        for baseline in ('e40', 'old', 'persistence', 'e51_global', 'e58_global'):
            result['paired_date_brier_differences'][name][baseline] = {}
            for label in ('nh4', 'do'):
                delta = errors[name][label] - errors[baseline][label]
                result['paired_date_brier_differences'][name][baseline][label] = {'mean': float(delta.mean()),
                    'circular_five_day_bootstrap95_ci': np.quantile(delta[indices].mean(1), [.025, .975]).tolist()}
    differences = result['paired_date_brier_differences']['e51_context5']
    do = result['metrics']['e51_context5']['do']
    checks = {'original_point_gate': source['primary_point_gate']['passed'],
        'nh4_brier_not_worse_e40': differences['e40']['nh4']['circular_five_day_bootstrap95_ci'][1] <= 0,
        'do_brier_not_worse_e40': differences['e40']['do']['circular_five_day_bootstrap95_ci'][1] <= 0,
        'do_brier_not_worse_persistence': differences['persistence']['do']['circular_five_day_bootstrap95_ci'][1] <= 0,
        'do_ap_above_prevalence': do['average_precision'] > do['prevalence'],
        'do_auc_above_half': do['roc_auc'] > .5,
        'at_least_one_new_do_onset_within_four_steps': do['onset_tolerance_steps']['4']['matched'] >= 1}
    result.update({'adoption': {'passed': all(checks.values()), 'checks': checks},
                   'independent_donors_and_paths_exact': True, 'independent_full_day_passed': True,
                   'source_sha256': digest(Path(__file__)), 'run_summary_sha256': digest(args.runs / 'summary.json')})
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('ADOPTION', json.dumps(result['adoption']), flush=True)


if __name__ == '__main__':
    main()

"""Compare upper-tail event timing and probabilities under fixed adoption rules."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

from upper_tail import ROOT, UpperDynamics, interpolate, q

sys.path.insert(0, str(ROOT / 'Lee/analysis'))
sys.path.insert(0, str(ROOT / 'Lee/experiments/e42'))
from audit_long_history_events import event_metrics
from quantile_env import FrozenDynamics, WORKSPACE
from cooperative import recursive


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'runs', 'freeze', 'audit', 'e51', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    independent = json.loads(args.audit.read_text())
    assert independent['reconstruction_passed']
    assert independent['independent_full_day_sklearn_numpy_interp_passed']
    raw = pd.read_pickle(args.data)
    days = raw['days']['val']
    assert independent['days'] == days
    batch = q.arrays(raw['data'], days)
    truth = batch[0]
    source = json.loads((args.runs / 'summary.json').read_text())
    assert source['data_sha256'] == hashlib.sha256(args.data.read_bytes()).hexdigest()
    assert source['test_evaluated'] is False
    assert set(source['fit_days'] + source['calibration_days']).isdisjoint(days + raw['days']['test'])
    for directory in (args.runs, args.e51):
        summary = json.loads((directory / 'summary.json').read_text())
        assert summary['test_evaluated'] is False
        assert summary['fit_days'] == source['fit_days']
        assert summary['calibration_days'] == source['calibration_days']
        for name, digest in summary['model_hashes'].items():
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
    mode, block, alpha = source['selected'].split('_')
    bank = np.load(args.runs / 'pit_bank.npz')['pit']
    dyn = UpperDynamics(q.QuantileDynamics(args.runs), q.GBMDynamics(args.runs / 'upper_do_0.99.pkl'))
    with patch.object(q, 'interpolate', interpolate):
        candidate = q.rollout(dyn, batch, q.paths(bank, days, int(block[1:]), uniform=mode == 'uniform'), float(alpha[1:]))
    parent = FrozenDynamics(args.freeze)
    frozen = q.rollout(parent.model, batch, q.paths(parent.bank, days, 95), .5)
    old = recursive(q.GBMDynamics(WORKSPACE / parent.manifest['original_environment_path']), batch)[:, None]
    persistence = np.broadcast_to(truth[None, None, :, 0, None, :], (1, 1, len(days), 95, 5))
    prior = json.loads((args.e51 / 'summary.json').read_text())
    mode, block, alpha = prior['selected'].split('_')
    previous = q.rollout(q.QuantileDynamics(args.e51), batch,
        q.paths(np.load(args.e51 / 'pit_bank.npz')['pit'], days, int(block[1:]), uniform=mode == 'uniform'), float(alpha[1:]))
    reference = json.loads((ROOT / 'Lee/experiments/e40/event_reference_audit.json').read_text())
    metrics, errors = {}, {}
    for name, trajectory in [('candidate', candidate), ('e40', frozen), ('old', old),
                              ('persistence', persistence), ('e51', previous)]:
        metrics[name], errors[name] = event_metrics(trajectory, truth)
        for label in ('nh4', 'do'):
            if name in ('e40', 'old', 'persistence'):
                for key in ('brier', 'average_precision', 'roc_auc'):
                    np.testing.assert_allclose(metrics[name][label][key], reference['metrics'][name][label][key], rtol=1e-12)
            if name in ('candidate', 'e51'):
                summary = source if name == 'candidate' else prior
                np.testing.assert_allclose(metrics[name][label]['brier'],
                    summary['validation'][summary['selected']][label + '_brier'], rtol=1e-12)
    starts = np.random.default_rng(20261112).integers(0, len(days), size=(10000, 8))
    indices = ((starts[:, :, None] + np.arange(5)) % len(days)).reshape(10000, -1)[:, :len(days)]
    differences = {}
    for name in ('e40', 'old', 'persistence', 'e51'):
        differences[name] = {}
        for label in ('nh4', 'do'):
            delta = errors['candidate'][label] - errors[name][label]
            differences[name][label] = {'mean': float(delta.mean()),
                'circular_five_day_bootstrap95_ci': np.quantile(delta[indices].mean(1), [.025, .975]).tolist()}
    do = metrics['candidate']['do']
    checks = {'original_point_gate': source['gate']['passed'],
              'nh4_brier_not_worse_e40': differences['e40']['nh4']['circular_five_day_bootstrap95_ci'][1] <= 0,
              'do_brier_not_worse_e40': differences['e40']['do']['circular_five_day_bootstrap95_ci'][1] <= 0,
              'do_brier_not_worse_persistence': differences['persistence']['do']['circular_five_day_bootstrap95_ci'][1] <= 0,
              'do_ap_above_prevalence': do['average_precision'] > do['prevalence'],
              'do_auc_above_half': do['roc_auc'] > .5,
              'at_least_one_new_do_onset_within_four_steps': do['onset_tolerance_steps']['4']['matched'] >= 1}
    result = {'selected': source['selected'], 'metrics': metrics,
              'paired_date_brier_differences': differences,
              'adoption': {'passed': all(checks.values()), 'checks': checks},
              'original_point_gate': source['gate'], 'test_evaluated': False,
              'model_refit': False, 'selection_changed': False, 'model_members': 5, 'policy_seeds': [],
              'scope': 'Exploratory reused40 validation days; no independent confirmation or plant/policy gain.',
              'onset_definition': 'Within-day one-to-one new onsets excluding initial ongoing events; probability threshold.5, tolerances0/1/4 steps.',
              'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
              'independent_audit_sha256': hashlib.sha256(args.audit.read_bytes()).hexdigest()}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('ADOPTION', json.dumps(result['adoption']), flush=True)


if __name__ == '__main__':
    main()

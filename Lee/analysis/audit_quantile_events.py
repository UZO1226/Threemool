"""Compare an independently reconstructed quantile candidate with fixed references."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Lee/analysis'))
from audit_long_history_events import event_metrics
from quantile_env import FrozenDynamics, WORKSPACE
from quantile_dynamics import QuantileDynamics, arrays, paths, rollout
from cooperative import recursive, GBMDynamics


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'runs', 'freeze', 'audit', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    independent = json.loads(args.audit.read_text())
    assert independent['reconstruction_passed']
    assert independent['independent_full_day_sklearn_numpy_interp_passed']
    source = json.loads((args.runs / 'summary.json').read_text())
    assert source['test_evaluated'] is False
    for name, digest in source['model_hashes'].items():
        assert hashlib.sha256((args.runs / name).read_bytes()).hexdigest() == digest
    raw = pd.read_pickle(args.data)
    days = raw['days']['val']
    assert days == independent['days']
    assert set(source['fit_days'] + source['calibration_days']).isdisjoint(days + raw['days']['test'])
    batch = arrays(raw['data'], days)
    parent = FrozenDynamics(args.freeze)
    mode, block, alpha = source['selected'].split('_')
    bank = np.load(args.runs / 'pit_bank.npz')['pit']
    candidate = rollout(QuantileDynamics(args.runs), batch,
        paths(bank, days, int(block[1:]), uniform=mode == 'uniform'), float(alpha[1:]))
    frozen = rollout(parent.model, batch, paths(parent.bank, days, 95), .5)
    old = recursive(GBMDynamics(WORKSPACE / parent.manifest['original_environment_path']), batch)[:, None]
    truth = batch[0]
    persistence = np.broadcast_to(truth[None, None, :, 0, None, :], (1, 1, len(days), 95, 5))
    prior = json.loads((ROOT / 'Lee/experiments/e40/event_reference_audit.json').read_text())
    result = {'test_evaluated': False, 'model_refit': False, 'selection_changed': False,
              'selected': source['selected'], 'source_commit': source['code_commit'],
              'model_members': 5, 'policy_seeds': [],
              'scope': 'Exploratory reused validation; no environment adoption or controller improvement claim.',
              'onset_definition': 'Within-day one-to-one onset matching, excluding initial ongoing events; threshold.5, tolerances0/1/4.',
              'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
              'metrics': {}, 'paired_date_brier_differences': {}}
    errors = {}
    for name, trajectory in [('candidate', candidate), ('e40', frozen), ('old', old), ('persistence', persistence)]:
        result['metrics'][name], errors[name] = event_metrics(trajectory, truth)
        if name != 'candidate':
            for label in ('nh4', 'do'):
                for key in ('brier', 'average_precision', 'roc_auc'):
                    np.testing.assert_allclose(result['metrics'][name][label][key], prior['metrics'][name][label][key], rtol=1e-12)
        else:
            for label in ('nh4', 'do'):
                np.testing.assert_allclose(result['metrics'][name][label]['brier'],
                    source['validation'][source['selected']][label + '_brier'], rtol=1e-12)
    rng = np.random.default_rng(20261112)
    starts = rng.integers(0, len(days), size=(10000, 8))
    indices = ((starts[:, :, None] + np.arange(5)) % len(days)).reshape(10000, -1)[:, :len(days)]
    for baseline in ('e40', 'old', 'persistence'):
        result['paired_date_brier_differences'][baseline] = {}
        for label in ('nh4', 'do'):
            difference = errors['candidate'][label] - errors[baseline][label]
            result['paired_date_brier_differences'][baseline][label] = {
                'mean': float(difference.mean()),
                'circular_five_day_bootstrap95_ci': np.quantile(difference[indices].mean(1), [.025, .975]).tolist()}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result['paired_date_brier_differences']), flush=True)


if __name__ == '__main__':
    main()

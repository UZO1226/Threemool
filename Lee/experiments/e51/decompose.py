"""Separate PIT-bank and alpha effects without reselecting the primary candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/analysis'))
from audit_long_history_events import event_metrics
from calibrate_blend import prepare
from quantile_env import FrozenDynamics
import quantile_dynamics as q


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'reference-summary', 'freeze', 'runs', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    summary_path = args.runs / 'summary.json'
    summary_digest = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    source = json.loads(summary_path.read_text())
    assert source['selected'] == 'pit_b95_a0.25' and source['weight_fitting'] is False
    parent = FrozenDynamics(args.freeze)
    original = json.loads((parent.path / 'summary.json').read_text())
    assert original['selected'] == 'pit_b95_a0.5'
    assert source['model_hashes'] == original['model_hashes']
    for name, digest in source['model_hashes'].items():
        assert hashlib.sha256((args.runs / name).read_bytes()).hexdigest() == digest
    raw = pd.read_pickle(args.data)
    ref = json.loads(args.reference_summary.read_text())
    banks, _ = prepare(parent, raw, ref)
    np.testing.assert_array_equal(np.load(args.runs / 'pit_bank.npz')['pit'], banks[.25])
    days = raw['days']['val']
    batch = q.arrays(raw['data'], days)
    sd = np.concatenate([raw['data'].loc[day][q.STATE].to_numpy(float) for day in ref['fit_days']]).std(0)
    result = {'test_evaluated': False, 'model_fitted': False, 'selection_changed': False,
              'diagnostic_defined_after_primary_outcome': True,
              'primary_summary_sha256': summary_digest,
              'diagnostic_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'data_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
              'primary_candidate': source['selected'], 'primary_parent': original['selected'],
              'scope': 'Exploratory fixed two-by-two bank/alpha decomposition. No cell is selected for adoption.',
              'metrics': {}, 'paired_date_brier_differences': {}}
    errors = {}
    for kind in ('original', 'matched'):
        for alpha in (.25, .5):
            key = f'{kind}_bank_alpha{alpha}'
            bank = parent.bank if kind == 'original' else banks[alpha]
            prediction = q.rollout(parent.model, batch, q.paths(bank, days, 95), alpha)
            score = q.summarize(prediction, batch[0], sd)
            events, errors[key] = event_metrics(prediction, batch[0])
            result['metrics'][key] = {'normalized_mae': score['normalized_mae'],
                'nh4_mae': score['mae']['nh4'], 'nh4_frequency': score['nh4_violation']['pred'],
                'do_frequency': score['do_violation']['pred'], 'events': events}
            if (kind, alpha) in [('original', .5), ('matched', .25)]:
                known_source = original if kind == 'original' else source
                known = known_source['validation'][known_source['selected']]
                for metric in ('normalized_mae', 'nh4_brier', 'do_brier'):
                    np.testing.assert_allclose(score[metric], known[metric], rtol=1e-12)
            print('DECOMPOSED', key, flush=True)
    rng = np.random.default_rng(20261112)
    starts = rng.integers(0, len(days), size=(10000, 8))
    indices = ((starts[:, :, None] + np.arange(5)) % len(days)).reshape(10000, -1)[:, :len(days)]
    comparisons = [('matched_bank_alpha0.25', 'original_bank_alpha0.25'),
                   ('matched_bank_alpha0.5', 'original_bank_alpha0.5'),
                   ('original_bank_alpha0.25', 'original_bank_alpha0.5'),
                   ('matched_bank_alpha0.25', 'matched_bank_alpha0.5')]
    for candidate, baseline in comparisons:
        row = {}
        for label in ('nh4', 'do'):
            difference = errors[candidate][label] - errors[baseline][label]
            row[label] = {'mean': float(difference.mean()),
                'circular_five_day_bootstrap95_ci': np.quantile(difference[indices].mean(1), [.025, .975]).tolist()}
        result['paired_date_brier_differences'][candidate + '_minus_' + baseline] = row
    assert hashlib.sha256(summary_path.read_bytes()).hexdigest() == summary_digest
    result['primary_summary_unchanged'] = True
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result['paired_date_brier_differences']), flush=True)


if __name__ == '__main__':
    main()

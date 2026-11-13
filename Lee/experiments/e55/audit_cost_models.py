"""Reconstruct held training-stream probabilities and date-paired diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np

from fit_models import read_capture, metrics, verify
from policy_io import load_correction
from cost_model import features


def main():
    parser = argparse.ArgumentParser()
    for name in ('run', 'data', 'correction', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--allow-smoke', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    config = json.loads((args.run / 'effective_config.json').read_text())
    models, stored = load_correction(args.correction, config, args.allow_smoke)
    directory = Path(config['experiment_metadata']['passive_capture']['directory'])
    fit, held, hashes = read_capture(directory, args.data, args.allow_smoke)
    assert hashes == stored['capture_file_sha256']
    for subset in (fit, held):
        np.testing.assert_allclose(subset['next_obs'][:, 10], (subset['action'] + 1) / 2,
                                   atol=1e-7, rtol=1e-7)
    x = features(held['obs'], held['action'])
    direct = np.column_stack((held['obs'], held['action']))
    direct[:, 1] = 0
    np.testing.assert_array_equal(x, direct)
    verification = verify(models, x[np.linspace(0, len(x) - 1, min(len(x), 10000), dtype=int)])
    days = np.unique(held['day'])
    rng = np.random.default_rng(5502)
    bootstrap = rng.integers(0, len(days), size=(10000, len(days)))
    reports = {}
    for index, label in enumerate(('nh4', 'do')):
        y = held[label]
        probability = models[index].predict_proba(direct)[:, 1]
        assert metrics(y, probability) == stored['label_diagnostics'][label]['held']
        current = held['obs'][:, 0] > 4 if index == 0 else held['obs'][:, 2:5].max(-1) > 2
        controls = {'current_state_persistence': current.astype(float),
                    'fit_prevalence_constant': np.full(len(x), fit[label].mean())}
        contrasts = {}
        for name, control in controls.items():
            assert metrics(y, control) == stored['label_diagnostics'][label][name]
            difference = np.square(probability - y) - np.square(control - y)
            sums = np.array([difference[held['day'] == day].sum() for day in days])
            counts = np.array([(held['day'] == day).sum() for day in days])
            distribution = sums[bootstrap].sum(1) / counts[bootstrap].sum(1)
            contrasts[name] = {'brier_difference': float(difference.mean()),
                               'paired_date_bootstrap95_ci': np.quantile(distribution, [.025, .975]).tolist()}
        new = ~current
        new_metrics = metrics(y[new], probability[new]) if new.any() else {
            'n': 0, 'positive': 0, 'prevalence': None, 'brier': None,
            'average_precision': None, 'roc_auc': None, 'recall_at_half': None}
        reports[label] = {'all': metrics(y, probability),
                          'new_event_subset': new_metrics,
                          'new_event_count': int(y[new].sum()),
                          'new_event_recall_count_at_half': int(np.count_nonzero((y[new] == 1) & (probability[new] >= .5))),
                          'brier_contrasts': contrasts}
    result = {'passed': True, 'policy_seed': config['seed'], 'test_evaluated': False,
              'smoke_only': args.allow_smoke, 'fit_transitions': len(fit['obs']),
              'held_transitions': len(x), 'held_training_dates': len(days),
              'public_features_exact': True, 'terminal_executed_air_exact_tolerance': 1e-7,
              'held_metrics_reproduced_exact': True, 'verification': verification,
              'labels': reports, 'selection_or_refitting': False,
              'uncertainty_scope': '10000 resamples of held on-policy training dates, conditional on this fitted policy/model. Not policy-seed or plant-generalization uncertainty.'}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('COST_MODEL_AUDIT_PASS', config['seed'], flush=True)


if __name__ == '__main__':
    main()

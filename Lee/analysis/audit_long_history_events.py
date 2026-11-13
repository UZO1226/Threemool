"""Audit causal-history event forecasts without selecting new settings."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e41'))
from long_context import LongContextData, LongContextModel, PersistentContextModel
from latent import Data, precompute, simulate, noise_paths, summarize
sys.path.insert(0, str(ROOT / 'Lee/experiments/e42'))
from quantile_env import FrozenDynamics
from quantile_dynamics import arrays, paths, rollout
from event_timing import onset_counts


def reconstruct(directory, data, model_class, days):
    source = json.loads((directory / 'summary.json').read_text())
    models = []
    for member in range(5):
        checkpoint = directory / f"epoch{source['selected_epoch']}_m{member}.pt"
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == source['checkpoint_hashes'][checkpoint.name]
        model = model_class(data.sd, data.mean, data.mean_air, data.hi)
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        model.eval()
        models.append(model)
    cache = precompute(models, data, days, 16, row_sum_bound=.995)
    residual = np.load(directory / 'residual_bank.npz')['residual']
    trajectory = simulate(cache, np.array(source['bias']), noise_paths(residual, days), np.array(source['scale']))
    score = summarize(trajectory, cache['true'], data.sd)
    np.testing.assert_allclose(score['normalized_mae'], source['validation']['normalized_mae'], rtol=1e-12)
    return trajectory, cache['true'], source


def event_metrics(trajectory, truth):
    metrics, errors = {}, {}
    for label in ('nh4', 'do'):
        events = trajectory[..., 0] > 4 if label == 'nh4' else trajectory[..., 2:].max(-1) > 2
        target = truth[:, 1:, 0] > 4 if label == 'nh4' else truth[:, 1:, 2:].max(-1) > 2
        initial = truth[:, 0, 0] > 4 if label == 'nh4' else truth[:, 0, 2:].max(-1) > 2
        probability = events.mean((0, 1))
        positive = probability >= .5
        errors[label] = ((probability - target) ** 2).mean(1)
        onsets = {}
        for tolerance in (0, 1, 4):
            counts = np.sum([onset_counts(y, p, i, tolerance)
                             for y, p, i in zip(target, positive, initial)], axis=0)
            actual, predicted, matched, delay = counts
            onsets[str(tolerance)] = {
                'actual_onsets': int(actual), 'predicted_onsets': int(predicted), 'matched': int(matched),
                'recall': float(matched / actual) if actual else None,
                'precision': float(matched / predicted) if predicted else None,
                'mean_absolute_delay_steps': float(delay / matched) if matched else None}
        tp = int((target & positive).sum())
        fp = int((~target & positive).sum())
        fn = int((target & ~positive).sum())
        metrics[label] = {
            'prevalence': float(target.mean()), 'brier': float(errors[label].mean()),
            'average_precision': float(average_precision_score(target.ravel(), probability.ravel())),
            'roc_auc': float(roc_auc_score(target.ravel(), probability.ravel())),
            'precision_at_half': tp / (tp + fp) if tp + fp else None,
            'recall_at_half': tp / (tp + fn) if tp + fn else None,
            'onset_tolerance_steps': onsets}
    return metrics, errors


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'reference-summary', 'runs', 'control', 'freeze', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(1)
    raw = pd.read_pickle(args.data)
    reference = json.loads(args.reference_summary.read_text())
    days = raw['days']['val']
    candidate, truth, source = reconstruct(args.runs, LongContextData(raw, reference['fit_days']), LongContextModel, days)
    control, control_truth, _ = reconstruct(args.control, Data(raw, reference['fit_days']), PersistentContextModel, days)
    np.testing.assert_array_equal(truth, control_truth)
    dyn = FrozenDynamics(args.freeze)
    batch = arrays(raw['data'], days)
    np.testing.assert_array_equal(truth, batch[0])
    frozen = rollout(dyn.model, batch, paths(dyn.bank, days, 95), .5)
    persistence = np.broadcast_to(truth[None, None, :, 0, None, :], (1, 1, len(days), 95, 5))
    prior = json.loads((ROOT / 'Lee/experiments/e40/event_reference_audit.json').read_text())
    result = {
        'test_evaluated': False, 'model_refit': False, 'selection_changed': False,
        'model_seeds': list(range(5)), 'policy_seeds': [],
        'selected_epoch': source['selected_epoch'], 'selected_scale': source['selected_scale'],
        'input_sha256': hashlib.sha256(args.data.read_bytes()).hexdigest(),
        'run_summary_sha256': {name: hashlib.sha256((directory / 'summary.json').read_bytes()).hexdigest()
                               for name, directory in [('e41', args.runs), ('e34', args.control)]},
        'freeze_file_sha256': hashlib.sha256(args.freeze.read_bytes()).hexdigest(),
        'scope': 'Exploratory reused validation. No adoption or policy improvement claim.',
        'onset_definition': 'Within-day one-to-one matching, excluding ongoing initial events, threshold.5 and tolerances0/1/4 steps.',
        'metrics': {}, 'paired_date_brier_differences': {}}
    errors = {}
    for name, trajectory in [('e41', candidate), ('e34', control), ('e40', frozen), ('persistence', persistence)]:
        result['metrics'][name], errors[name] = event_metrics(trajectory, truth)
        if name in ('e40', 'persistence'):
            for label in ('nh4', 'do'):
                for key in ('brier', 'average_precision', 'roc_auc'):
                    np.testing.assert_allclose(result['metrics'][name][label][key], prior['metrics'][name][label][key], rtol=1e-12)
    rng = np.random.default_rng(20261112)
    starts = rng.integers(0, len(days), size=(10000, 8))
    indices = ((starts[:, :, None] + np.arange(5)) % len(days)).reshape(10000, -1)[:, :len(days)]
    for baseline in ('e34', 'e40', 'persistence'):
        result['paired_date_brier_differences'][baseline] = {}
        for label in ('nh4', 'do'):
            difference = errors['e41'][label] - errors[baseline][label]
            result['paired_date_brier_differences'][baseline][label] = {
                'mean': float(difference.mean()),
                'circular_five_day_bootstrap95_ci': np.quantile(difference[indices].mean(1), [.025, .975]).tolist()}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result['paired_date_brier_differences']), flush=True)


if __name__ == '__main__':
    main()

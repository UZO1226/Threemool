"""Physical next-DO regression, empirical upper margins and held diagnostics."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

from forecast_data import ROOT
from rate_model import FastRate, ConstantRate


def fit_model(x, y):
    if np.ptp(y) == 0:
        return ConstantRate(y[0])
    signs = np.zeros(12, dtype=int)
    signs[0] = -1
    signs[[2, 3, 4, 10, 11]] = 1
    return HistGradientBoostingRegressor(loss='squared_error', max_iter=200, learning_rate=.05,
        max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1, early_stopping=False,
        random_state=7500, monotonic_cst=signs).fit(x, y)


def margin(y, point):
    return float(max(0., np.quantile(np.asarray(y) - np.asarray(point), .95, method='higher')))


def mean_prediction(model, x):
    return np.maximum(model.predict(x), 0.)


def compiled_mean(model, x):
    return np.maximum(FastRate([model]).raw(x)[:, 0], 0.)


def upper_admissible(upper, reference):
    return np.asarray(upper) <= np.maximum(2., np.asarray(reference)) + 1e-12


def diagnostics(y, point, upper, current):
    error = point - y
    residual = y - upper
    event, alarm = y > 2, upper > 2
    safe = current <= 2
    onset = safe & event
    return {'n': len(y), 'mse': float(np.mean(error ** 2)), 'mae': float(np.mean(np.abs(error))),
        'mean_bias': float(error.mean()), 'target_mean': float(y.mean()), 'prediction_mean': float(point.mean()),
        'upper_pinball95': float(np.mean(np.maximum(.95 * residual, -.05 * residual))),
        'upper_coverage': float(np.mean(y <= upper)), 'next_do_events': int(event.sum()),
        'false_negatives': int(np.count_nonzero(event & ~alarm)),
        'false_positives': int(np.count_nonzero(~event & alarm)),
        'onset_events': int(onset.sum()), 'onset_false_negatives': int(np.count_nonzero(onset & ~alarm)),
        'onset_true_positives': int(np.count_nonzero(onset & alarm)),
        'upper_ranking_ap': float(average_precision_score(event, upper)) if event.any() else None,
        'upper_ranking_auc': float(roc_auc_score(event, upper)) if len(np.unique(event)) == 2 else None}


def held_report(fit, calibration, held, model, adjustment):
    y, x = held['y'], held['x']
    point = mean_prediction(model, x)
    current = x[:, 2:5].max(-1)
    persistence_margin = margin(calibration['y'], calibration['x'][:, 2:5].max(-1))
    constant = np.full(len(y), fit['y'].mean())
    constant_margin = margin(calibration['y'], np.full(len(calibration['y']), fit['y'].mean()))
    forecasts = {'model': (point, point + adjustment),
        'current_state_persistence': (current, current + persistence_margin),
        'fitting_mean_constant': (constant, constant + constant_margin)}
    values = {name: diagnostics(y, p, u, current) for name, (p, u) in forecasts.items()}
    days, codes = np.unique(held['day'], return_inverse=True)
    counts = np.bincount(codes)
    ix = np.random.default_rng(7502).integers(0, len(days), size=(10000, len(days)))
    contrasts = {}
    for name in ('current_state_persistence', 'fitting_mean_constant'):
        difference = (point - y) ** 2 - (forecasts[name][0] - y) ** 2
        sums = np.bincount(codes, weights=difference)
        samples = sums[ix].sum(1) / counts[ix].sum(1)
        contrasts[name] = {'mean_mse_difference': float(difference.mean()),
            'paired_training_date95_ci': np.quantile(samples, [.025, .975]).tolist()}
    covered = np.bincount(codes, weights=(y <= point + adjustment).astype(float))
    coverage_interval = np.quantile(covered[ix].sum(1) / counts[ix].sum(1), [.025, .975])
    return {'forecasts': values, 'mse_contrasts': contrasts, 'held_training_dates': len(days),
        'upper_coverage_training_date95_ci': coverage_interval.tolist(),
        'persistence_margin': persistence_margin, 'constant_margin': constant_margin,
        'uncertainty_scope': 'Conditional held-training-date resamples, not policy seeds, off-policy coverage or physical safety.'}

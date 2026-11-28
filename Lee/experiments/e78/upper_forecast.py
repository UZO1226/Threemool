"""Calibrate conditional uncertainty while preserving persistence alarms."""
import numpy as np

EDGES = np.array([1., 1.5, 2.])


def strata(current):
    current = np.asarray(current)
    assert np.isfinite(current).all() and (current >= 0).all()
    return np.searchsorted(EDGES, current, side='right')


def positive_quantile(values, q):
    return float(max(0., np.quantile(values, q, method='higher')))


def calibrate_upper(y, point, current):
    y, point, current = map(np.asarray, (y, point, current))
    assert y.shape == point.shape == current.shape and len(y)
    assert np.isfinite(y).all() and np.isfinite(point).all()
    residual = y - point
    global_margin = positive_quantile(residual, .975)
    groups = strata(current)
    margins = [max(global_margin, positive_quantile(residual[groups == k], .975))
               if np.any(groups == k) else global_margin for k in range(4)]
    return {'version': 1, 'quantile': .975, 'edges': EDGES.tolist(),
        'global_margin': global_margin, 'stratum_margins': margins,
        'stratum_counts': [int(np.count_nonzero(groups == k)) for k in range(4)],
        'persistence_margin': positive_quantile(y - current, .95)}


def upper_prediction(point, current, config):
    assert config['version'] == 1 and config['quantile'] == .975
    assert config['edges'] == EDGES.tolist()
    margins = np.asarray(config['stratum_margins'])
    assert margins.shape == (4,) and np.isfinite(margins).all()
    assert np.all(margins >= config['global_margin']) and config['global_margin'] >= 0
    assert np.isfinite(config['persistence_margin']) and config['persistence_margin'] >= 0
    current, point = np.asarray(current), np.asarray(point)
    assert current.shape == point.shape and np.isfinite(point).all()
    return np.maximum(point + margins[strata(current)], current + config['persistence_margin'])

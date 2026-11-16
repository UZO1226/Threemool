"""Bounded squared-loss cost-rate models and the fixed local controller."""
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT / 'Lee/experiments/e55'))
from fast_cost import logits
from targets import public_features


class ConstantRate:
    def __init__(self, value):
        self.value = float(value)

    def predict(self, x):
        return np.full(len(x), self.value)


def fit_models(x, targets):
    assert x.shape[1] == 12 and targets.shape == (len(x), 2)
    assert np.isfinite(x).all() and np.isfinite(targets).all()
    assert np.all((targets >= 0) & (targets <= 1))
    signs = np.zeros(12, dtype=int)
    signs[0] = 1
    signs[[2, 3, 4, 10, 11]] = -1
    models = []
    for index, constraint in enumerate((signs, -signs)):
        y = targets[:, index]
        if np.ptp(y) == 0:
            model = ConstantRate(y[0])
        else:
            model = HistGradientBoostingRegressor(
                loss='squared_error', max_iter=200, learning_rate=.05,
                max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1,
                early_stopping=False, random_state=5600 + index,
                monotonic_cst=constraint,
            ).fit(x, y)
        models.append(model)
    return models


class FastRate:
    def __init__(self, models):
        nodes, roots, offsets, baseline = [], [], [0], []
        count = 0
        for model in models:
            if isinstance(model, ConstantRate):
                baseline.append(model.value)
            else:
                assert model.is_categorical_ is None or not np.any(model.is_categorical_)
                baseline.append(float(model._baseline_prediction[0, 0]))
                for predictors in model._predictors:
                    assert len(predictors) == 1
                    tree = predictors[0].nodes
                    assert not tree['is_categorical'].any()
                    roots.append(count)
                    nodes.append(tree)
                    count += len(tree)
            offsets.append(len(roots))
        if nodes:
            joined = np.concatenate(nodes)
            arrays = [np.ascontiguousarray(joined[key]) for key in
                      ('feature_idx', 'num_threshold', 'left', 'right', 'value', 'is_leaf')]
        else:
            arrays = [np.empty(0, dtype=dtype) for dtype in (np.int64, float, np.int64, np.int64, float, bool)]
        self.args = [np.asarray(offsets), np.asarray(roots, dtype=np.int64), *arrays, np.asarray(baseline)]

    def raw(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 12 or not np.isfinite(x).all():
            raise ValueError('Finite public12 features required')
        return logits(x, *self.args)

    def predict(self, x):
        return np.clip(self.raw(x), 0, 1)


def slow_rates(models, x):
    return np.clip(np.column_stack([model.predict(x) for model in models]), 0, 1)


class RatePolicy:
    def __init__(self, base, models):
        self.base = base
        self.risk = FastRate(models)

    def act_det(self, obs):
        obs = np.asarray(obs)
        base = np.asarray(self.base.act_det(obs)).reshape(-1)
        candidates = np.clip(base[:, None] + np.array([0, -.125, .125, -.25, .25]), -1, 1)
        x = public_features(np.repeat(obs, 5, axis=0), candidates.reshape(-1))
        score = self.risk.predict(x).sum(-1).reshape(-1, 5) + (candidates - base[:, None]) ** 2
        return candidates[np.arange(len(obs)), np.argmin(score, axis=1)]

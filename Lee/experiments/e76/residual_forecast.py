"""Preserve current DO and learn its next-step change from existing captures."""
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e75'))
from forecast_data import read_blocks, sha
from do_forecast import margin, diagnostics, upper_admissible, held_report as absolute_held_report
from rate_model import FastRate, ConstantRate


def current_do(x):
    return np.asarray(x)[:, 2:5].max(-1)


def fit_model(x, y):
    change = y - current_do(x)
    if np.ptp(change) == 0:
        return ConstantRate(change[0])
    signs = np.zeros(12, dtype=int)
    signs[0] = -1
    signs[[10, 11]] = 1
    return HistGradientBoostingRegressor(loss='squared_error', max_iter=200, learning_rate=.05,
        max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1, early_stopping=False,
        random_state=7500, monotonic_cst=signs).fit(x, change)


def mean_prediction(model, x):
    return np.maximum(current_do(x) + model.predict(x), 0.)


def compiled_mean(model, x):
    return np.maximum(current_do(x) + FastRate([model]).raw(x)[:, 0], 0.)


class PhysicalView:
    def __init__(self, change_model):
        self.change_model = change_model

    def predict(self, x):
        return current_do(x) + self.change_model.predict(x)


def held_report(fit, calibration, held, model, adjustment):
    return absolute_held_report(fit, calibration, held, PhysicalView(model), adjustment)

"""Fixed training-stream classifiers and deterministic local action correction."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


def features(obs, action):
    obs = np.asarray(obs, float)
    assert obs.ndim == 2 and obs.shape[1] == 11
    x = np.column_stack((obs, np.asarray(action).reshape(-1)))
    x[:, 1] = 0
    assert np.isfinite(x).all()
    return x


class ConstantProbability:
    def __init__(self, value):
        self.value = float(value)

    def predict_proba(self, x):
        return np.tile([1 - self.value, self.value], (len(x), 1))


def fit_cost_models(obs, action, nh4, do):
    x = features(obs, action)
    signs = np.zeros(12, dtype=int)
    signs[0] = 1
    signs[[2, 3, 4, 10, 11]] = -1
    models = []
    for label, seed, constraint in ((nh4, 5500, signs), (do, 5501, -signs)):
        y = np.asarray(label).reshape(-1)
        assert len(y) == len(x) and np.isin(y, [0, 1]).all()
        if len(np.unique(y)) == 1:
            model = ConstantProbability(y[0])
        else:
            model = HistGradientBoostingClassifier(
                max_iter=200, learning_rate=.05, max_leaf_nodes=15,
                min_samples_leaf=40, l2_regularization=1,
                early_stopping=False, random_state=seed,
                monotonic_cst=constraint,
            ).fit(x, y)
            for trees in model._predictors:
                for tree in trees:
                    nodes = tree.nodes
                    assert not np.any((nodes['is_leaf'] == 0) & (nodes['feature_idx'] == 1))
        models.append(model)
    return models


class CorrectedPolicy:
    def __init__(self, base, models):
        self.base = base
        self.models = models

    def act_det(self, obs):
        obs = np.asarray(obs)
        base = np.asarray(self.base.act_det(obs)).reshape(-1)
        candidates = np.clip(base[:, None] + np.array([0, -.125, .125, -.25, .25]), -1, 1)
        x = features(np.repeat(obs, 5, axis=0), candidates.reshape(-1))
        risk = sum(model.predict_proba(x)[:, 1] for model in self.models).reshape(-1, 5)
        score = risk + (candidates - base[:, None]) ** 2
        return candidates[np.arange(len(obs)), np.argmin(score, axis=1)]

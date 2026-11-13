"""Finite numeric binary-HGB inference, with independently checked sklearn logits."""
import numpy as np
from numba import njit


@njit(cache=True)
def logits(x, offsets, roots, feature, threshold, left, right, value, leaf, baseline):
    out = np.empty((len(x), len(baseline)))
    for i in range(len(x)):
        for model in range(len(baseline)):
            total = baseline[model]
            for tree in range(offsets[model], offsets[model + 1]):
                root = roots[tree]
                node = root
                while not leaf[node]:
                    node = root + (left[node] if x[i, feature[node]] <= threshold[node] else right[node])
                total += value[node]
            out[i, model] = total
    return out


class FastCost:
    def __init__(self, models):
        self.constants = {}
        nodes, roots, offsets, baseline = [], [], [0], []
        count = 0
        for index, model in enumerate(models):
            if hasattr(model, 'value'):
                self.constants[index] = model.value
                baseline.append(0.)
            else:
                assert np.array_equal(model.classes_, [0, 1])
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

    def decision_function(self, x):
        x = np.asarray(x, dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError('Fast classifier requires finite public features')
        return logits(x, *self.args)

    def probabilities(self, x):
        z = self.decision_function(x)
        p = np.empty_like(z)
        positive = z >= 0
        p[positive] = 1 / (1 + np.exp(-z[positive]))
        exp_negative = np.exp(z[~positive])
        p[~positive] = exp_negative / (1 + exp_negative)
        for index, value in self.constants.items():
            p[:, index] = value
        return p


class FastCorrectedPolicy:
    def __init__(self, base, models):
        self.base = base
        self.risk = FastCost(models)

    def act_det(self, obs):
        from cost_model import features
        obs = np.asarray(obs)
        base = np.asarray(self.base.act_det(obs)).reshape(-1)
        candidates = np.clip(base[:, None] + np.array([0, -.125, .125, -.25, .25]), -1, 1)
        x = features(np.repeat(obs, 5, axis=0), candidates.reshape(-1))
        probability = self.risk.probabilities(x).sum(-1).reshape(-1, 5)
        score = probability + (candidates - base[:, None]) ** 2
        return candidates[np.arange(len(obs)), np.argmin(score, axis=1)]

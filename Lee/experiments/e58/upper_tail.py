"""Fixed DO-only upper-tail kernel; historical backbone remains unchanged."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e38'))
import quantile_dynamics as q

LEVELS = np.r_[q.LEVELS, .99]
BACKBONE_INTERPOLATE = q.interpolate


def interpolate(values, u):
    """Retain five-level arithmetic exactly and add only the DO extension."""
    if len(values) == 5:
        return BACKBONE_INTERPOLATE(values, u)
    if len(values) != 6:
        raise ValueError('Expected five backbone levels and one upper endpoint')
    result = BACKBONE_INTERPOLATE(values[:5], u)
    fraction = np.clip((u[:, 2:] - .95) / (.99 - .95), 0, 1)
    result[:, 2:] += fraction * np.maximum(values[5, :, 2:] - values[4, :, 2:], 0)
    return result


def pit(values, target, rng):
    """Randomized CDF with explicit atoms at equal adjacent endpoints."""
    if len(values) != 6:
        raise ValueError('Expected six conditional quantile endpoints')
    index = np.clip((values <= target[None]).sum(0) - 1, 0, len(LEVELS) - 2)
    low = np.take_along_axis(values, index[None], axis=0)[0]
    high = np.take_along_axis(values, (index + 1)[None], axis=0)[0]
    fraction = np.clip((target - low) / np.maximum(high - low, 1e-12), 0, 1)
    u = LEVELS[index] + fraction * (LEVELS[index + 1] - LEVELS[index])
    u = np.where(target < values[0], 0, np.where(target > values[-1], 1, u))
    equal = np.isclose(values, target[None], rtol=0, atol=1e-10)
    ties = equal.any(0)
    left = np.where(equal, LEVELS[:, None, None], np.inf).min(0)
    right = np.where(equal, LEVELS[:, None, None], -np.inf).max(0)
    left = np.where(equal[0], 0, left)
    right = np.where(equal[-1], 1, right)
    left = np.where(ties, left, 0)
    right = np.where(ties, right, 0)
    return np.where(ties, left + rng.random(target.shape) * (right - left), u)


class UpperDynamics:
    def __init__(self, backbone, upper=None):
        self.backbone = backbone
        self.upper = upper
        self.hi = backbone.hi
        self.n_members = backbone.n_members
        self.models = backbone.models + ([] if upper is None else [upper])

    def quantiles(self, x, members):
        central = self.backbone.quantiles(x, members)
        endpoint = central[-1].copy()
        if self.upper is not None:
            proposed = np.clip(x[:, :5] + self.upper.delta(x, members), 0, self.hi)
            endpoint[:, 2:] = np.maximum(endpoint[:, 2:], proposed[:, 2:])
        return np.concatenate([central, endpoint[None]], axis=0)

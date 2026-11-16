"""Local common rollout for unchanged five-level and extended six-level models."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e58'))
from upper_tail import UpperDynamics, interpolate, q


def load_model(directory, parent):
    backbone = q.QuantileDynamics(directory)
    return backbone if parent == 'e51' else UpperDynamics(backbone, q.GBMDynamics(directory / 'upper_do_0.99.pkl'))


def rollout(dyn, batch, noise, draws=8, air_scale=1.):
    true, exo, zones = batch
    n = len(true)
    members = np.repeat(np.arange(5), draws * n)
    state = np.tile(true[:, 0], (5 * draws, 1))
    exo = np.tile(exo, (5 * draws, 1, 1))
    zones = zones * air_scale
    zones *= np.minimum(1, 9000 / np.maximum(zones.sum(-1, keepdims=True), 1))
    zones = np.tile(zones, (5 * draws, 1, 1))
    values = []
    for t in range(95):
        x = q.features(state, exo[:, t], zones[:, t], zones[:, max(0, t - 1)])
        sampled = interpolate(dyn.quantiles(x, members), noise[:, t])
        state = np.clip(.25 * state + .75 * sampled, 0, dyn.hi)
        values.append(state.copy())
    return np.stack(values, 1).reshape(5, draws, n, 95, 5)

"""Public current features and episode-safe short-horizon violation rates."""
import numpy as np


def public_features(obs, action):
    obs = np.asarray(obs, dtype=float)
    assert obs.ndim == 2 and obs.shape[1] == 11
    x = np.column_stack((obs, np.asarray(action).reshape(-1)))
    assert np.isfinite(x).all() and np.all(np.abs(x[:, -1]) <= 1)
    return x


def cost_rates(cost, done, horizon):
    cost = np.asarray(cost, dtype=float)
    done = np.asarray(done, dtype=bool)
    assert cost.ndim == 2 and cost.shape == done.shape
    assert isinstance(horizon, int) and 1 <= horizon <= 95
    assert np.isfinite(cost).all() and np.isin(cost, [0, 1]).all()
    total = np.zeros_like(cost)
    count = np.zeros(cost.shape, dtype=np.int64)
    active = np.ones(cost.shape, dtype=bool)
    for offset in range(min(horizon, len(cost))):
        length = len(cost) - offset
        eligible = active[:length]
        total[:length] += np.where(eligible, cost[offset:], 0)
        count[:length] += eligible
        active[:length] &= ~done[offset:]
        active[length:] = False
    assert np.all(count >= 1)
    rate = total / count
    assert np.isfinite(rate).all() and np.all((rate >= 0) & (rate <= 1))
    return rate, count

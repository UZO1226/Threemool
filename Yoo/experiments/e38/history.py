"""NH4 전이에만 15/30/45/60분 전 수질을 추가한다."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'e30'))
from surrogate_dw import features, build_arrays, STATE


def initial_history(state, t):
    return np.stack([state[max(t-lag, 0)] for lag in range(1, 5)])


def advance_history(history, current):
    return np.concatenate([current[:, None, :], history[:, :3]], axis=1)


def build_history(table, dates):
    x, y = build_arrays(table, dates)
    h = np.stack([initial_history(table.loc[d][STATE].to_numpy(float), t)
                  for d in dates for t in range(95)])
    return np.column_stack([x, h.reshape(len(x), -1)]), y


class HistoryDynamics:
    def __init__(self, base, history_models=None):
        self.base = base
        self.models = base['models']
        self.hi = np.asarray(base['hi'])
        self.n_members = len(self.models)
        self.history_models = history_models

    def step(self, state, exo, zone, previous_zone, members, history):
        x = features(state, exo, zone, previous_zone)
        xh = np.column_stack([x, history.reshape(len(x), -1)])
        delta = np.zeros_like(state)
        for m in np.unique(members):
            ids = members == m
            for j in range(5):
                model = self.history_models[m] if j == 0 and self.history_models is not None else self.models[m][j]
                delta[ids, j] = model.predict(xh[ids] if j == 0 and self.history_models is not None else x[ids])
        return np.clip(state + delta, 0, self.hi)


def replay(windows, days, dyn, horizon):
    n, m = len(windows), dyn.n_members
    initial = np.stack([days[d]['state'][t] for d, t in windows])
    state = np.repeat(initial, m, axis=0)
    history = np.repeat(np.stack([initial_history(days[d]['state'], t) for d, t in windows]), m, axis=0)
    previous = np.repeat(np.stack([days[d]['zone'][max(t-1, 0)] for d, t in windows]), m, axis=0)
    members = np.tile(np.arange(m), n)
    traj = [state.reshape(n, m, 5).copy()]
    for step in range(horizon):
        exo = np.repeat(np.stack([days[d]['exo'][t+step] for d,t in windows]), m, axis=0)
        zone = np.repeat(np.stack([days[d]['zone'][t+step] for d,t in windows]), m, axis=0)
        nxt = dyn.step(state, exo, zone, previous, members, history)
        history = advance_history(history, state)
        state, previous = nxt, zone
        assert np.isfinite(state).all()
        traj.append(state.reshape(n, m, 5).copy())
    return np.stack(traj, axis=2)

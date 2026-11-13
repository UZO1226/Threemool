"""Strict public-observation history with reset-safe terminal observations."""
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e42'))
import quantile_env
from quantile_env import FrozenDynamics, FrozenQuantileEnv as OriginalFrozenEnv
from history_agent import LAG_COLUMNS, MODES


class HistoryEnv:
    def __init__(self, data_path, dynamics, split='train', n_env=8, seed=0, mode='history'):
        assert mode in MODES
        self.base = OriginalFrozenEnv(data_path, dynamics, split=split, n_env=n_env, seed=seed)
        self.mode = mode
        self.current = None

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, 'base'), name)

    @property
    def fixed(self):
        return self.base.fixed

    @fixed.setter
    def fixed(self, value):
        self.base.fixed = value

    @property
    def fixed_draws(self):
        return self.base.fixed_draws

    @fixed_draws.setter
    def fixed_draws(self, value):
        self.base.fixed_draws = value

    def reset(self):
        observation = self.base.reset()
        self.current = observation.copy()
        return np.c_[observation, observation[:, LAG_COLUMNS]]

    def step(self, action):
        assert self.current is not None
        previous = self.current.copy()
        observation, reward, cost, done, terminal, info = self.base.step(action)
        if self.mode == 'history':
            lag = previous[:, LAG_COLUMNS].copy()
            terminal_lag = lag.copy()
            lag[done] = observation[done][:, LAG_COLUMNS]
        else:
            lag = observation[:, LAG_COLUMNS]
            terminal_lag = terminal[:, LAG_COLUMNS]
        self.current = observation.copy()
        return np.c_[observation, lag], reward, cost, done, np.c_[terminal, terminal_lag], info


def evaluate_history(data_path, dynamics, act_batch=None, replay=False, draws=8,
                     split='val', air_offset=0, mode='history'):
    assert isinstance(dynamics, FrozenDynamics) and mode in MODES
    class EvaluationHistoryEnv(HistoryEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, mode=mode)
    with patch.object(quantile_env, 'FrozenQuantileEnv', EvaluationHistoryEnv):
        return quantile_env.evaluate(data_path, dynamics, act_batch=act_batch, replay=replay,
            draws=draws, split=split, air_offset=air_offset)

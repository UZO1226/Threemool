"""Passive capture of transitions already consumed by the unchanged collector."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/code/darrow'))
sys.path.insert(0, str(ROOT / 'Lee/experiments/e42'))
from quantile_env import FrozenQuantileEnv


class CaptureEnv(FrozenQuantileEnv):
    def __init__(self, *args, capture_path, horizon=760, **kwargs):
        if kwargs.get('split', 'train') != 'train':
            raise ValueError('Passive captures are restricted to policy training')
        super().__init__(*args, **kwargs)
        self.capture_path = Path(capture_path)
        self.capture_horizon = horizon
        self.capture_steps = 0
        self.capture_chunks = []
        self.capture_current = []

    def step(self, action):
        before = self._obs().copy()
        day = np.asarray(self.day).copy()
        episode = self.episode_count.copy()
        within = self.t.copy()
        executed = np.clip(np.asarray(action, float).reshape(self.n), -1, 1)
        result = super().step(action)
        obs, reward, cost, done, terminal, info = result
        self.capture_current.append({
            'obs': before, 'action': executed.copy(), 'next_obs': terminal.copy(),
            'nh4': info['v_nh4'].copy(), 'do': info['v_do'].copy(),
            'done': done.copy(), 'day': day.astype(str), 'episode': episode,
            'within_episode': within,
        })
        self.capture_steps += 1
        if self.capture_steps % self.capture_horizon == 0:
            update = self.capture_steps // self.capture_horizon
            directory = self.capture_path
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f'update_{update:03d}.npz'
            if path.exists():
                raise FileExistsError(path)
            arrays = {key: np.stack([row[key] for row in self.capture_current])
                      for key in self.capture_current[0]}
            arrays['update'] = np.asarray(update)
            np.savez_compressed(path, **arrays)
            self.capture_chunks.append(path)
            self.capture_current.clear()
        return result

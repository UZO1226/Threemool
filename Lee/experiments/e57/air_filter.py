"""A fixed public-observation actuator filter, independent of the frozen world."""
import json
from pathlib import Path

import numpy as np


class AirFilter:
    def __init__(self, base, gain=.5):
        if gain not in (.5, 1.):
            raise ValueError('Only the fixed primary gain or identity null is allowed')
        self.base = base
        self.gain = float(gain)

    def act_det(self, obs):
        obs = np.atleast_2d(np.asarray(obs))
        if obs.shape[1] != 11 or not np.isfinite(obs).all():
            raise ValueError('Finite original11 public observations required')
        original = np.asarray(self.base.act_det(obs)).reshape(-1)
        assert len(original) == len(obs) and np.isfinite(original).all()
        assert np.all(np.abs(original) <= 1)
        if self.gain == 1.:
            # Preserve dtype too, including the original offset-rounding path.
            return original
        previous = np.clip(2 * obs[:, 10].astype(float) - 1, -1, 1)
        return np.clip(self.gain * original.astype(float) + (1 - self.gain) * previous, -1, 1)

    def config(self):
        return {'kind': 'executed_air_smoothing', 'gain': self.gain,
                'public_previous_air_index': 10, 'previous_air_normalization_m3_h': 9000,
                'clip_previous_normalized_action': [-1, 1], 'offset_applied_after_filter': True,
                'additional_policy_training': False, 'extra_environment_training_steps': 0}

    def save_config(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.config(), indent=2))

    @classmethod
    def load_config(cls, base, path):
        payload = json.loads(Path(path).read_text())
        policy = cls(base, payload['gain'])
        assert payload == policy.config(), 'Filter configuration changed'
        return policy

"""Nine nested fixed offsets inside a prospectively specified half radius."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e63'))
from trust_policy import TrustPolicy
from rate_model import FastRate
from targets import public_features

OFFSETS = [0, -.125, .125, -.25, .25, -.375, .375, -.5, .5]


def configuration():
    return {'version': 1, 'horizon': 4, 'candidate_offsets': OFFSETS.copy(),
            'action_clip': [-1, 1], 'tie_rule': 'first_candidate',
            'observation_dimension': 11, 'rate_output_clip': [0, 1],
            'penalty': 0, 'hard_action_radius': .5}


class RadiusPolicy:
    def __init__(self, base, models):
        self.base = base
        self.risk = FastRate(models)
        self.config = configuration()

    def act_det(self, obs):
        obs = np.asarray(obs)
        original = np.asarray(self.base.act_det(obs)).reshape(-1)
        if not np.isfinite(original).all() or np.any(np.abs(original) > 1):
            raise ValueError('Base policy must return finite bounded actions')
        candidates = np.clip(original[:, None] + np.asarray(OFFSETS), -1, 1)
        features = public_features(np.repeat(obs, len(OFFSETS), axis=0), candidates.reshape(-1))
        risk = self.risk.predict(features).sum(-1).reshape(-1, len(OFFSETS))
        return candidates[np.arange(len(obs)), risk.argmin(-1)]

    def save_config(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(self.config, indent=2, allow_nan=False))


def load_policy(base, models, path):
    if json.loads(Path(path).read_text()) != configuration():
        raise ValueError('Controller configuration differs from fixed specification')
    return RadiusPolicy(base, models)

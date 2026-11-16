"""Fixed risk-only correction inside the unchanged discrete action radius."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e56'))
from rate_model import FastRate
from targets import public_features

OFFSETS = [0, -.125, .125, -.25, .25]


def configuration(penalty=0):
    if isinstance(penalty, bool) or penalty not in (0, 1):
        raise ValueError('Only fixed primary0 and identity-null1 are supported')
    return {'version': 1, 'horizon': 4, 'penalty': penalty, 'candidate_offsets': OFFSETS.copy(),
            'action_clip': [-1, 1], 'tie_rule': 'first_candidate', 'observation_dimension': 11,
            'rate_output_clip': [0, 1]}


class TrustPolicy:
    def __init__(self, base, models, penalty=0):
        self.base = base
        self.risk = FastRate(models)
        self.config = configuration(penalty)

    def act_det(self, obs):
        obs = np.asarray(obs)
        original = np.asarray(self.base.act_det(obs)).reshape(-1)
        if not np.isfinite(original).all() or np.any(np.abs(original) > 1):
            raise ValueError('Base policy must return finite bounded actions')
        candidates = np.clip(original[:, None] + np.array(OFFSETS), -1, 1)
        x = public_features(np.repeat(obs, 5, axis=0), candidates.reshape(-1))
        score = self.risk.predict(x).sum(-1).reshape(-1, 5)
        if self.config['penalty'] == 1:
            score = score + np.square(candidates - original[:, None])
        return candidates[np.arange(len(obs)), np.argmin(score, axis=1)]

    def save_config(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(self.config, indent=2, allow_nan=False))


def load_policy(base, models, path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict) or value != configuration(value.get('penalty')):
        raise ValueError('Controller configuration differs from fixed specification')
    return TrustPolicy(base, models, value['penalty'])

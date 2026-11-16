"""Risk-guided policies with a training-only mean-action constraint."""
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
BOUND = 2 ** 26


def configuration(low, high, weight):
    if any(isinstance(value, (bool, np.bool_)) for value in (low, high, weight)):
        raise ValueError('Numeric projection parameters required')
    low, high, weight = map(float, (low, high, weight))
    if not all(np.isfinite([low, high, weight])) or not -BOUND <= low <= high <= BOUND or not 0 <= weight <= 1:
        raise ValueError('Invalid bounded projection parameters')
    return {'version': 1, 'horizon': 4, 'candidate_offsets': OFFSETS.copy(),
        'hard_action_radius': .5, 'action_clip': [-1, 1], 'tie_rule': 'first_candidate',
        'observation_dimension': 11, 'rate_output_clip': [0, 1],
        'lambda_low': low, 'lambda_high': high, 'low_policy_weight': weight,
        'calibration_scope': 'Updates1 through52, e63 native mean-action target only.'}


def choose(candidates, risk, multiplier):
    score = risk + multiplier * (candidates - candidates[:, :1])
    return candidates[np.arange(len(candidates)), score.argmin(-1)]


def project(candidates, risk, config):
    low = choose(candidates, risk, config['lambda_low'])
    if config['lambda_low'] == config['lambda_high']:
        return low
    high = choose(candidates, risk, config['lambda_high'])
    weight = config['low_policy_weight']
    return weight * low + (1 - weight) * high


def calibrate(candidates, risk, target):
    if candidates.ndim != 2 or candidates.shape != risk.shape or candidates.shape[1] != 9:
        raise ValueError('Expected matching nine-candidate arrays')
    assert np.isfinite(candidates).all() and np.isfinite(risk).all() and np.isfinite(target)
    zero = choose(candidates, risk, 0.)
    trace = [{'lambda': 0., 'mean_action': float(zero.mean())}]
    if abs(zero.mean() - target) <= 1e-12:
        config = configuration(0., 0., 1.)
    else:
        low, high = float(-BOUND), float(BOUND)
        a_low, a_high = choose(candidates, risk, low), choose(candidates, risk, high)
        trace.extend([{'lambda': low, 'mean_action': float(a_low.mean())},
                      {'lambda': high, 'mean_action': float(a_high.mean())}])
        assert a_high.mean() - 1e-12 <= target <= a_low.mean() + 1e-12
        for _ in range(64):
            middle = (low + high) / 2
            action = choose(candidates, risk, middle)
            mean = float(action.mean())
            trace.append({'lambda': middle, 'mean_action': mean})
            if mean > target:
                low, a_low = middle, action
            else:
                high, a_high = middle, action
        gap = float(a_low.mean() - a_high.mean())
        if gap == 0:
            assert abs(float(a_low.mean()) - target) <= 1e-10
            config = configuration(low, low, 1.)
        else:
            assert gap > 0
            weight = float(np.clip((target - a_high.mean()) / gap, 0, 1))
            config = configuration(low, high, weight)
    achieved = float(project(candidates, risk, config).mean())
    assert abs(achieved - target) <= 1e-10
    return config, {'target_mean_action': float(target), 'achieved_mean_action': achieved,
                    'absolute_mean_action_error': abs(achieved - target), 'trace': trace}


class BudgetPolicy:
    def __init__(self, base, models, config):
        if config != configuration(config.get('lambda_low'), config.get('lambda_high'), config.get('low_policy_weight')):
            raise ValueError('Controller specification differs')
        self.base, self.risk, self.config = base, FastRate(models), config.copy()

    def geometry(self, obs):
        obs = np.asarray(obs)
        original = np.asarray(self.base.act_det(obs)).reshape(-1)
        assert np.isfinite(original).all() and np.all(np.abs(original) <= 1)
        candidates = np.clip(original[:, None] + np.asarray(OFFSETS), -1, 1)
        x = public_features(np.repeat(obs, 9, axis=0), candidates.reshape(-1))
        risk = self.risk.predict(x).sum(-1).reshape(-1, 9)
        return candidates, risk

    def act_det(self, obs):
        candidates, risk = self.geometry(obs)
        return project(candidates, risk, self.config)

    def save_config(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        path.write_text(json.dumps(self.config, indent=2, allow_nan=False))


def load_policy(base, models, path):
    return BudgetPolicy(base, models, json.loads(Path(path).read_text()))

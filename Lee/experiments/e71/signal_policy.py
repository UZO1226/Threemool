"""Preserve e63 when expanded actions offer no strict predicted-rate gain."""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e69'))
from budget_policy import (BudgetPolicy, TrustPolicy, calibrate, public_features,
                           configuration as budget_configuration, project as budget_project)

GAIN = 1e-12


def configuration(low, high, weight, enabled=True):
    if type(enabled) is not bool:
        raise ValueError('Boolean calibration mode required')
    value = budget_configuration(low, high, weight)
    value.update({'version': 2, 'expanded_signal_gate': True, 'minimum_predicted_gain': GAIN,
        'reference_only': not enabled,
        'calibration_scope': 'Eligible states in updates1 through52, e63 native mean-action target only.'})
    return value


def eligible(risk):
    return risk[:, :5].min(-1) - risk.min(-1) > GAIN


def project(candidates, risk, config):
    previous = candidates[np.arange(len(candidates)), risk[:, :5].argmin(-1)]
    if config['reference_only']:
        return previous
    mask = eligible(risk)
    action = previous.copy()
    if mask.any():
        action[mask] = budget_project(candidates[mask], risk[mask], config)
    return action


class SignalPolicy(BudgetPolicy):
    def __init__(self, base, models, config):
        expected = configuration(config.get('lambda_low'), config.get('lambda_high'),
                                 config.get('low_policy_weight'), not config.get('reference_only'))
        if config != expected:
            raise ValueError('Signal controller specification differs')
        super().__init__(base, models, budget_configuration(config['lambda_low'],
            config['lambda_high'], config['low_policy_weight']))
        self.config = config.copy()

    def act_det(self, obs):
        candidates, risk = self.geometry(obs)
        return project(candidates, risk, self.config)


def load_policy(base, models, path):
    return SignalPolicy(base, models, json.loads(Path(path).read_text()))

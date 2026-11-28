"""Center admissible mean-preserving proposals on authenticated matched e63."""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e69'))
from budget_policy import (BudgetPolicy, TrustPolicy, calibrate, public_features,
                           project as budget_project, configuration as budget_configuration, OFFSETS)

TOL = 1e-12


def configuration(low, high, weight, reference_offset, enabled=True):
    if type(enabled) is not bool:
        raise ValueError('Boolean reference mode required')
    if isinstance(reference_offset, (bool, np.bool_)) or not isinstance(reference_offset, (int, float, np.integer, np.floating)):
        raise ValueError('Numeric reference offset required')
    reference_offset = float(reference_offset)
    if not np.isfinite(reference_offset) or not -2 <= reference_offset <= 2:
        raise ValueError('Invalid authenticated reference offset')
    value = budget_configuration(low, high, weight)
    value.update({'version': 4, 'anchor_reference': 'authenticated_matched_e63',
        'candidate_center': 'matched_reference_action', 'reference_fixed_offset': reference_offset,
        'componentwise_predicted_guard': True, 'component_tolerance': TOL,
        'minimum_predicted_gain': TOL, 'reference_only': not enabled,
        'calibration_scope': 'Admissible column0-gain states in updates1 through52, authenticated matched-reference mean-action target only.'})
    return value


def guard_geometry(candidates, components):
    candidates, components = np.asarray(candidates), np.asarray(components)
    assert candidates.ndim == 2 and candidates.shape[1] == 9
    assert components.shape == (*candidates.shape, 2)
    assert np.isfinite(candidates).all() and np.isfinite(components).all()
    reference = components[:, 0]
    total = components.sum(-1)
    admissible = np.all(components <= reference[:, None, :] + TOL, axis=-1)
    return (np.where(admissible, candidates, candidates[:, :1]),
            np.where(admissible, total, total[:, :1]))


def eligible(risk):
    return risk[:, 0] - risk.min(-1) > TOL


def project(candidates, risk, config):
    action = candidates[:, 0].copy()
    if config['reference_only']:
        return action
    mask = eligible(risk)
    if mask.any():
        action[mask] = budget_project(candidates[mask], risk[mask], config)
    return action


class AnchorPolicy(BudgetPolicy):
    def __init__(self, base, models, config):
        if type(config.get('reference_only')) is not bool:
            raise ValueError('Boolean reference mode required')
        expected = configuration(config.get('lambda_low'), config.get('lambda_high'),
            config.get('low_policy_weight'), config.get('reference_fixed_offset'), not config['reference_only'])
        if config != expected:
            raise ValueError('Anchored controller specification differs')
        super().__init__(base, models, budget_configuration(config['lambda_low'], config['lambda_high'], config['low_policy_weight']))
        self.reference = TrustPolicy(base, models, 0)
        self.config = config.copy()

    def reference_action(self, obs):
        return np.clip(self.reference.act_det(obs) + self.config['reference_fixed_offset'], -1, 1)

    def raw_geometry(self, obs):
        obs = np.asarray(obs)
        reference = self.reference_action(obs)
        assert np.isfinite(reference).all() and np.all(np.abs(reference) <= 1)
        candidates = np.clip(reference[:, None] + np.asarray(OFFSETS), -1, 1)
        x = public_features(np.repeat(obs, 9, axis=0), candidates.reshape(-1))
        components = self.risk.predict(x).reshape(-1, 9, 2)
        return candidates, components

    def geometry(self, obs):
        return guard_geometry(*self.raw_geometry(obs))

    def act_det(self, obs):
        if self.config['reference_only']:
            return self.reference_action(obs)
        return project(*self.geometry(obs), self.config)


def load_policy(base, models, path):
    return AnchorPolicy(base, models, json.loads(Path(path).read_text()))

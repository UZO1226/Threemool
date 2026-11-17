"""Exclude proposals increasing either predicted component before mean calibration."""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e71'))
from signal_policy import eligible, project, calibrate, public_features, TrustPolicy
from budget_policy import BudgetPolicy, configuration as budget_configuration, OFFSETS

TOL = 1e-12


def configuration(low, high, weight, enabled=True):
    if type(enabled) is not bool:
        raise ValueError('Boolean reference mode required')
    value = budget_configuration(low, high, weight)
    value.update({'version': 3, 'expanded_signal_gate': True, 'minimum_predicted_gain': TOL,
        'componentwise_predicted_guard': True, 'component_tolerance': TOL,
        'reference_only': not enabled,
        'calibration_scope': 'Admissible expanded-gain states in updates1 through52, e63 native mean-action target only.'})
    return value


def guard_geometry(candidates, components):
    candidates, components = np.asarray(candidates), np.asarray(components)
    assert candidates.ndim == 2 and candidates.shape[1] == 9
    assert components.shape == (*candidates.shape, 2)
    assert np.isfinite(candidates).all() and np.isfinite(components).all()
    total = components.sum(-1)
    selected = total[:, :5].argmin(-1)
    rows = np.arange(len(candidates))
    reference_action = candidates[rows, selected]
    reference_components = components[rows, selected]
    admissible = np.all(components <= reference_components[:, None, :] + TOL, axis=-1)
    guarded_candidates = np.where(admissible, candidates, reference_action[:, None])
    guarded_total = np.where(admissible, total, reference_components.sum(-1)[:, None])
    return guarded_candidates, guarded_total


class GuardPolicy(BudgetPolicy):
    def __init__(self, base, models, config):
        if type(config.get('reference_only')) is not bool:
            raise ValueError('Boolean reference mode required')
        expected = configuration(config.get('lambda_low'), config.get('lambda_high'),
                                 config.get('low_policy_weight'), not config['reference_only'])
        if config != expected:
            raise ValueError('Guard controller specification differs')
        super().__init__(base, models, budget_configuration(config['lambda_low'],
            config['lambda_high'], config['low_policy_weight']))
        self.config = config.copy()

    def raw_geometry(self, obs):
        obs = np.asarray(obs)
        native = np.asarray(self.base.act_det(obs)).reshape(-1)
        assert np.isfinite(native).all() and np.all(np.abs(native) <= 1)
        candidates = np.clip(native[:, None] + np.asarray(OFFSETS), -1, 1)
        x = public_features(np.repeat(obs, 9, axis=0), candidates.reshape(-1))
        components = self.risk.predict(x).reshape(-1, 9, 2)
        return candidates, components

    def geometry(self, obs):
        return guard_geometry(*self.raw_geometry(obs))

    def act_det(self, obs):
        return project(*self.geometry(obs), self.config)


def load_policy(base, models, path):
    return GuardPolicy(base, models, json.loads(Path(path).read_text()))

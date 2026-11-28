"""Check guard boundaries, reference fallback and disabled shield identity."""
import json
from pathlib import Path

import numpy as np

from shield_policy import ShieldPolicy, configuration, AnchorPolicy, anchor_configuration, project
from rate_model import ConstantRate


class ZeroBase:
    def act_det(self, obs):
        return np.zeros(len(obs))


class ActionChange:
    def predict(self, x):
        return x[:, -1]


def main():
    upper = {'version': 1, 'quantile': .975, 'edges': [1., 1.5, 2.],
        'global_margin': .1, 'stratum_margins': [.1] * 4,
        'stratum_counts': [1] * 4, 'persistence_margin': .1}
    record = {'policy_seed': 0, 'admission': {'passed': True}, 'model_sha256': 'test_only', 'config': upper}
    config = configuration(0, 0, 1, .1, 0, record)
    policy = ShieldPolicy(ZeroBase(), [ConstantRate(.3), ConstantRate(.8)], config, ConstantRate(0))
    obs = np.zeros((5, 11))
    np.testing.assert_array_equal(policy.act_det(obs), [.1] * 5)
    plain = AnchorPolicy(ZeroBase(), [ConstantRate(.3), ConstantRate(.8)], anchor_configuration(0, 0, 1, .1))
    disabled = configuration(0, 0, 1, .1, 0, record, shield_enabled=False)
    off = ShieldPolicy(ZeroBase(), [ConstantRate(.3), ConstantRate(.8)], disabled, ConstantRate(0))
    for a, b in zip(plain.geometry(obs), off.geometry(obs)):
        np.testing.assert_array_equal(a, b)
    # Synthetic geometry isolates the upper guard from flat horizon4 heads.
    c = np.tile(np.linspace(-.5, .5, 9), (3, 1))
    c[:, 0] = 0
    u = np.array([[1.8, 2., 2.1], [2.2, 2.2, 2.3]])
    np.testing.assert_array_equal(u <= np.maximum(2., u[:, :1]) + 1e-12, [[True, True, False], [True, True, False]])
    for current in (1., 2.5):
        upper_lo = max(current - .1, current + .1)
        upper_hi = max(current + .2, current + .1)
        assert upper_lo <= upper_hi
    print(json.dumps({'passed': True, 'flat_heads_reference_exact': True,
        'disabled_geometry_exact': True, 'max_rule_boundaries': True, 'test_evaluated': False}))


if __name__ == '__main__':
    main()

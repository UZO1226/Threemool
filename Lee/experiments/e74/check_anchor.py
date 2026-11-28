"""Known-answer controls for the matched anchor and column0 fallback."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from anchor_policy import ROOT, AnchorPolicy, configuration, guard_geometry, eligible, project, calibrate
from rate_model import ConstantRate


class ZeroBase:
    def act_det(self, obs):
        return np.zeros(len(obs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    candidates = np.tile([.1, -.025, .225, -.15, .35, -.275, .475, -.4, .6], (2, 1))
    components = np.ones((2, 9, 2)) * .5
    components[0, 8] = [.6, .1]
    components[1, 2] = [.4, .5]
    c, risk = guard_geometry(candidates, components)
    assert c[0, 8] == .1 and risk[0, 8] == 1
    np.testing.assert_array_equal(eligible(risk), [False, True])
    np.testing.assert_array_equal(project(c, risk, configuration(0, 0, 1, .1)), [.1, .225])
    np.testing.assert_array_equal(project(c, risk, configuration(0, 0, 1, .1, False)), [.1, .1])
    components[0, 8] = [.1, .6]
    assert guard_geometry(candidates, components)[0][0, 8] == .1
    monotone = np.empty((1, 9, 2))
    monotone[0, :, 0] = 1 - .8 * candidates[0]
    monotone[0, :, 1] = .5
    c, risk = guard_geometry(candidates[:1], monotone)
    fitted, calibration = calibrate(c, risk, .2875)
    config = configuration(fitted['lambda_low'], fitted['lambda_high'], fitted['low_policy_weight'], .1)
    mixed = project(c, risk, config)
    np.testing.assert_allclose(mixed, [.2875], atol=1e-10, rtol=0)
    assert 1 - .8 * mixed[0] <= monotone[0, 0, 0] + 1e-12
    policy = AnchorPolicy(ZeroBase(), [ConstantRate(.3), ConstantRate(.8)], config)
    np.testing.assert_array_equal(policy.act_det(np.zeros((3, 11))), [.1, .1, .1])
    restored = AnchorPolicy(ZeroBase(), [ConstantRate(.3), ConstantRate(.8)], json.loads(json.dumps(config)))
    np.testing.assert_array_equal(restored.act_det(np.zeros((3, 11))), [.1, .1, .1])
    invalid = dict(config, unregistered_option=True)
    try:
        AnchorPolicy(ZeroBase(), [ConstantRate(.3), ConstantRate(.8)], invalid)
    except ValueError:
        pass
    else:
        raise AssertionError('Unknown controller option accepted')
    sources = [Path(__file__), Path(__file__).with_name('anchor_policy.py'), ROOT / 'Lee/experiments/e69/budget_policy.py']
    result = {'passed': True, 'column0_reference_preserved': True, 'inner_grid_gain_eligible': True,
        'both_component_increases_excluded': True, 'monotone_interpolation_control': True,
        'flat_risk_fixed_offset_nn_identity': True, 'strict_config_roundtrip_and_unknown_option_rejection': True,
        'calibration_error': calibration['absolute_mean_action_error'],
        'test_evaluated': False, 'new_environment_queries': 0,
        'source_sha256': {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('ANCHOR_CONTROL_CHECK_PASSED', flush=True)


if __name__ == '__main__':
    main()

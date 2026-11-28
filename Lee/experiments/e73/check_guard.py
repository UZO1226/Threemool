"""Known-answer exclusion, monotone interpolation and fallback controls."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from guard_policy import ROOT, configuration, guard_geometry, eligible, project, calibrate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    candidates = np.tile([0, -.125, .125, -.25, .25, -.375, .375, -.5, .5], (2, 1))
    components = np.ones((2, 9, 2)) * .5
    # A lower sum cannot make a proposal admissible when NH4 increases.
    components[0, 8] = [.6, .1]
    # Monotone NH4 with plateau DO gives an admissible larger-air proposal.
    components[1, :, 0] = [1., 1., .9, 1., .8, 1., .7, 1., .6]
    guarded, risks = guard_geometry(candidates, components)
    assert guarded[0, 8] == 0 and risks[0, 8] == 1
    np.testing.assert_array_equal(eligible(risks), [False, True])
    np.testing.assert_array_equal(project(guarded, risks, configuration(0, 0, 1)), [0, .5])
    np.testing.assert_array_equal(project(guarded, risks, configuration(0, 0, 1, False)), [0, .25])
    fitted, calibration = calibrate(guarded[1:], risks[1:], .25)
    config = configuration(fitted['lambda_low'], fitted['lambda_high'], fitted['low_policy_weight'])
    np.testing.assert_allclose(project(guarded, risks, config), [0, .25], atol=1e-10, rtol=0)
    flat_candidates, flat_risk = guard_geometry(candidates, np.ones_like(components))
    np.testing.assert_array_equal(project(flat_candidates, flat_risk, config), [0, 0])
    # Check both directions of exclusion independently of summed-risk ordering.
    components[0, 8] = [.1, .6]
    excluded, _ = guard_geometry(candidates, components)
    assert excluded[0, 8] == 0
    np.testing.assert_array_equal(excluded[0, :8], candidates[0, :8])
    mixture, _ = calibrate(guarded[1:], risks[1:], .3125)
    mix_config = configuration(mixture['lambda_low'], mixture['lambda_high'], mixture['low_policy_weight'])
    mixed_action = project(guarded[1:], risks[1:], mix_config)
    np.testing.assert_allclose(mixed_action, [.3125], atol=1e-10, rtol=0)
    assert 1 - .8 * mixed_action[0] <= .8 + 1e-12
    sources = [Path(__file__), Path(__file__).with_name('guard_policy.py'),
               ROOT / 'Lee/experiments/e71/signal_policy.py', ROOT / 'Lee/experiments/e69/budget_policy.py']
    result = {'passed': True, 'nh4_increase_excluded': True, 'do_increase_excluded': True,
        'active_only_mean_constraint': calibration['absolute_mean_action_error'] <= 1e-10,
        'flat_risk_fitted_parameter_identity': True, 'reference_only_identity': True,
        'test_evaluated': False, 'new_environment_queries': 0,
        'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('GUARD_CONTROL_CHECK_PASSED', flush=True)


if __name__ == '__main__':
    main()

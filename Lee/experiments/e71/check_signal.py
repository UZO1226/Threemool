"""Known-action tests of fallback and active-only mean preservation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from signal_policy import ROOT, calibrate, configuration, eligible, project


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    candidates = np.tile([0, -.125, .125, -.25, .25, -.375, .375, -.5, .5], (2, 1))
    risk = np.array([[1.] * 9, [1., 1., .9, 1., .8, 1., .7, 1., .6]])
    np.testing.assert_array_equal(eligible(risk), [False, True])
    np.testing.assert_array_equal(project(candidates, risk, configuration(0, 0, 1)), [0, .5])
    np.testing.assert_array_equal(project(candidates, risk, configuration(0, 0, 1, enabled=False)), [0, .25])
    fitted, _ = calibrate(candidates[1:], risk[1:], .25)
    config = configuration(fitted['lambda_low'], fitted['lambda_high'], fitted['low_policy_weight'])
    action = project(candidates, risk, config)
    assert action[0] == 0
    np.testing.assert_allclose(action.mean(), .125, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(project(candidates, np.ones_like(risk), config), np.zeros(2))
    sources = [Path(__file__), Path(__file__).with_name('signal_policy.py'), ROOT / 'Lee/experiments/e69/budget_policy.py']
    result = {'passed': True, 'known_reference_fallback': True, 'active_only_mean_constraint': True,
        'flat_risk_identity_with_fitted_parameters': True, 'reference_only_identity': True,
        'test_evaluated': False, 'new_environment_queries': 0,
        'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        'scope': 'Synthetic software controls only, not policy performance.'}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('SIGNAL_CONTROL_CHECK_PASSED', flush=True)


if __name__ == '__main__':
    main()

"""Check the constrained calibration with independently known optima."""
import json

import numpy as np

from budget_policy import calibrate, choose, project


def main():
    candidates = np.tile(np.linspace(0, 1, 9), (2, 1))
    risk = np.array([np.linspace(0, .2, 9), np.linspace(.6, 0, 9)])
    config, report = calibrate(candidates, risk, .25)
    action = project(candidates, risk, config)
    np.testing.assert_allclose(action, [0., .5], rtol=0, atol=1e-10)
    assert abs(report['achieved_mean_action'] - .25) <= 1e-10
    assert config['lambda_low'] <= .6 <= config['lambda_high']
    values = [choose(candidates, risk, lam).mean() for lam in (-1, 0, .3, 1)]
    assert all(first >= second for first, second in zip(values, values[1:]))
    config, _ = calibrate(candidates, risk, .5)
    assert config['lambda_low'] == config['lambda_high'] == 0.
    flat, _ = calibrate(candidates, np.zeros_like(risk), 0.)
    np.testing.assert_array_equal(project(candidates, np.zeros_like(risk), flat), np.zeros(2))
    print(json.dumps({'passed': True, 'known_discrete_jump_mixture': True,
                      'air_multiplier_sign_monotone': True, 'zero_identity_and_flat_null': True}), flush=True)


if __name__ == '__main__':
    main()

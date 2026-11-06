"""보수성 설정과 critic 진단 목표가 잘못 적용되지 않는지 확인한다."""
from pathlib import Path
import sys
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'experiments/e04'))
from agents import make_agent
from env import AerationEnv
from diagnose import discounted_targets


class PreflightTests(unittest.TestCase):
    def test_invalid_conservatism_is_rejected_before_loading(self):
        for k in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                AerationEnv('missing', 'missing', pess_k=k)
            with self.assertRaises(ValueError):
                AerationEnv('missing', 'missing', unc_pen=k)
            with self.assertRaises(ValueError):
                make_agent('fcsrl_ucb', {}, None, ucb_k=k)
        for n in (0, 1, 2.5, True):
            with self.assertRaises(ValueError):
                make_agent('fcsrl_ucb', {}, None, ucb_n=n)

    def test_bootstrap_target_matches_analytic_value(self):
        cost = np.ones((7, 2))
        q = {t: np.full(2, 10.) for t in (0, 3, 6)}
        starts, target = discounted_targets(cost, q, horizon=3, gamma=.9)
        np.testing.assert_array_equal(starts, [0, 3])
        np.testing.assert_allclose(target, 10.)
        # 날짜 경계를 넘지 않고 비용과 Q의 할인을 맞춘다.
        q[3] = np.zeros(2)
        _, target = discounted_targets(cost, q, horizon=3, gamma=.9)
        np.testing.assert_allclose(target[0], 1 + .9 + .9**2)


if __name__ == '__main__':
    unittest.main()

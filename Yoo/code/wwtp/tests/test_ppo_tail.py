"""표본 CVaR, trajectory score 경사와 beta=0 제거 검증."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppo_config import PPOConfig
from ppo_fcsrl import PPOFCSRLAgent, RepresentationConfig
from ppo_lag import PPOLagAgent
from ppo_tail import TailConfig, PPOTailAgent, PPOFCSRLTailAgent, empirical_cvar, episode_tail_scores


class TailTests(unittest.TestCase):
    def test_fractional_tail_and_ties(self):
        self.assertEqual(empirical_cvar([0, 1, 2, 3, 4], 0.2), (3., 4.))
        self.assertAlmostEqual(empirical_cvar([0, 1, 2, 3, 4], 0.3)[1], (4 + 0.5 * 3) / 1.5)
        self.assertEqual(empirical_cvar([0, 1, 2, 3, 4], 1), (0., 2.))
        self.assertEqual(empirical_cvar([2, 2, 2, 2], 0.2), (2., 2.))
        # CVaR는 상위 표본을 ceil개 단순 평균하는 방식과 다르다.
        costs = np.arange(32.)
        self.assertAlmostEqual(empirical_cvar(costs, .2)[1], (sum(range(26, 32)) + .4 * 25) / 6.4)

    def test_daily_scores_do_not_cross_resets_or_environments(self):
        cost = np.array([[0, 1], [0, 1], [2, 0], [2, 0]], dtype='float32')
        end = np.array([[0, 0], [1, 1], [0, 0], [1, 1]])
        score, info = episode_tail_scores(cost, end, TailConfig(tail_fraction=.5, cost_scale=1))
        np.testing.assert_array_equal(score, [[0, 4], [0, 4], [8, 0], [8, 0]])
        self.assertEqual(info['train_cost_cvar'], 3)
        self.assertEqual(info['tail_episode_count'], 4)
        with self.assertRaises(ValueError):
            episode_tail_scores(cost[:-1], end[:-1], TailConfig())
        with self.assertRaises(ValueError):
            episode_tail_scores(cost, np.array([[1, 0], [1, 1], [0, 0], [1, 1]]), TailConfig())

    def test_upper_tail_gradient_matches_enumerated_distribution(self):
        # 2단계 Bernoulli 비용. p=.2, upper 50% VaR=0, CVaR=2*p/.5.
        logit = torch.tensor(np.log(.2/.8), dtype=torch.float64, requires_grad=True)
        actions = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=torch.float64)
        dist = torch.distributions.Bernoulli(logits=logit)
        logp = dist.log_prob(actions).sum(-1)
        probs = logp.detach().exp()
        excess_score = actions.sum(-1) / .5
        surrogate = (probs * logp * excess_score).sum()
        surrogate.backward()
        self.assertAlmostEqual(logit.grad.item(), 2 / .5 * .2 * .8, places=12)
        # 비용 경사를 빼는 policy ascent는 위험 행동 확률을 낮춰야 한다.
        self.assertGreater(logit.grad.item(), 0)

    def test_zero_weight_is_exact_for_both_agents(self):
        torch.set_num_threads(1)
        c = PPOConfig(total_steps=32, n_chunks=1, update_epochs=2, minibatch_size=8, hidden_dim=16)
        r = RepresentationConfig(z_dim=8, hidden_dim=16)
        rng = np.random.default_rng(9)
        buf = {key: rng.normal(size=shape).astype('float32') for key, shape in
               [('obs', (32, 11)), ('obs_next', (32, 11)), ('act', (32, 1)), ('adv_r', (32,)),
                ('adv_c', (32,)), ('ret_r', (32,)), ('ret_c', (32,))]}
        buf['act_exec'] = np.clip(buf['act'], -1, 1)
        buf['feasi'] = np.abs(buf['ret_c'])
        for original, extension in [(PPOLagAgent, PPOTailAgent), (PPOFCSRLAgent, PPOFCSRLTailAgent)]:
            extra = {'repr_config': r} if original is PPOFCSRLAgent else {}
            torch.manual_seed(10)
            base = original(11, 1, config=c, **extra)
            torch.manual_seed(10)
            risk = extension(11, 1, config=c, tail_config=TailConfig(weight=0), **extra)
            with torch.no_grad():
                buf['logp'] = base.dist(base._t(buf['obs'])).log_prob(base._t(buf['act'])).sum(-1).numpy()
            torch.manual_seed(20)
            expected = base.update(buf)
            torch.manual_seed(20)
            actual = risk.update(buf)
            self.assertEqual(expected, actual)
            names = ['pi', 'vr', 'vc'] + (['encoder', 'fixed_encoder', 'feasi_head'] if extra else [])
            for name in names:
                for x, y in zip(getattr(base, name).parameters(), getattr(risk, name).parameters()):
                    torch.testing.assert_close(x, y, rtol=0, atol=0)
            torch.testing.assert_close(base.log_std, risk.log_std, rtol=0, atol=0)

    def test_invalid_settings_and_costs(self):
        for kw in ({'tail_fraction': 0}, {'tail_fraction': 2}, {'weight': -1}, {'cost_scale': 0},
                   {'weight': float('nan')}):
            with self.assertRaises(ValueError):
                TailConfig(**kw)
        for costs in ([], [-1], [float('inf')], [[1]]):
            with self.assertRaises(ValueError):
                empirical_cvar(costs)


if __name__ == '__main__':
    unittest.main()

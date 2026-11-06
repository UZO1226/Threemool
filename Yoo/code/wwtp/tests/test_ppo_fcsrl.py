"""원본 표현 손실, rollout 경계 및 PPO 정책 분리의 수치적 검증."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fcsrl.agent import PPOLagReprAgent
from fcsrl.data import Batch
from ppo_config import PPOConfig
from ppo_fcsrl import PPOFCSRLAgent, RepresentationConfig, feasibility_targets


class RepresentationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(10)
        self.c = PPOConfig(total_steps=32, n_env=2, horizon=16, n_chunks=1, hidden_dim=16,
                           minibatch_size=8, update_epochs=2)
        self.r = RepresentationConfig(z_dim=8, hidden_dim=16)
        self.agent = PPOFCSRLAgent(11, 1, config=self.c, repr_config=self.r)
        rng = np.random.default_rng(8)
        self.buf = {'obs': rng.normal(size=(16, 11)).astype('float32'),
                    'obs_next': rng.normal(size=(16, 11)).astype('float32'),
                    'act_exec': rng.uniform(-1, 1, (16, 1)).astype('float32'),
                    'feasi': rng.uniform(0, 2, 16).astype('float32')}

    def test_original_loss_and_gradient_step_are_identical(self):
        reference = deepcopy(self.agent)
        before = self.agent.act_det(self.buf['obs'])
        actual = self.agent.update_aux(self.buf, torch.arange(16))
        batch = Batch(obs=reference._t(self.buf['obs']), obs_next=reference._t(self.buf['obs_next']),
                      act=reference._t(self.buf['act_exec']), feasi_score=reference._t(self.buf['feasi']).unsqueeze(-1))
        expected = PPOLagReprAgent.train_encoder(reference, batch)
        self.assertEqual(actual, expected)
        for name in ('encoder', 'feasi_head', 'proj_layer', 'post_proj'):
            for p, q in zip(getattr(self.agent, name).parameters(), getattr(reference, name).parameters()):
                torch.testing.assert_close(p, q, rtol=0, atol=0)
        np.testing.assert_array_equal(before, self.agent.act_det(self.buf['obs']))
        self.agent.finish_update()
        self.assertFalse(np.array_equal(before, self.agent.act_det(self.buf['obs'])))

    def test_feasibility_matches_original_separately_for_each_environment(self):
        T, N = 5, 2
        rng = np.random.default_rng(5)
        obs_next = rng.normal(size=(T, N, 11)).astype('float32')
        cost = rng.integers(0, 3, size=(T, N)).astype('float32')
        end = np.zeros((T, N), dtype=bool)
        end[1, 0], end[3, 1] = True, True
        term = np.zeros_like(end)
        term[3, 1] = True
        bootstrap = self.agent.next_feasibility(obs_next.reshape(-1, 11)).reshape(T, N)
        actual = feasibility_targets(cost, bootstrap, end, term, self.r.feasibility_discount)
        self.agent._f_discount = self.r.feasibility_discount
        for i in range(N):
            batch = Batch(cost=cost[:, i], obs_next=obs_next[:, i], trunc=end[:, i], terminate=term[:, i])
            expected = PPOLagReprAgent._get_feasibility(self.agent, batch)
            np.testing.assert_allclose(actual[:, i], expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(actual[3, 1], cost[3, 1])
        changed = cost.copy()
        changed[2:, 0] = 100
        second = feasibility_targets(changed, bootstrap, end, term)
        np.testing.assert_array_equal(actual[:2, 0], second[:2, 0])

    def test_frozen_control_and_policy_ratio(self):
        frozen = PPOFCSRLAgent(11, 1, config=self.c,
            repr_config=RepresentationConfig(z_dim=8, hidden_dim=16, frozen=True))
        before = deepcopy(frozen.encoder.state_dict())
        self.assertEqual(frozen.update_aux(self.buf, torch.arange(16)), {})
        frozen.finish_update()
        for key, value in frozen.fixed_encoder.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        obs = self.agent._t(self.buf['obs'])
        dist = self.agent.dist(obs)
        raw = torch.full((16, 1), 2.0)
        old = dist.log_prob(raw).detach()
        self.agent.update_aux(self.buf, torch.arange(16))
        torch.testing.assert_close((self.agent.dist(obs).log_prob(raw) - old).exp(), torch.ones_like(old))
        self.assertFalse(torch.equal(old, dist.log_prob(raw.clamp(-1, 1))))

    def test_update_and_checkpoint_roundtrip(self):
        buf = dict(self.buf)
        with torch.no_grad():
            d = self.agent.dist(self.agent._t(buf['obs']))
            act = d.sample()
            buf.update(act=act.numpy(), logp=d.log_prob(act).sum(-1).numpy())
        buf.update(adv_r=np.linspace(-1, 1, 16), adv_c=np.linspace(1, -1, 16),
                   ret_r=np.ones(16), ret_c=np.ones(16))
        metrics = self.agent.update(buf)
        self.assertTrue(all(np.isfinite(x) for x in metrics.values()))
        self.assertIn('encoder_pred_feasi', metrics)
        with tempfile.TemporaryDirectory() as tmp:
            self.agent.save(tmp)
            restored = PPOFCSRLAgent(11, 1, config=self.c, repr_config=self.r)
            restored.load(tmp)
            np.testing.assert_array_equal(self.agent.act_det(buf['obs']), restored.act_det(buf['obs']))
            np.testing.assert_array_equal(self.agent.next_feasibility(buf['obs']), restored.next_feasibility(buf['obs']))


if __name__ == '__main__':
    unittest.main()

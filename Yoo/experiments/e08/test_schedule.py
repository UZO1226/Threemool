"""고정 경계 전 계산 일치와 경계 후 정책 및 표현 분리를 확인한다."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scheduled_agent import ScheduledAgent, representation_digest
from ppo_fcsrl import PPOFCSRLAgent, RepresentationConfig
from ppo_config import PPOConfig


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.c = PPOConfig(total_steps=64, n_env=2, horizon=16, n_chunks=1, hidden_dim=16,
                           minibatch_size=8, update_epochs=2)
        self.r = RepresentationConfig(z_dim=8, hidden_dim=16)

    def pair(self, boundary):
        torch.manual_seed(19)
        base = PPOFCSRLAgent(11, 1, config=self.c, repr_config=self.r)
        torch.manual_seed(19)
        controlled = ScheduledAgent(11, 1, config=self.c, repr_config=self.r, freeze_after=boundary)
        rng = np.random.default_rng(11)
        buf = {k:rng.normal(size=(16,11)).astype('float32') for k in ('obs','obs_next')}
        with torch.no_grad():
            dist = base.dist(base._t(buf['obs']))
            action = dist.sample()
            buf.update(act=action.numpy(), act_exec=action.clamp(-1,1).numpy(),
                       logp=dist.log_prob(action).sum(-1).numpy())
        for k in ('adv_r','adv_c','ret_r','ret_c','feasi'):
            buf[k] = rng.uniform(0,1,16).astype('float32')
        return base, controlled, buf

    def test_disabled_schedule_and_boundary_match_original(self):
        for boundary in (0, 2):
            base, controlled, buf = self.pair(boundary)
            for u in range(2):
                torch.manual_seed(100+u)
                expected = base.update(buf)
                torch.manual_seed(100+u)
                actual = controlled.update(buf)
                actual.pop('representation_frozen_next')
                self.assertEqual(expected, actual)
                self.assertEqual(representation_digest(base), representation_digest(controlled))
                for name in ('pi','vr','vc'):
                    for p,q in zip(getattr(base,name).parameters(),getattr(controlled,name).parameters()):
                        self.assertTrue(torch.equal(p,q))
                self.assertTrue(torch.equal(base.log_std,controlled.log_std))

    def test_freeze_stops_all_representation_updates_but_not_policy(self):
        _, agent, buf = self.pair(1)
        before = representation_digest(agent)
        agent.update(buf)
        self.assertNotEqual(before,representation_digest(agent))
        digest = representation_digest(agent)
        policy = deepcopy(agent.pi.state_dict())
        metrics = agent.update(buf)
        self.assertEqual(digest,representation_digest(agent))
        self.assertNotIn('encoder_pred_feasi',metrics)
        self.assertTrue(any(not torch.equal(v,policy[k]) for k,v in agent.pi.state_dict().items()))
        with tempfile.TemporaryDirectory() as directory:
            agent.save(directory)
            restored = PPOFCSRLAgent(11,1,config=self.c,repr_config=self.r)
            restored.load(directory)
            np.testing.assert_array_equal(agent.act_det(buf['obs']),restored.act_det(buf['obs']))

    def test_invalid_schedule_rejected(self):
        for boundary in (-1,1.5,True):
            with self.assertRaises(ValueError):
                ScheduledAgent(11,1,config=self.c,repr_config=self.r,freeze_after=boundary)


if __name__ == '__main__':
    unittest.main()

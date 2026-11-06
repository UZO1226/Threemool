"""설정 변경이 실제 학습량, 신경망, 저장 및 평가에 반영되는지 검증한다."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ppo_config import PPOConfig
from ppo_lag import PPOLagAgent, train


class TinyEnv:
    steps = {}

    def __init__(self, data, surrogate, split, seed):
        self.split = split
        self.t = 0

    def reset(self):
        self.t = 0
        return np.ones(11, np.float32), {}

    def step(self, action):
        assert np.all(np.abs(action) <= 1)
        self.t += 1
        self.steps[self.split] = self.steps.get(self.split, 0) + 1
        return np.full(11, 1 + self.t / 10, np.float32), -0.2, False, self.t == 2, {"cost": 0.0}


class ConfigTests(unittest.TestCase):
    def test_yaml_matches_historical_defaults(self):
        path = Path(__file__).resolve().parents[1] / "ppo_wwtp.yaml"
        c = PPOConfig.from_dict(yaml.safe_load(path.read_text()))
        self.assertEqual(c, PPOConfig())
        self.assertEqual(c.actual_steps, 2_002_112)
        self.assertEqual(c.n_updates, 218)

    def test_invalid_or_unused_settings_fail(self):
        for values in ({"trainer": {}}, {"total_steps": 0}, {"minibatch_size": True},
                       {"actor_lr": float("nan")}, {"pid_ki": 0}, {"n_chunks": 1000}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                PPOConfig.from_dict(values)

    def test_custom_config_reaches_training_and_checkpoint(self):
        c = PPOConfig(total_steps=9, n_env=2, horizon=3, n_chunks=2, test_episodes=1,
                      hidden_dim=16, actor_lr=0.0007, critic_lr=0.002, init_log_std=-1.2,
                      update_epochs=1, minibatch_size=3, gamma=0.8, gae_lambda=0.7,
                      max_lambda=3, pid_kp=0.1, pid_ki=0.01, pid_kd=0.02)
        TinyEnv.steps = {}
        with tempfile.TemporaryDirectory() as tmp, patch("ppo_lag.AerationEnv", TinyEnv):
            run = Path(tmp)
            agent, best = train("data.pkl", "surrogate.json", 7, run, config=c)
            self.assertEqual(TinyEnv.steps["train"], 12)
            self.assertEqual(TinyEnv.steps["val"], 4)
            self.assertEqual(agent.pi[0].out_features, 16)
            self.assertEqual(agent.opt_pi.param_groups[0]["lr"], c.actor_lr)
            self.assertEqual(agent.opt_v.param_groups[0]["lr"], c.critic_lr)
            self.assertEqual(agent.pid.max_lag, 3)
            self.assertEqual(agent.pid.KP, 0.1)
            saved = json.loads((run / "effective_config.json").read_text())
            self.assertEqual(saved["config"], c.to_dict())
            self.assertEqual(saved["actual_steps"], 12)
            self.assertEqual(len(json.loads((run / "train_log.json").read_text())), 2)
            final = run / "model_final"
            restored = PPOLagAgent(11, 1, config=PPOConfig.from_dict(json.loads((final / "config.json").read_text())))
            restored.load(final)
            np.testing.assert_array_equal(agent.act_det(np.ones(11)), restored.act_det(np.ones(11)))
            self.assertTrue(all(torch.isfinite(x).all() for x in restored.pi.parameters()))
            self.assertTrue((run / "model" / f"epoch_{best['epoch']}" / "model.pt").is_file())


if __name__ == "__main__":
    unittest.main()

"""Check passive recording against the frozen environment, including resets."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from capture_env import CaptureEnv, FrozenQuantileEnv
from quantile_env import FrozenDynamics, frozen_digest
from cost_model import ConstantProbability, CorrectedPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    dynamics = FrozenDynamics(args.freeze)
    with tempfile.TemporaryDirectory() as directory:
        plain = FrozenQuantileEnv(args.data, dynamics, n_env=8, seed=5500)
        capture = CaptureEnv(args.data, dynamics, n_env=8, seed=5500,
                             capture_path=directory, horizon=190)
        np.testing.assert_array_equal(plain.reset(), capture.reset())
        rng = np.random.default_rng(55)
        for t in range(190):
            if t % 95 == 0:
                plain.set_member(2)
                capture.set_member(2)
            action = rng.uniform(-1, 1, 8)
            before = np.random.get_state()
            left, right = plain.step(action), capture.step(action)
            after = np.random.get_state()
            assert before[0] == after[0] and np.array_equal(before[1], after[1])
            assert before[2:] == after[2:]
            for a, b in zip(left[:5], right[:5]):
                np.testing.assert_array_equal(a, b)
            for key in left[5]:
                np.testing.assert_array_equal(left[5][key], right[5][key])
            assert plain.rng.bit_generator.state == capture.rng.bit_generator.state
        with np.load(Path(directory) / 'update_001.npz') as saved:
            assert saved['obs'].shape == (190, 8, 11)
            assert saved['done'].sum() == 16
            np.testing.assert_array_equal(saved['nh4'], saved['next_obs'][:, :, 0] > 4)
            np.testing.assert_array_equal(saved['do'], saved['next_obs'][:, :, 2:5].max(-1) > 2)
    class Base:
        def act_det(self, obs):
            return obs[:, :1]
    obs = np.zeros((1001, 11))
    obs[:, 0] = np.linspace(-1, 1, len(obs))
    base = Base()
    corrected = CorrectedPolicy(base, [ConstantProbability(.3), ConstantProbability(.8)])
    np.testing.assert_array_equal(corrected.act_det(obs), base.act_det(obs).reshape(-1))
    result = {'passed': True, 'identical_physical_steps': 190,
              'transitions': 1520, 'completed_episodes': 16,
              'rng_unchanged': True, 'terminal_labels_exact': True,
              'constant_risk_null_action_identity': True,
              'scope': 'capture and constant-risk actions only; not full training or learned predictor',
              'runtime_sources': {name: frozen_digest(Path(__file__).with_name(name))
                                  for name in ('capture_env.py', 'cost_model.py', 'check_capture.py')}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

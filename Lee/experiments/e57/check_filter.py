"""Verify gain1 identity and gain.5 public-air feedback before primary outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

from air_filter import AirFilter

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Lee/experiments/e55'))
from policy_io import load_captured, file_hashes
from quantile_env import FrozenDynamics, FrozenQuantileEnv, evaluate, frozen_digest


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'freeze', 'run', 'baseline', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    class SyntheticBase:
        def act_det(self, obs):
            self.seen = obs.copy()
            return np.clip(obs[:, :1], -1, 1).astype(np.float32)
    obs = np.zeros((2049, 11), dtype=np.float32)
    obs[:, 0] = np.linspace(-1.5, 1.5, len(obs))
    obs[:, 10] = np.linspace(-.25, 1.25, len(obs))
    before = obs.copy()
    synthetic = SyntheticBase()
    filtered = AirFilter(synthetic)
    expected_base = synthetic.act_det(obs).reshape(-1)
    previous = np.minimum(1, np.maximum(-1, 2 * obs[:, 10].astype(float) - 1))
    expected = np.minimum(1, np.maximum(-1, .5 * expected_base.astype(float) + .5 * previous))
    np.testing.assert_array_equal(filtered.act_det(obs), expected)
    np.testing.assert_array_equal(obs, before)
    np.testing.assert_array_equal(synthetic.seen, obs)
    null = AirFilter(synthetic, 1.)
    actual = null.act_det(obs)
    np.testing.assert_array_equal(actual, expected_base)
    assert actual.dtype == expected_base.dtype
    np.testing.assert_array_equal(actual + .125, expected_base + .125)
    assert np.all(np.abs(expected) <= 1)
    with tempfile.TemporaryDirectory(prefix='e57_filter_config_') as directory:
        assert Path(directory).resolve().is_relative_to(Path(tempfile.gettempdir()).resolve())
        config_path = Path(directory) / 'filter.json'
        filtered.save_config(config_path)
        np.testing.assert_array_equal(AirFilter.load_config(synthetic, config_path).act_det(obs), expected)
        payload = json.loads(config_path.read_text())
        payload['offset_applied_after_filter'] = False
        config_path.write_text(json.dumps(payload))
        try:
            AirFilter.load_config(synthetic, config_path)
        except AssertionError:
            pass
        else:
            raise AssertionError('Changed filter operation order accepted')
    torch.set_num_threads(1)
    base, config, metrics, hashes = load_captured(args.run, args.freeze, args.baseline, False)
    dynamics = FrozenDynamics(args.freeze)
    original_env = FrozenQuantileEnv(args.data, dynamics, n_env=3, seed=5701)
    identity_env = FrozenQuantileEnv(args.data, dynamics, n_env=3, seed=5701)
    primary_env = FrozenQuantileEnv(args.data, dynamics, n_env=3, seed=5701)
    original_obs, identity_obs, primary_obs = original_env.reset(), identity_env.reset(), primary_env.reset()
    null, primary = AirFilter(base, 1.), AirFilter(base, .5)
    reset_count = 0
    for step in range(190):
        if step % 95 == 0:
            for env in (original_env, identity_env, primary_env):
                env.set_member(2)
        # This offset exercises actual executed-air feedback and original dtype.
        original_action = np.asarray(base.act_det(original_obs)).reshape(-1) + .125
        identity_action = null.act_det(identity_obs) + .125
        np.testing.assert_array_equal(original_action, identity_action)
        left, right = original_env.step(original_action), identity_env.step(identity_action)
        for a, b in zip(left[:5], right[:5]):
            np.testing.assert_array_equal(a, b)
        for key in left[5]:
            np.testing.assert_array_equal(left[5][key], right[5][key])
        original_obs, identity_obs = left[0], right[0]
        proposed = primary.act_det(primary_obs)
        raw_base = np.asarray(base.act_det(primary_obs)).reshape(-1)
        np.testing.assert_array_equal(proposed, .5 * raw_base.astype(float) +
            .5 * np.clip(2 * primary_obs[:, 10].astype(float) - 1, -1, 1))
        output = primary_env.step(proposed + .125)
        executed = (np.clip(proposed + .125, -1, 1) + 1) / 2 * 9000
        np.testing.assert_array_equal(output[5]['air'], executed)
        np.testing.assert_allclose(output[4][:, 10], executed / 9000, atol=1e-7, rtol=1e-7)
        done = output[3]
        primary_obs = output[0]
        if done.any():
            reset_count += int(done.sum())
            initial = np.array([primary_env.day_arrays[primary_env.day[i]]['air'][0] for i in np.flatnonzero(done)])
            np.testing.assert_allclose(primary_obs[done, 10], initial / 9000, atol=1e-7, rtol=1e-7)
        if (~done).any():
            np.testing.assert_allclose(primary_obs[~done, 10], executed[~done] / 9000, atol=1e-7, rtol=1e-7)
    assert reset_count == 6
    native, _ = evaluate(args.data, dynamics, act_batch=null.act_det)
    for key, value in metrics['val_final'].items():
        np.testing.assert_allclose(native[key], value, atol=1e-10, rtol=1e-10)
    assert file_hashes(args.run / 'model_final') == hashes
    result = {'passed': True, 'policy_seed': config['seed'], 'base_training_steps': 401280,
              'test_evaluated': False, 'primary_performance_evaluated': False,
              'synthetic_conversion_clipping_and_convex_combination_exact': True,
              'original_nn_observation_unchanged': True, 'identity_action_and_dtype_exact': True,
              'config_roundtrip_exact_and_operation_order_tamper_rejected': True,
              'identity_offset_reset_terminal_rollout_steps': 190,
              'primary_executed_air_feedback_rollout_steps': 190, 'primary_episode_resets_checked': 6,
              'full_identity_native_metrics_match_original_tolerance': 1e-10,
              'base_checkpoint_hashes': hashes, 'extra_environment_training_steps': 0,
              'diagnostic_environment_transitions': 3 * 3 * 190 + 1600 * 95,
              'source_sha256': {name: frozen_digest(Path(__file__).with_name(name))
                                for name in ('air_filter.py', 'check_filter.py')},
              'freeze_canonical_sha256': frozen_digest(args.freeze),
              'scope': 'Synthetic and training-rollout filter checks, original gain1 native validation replay. No gain.5 validation outcome, new PPO fit or parameter selection.'}
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print('E57_FILTER_PREFLIGHT_PASS', flush=True)


if __name__ == '__main__':
    main()

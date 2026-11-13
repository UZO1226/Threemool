"""Check initialization, optimizer references, causal observations and serialization."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from history_agent import HistoryFCSRL, ExpandedLinear, LAG_COLUMNS, digest
from history_env import HistoryEnv, OriginalFrozenEnv, quantile_env
from train_history import history_runtime, train_dw, train_policy
from quantile_env import FrozenDynamics
from ppo_fcsrl import PPOFCSRLAgent, RepresentationConfig
from ppo_config import PPOConfig
from fcsrl.utils import DeviceConfig


def parameters_in(optimizer):
    return {id(p) for group in optimizer.param_groups for p in group['params']}


def main():
    parser = argparse.ArgumentParser()
    for name in ('data', 'freeze', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    DeviceConfig().select_device(-1)
    torch.set_num_threads(1)
    config = PPOConfig(total_steps=6080, n_env=8, horizon=760, n_chunks=1, cost_limit=12.)
    torch.manual_seed(54)
    base = PPOFCSRLAgent(11, 1, config=config, repr_config=RepresentationConfig())
    baseline_rng = torch.random.get_rng_state().clone()
    original_sources = (train_dw.OBS_DIM, train_policy.FrozenQuantileEnv, train_policy.evaluate, quantile_env.FrozenQuantileEnv)
    raw = np.random.default_rng(54).normal(size=(128, 11)).astype(np.float32)
    past = np.random.default_rng(5401).normal(size=(128, 6)).astype(np.float32)
    probe = torch.tensor(raw)
    result = {'passed': False, 'policy_training': False, 'test_evaluated': False, 'modes': {}}
    for mode in ('history', 'repeated'):
        torch.manual_seed(54)
        agent = HistoryFCSRL(17, 1, config=config, repr_config=RepresentationConfig(), mode=mode)
        np.testing.assert_array_equal(torch.random.get_rng_state().numpy(), baseline_rng.numpy())
        augmented = np.c_[raw, past if mode == 'history' else raw[:, LAG_COLUMNS]]
        ot = torch.tensor(augmented)
        common_exact = True
        for name in ('pi', 'vr', 'vc'):
            original_layer = getattr(base, name)[0]
            widened = getattr(agent, name)[0]
            assert isinstance(widened, ExpandedLinear)
            assert torch.equal(widened.weight[:, widened.common_columns], original_layer.weight)
            assert torch.count_nonzero(widened.weight[:, widened.extra_columns]) == 0
            assert torch.equal(widened.bias, original_layer.bias)
        for current, reference in [(agent.encoder, base.encoder), (agent.fixed_encoder, base.fixed_encoder)]:
            for name, parameter in reference.named_parameters():
                actual = dict(current.named_parameters())[name]
                if name == 'zs_net.layers.0.weight':
                    assert torch.equal(actual[:, :11], parameter) and torch.count_nonzero(actual[:, 11:]) == 0
                else:
                    assert torch.equal(actual, parameter)
        maximum = 0.
        with torch.no_grad():
            comparisons = [(agent.dist(ot).loc, base.dist(probe).loc),
                (agent.values(ot)[0], base.values(probe)[0]), (agent.values(ot)[1], base.values(probe)[1]),
                (agent.encoder.zs(ot), base.encoder.zs(probe)),
                (agent.fixed_encoder.zs(ot), base.fixed_encoder.zs(probe))]
            for actual, expected in comparisons:
                maximum = max(maximum, float((actual - expected).abs().max()))
                common_exact = common_exact and torch.equal(actual, expected)
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        assert parameters_in(agent.opt_pi) == {id(p) for p in [*agent.pi.parameters(), agent.log_std]}
        assert parameters_in(agent.opt_v) == {id(p) for p in [*agent.vr.parameters(), *agent.vc.parameters()]}
        expected_encoder = {id(p) for module in (agent.encoder, agent.feasi_head, agent.proj_layer, agent.post_proj)
                            for p in module.parameters()}
        assert parameters_in(agent.encoder_optim) == expected_encoder
        assert not any(p.requires_grad for p in agent.fixed_encoder.parameters())
        agent.opt_pi.zero_grad()
        agent.dist(ot).loc.square().mean().backward()
        actor_gradient = float(agent.pi[0].weight.grad[:, 11:17].abs().max())
        assert actor_gradient > 0
        agent.encoder_optim.zero_grad()
        agent.encoder.zs(ot).square().mean().backward()
        encoder_gradient = float(agent.encoder.zs_net.layers[0].weight.grad[:, 11:].abs().max())
        assert encoder_gradient > 0
        base.obs_normalizer = type(base.obs_normalizer)(11)
        base.obs_normalizer.update(raw)
        agent.obs_normalizer.update(augmented)
        np.testing.assert_array_equal(agent.obs_normalizer.mean[:11], base.obs_normalizer.mean)
        np.testing.assert_array_equal(agent.obs_normalizer.m2[:11], base.obs_normalizer.m2)
        with tempfile.TemporaryDirectory(prefix='e54_history_') as folder:
            agent.save(folder)
            restored = HistoryFCSRL(17, 1, config=config, repr_config=RepresentationConfig(), mode=mode)
            restored.load(folder)
            np.testing.assert_array_equal(restored.act_det(augmented), agent.act_det(augmented))
        with history_runtime(mode):
            assert train_dw.OBS_DIM == 17 and train_policy.FrozenQuantileEnv is not OriginalFrozenEnv
            assert quantile_env.FrozenQuantileEnv is OriginalFrozenEnv
        assert (train_dw.OBS_DIM, train_policy.FrozenQuantileEnv, train_policy.evaluate, quantile_env.FrozenQuantileEnv) == original_sources
        result['modes'][mode] = {'common_parameters_exact': True, 'rng_exact': True,
            'initial_outputs_bitwise_equal': common_exact, 'initial_outputs_max_difference': maximum,
            'actor_new_column_gradient_max': actor_gradient, 'encoder_new_column_gradient_max': encoder_gradient,
            'optimizer_membership_exact': True, 'fixed_encoder_frozen': True,
            'first11_normalizer_statistics_exact': True, 'serialization_replay_exact': True,
            'scoped_globals_restored': True,
            'trainable_parameter_count': sum(p.numel() for module in (agent.pi, agent.vr, agent.vc, agent.encoder,
                agent.feasi_head, agent.proj_layer, agent.post_proj) for p in module.parameters() if p.requires_grad) + agent.log_std.numel()}
    dynamics = FrozenDynamics(args.freeze)
    for mode in ('history', 'repeated'):
        baseline = OriginalFrozenEnv(args.data, dynamics, n_env=4, seed=54)
        history = HistoryEnv(args.data, dynamics, n_env=4, seed=54, mode=mode)
        current, extended = baseline.reset(), history.reset()
        np.testing.assert_array_equal(extended[:, :11], current)
        np.testing.assert_array_equal(extended[:, 11:], current[:, LAG_COLUMNS])
        rng = np.random.default_rng(5402)
        terminals = 0
        for _ in range(190):
            previous = current.copy()
            action = rng.uniform(-1.3, 1.3, 4)
            first, second = baseline.step(action), history.step(action)
            for index in (1, 2, 3):
                np.testing.assert_array_equal(first[index], second[index])
            np.testing.assert_array_equal(first[0], second[0][:, :11])
            np.testing.assert_array_equal(first[4], second[4][:, :11])
            for key in first[5]:
                np.testing.assert_array_equal(first[5][key], second[5][key])
            done = first[3]
            expected = previous[:, LAG_COLUMNS].copy() if mode == 'history' else first[0][:, LAG_COLUMNS].copy()
            expected[done] = first[0][done][:, LAG_COLUMNS]
            terminal = previous[:, LAG_COLUMNS] if mode == 'history' else first[4][:, LAG_COLUMNS]
            np.testing.assert_array_equal(second[0][:, 11:], expected)
            np.testing.assert_array_equal(second[4][:, 11:], terminal)
            current = first[0]
            terminals += int(done.sum())
        assert terminals == 8
        result['modes'][mode]['physical_environment_190_steps_exact'] = True
        result['modes'][mode]['reset_and_terminal_history_exact'] = True
    assert result['modes']['history']['trainable_parameter_count'] == result['modes']['repeated']['trainable_parameter_count']
    result['same_capacity_exact'] = True
    result['runtime_sources'] = {name: digest(Path(__file__).with_name(name))
        for name in ('history_agent.py', 'history_env.py', 'train_history.py')}
    result['passed'] = True
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

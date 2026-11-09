"""실제 Kaggle 관측으로 초기 정책, 원시 행동 확률, 고정 prior, 갱신 및 복원을 검사한다."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
from fcsrl.utils import DeviceConfig
from behavior import make_agent, behavior_penalty, observations
from env import AerationEnv
from evaluate import batched_rollout, run_policy, summarize_rows
from ppo_config import PPOConfig
from ppo_lag import gae
from ppo_fcsrl import feasibility_targets


def main(a):
    torch.set_num_threads(1)
    DeviceConfig().select_device(-1)
    c = PPOConfig(total_steps=256, n_env=1, horizon=256, n_chunks=1, update_epochs=2, init_log_std=-1.6)
    env = AerationEnv(a.data, a.model, split='train', seed=0)
    sample = np.concatenate([observations(env.day_data[d]) for d in sorted(env.days)[:3]])
    agents = [make_agent(m, 0, a.prepared, c) for m in ('ppo', 'fcsrl', 'ppo_anchor', 'fcsrl_anchor')]
    original = agents[0].act_det(sample)
    assert all(np.array_equal(original, agent.act_det(sample)) for agent in agents)
    assert np.allclose(behavior_penalty(np.array([.75, -2., 1.]), np.array([.5, -.5, -1.]), [1, 1, 0]), [.0625, .25, 0.])
    x = env.day_data[sorted(env.days)[0]]
    altered = {k: v.copy() for k, v in x.items()}
    altered['air'][10:] += 2000
    assert np.array_equal(observations(x)[:11], observations(altered)[:11])
    agent = agents[1]
    prior_before = {k: v.clone() for k, v in agent.prior.state_dict().items()}
    norm_before = agent.obs_normalizer.state()
    obs, _ = env.reset()
    shape = [('obs', (11,)), ('obs_next', (11,)), ('act', (1,)), ('act_exec', (1,)), ('logp', ()),
             ('rew', ()), ('cost', ()), ('vr', ()), ('vc', ()), ('nvr', ()), ('nvc', ()),
             ('end', ()), ('terminated', ()), ('next_feasi', ())]
    b = {k: np.zeros((256, 1) + sh, np.float32) for k, sh in shape}
    for t in range(256):
        on = agent.obs_normalizer(obs[None])
        with torch.no_grad():
            ot = agent._t(on)
            dist = agent.dist(ot)
            act = dist.sample()
            b['act'][t] = act.numpy()
            b['logp'][t] = dist.log_prob(act).sum(-1).numpy()
            vr, vc = agent.values(ot)
            b['vr'][t], b['vc'][t] = vr.numpy(), vc.numpy()
        b['obs'][t], b['act_exec'][t] = on, np.clip(b['act'][t], -1, 1)
        obs, rew, term, trunc, info = env.step(b['act_exec'][t, 0])
        assert not term and not trunc
        b['rew'][t], b['cost'][t] = rew, info['cost']
        no = agent.obs_normalizer(obs[None])
        b['obs_next'][t] = no
        with torch.no_grad():
            vr, vc = agent.values(agent._t(no))
            b['nvr'][t], b['nvc'][t] = vr.numpy(), vc.numpy()
            b['next_feasi'][t] = agent.next_feasibility(no)
    b['adv_r'], b['ret_r'] = gae(b['rew'], b['vr'], b['nvr'], b['end'])
    b['adv_c'], b['ret_c'] = gae(b['cost'], b['vc'], b['nvc'], b['end'])
    b['feasi'] = feasibility_targets(b['cost'], b['next_feasi'], b['end'], b['terminated'])
    flat = {k: v.reshape(256, *v.shape[2:]) for k, v in b.items()}
    with torch.no_grad():
        now = agent.dist(agent._t(flat['obs'])).log_prob(agent._t(flat['act'])).sum(-1).numpy()
    assert np.allclose(now, flat['logp'], atol=1e-6)
    metrics = agent.update(flat)
    assert all(np.isfinite(v) for v in metrics.values())
    assert all(torch.equal(v, prior_before[k]) for k, v in agent.prior.state_dict().items())
    assert agent.obs_normalizer.state() == norm_before
    with tempfile.TemporaryDirectory() as td:
        agent.save(td)
        restored = make_agent('fcsrl', 8, a.prepared, c)
        restored.load(td)
        assert np.array_equal(agent.act_det(sample), restored.act_det(sample))
    eps = [(sorted(env.days)[0], 0), (sorted(env.days)[1], 1)]
    seq = AerationEnv(a.data, a.model, split='train', fixed_episodes=eps)
    scalar = summarize_rows(run_policy(seq, lambda o, e: agent.act_det(o)[0], 2), 15.)
    bat = AerationEnv(a.data, a.model, split='train', fixed_episodes=eps)
    vector = batched_rollout(bat, bat.sur, lambda o: agent.act_det(o)[:, 0], 15.)[0]
    # 배치 크기에 따른 float32 추론 차이를 고려해 공기량 차이 0.1 Nm3/일 이내를 검사한다.
    assert abs(scalar['air_m3'] - vector['air_m3']) < .1
    assert scalar['cost'] == vector['cost']
    report = {'passed': True, 'initial_policy_equal_all_four': True, 'current_air_absent_from_input': True,
              'penalty_units_and_aerobic_mask': True, 'raw_log_probability': True,
              'original_fcsrl_update_finite': True, 'prior_and_normalizer_frozen': True,
              'checkpoint_exact_restore': True, 'scalar_batched_air_difference': scalar['air_m3'] - vector['air_m3'],
              'scalar_batched_cost_equal': True, 'update': metrics}
    a.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('data', 'model', 'prepared', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    main(p.parse_args())

"""e63: 정책이 총 공기와 구간 배분 기울기를 함께 정한다.

행동 a = [a0, a1] in [-1, 1]^2.
- a0: 총 공기 (기존과 같음, 0 ~ 9,000 m³/h)
- a1: 배분 기울기 w = W_MAX x clip(a1, 0, 1). 그 시각 배분 = (1 - w) x 기록 배분 + w x BACK.
  a1 <= 0 이면 기록 배분 그대로다. 초기 정책 평균은 a1 = 0 이라 기록 배분에서 시작한다.
W_MAX 0.5 와 BACK [0, 0, .2, .3, .5] 는 e62 에서 coverage 0.979 였던 back_w0.5 와 같고 결과 전에 고정했다.

배분은 환경과 차단이 모두 읽는 env.x['frac'][:, t] 를 그 스텝에 덮어써 넣는다. 그래서 차단은 정책이 고른
배분으로 필요 공기를 계산한다. 원래 기록 배분은 reset 직후 복사해 둔다.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e60'))
import train_shielded as t60  # noqa: E402
se, base, h = t60.se, t60.base, t60.h
from behavior import PriorFCSRL, PriorPPO  # noqa: E402  (e23, e30 common 이 경로를 넣는다)
from ppo_lag import gae, mlp  # noqa: E402
from ppo_fcsrl import PPOFCSRLAgent, feasibility_targets, RepresentationConfig  # noqa: E402
from darrow_env import all_episodes, action_to_air  # noqa: E402

W_MAX = 0.5
BACK = np.array([0., 0., .2, .3, .5])
A_DIM = 2


def weight(a1):
    return W_MAX * np.clip(a1, 0., 1.)


class Prior2(nn.Module):
    """기존 1차원 모방 prior 를 총 공기 축에만 쓰고 배분 축은 0 을 낸다."""

    def __init__(self):
        super().__init__()
        self.base = mlp(11, 1)

    def forward(self, o):
        p = self.base(o)
        return torch.cat([p, torch.zeros_like(p)], dim=-1)


def make_agent(method, seed, prepared, config):
    """e23 behavior.make_agent 와 같은 절차, 행동 차원만 2."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    cls = PriorFCSRL if method == 'fcsrl' else PriorPPO
    agent = cls(11, A_DIM, config=config)
    agent.prior = Prior2().to(agent.dev)
    agent.prior.base.load_state_dict(torch.load(prepared / 'prior.pt', map_location=agent.dev, weights_only=True))
    agent.prior.requires_grad_(False).eval()
    agent.obs_normalizer.load(json.loads((prepared / 'normalizer.json').read_text()))
    with torch.no_grad():
        agent.pi[-1].weight.zero_()
        agent.pi[-1].bias.zero_()
    return agent


def make_pair(method, seed, prepared, c, device):
    """e56 hybrid.make_pair 와 같고 make_agent 만 2차원이다."""
    h.configure('cpu')
    learner = make_agent(method, seed, prepared, c)
    collector = deepcopy(learner)
    if device != 'cpu':
        h.configure(device)
        learner.dev = torch.device(device)
        for module in h.modules(learner).values():
            module.to(device)
        learner.log_std = nn.Parameter(learner.log_std.detach().to(device))
        learner.opt_pi = torch.optim.Adam(list(learner.pi.parameters()) + [learner.log_std], lr=c.actor_lr)
        learner.opt_v = torch.optim.Adam(list(learner.vr.parameters()) + list(learner.vc.parameters()), lr=c.critic_lr)
        if hasattr(learner, 'encoder_optim'):
            params = [p for m in (learner.encoder, learner.feasi_head, learner.proj_layer, learner.post_proj) for p in m.parameters()]
            learner.encoder_optim = torch.optim.Adam(params, lr=learner.repr_config.encoder_lr)
    return learner, collector


def apply_allocation(env, record_frac, a1):
    """그 스텝의 배분을 덮어쓰고 w 를 돌려준다."""
    w = weight(a1)
    t = env.t
    env.x['frac'][:, t] = (1 - w)[:, None] * record_frac[:, t] + w[:, None] * BACK[None, :]
    return w


def collect(agent, c, env, days, rng, shield):
    """e60 collect 와 같고 행동이 2차원이며 배분을 덮어쓴다."""
    T, N = c.horizon, c.n_env
    assert T % env.T == 0
    member = int(rng.integers(env.dyn.n_members))
    b = {k: np.zeros((T, N), np.float32) for k in
         ('logp', 'rew', 'cost', 'vr', 'vc', 'nvr', 'nvc', 'end', 'terminated', 'next_feasi', 'w')}
    for k in ('obs', 'obs_next'):
        b[k] = np.zeros((T, N, 11), np.float32)
    for k in ('act', 'act_exec'):
        b[k] = np.zeros((T, N, A_DIM), np.float32)

    def reset():
        o = env.reset([(days[i], member) for i in rng.integers(len(days), size=N)])
        return o, env.x['frac'].copy()
    obs, record_frac = reset()
    used_steps = 0
    for t in range(T):
        on = agent.obs_normalizer(obs)
        with torch.no_grad():
            ot = agent._t(on)
            dist = agent.dist(ot)
            action = dist.sample()
            b['logp'][t] = dist.log_prob(action).sum(-1).numpy()
            vr, vc = agent.values(ot)
            b['vr'][t], b['vc'][t] = vr.numpy(), vc.numpy()
        b['obs'][t], b['act'][t] = on, action.numpy()
        a = np.clip(b['act'][t], -1, 1)
        b['w'][t] = apply_allocation(env, record_frac, a[:, 1])
        total = a[:, 0]
        if shield is not None:
            total, used = shield(obs, total, env)
            used_steps += int(used.sum())
        b['act_exec'][t, :, 0], b['act_exec'][t, :, 1] = total, a[:, 1]
        nxt, reward, cost, trunc, info = env.step(b['act_exec'][t, :, 0])
        b['rew'][t], b['cost'][t], b['end'][t] = reward, cost, float(trunc)
        no = agent.obs_normalizer(nxt)
        b['obs_next'][t] = no
        with torch.no_grad():
            vr, vc = agent.values(agent._t(no))
            b['nvr'][t], b['nvc'][t] = vr.numpy(), vc.numpy()
            if isinstance(agent, PPOFCSRLAgent):
                b['next_feasi'][t] = agent.next_feasibility(no)
        if trunc and t + 1 < T:
            obs, record_frac = reset()
        else:
            obs = nxt
    b['adv_r'], b['ret_r'] = gae(b['rew'], b['vr'], b['nvr'], b['end'], c.gamma, c.gae_lambda)
    b['adv_c'], b['ret_c'] = gae(b['cost'], b['vc'], b['nvc'], b['end'], c.gamma, c.gae_lambda)
    b['feasi'] = feasibility_targets(b['cost'], b['next_feasi'], b['end'], b['terminated'], RepresentationConfig().feasibility_discount)
    diag = {'member': member, 'daily_cost_mean': float(b['cost'].reshape(-1, env.T, N).sum(1).mean()),
            'raw_action_clip_percent': float((abs(b['act']) > 1).mean() * 100), 'reward_mean': float(b['rew'].mean()),
            'intervention_percent': 100 * used_steps / (T * N), 'w_mean': float(b['w'].mean()), 'w_positive_percent': float((b['w'] > 0).mean() * 100),
            'executed_air_mean': float(action_to_air(b['act_exec'][:, :, 0]).mean())}
    w = b.pop('w')
    return {k: v.reshape(T * N, *v.shape[2:]) for k, v in b.items()}, diag


def rollout(env, dates, act, shield):
    """결정적 평가. act(obs) -> (N, 2) 또는 (N,) (1차원이면 기록 배분). 상태, 실행 공기, 구간 공기, w."""
    episodes = all_episodes(dates, env.dyn.n_members)
    obs = env.reset(episodes)
    record_frac = env.x['frac'].copy()
    N = len(episodes)
    states, airs, zones, ws, used_steps = [], [], [], [], 0
    for _ in range(env.T):
        a = np.asarray(act(obs), dtype=float).reshape(N, -1)
        w = apply_allocation(env, record_frac, a[:, 1]) if a.shape[1] == 2 else np.zeros(N)
        total = np.clip(a[:, 0], -1, 1)
        if shield is not None:
            total, used = shield(obs, total, env)
            used_steps += int(used.sum())
        frac_t = env.x['frac'][:, env.t].copy()
        obs, reward, cost, done, info = env.step(total)
        assert np.array_equal(cost, info['v_nh4'] + info['v_do'])
        states.append(env.state.copy())
        airs.append(info['air'].copy())
        zones.append(info['air'][:, None] * frac_t)
        ws.append(w)
    assert done
    states, airs, zones, ws = (np.stack(x, 1) for x in (states, airs, zones, ws))
    result, rows, members = base.metrics(states, airs, episodes)
    result.pop('both_mean_limits')
    result.pop('both_limits_episode_pct')
    n = (states[:, :, 0] > 4).sum(1)
    d = (states[:, :, 2:].max(2) > 2).sum(1)
    v = states[:, :, 0] > 4
    carry = np.where(v.all(1), v.shape[1], np.argmin(v, axis=1))
    result.update(cost=float((n + d).mean()), nh4_viol_pct=float((100 * n / 95).mean()), do_viol_pct=float((100 * d / 95).mean()),
                  days_within_limit_pct=float(100 * np.mean(n + d <= 12)), n_episodes=N,
                  intervention_pct=0.0 if shield is None else 100 * used_steps / (N * env.T),
                  nh4_carryover=float(carry.mean()), nh4_controllable=float((n - carry).mean()),
                  w_mean=float(ws.mean()), w_positive_pct=float(100 * (ws > 0).mean()))
    return result, rows, states, airs, zones

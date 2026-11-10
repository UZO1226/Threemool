"""DARROW에서 e29의 PPO/FCSRL 및 출력 보정을 같은 예산으로 재학습한다."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from tqdm import tqdm

from common import config, make_agent, METHODS, save
from fcsrl.utils import DeviceConfig
from ppo_lag import gae
from ppo_fcsrl import PPOFCSRLAgent, feasibility_targets, RepresentationConfig
from darrow_env import BatchEnv, Dynamics, load_days, evaluate


def collect(agent, c, env, days, rng):
    T, N = c.horizon, c.n_env
    assert T % env.T == 0
    member = int(rng.integers(env.dyn.n_members))
    b = {k: np.zeros((T, N), np.float32) for k in
         ('logp', 'rew', 'cost', 'vr', 'vc', 'nvr', 'nvc', 'end', 'terminated', 'next_feasi')}
    for k in ('obs', 'obs_next'):
        b[k] = np.zeros((T, N, 11), np.float32)
    for k in ('act', 'act_exec'):
        b[k] = np.zeros((T, N, 1), np.float32)
    def reset():
        return env.reset([(days[i], member) for i in rng.integers(len(days), size=N)])
    obs = reset()
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
        b['act_exec'][t] = np.clip(b['act'][t], -1, 1)
        nxt, reward, cost, trunc, info = env.step(b['act_exec'][t, :, 0])
        b['rew'][t], b['cost'][t], b['end'][t] = reward, cost, float(trunc)
        no = agent.obs_normalizer(nxt)
        b['obs_next'][t] = no
        with torch.no_grad():
            vr, vc = agent.values(agent._t(no))
            b['nvr'][t], b['nvc'][t] = vr.numpy(), vc.numpy()
            if isinstance(agent, PPOFCSRLAgent):
                b['next_feasi'][t] = agent.next_feasibility(no)
        # Terminal observation is bootstrapped before reset; no next-day values leak into it.
        obs = reset() if trunc and t + 1 < T else nxt
    b['adv_r'], b['ret_r'] = gae(b['rew'], b['vr'], b['nvr'], b['end'], c.gamma, c.gae_lambda)
    b['adv_c'], b['ret_c'] = gae(b['cost'], b['vc'], b['nvc'], b['end'], c.gamma, c.gae_lambda)
    b['feasi'] = feasibility_targets(b['cost'], b['next_feasi'], b['end'], b['terminated'],
                                     RepresentationConfig().feasibility_discount)
    diag = {'member': member, 'daily_cost_mean': float(b['cost'].reshape(-1, env.T, N).sum(1).mean()),
            'raw_action_clip_percent': float((abs(b['act']) > 1).mean() * 100),
            'reward_mean': float(b['rew'].mean())}
    return {k: v.reshape(T*N, *v.shape[2:]) for k, v in b.items()}, diag


def train(a):
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    DeviceConfig().select_device(-1)
    c = config(a.updates)
    agent = make_agent(a.method, a.seed, a.prepared, c)
    splits, days = load_days(a.prepared / 'data.pkl')
    env = BatchEnv(days, Dynamics(a.prepared / 'surrogate.pkl'))
    rng = np.random.default_rng(a.seed)
    save(a.out / 'effective_config.json', {'method': a.method, 'seed': a.seed, 'config': c.to_dict(),
         'prepared': str(a.prepared), 'evaluation_split': 'val', 'model_selection': 'last',
         'old_weights_reused': False, 'behavior_penalty': 0})
    start, records = time.monotonic(), []
    norm_before = agent.obs_normalizer.state()
    prior_before = {k: v.clone() for k, v in agent.prior.state_dict().items()}
    for u in tqdm(range(1, c.n_updates + 1), desc=f'{a.method} s{a.seed}', mininterval=10):
        buf, diag = collect(agent, c, env, splits['train'], rng)
        update = agent.update(buf)
        assert all(np.isfinite(v) for v in update.values()), update
        modules = [agent.pi, agent.vr, agent.vc]
        if isinstance(agent, PPOFCSRLAgent):
            modules += [agent.encoder, agent.fixed_encoder, agent.feasi_head, agent.proj_layer, agent.post_proj]
        assert all(torch.isfinite(p).all() for m in modules for p in m.parameters())
        agent.pid.update(diag['daily_cost_mean'], c.cost_limit)
        agent.lagrg = agent.pid.get_lagrg()
        records.append({'update': u, 'lambda': agent.lagrg, **diag, **update})
        elapsed = time.monotonic() - start
        save(a.out / 'train_log.json', records)
        save(a.out / 'progress.json', {'phase': 'training', 'update': u, 'updates': c.n_updates,
             'steps': u*6080, 'total_steps': c.actual_steps, 'elapsed_seconds': elapsed,
             'estimated_remaining_seconds': elapsed/u*(c.n_updates-u), 'latest': records[-1]})
    assert norm_before == agent.obs_normalizer.state()
    assert all(torch.equal(v, prior_before[k]) for k, v in agent.prior.state_dict().items())
    agent.save(a.out / 'model_final')
    save(a.out / 'progress.json', {'phase': 'validation', 'steps': c.actual_steps})
    result = evaluate(env, splits['val'], lambda x: agent.act_det(x)[:, 0])
    save(a.out / 'metrics.json', {'method': a.method, 'seed': a.seed, 'val_final': result,
         'steps': c.actual_steps, 'elapsed_seconds': time.monotonic()-start,
         'normalizer_and_prior_frozen': True})
    save(a.out / 'progress.json', {'phase': 'complete', 'steps': c.actual_steps})
    print('DONE', result, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--updates', type=int, default=66)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    train(p.parse_args())

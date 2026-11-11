"""DARROW 환경에서 PPO-Lag 와 FCSRL + PPO-Lag 를 학습한다 (Yoo/e06 train_ppo.py 의 수집, 갱신, 선택 흐름을 따름).

바뀐 점: 환경이 벡터화된 VecAerationEnv 이고, 비용 한도를 인자로 받으며, 관측 11차원은 같다.
Yoo 의 PPO, FCSRL 구현(ppo_lag.py, ppo_fcsrl.py)과 원본 FCSRL 은 수정하지 않는다.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import fcsrl_compat  # noqa: F401,E402  Windows 에서 safety_gymnasium 없이 원본 FCSRL 을 가져온다
import numpy as np  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402
from fcsrl.utils import DeviceConfig  # noqa: E402
from env_dw import OBS_DIM, T_STEPS, VecAerationEnv, batched_rollout  # noqa: E402
from feasibility_policy import FeasibilityPenaltyConfig, PPOFCSRLFeasibilityPenaltyAgent  # noqa: E402
from ppo_config import PPOConfig  # noqa: E402
from ppo_fcsrl import PPOFCSRLAgent, RepresentationConfig, feasibility_targets  # noqa: E402
from ppo_lag import PPOLagAgent, gae  # noqa: E402
from surrogate_dw import GBMDynamics  # noqa: E402


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def prefer(candidate, best, limit):
    """공통 한도 만족 시 최고 보상, 만족 모델이 없으면 최소 비용을 고른다 (Yoo 규칙)."""
    if best is None:
        return True
    feasible, old_feasible = candidate['cost'] <= limit, best['cost'] <= limit
    if feasible != old_feasible:
        return feasible
    return candidate['reward'] > best['reward'] if feasible else candidate['cost'] < best['cost']


def make_agent(a, c):
    r = RepresentationConfig()
    if a.method == 'ppolag':
        return PPOLagAgent(OBS_DIM, 1, config=c), r
    if a.method in ('ppolag_fcsrl', 'ppolag_fcsrl_curriculum'):
        return PPOFCSRLAgent(OBS_DIM, 1, config=c, repr_config=r), r
    if a.method == 'ppolag_fcsrl_fpen':
        return PPOFCSRLFeasibilityPenaltyAgent(
            OBS_DIM, 1, config=c, repr_config=r,
            feasibility_penalty=FeasibilityPenaltyConfig(weight=a.fpen_weight, relative=True)), r
    raise ValueError(a.method)


def train(a):
    run = a.out
    if run.exists() and any(run.iterdir()):
        raise ValueError(f'Refusing to overwrite {run}')
    run.mkdir(parents=True, exist_ok=True)
    kw = {}
    if a.pid_mult != 1.0:
        base = PPOConfig()
        kw = dict(pid_kp=base.pid_kp * a.pid_mult, pid_ki=base.pid_ki * a.pid_mult, pid_kd=base.pid_kd * a.pid_mult)
    c = PPOConfig(total_steps=a.steps, n_env=8, horizon=8 * T_STEPS, n_chunks=a.evaluations,
                  cost_limit=a.limit, **kw)
    DeviceConfig().select_device(-1)
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    agent, r = make_agent(a, c)
    dyn = GBMDynamics(a.surrogate)
    env = VecAerationEnv(a.data, None, split='train', n_env=c.n_env, seed=a.seed * 1000, dynamics=dyn)
    curriculum = None
    if a.method == 'ppolag_fcsrl_curriculum':
        from curriculum_dw import risk_day_weights
        env.day_weights, curriculum = risk_day_weights(env.table, env.days)
    write_json(run / 'effective_config.json', {
        'method': a.method, 'seed': a.seed, 'config': c.to_dict(), 'pid_mult': a.pid_mult,
        'representation': asdict(r) if isinstance(agent, PPOFCSRLAgent) else None,
        'feasibility_penalty': asdict(agent.feasibility_penalty) if hasattr(agent, 'feasibility_penalty') else None,
        'cost_limit': a.limit, 'data': str(a.data.resolve()), 'surrogate': str(a.surrogate.resolve()),
        'actual_steps': c.actual_steps, 'curriculum': curriculum})
    obs = env.reset()
    agent.obs_normalizer.update(obs)
    ep_cost, ep_reward = np.zeros(c.n_env), np.zeros(c.n_env)
    log, pid_trace, best, final = [], [], None, None
    eval_at = set(np.ceil(np.linspace(0, c.n_updates, c.n_chunks + 1)[1:]).astype(int))
    start = time.monotonic()
    bar = tqdm(total=c.actual_steps, desc=f'{a.method} seed {a.seed}', mininterval=10)
    try:
        for u in range(1, c.n_updates + 1):
            B = {key: np.zeros((c.horizon, c.n_env) + shape, dtype=np.float32) for key, shape in
                 (('obs', (OBS_DIM,)), ('obs_next', (OBS_DIM,)), ('act', (1,)), ('act_exec', (1,)),
                  ('logp', ()), ('rew', ()), ('cost', ()), ('vr', ()), ('vc', ()), ('nvr', ()),
                  ('nvc', ()), ('end', ()), ('terminated', ()), ('next_feasi', ()))}
            completed = []
            env.set_member(np.random.randint(dyn.n_members))
            for t in range(c.horizon):
                on = agent.obs_normalizer(obs)
                with torch.no_grad():
                    ot = agent._t(on)
                    dist = agent.dist(ot)
                    actions = dist.sample()
                    B['logp'][t] = dist.log_prob(actions).sum(-1).cpu().numpy()
                    vr, vc = agent.values(ot)
                    B['vr'][t], B['vc'][t] = vr.cpu().numpy(), vc.cpu().numpy()
                actions = actions.cpu().numpy()
                B['obs'][t], B['act'][t] = on, actions
                B['act_exec'][t] = np.clip(actions, -1, 1)
                new_obs, reward, cost, trunc, terminal_obs, _ = env.step(B['act_exec'][t, :, 0])
                B['rew'][t], B['cost'][t] = reward, cost
                B['end'][t], B['terminated'][t] = trunc.astype(np.float32), 0.0
                ep_cost += cost
                ep_reward += reward
                for i in np.flatnonzero(trunc):
                    completed.append((float(ep_reward[i]), float(ep_cost[i])))
                    ep_reward[i] = ep_cost[i] = 0
                next_on = agent.obs_normalizer(terminal_obs)
                B['obs_next'][t] = next_on
                with torch.no_grad():
                    vr, vc = agent.values(agent._t(next_on))
                    B['nvr'][t], B['nvc'][t] = vr.cpu().numpy(), vc.cpu().numpy()
                    if isinstance(agent, PPOFCSRLAgent):
                        B['next_feasi'][t] = agent.next_feasibility(next_on)
                obs = new_obs
                agent.obs_normalizer.update(obs)
            B['adv_r'], B['ret_r'] = gae(B['rew'], B['vr'], B['nvr'], B['end'], c.gamma, c.gae_lambda)
            B['adv_c'], B['ret_c'] = gae(B['cost'], B['vc'], B['nvc'], B['end'], c.gamma, c.gae_lambda)
            B['feasi'] = feasibility_targets(B['cost'], B['next_feasi'], B['end'], B['terminated'],
                                             r.feasibility_discount)
            flat = {key: value.reshape(c.horizon * c.n_env, *value.shape[2:]) for key, value in B.items()}
            info = agent.update(flat)
            if hasattr(agent, 'last_penalty_rms'):
                info['fpen_rms'] = agent.last_penalty_rms
                info['fpen_to_base_rms'] = agent.last_penalty_rms / max(agent.last_base_rms, 1e-8)
            if not all(np.isfinite(v) for v in info.values()):
                raise FloatingPointError(f'Nonfinite update metrics: {info}')
            modules = [agent.pi, agent.vr, agent.vc]
            if isinstance(agent, PPOFCSRLAgent):
                modules += [agent.encoder, agent.fixed_encoder, agent.feasi_head, agent.proj_layer, agent.post_proj]
            if any(not torch.isfinite(p).all() for module in modules for p in module.parameters()):
                raise FloatingPointError('Nonfinite model parameter')
            if completed:
                cost_mean = float(np.mean([x[1] for x in completed]))
                agent.pid.update(cost_mean, c.cost_limit)
                agent.lagrg = agent.pid.get_lagrg()
                pid_trace.append({'update': u, 'cost': cost_mean, 'lambda': agent.lagrg})
            bar.update(c.horizon * c.n_env)
            bar.set_postfix(lam=f'{agent.lagrg:.2f}', cost=f'{cost_mean:.1f}')
            record = {'update': u, 'steps': u * c.horizon * c.n_env, 'lambda': agent.lagrg,
                      'train_episode_cost': cost_mean, **info}
            if u in eval_at:
                epoch = len([x for x in log if 'val' in x]) + 1
                final, rows = batched_rollout(a.data, dyn, 'val',
                                              lambda x: agent.act_det(x)[:, 0], c.cost_limit)
                candidate = {'epoch': epoch, **final}
                agent.save(run / 'model' / f'epoch_{epoch}')
                if prefer(candidate, best, c.cost_limit):
                    best = candidate
                record.update(epoch=epoch, val=final)
                print(f'\nValidation {epoch}: cost={final["cost"]:.2f}, air={final["air_m3"]:.0f}', flush=True)
            log.append(record)
            elapsed = time.monotonic() - start
            write_json(run / 'progress.json', {'phase': 'training', 'update': u, 'updates': c.n_updates,
                                               'steps': record['steps'], 'total_steps': c.actual_steps,
                                               'elapsed_seconds': elapsed,
                                               'estimated_remaining_seconds': elapsed / u * (c.n_updates - u)})
            write_json(run / 'train_log.json', log)
        agent.save(run / 'model_final')
        write_json(run / 'metrics.json', {
            'method': a.method, 'seed': a.seed, 'best_epoch': best['epoch'], 'best_feasible': best['cost'] <= c.cost_limit,
            'evaluation_split': 'val', 'val_best': best, 'val_final': final, 'steps': c.actual_steps,
            'report_cost_limit': c.cost_limit, 'elapsed_seconds': time.monotonic() - start})
        write_json(run / 'progress.json', {'phase': 'complete', 'steps': c.actual_steps, 'total_steps': c.actual_steps,
                                           'elapsed_seconds': time.monotonic() - start})
    finally:
        bar.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['ppolag', 'ppolag_fcsrl', 'ppolag_fcsrl_fpen', 'ppolag_fcsrl_curriculum'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--steps', type=int, default=400_000)
    p.add_argument('--evaluations', type=int, default=5)
    p.add_argument('--limit', type=float, required=True, help='하루 비용 한도')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--surrogate', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--fpen-weight', type=float, default=0.1)
    p.add_argument('--pid-mult', type=float, default=1.0)
    train(p.parse_args())

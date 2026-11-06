"""학습 전반부의 표현을 고정하는 대조. e05의 수집 및 PPO 계산을 유지한다."""
import argparse
from dataclasses import asdict
import json
import hashlib
from pathlib import Path
import sys
import time

CODE = Path(__file__).resolve().parents[2] / 'code/wwtp'
sys.path.insert(0, str(CODE))

import numpy as np
import torch
from tqdm import tqdm
from fcsrl.utils import DeviceConfig
from env import AerationEnv
from evaluate import batched_rollout, episodes
from ppo_config import PPOConfig
from ppo_lag import PPOLagAgent, gae
from ppo_fcsrl import PPOFCSRLAgent, RepresentationConfig, feasibility_targets
from ppo_tail import risk_metrics
from scheduled_agent import ScheduledAgent, representation_digest


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def prefer(candidate, best, limit):
    """공통 한도 만족 시 최고 보상, 만족 모델이 없으면 최소 비용을 고른다."""
    if best is None:
        return True
    feasible, old_feasible = candidate['cost'] <= limit, best['cost'] <= limit
    if feasible != old_feasible:
        return feasible
    return candidate['reward'] > best['reward'] if feasible else candidate['cost'] < best['cost']


def train(a):
    run = a.out
    if run.exists() and any(run.iterdir()):
        raise ValueError(f'Refusing to overwrite {run}')
    c = PPOConfig(total_steps=a.steps, n_chunks=a.evaluations)
    run.mkdir(parents=True, exist_ok=True)
    DeviceConfig().select_device(-1)
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if a.freeze_after and not 0 < a.freeze_after < c.n_updates:
        raise ValueError('Freeze boundary must precede the final update')
    r = RepresentationConfig()
    agent = ScheduledAgent(11, 1, config=c, repr_config=r, freeze_after=a.freeze_after)
    envs = [AerationEnv(a.data, a.model, split='train', seed=a.seed * 1000 + i) for i in range(c.n_env)]
    val = AerationEnv(a.data, a.model, split='val')
    val.fixed = episodes(val)
    validation = {'split': 'val', 'episodes': len(val.fixed), 'days': val.days, 'members': val.n_members}
    write_json(run / 'effective_config.json', {'method': 'ppolag_fcsrl', 'variant': a.variant, 'freeze_after': a.freeze_after, 'seed': a.seed, 'config': c.to_dict(),
        'representation': asdict(r), 'validation': validation,
        'report_cost_limit': c.cost_limit, 'device': 'cpu', 'data': str(a.data.resolve()),
        'surrogate': str(a.model.resolve()), 'actual_steps': c.actual_steps})
    obs = np.stack([e.reset()[0] for e in envs])
    agent.obs_normalizer.update(obs)
    ep_cost = np.zeros(c.n_env)
    ep_reward = np.zeros(c.n_env)
    log, pid_trace, best, final = [], [], None, None
    eval_at = set(np.ceil(np.linspace(0, c.n_updates, c.n_chunks + 1)[1:]).astype(int))
    start = time.monotonic()
    bar = tqdm(total=c.actual_steps, desc=f'{a.variant} seed {a.seed}', mininterval=5)
    try:
        for u in range(1, c.n_updates + 1):
            B = {key: np.zeros((c.horizon, c.n_env) + shape, dtype=np.float32) for key, shape in
                 (('obs', (11,)), ('obs_next', (11,)), ('act', (1,)), ('act_exec', (1,)),
                  ('logp', ()), ('rew', ()), ('cost', ()), ('vr', ()), ('vc', ()), ('nvr', ()),
                  ('nvc', ()), ('end', ()), ('terminated', ()), ('next_feasi', ()))}
            completed = []
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
                nxt = np.empty_like(obs)
                for i, env in enumerate(envs):
                    o2, reward, term, trunc, info = env.step(B['act_exec'][t, i])
                    if term:
                        raise RuntimeError('This protocol expects time-limit truncation only')
                    B['rew'][t, i], B['cost'][t, i] = reward, info['cost']
                    B['end'][t, i], B['terminated'][t, i] = term or trunc, term
                    ep_cost[i] += info['cost']
                    ep_reward[i] += reward
                    nxt[i] = o2
                    if term or trunc:
                        completed.append((float(ep_reward[i]), float(ep_cost[i])))
                        ep_reward[i] = ep_cost[i] = 0
                        o2, _ = env.reset()
                    obs[i] = o2
                next_on = agent.obs_normalizer(nxt)
                B['obs_next'][t] = next_on
                with torch.no_grad():
                    vr, vc = agent.values(agent._t(next_on))
                    B['nvr'][t], B['nvc'][t] = vr.cpu().numpy(), vc.cpu().numpy()
                    if isinstance(agent, PPOFCSRLAgent):
                        B['next_feasi'][t] = agent.next_feasibility(next_on)
                agent.obs_normalizer.update(obs)
            B['adv_r'], B['ret_r'] = gae(B['rew'], B['vr'], B['nvr'], B['end'], c.gamma, c.gae_lambda)
            B['adv_c'], B['ret_c'] = gae(B['cost'], B['vc'], B['nvc'], B['end'], c.gamma, c.gae_lambda)
            B['feasi'] = feasibility_targets(B['cost'], B['next_feasi'], B['end'], B['terminated'], r.feasibility_discount)
            flat = {key: value.reshape(c.horizon * c.n_env, *value.shape[2:]) for key, value in B.items()}
            info = agent.update(flat)
            if u == a.boundary_update:
                agent.save(run / 'model_boundary')
                write_json(run / 'boundary.json', {'update': u, 'steps': u * c.horizon * c.n_env,
                    'representation_sha256': representation_digest(agent),
                    'torch_rng_sha256': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()})
            if not all(np.isfinite(v) for v in info.values()):
                raise FloatingPointError(f'Nonfinite update metrics: {info}')
            modules = [agent.pi, agent.vr, agent.vc]
            if isinstance(agent, PPOFCSRLAgent):
                modules += [agent.encoder, agent.fixed_encoder, agent.feasi_head, agent.proj_layer, agent.post_proj]
            if any(not torch.isfinite(p).all() for module in modules for p in module.parameters()):
                raise FloatingPointError('Nonfinite model parameter')
            if completed:
                cost = float(np.mean([x[1] for x in completed]))
                agent.pid.update(cost, c.cost_limit)
                agent.lagrg = agent.pid.get_lagrg()
                pid_trace.append({'update': u, 'cost': cost, 'lambda': agent.lagrg,
                                  'integral': float(agent.pid.error_integral)})
            bar.update(c.horizon * c.n_env)
            bar.set_postfix(lam=f'{agent.lagrg:.2f}', std=f'{info["std"]:.2f}')
            record = {'update': u, 'steps': u * c.horizon * c.n_env, 'lambda': agent.lagrg, **info}
            if u in eval_at:
                epoch = len([x for x in log if 'val' in x]) + 1
                final, rows = batched_rollout(val, val.sur, lambda x: agent.act_det(x)[:, 0], c.cost_limit)
                final.update(risk_metrics(rows))
                candidate = {'epoch': epoch, **final}
                agent.save(run / 'model' / f'epoch_{epoch}')
                if prefer(candidate, best, c.cost_limit):
                    best = candidate
                    write_json(run / 'val_best_episodes.json', rows)
                record.update(epoch=epoch, val=final)
                print(f'\nValidation {epoch}: cost={final["cost"]:.3f}, air={final["air_m3"]:.1f}', flush=True)
            log.append(record)
            elapsed = time.monotonic() - start
            write_json(run / 'progress.json', {'phase': 'training', 'update': u, 'updates': c.n_updates,
                'steps': record['steps'], 'total_steps': c.actual_steps, 'elapsed_seconds': elapsed,
                'estimated_remaining_seconds': elapsed / u * (c.n_updates - u), 'latest': record})
            write_json(run / 'train_log.json', log)
            write_json(run / 'pid_trace.json', pid_trace)
        agent.save(run / 'model_final')
        write_json(run / 'val_final_episodes.json', rows)
        write_json(run / 'metrics.json', {'method': 'ppolag_fcsrl', 'variant': a.variant, 'freeze_after': a.freeze_after, 'seed': a.seed, 'best_epoch': best['epoch'],
            'best_feasible': best['cost'] <= c.cost_limit, 'evaluation_split': 'val', 'validation': validation,
            'val_best': best, 'val_final': final, 'steps': c.actual_steps, 'report_cost_limit': c.cost_limit,
            'elapsed_seconds': time.monotonic() - start})
        write_json(run / 'progress.json', {'phase': 'complete', 'steps': c.actual_steps,
            'total_steps': c.actual_steps, 'elapsed_seconds': time.monotonic() - start})
    finally:
        bar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['continuous', 'freeze_half'], required=True)
    parser.add_argument('--freeze-after', type=int, default=0)
    parser.add_argument('--boundary-update', type=int, default=22)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--steps', type=int, default=400_000)
    parser.add_argument('--evaluations', type=int, default=5)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    if (a.variant == 'continuous') != (a.freeze_after == 0):
        parser.error('continuous requires freeze-after 0; freeze_half requires a positive value')
    train(a)

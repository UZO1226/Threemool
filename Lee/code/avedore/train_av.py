"""Lee/e06: Yoo/e06 train_ppo.py (Avedøre 대리 환경, PPO-Lag 와 FCSRL + PPO-Lag)에
feasibility 벌점 방법(ppolag_fcsrl_fpen)을 추가한 학습 스크립트.

수집, 검증, 모델 선택, 기록은 Yoo 의 코드를 그대로 따르고 바뀐 곳은 아래 세 군데다.
1. 코드 경로를 Lee/code/avedore 로 둔다 (Yoo 코드의 수정 없는 복사본)
2. 방법 ppolag_fcsrl_fpen 과 --fpen-weight 인자를 더한다
3. 벌점 크기를 기록한다 (fpen_rms, fpen_to_base_rms)
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

CODE = Path(__file__).resolve().parents[2] / 'code/avedore'
sys.path.insert(0, str(CODE))

import fcsrl_compat  # noqa: F401  Windows 에서 safety_gymnasium 없이 원본 FCSRL 을 가져온다
import numpy as np
import torch
from tqdm import tqdm
from fcsrl.utils import DeviceConfig
from env import AerationEnv
from evaluate import batched_rollout, episodes
from ppo_config import PPOConfig
from ppo_lag import PPOLagAgent, gae
from ppo_fcsrl import PPOFCSRLAgent, RepresentationConfig, feasibility_targets
from ppo_tail import TailConfig, PPOTailAgent, PPOFCSRLTailAgent, episode_tail_scores, risk_metrics
from feasibility_policy import FeasibilityPenaltyConfig, PPOFCSRLFeasibilityPenaltyAgent


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
    r = RepresentationConfig()
    tail = TailConfig(tail_fraction=a.tail_fraction, weight=a.tail_weight, cost_scale=c.cost_limit)
    if a.method == 'ppolag':
        agent = PPOLagAgent(11, 1, config=c)
    elif a.method == 'ppolag_fcsrl':
        agent = PPOFCSRLAgent(11, 1, config=c, repr_config=r)
    elif a.method == 'ppolag_fcsrl_fpen':
        agent = PPOFCSRLFeasibilityPenaltyAgent(
            11, 1, config=c, repr_config=r, feasibility_penalty=FeasibilityPenaltyConfig(weight=a.fpen_weight, relative=a.fpen_relative))
    elif a.method == 'ppolag_tail':
        agent = PPOTailAgent(11, 1, config=c, tail_config=tail)
    else:
        agent = PPOFCSRLTailAgent(11, 1, config=c, repr_config=r, tail_config=tail)
    envs = [AerationEnv(a.data, a.model, split='train', seed=a.seed * 1000 + i) for i in range(c.n_env)]
    if len({e.T for e in envs}) != 1 or c.horizon % envs[0].T:
        raise ValueError('CVaR collection must contain complete equal-length daily episodes')
    if c.n_env * c.horizon / envs[0].T * tail.tail_fraction < 4:
        raise ValueError('At least four episodes of nominal tail mass are required per update')
    val = AerationEnv(a.data, a.model, split='val')
    val.fixed = episodes(val)
    validation = {'split': 'val', 'episodes': len(val.fixed), 'days': val.days, 'members': val.n_members}
    write_json(run / 'effective_config.json', {'method': a.method, 'seed': a.seed, 'config': c.to_dict(),
        'representation': asdict(r) if isinstance(agent, PPOFCSRLAgent) else None, 'validation': validation,
        'tail': asdict(tail), 'tail_enabled': a.method.endswith('_tail'),
        'feasibility_penalty': asdict(agent.feasibility_penalty) if hasattr(agent, 'feasibility_penalty') else None,
        'report_cost_limit': c.cost_limit, 'device': 'cpu', 'data': str(a.data.resolve()),
        'surrogate': str(a.model.resolve()), 'actual_steps': c.actual_steps})
    obs = np.stack([e.reset()[0] for e in envs])
    agent.obs_normalizer.update(obs)
    ep_cost = np.zeros(c.n_env)
    ep_reward = np.zeros(c.n_env)
    log, pid_trace, best, final = [], [], None, None
    eval_at = set(np.ceil(np.linspace(0, c.n_updates, c.n_chunks + 1)[1:]).astype(int))
    start = time.monotonic()
    bar = tqdm(total=c.actual_steps, desc=f'{a.method} seed {a.seed}', mininterval=5)
    try:
        for u in range(1, c.n_updates + 1):
            if any(e.t != 0 for e in envs):
                raise RuntimeError('Daily CVaR rollout started inside an episode')
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
            B['tail_score'], tail_info = episode_tail_scores(B['cost'], B['end'], tail)
            flat = {key: value.reshape(c.horizon * c.n_env, *value.shape[2:]) for key, value in B.items()}
            ar, ac = flat['adv_r'], flat['adv_c']
            base_adv = ((ar - ar.mean()) / (ar.std() + 1e-8) - agent.lagrg * (ac - ac.mean())) / (1 + agent.lagrg)
            component = tail.weight * flat['tail_score'] / (1 + agent.lagrg) if a.method.endswith('_tail') else np.zeros_like(base_adv)
            tail_info['tail_component_rms'] = float(np.sqrt(np.mean(component ** 2)))
            tail_info['base_advantage_rms'] = float(np.sqrt(np.mean(base_adv ** 2)))
            tail_info['tail_to_base_rms'] = tail_info['tail_component_rms'] / max(tail_info['base_advantage_rms'], 1e-8)
            info = agent.update(flat)
            info.update(tail_info)
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
                final.update(risk_metrics(rows, tail.tail_fraction))
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
        write_json(run / 'metrics.json', {'method': a.method, 'seed': a.seed, 'best_epoch': best['epoch'],
            'best_feasible': best['cost'] <= c.cost_limit, 'evaluation_split': 'val', 'validation': validation,
            'val_best': best, 'val_final': final, 'steps': c.actual_steps, 'report_cost_limit': c.cost_limit,
            'tail': asdict(tail), 'tail_enabled': a.method.endswith('_tail'),
            'elapsed_seconds': time.monotonic() - start})
        write_json(run / 'progress.json', {'phase': 'complete', 'steps': c.actual_steps,
            'total_steps': c.actual_steps, 'elapsed_seconds': time.monotonic() - start})
    finally:
        bar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['ppolag', 'ppolag_fcsrl', 'ppolag_fcsrl_fpen', 'ppolag_tail', 'ppolag_fcsrl_tail'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--steps', type=int, default=400_000)
    parser.add_argument('--evaluations', type=int, default=5)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--fpen-weight', type=float, default=0.1)
    parser.add_argument('--fpen-relative', action='store_true', help='벌점 크기를 기본 이점 크기의 weight 배로 맞춘다')
    parser.add_argument('--tail-fraction', type=float, default=0.2)
    parser.add_argument('--tail-weight', type=float, default=0.03)
    train(parser.parse_args())

"""e17 고정 예산 시험. 기존 agent와 tail 구현을 그대로 호출한다."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import warnings

import numpy as np
from scipy.integrate import ODEintWarning
import torch
from tqdm import tqdm
from bsm_mean import MeanEnv, normalized_cost, evaluate, ROOT
from fcsrl.utils import DeviceConfig
from ppo_config import PPOConfig
from ppo_lag import PPOLagAgent, gae
from ppo_fcsrl import PPOFCSRLAgent, RepresentationConfig, feasibility_targets
from ppo_tail import PPOTailAgent, PPOFCSRLTailAgent, TailConfig, episode_tail_scores

METHODS = {'ppo': PPOLagAgent, 'ppo_tail': PPOTailAgent,
           'fcsrl': PPOFCSRLAgent, 'fcsrl_tail': PPOFCSRLTailAgent}


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def config(updates=4):
    return PPOConfig(total_steps=updates * 5760, n_env=4, horizon=1440, n_chunks=1,
        cost_limit=288., init_log_std=-1.5,
        pid_kp=.02 * 15 / 288, pid_ki=.005 * 15 / 288, pid_kd=.01 * 15 / 288)


def make_agent(method, seed, observations, updates=4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    c = config(updates)
    tail = TailConfig(cost_scale=c.cost_limit)
    extra = {'tail_config': tail} if method.endswith('tail') else {}
    agent = METHODS[method](11, 1, config=c, **extra)
    agent.obs_normalizer.update(observations)
    # 기준 운전에서 상수인 특징도 다른 행동을 표현할 수 있도록 물리 척도 하한을 둔다.
    floors = np.array([1., .5, 1., 1., .5, .2, 5., .1, .5, .5, .25])
    agent.obs_normalizer.m2 = np.maximum(agent.obs_normalizer.m2, agent.obs_normalizer.n * floors ** 2)
    return agent, c, tail


def collect(agent, c, tail, prepared, rng, bar=None):
    states = [dict(np.load(prepared / f'day_{d}.npz')) for d in range(7)]
    envs = [MeanEnv(states[0]) for _ in range(c.n_env)]
    schedule = rng.integers(0, 7, size=(c.horizon // 288, c.n_env))
    obs = np.empty((c.n_env, 11), np.float32)
    def reset_day(i, block):
        day = int(schedule[block, i])
        envs[i].initial = states[day]
        envs[i].start_index = day * 1440
        return envs[i].reset()[0]
    for i in range(c.n_env):
        obs[i] = reset_day(i, 0)
    shapes = (('obs', (11,)), ('obs_next', (11,)), ('act', (1,)), ('act_exec', (1,)),
              ('logp', ()), ('rew', ()), ('cost', ()), ('flow', ()), ('vr', ()), ('vc', ()),
              ('nvr', ()), ('nvc', ()), ('end', ()), ('terminated', ()), ('next_feasi', ()))
    b = {k: np.zeros((c.horizon, c.n_env) + shape, np.float32) for k, shape in shapes}
    for t in range(c.horizon):
        on = agent.obs_normalizer(obs)
        with torch.no_grad():
            ot = agent._t(on)
            dist = agent.dist(ot)
            actions = dist.sample()
            b['logp'][t] = dist.log_prob(actions).sum(-1).cpu().numpy()
            vr, vc = agent.values(ot)
            b['vr'][t], b['vc'][t] = vr.cpu().numpy(), vc.cpu().numpy()
        b['obs'][t], b['act'][t] = on, actions.cpu().numpy()
        b['act_exec'][t] = np.clip(b['act'][t], -1, 1)
        nxt = np.empty_like(obs)
        for i, env in enumerate(envs):
            nxt[i], reward, term, trunc, info = env.step(b['act_exec'][t, i])
            assert not trunc and term == ((t + 1) % 288 == 0)
            b['rew'][t, i], b['cost'][t, i], b['flow'][t, i] = reward, info['load'], info['flow_ratio']
            b['end'][t, i] = b['terminated'][t, i] = term
        next_on = agent.obs_normalizer(nxt)
        b['obs_next'][t] = next_on
        with torch.no_grad():
            vr, vc = agent.values(agent._t(next_on))
            b['nvr'][t] = vr.cpu().numpy() * (1 - b['terminated'][t])
            b['nvc'][t] = vc.cpu().numpy() * (1 - b['terminated'][t])
            if isinstance(agent, PPOFCSRLAgent):
                b['next_feasi'][t] = agent.next_feasibility(next_on)
        obs = nxt
        if (t + 1) % 288 == 0:
            lo = t + 1 - 288
            b['cost'][lo:t + 1] = normalized_cost(b['cost'][lo:t + 1], b['flow'][lo:t + 1])
            if t + 1 < c.horizon:
                for i in range(c.n_env):
                    obs[i] = reset_day(i, (t + 1) // 288)
        if bar is not None:
            bar.update(c.n_env)
    b['adv_r'], b['ret_r'] = gae(b['rew'], b['vr'], b['nvr'], b['end'], c.gamma, c.gae_lambda)
    b['adv_c'], b['ret_c'] = gae(b['cost'], b['vc'], b['nvc'], b['end'], c.gamma, c.gae_lambda)
    r = RepresentationConfig()
    b['feasi'] = feasibility_targets(b['cost'], b['next_feasi'], b['end'], b['terminated'], r.feasibility_discount)
    b['tail_score'], tail_info = episode_tail_scores(b['cost'], b['end'], tail)
    assert tail_info['tail_effective_mass'] >= 4
    daily_cost = b['cost'].reshape(-1, 288, c.n_env).sum(axis=1).ravel()
    diagnostics = {'daily_nh4_mean': float(daily_cost.mean() / 288 * 4),
        'daily_cost_mean': float(daily_cost.mean()), 'stage_cost_max': float(b['cost'].max()),
        'feasibility_symlog_outside_fraction': float((np.log1p(b['feasi']) > r.bucket_high).mean()),
        'raw_action_clip_fraction': float((np.abs(b['act']) > 1).mean()),
        'observation_clip_fraction': float((np.abs(b['obs']) >= 10).mean()),
        'training_days': schedule.tolist(), **tail_info}
    flat = {k: v.reshape(c.horizon * c.n_env, *v.shape[2:]) for k, v in b.items()}
    return flat, diagnostics


def checked_update(agent, buf):
    norms = []
    original = torch.nn.utils.clip_grad_norm_
    def capture(*args, **kwargs):
        value = original(*args, **kwargs)
        norms.append(float(value))
        if not np.isfinite(norms[-1]):
            raise FloatingPointError('Nonfinite gradient')
        return value
    probe = agent._t(buf['obs'][:512])
    with torch.no_grad():
        d = agent.dist(probe)
        before = torch.distributions.Normal(d.mean.clone(), d.stddev.clone())
        pi_before = torch.cat([p.detach().flatten() for p in agent.pi.parameters()]).clone()
    torch.nn.utils.clip_grad_norm_ = capture
    try:
        info = agent.update(buf)
    finally:
        torch.nn.utils.clip_grad_norm_ = original
    modules = [agent.pi, agent.vr, agent.vc]
    if isinstance(agent, PPOFCSRLAgent):
        modules += [agent.encoder, agent.fixed_encoder, agent.feasi_head, agent.proj_layer, agent.post_proj]
    assert all(torch.isfinite(p).all() for m in modules for p in m.parameters())
    assert torch.isfinite(agent.log_std).all()
    with torch.no_grad():
        info['full_update_probe_kl'] = float(torch.distributions.kl_divergence(before, agent.dist(probe)).sum(-1).mean())
        pi_after = torch.cat([p.detach().flatten() for p in agent.pi.parameters()])
        info['policy_parameter_delta_l2'] = float(torch.linalg.vector_norm(pi_after - pi_before))
    info['gradient_norm_max'] = max(norms)
    info['gradient_clip_fraction'] = float(np.mean(np.array(norms) > agent.config.max_grad_norm))
    assert all(np.isfinite(v) for v in info.values())
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--updates', type=int, default=4)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    warnings.filterwarnings('error', category=ODEintWarning)
    torch.set_num_threads(1)
    DeviceConfig().select_device(-1)
    agent, c, tail = make_agent(a.method, a.seed, np.load(a.prepared / 'train_observations.npy'), a.updates)
    initial = dict(np.load(a.prepared / 'initial.npz'))
    save(a.out / 'config.json', {'method': a.method, 'seed': a.seed, 'config': asdict(c),
        'tail': asdict(tail), 'representation': asdict(RepresentationConfig()),
        'started_at': datetime.now(timezone.utc).isoformat(),
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'prepared_metadata_sha256': hashlib.sha256((a.prepared / 'metadata.json').read_bytes()).hexdigest(),
        'normalizer': agent.obs_normalizer.state()})
    start = time.monotonic()
    save(a.out / 'progress.json', {'phase': 'initial_evaluation', 'steps': 0, 'total_steps': c.actual_steps})
    baseline = evaluate(initial, lambda o: agent.act_det(o))
    save(a.out / 'initial_evaluation.json', baseline)
    rng = np.random.default_rng(10000 + a.seed)
    records = []
    bar = tqdm(total=c.actual_steps, desc=f'{a.method} seed={a.seed}', mininterval=5)
    for update in range(1, c.n_updates + 1):
        tick = time.monotonic()
        buf, diag = collect(agent, c, tail, a.prepared, rng, bar)
        ar, ac = buf['adv_r'], buf['adv_c']
        base_adv = ((ar - ar.mean()) / (ar.std() + 1e-8) - agent.lagrg * (ac - ac.mean())) / (1 + agent.lagrg)
        component = tail.weight * buf['tail_score'] / (1 + agent.lagrg) if a.method.endswith('tail') else np.zeros_like(base_adv)
        diag['tail_to_base_rms'] = float(np.sqrt(np.mean(component ** 2)) / max(np.sqrt(np.mean(base_adv ** 2)), 1e-8))
        info = checked_update(agent, buf)
        agent.pid.update(diag['daily_cost_mean'], c.cost_limit)
        agent.lagrg = agent.pid.get_lagrg()
        record = {'update': update, 'seconds': time.monotonic() - tick, 'lambda': agent.lagrg, **diag, **info}
        records.append(record)
        save(a.out / 'training.json', records)
        save(a.out / 'progress.json', {'phase': 'training', 'update': update, 'updates': c.n_updates,
            'steps': update * c.n_env * c.horizon, 'total_steps': c.actual_steps,
            'elapsed_seconds': time.monotonic() - start,
            'remaining_training_seconds': np.mean([r['seconds'] for r in records]) * (c.n_updates - update),
            'latest': record})
        bar.set_postfix(lam=f'{agent.lagrg:.3f}', nh4=f'{diag["daily_nh4_mean"]:.2f}')
    bar.close()
    save(a.out / 'progress.json', {'phase': 'final_evaluation', 'steps': c.actual_steps, 'total_steps': c.actual_steps})
    final = evaluate(initial, lambda o: agent.act_det(o))
    agent.save(a.out / 'model_final')
    result = {'method': a.method, 'seed': a.seed, 'steps': c.actual_steps, 'updates': c.n_updates,
        'initial': baseline, 'final': final, 'training': records,
        'elapsed_seconds': time.monotonic() - start, 'finished_at': datetime.now(timezone.utc).isoformat()}
    save(a.out / 'metrics.json', result)
    save(a.out / 'progress.json', {'phase': 'complete', 'steps': c.actual_steps,
        'total_steps': c.actual_steps, 'elapsed_seconds': result['elapsed_seconds']})
    print('COMPLETE', a.method, a.seed, result['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()

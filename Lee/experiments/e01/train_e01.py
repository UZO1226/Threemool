"""한 조건(비용 모드 x 방법 x seed)을 학습하고 평가한다.

Yoo/experiments/e17/train_pilot.py 의 수집, 갱신 점검, 기록 구조를 따르고 비용
정의만 `Lee/code/bsm/bsm_binary.py` 의 모드로 바꾼다. agent 구현(PPO-Lag, FCSRL
표현 학습)과 원본 FCSRL 코드는 수정하지 않는다.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import warnings

import numpy as np
from scipy.integrate import ODEintWarning
import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'Lee/code/bsm'))
import bsm_binary as cost_modes  # noqa: E402
from bsm_binary import BinaryEnv, evaluate, satisfied  # noqa: E402
from fcsrl.utils import DeviceConfig  # noqa: E402
from ppo_config import PPOConfig  # noqa: E402
from ppo_lag import PPOLagAgent, gae  # noqa: E402
from ppo_fcsrl import PPOFCSRLAgent, RepresentationConfig, feasibility_targets  # noqa: E402

METHODS = {'ppo': PPOLagAgent, 'fcsrl': PPOFCSRLAgent}
STEPS_PER_DAY = 288
OBS_DIM, ACT_DIM = 11, 1
# 기준 운전에서 상수인 특징도 다른 행동을 표현할 수 있도록 두는 물리 척도 하한 (Yoo/e17 과 동일)
OBS_STD_FLOORS = np.array([1., .5, 1., 1., .5, .2, 5., .1, .5, .5, .25])


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def config(mode, updates):
    return PPOConfig(total_steps=updates * 5760, n_env=4, horizon=1440, n_chunks=1,
                     cost_limit=cost_modes.daily_limit(mode), init_log_std=-1.5,
                     **cost_modes.pid_gains(mode))


def make_agent(method, mode, seed, observations, updates):
    torch.manual_seed(seed)
    np.random.seed(seed)
    c = config(mode, updates)
    agent = METHODS[method](OBS_DIM, ACT_DIM, config=c)
    agent.obs_normalizer.update(observations)
    agent.obs_normalizer.m2 = np.maximum(agent.obs_normalizer.m2,
                                         agent.obs_normalizer.n * OBS_STD_FLOORS ** 2)
    return agent, c


def collect(agent, c, mode, prepared, rng, bar=None):
    states = [dict(np.load(prepared / f'day_{d}.npz')) for d in range(7)]
    envs = [BinaryEnv(states[0]) for _ in range(c.n_env)]
    schedule = rng.integers(0, 7, size=(c.horizon // STEPS_PER_DAY, c.n_env))
    obs = np.empty((c.n_env, OBS_DIM), np.float32)

    def reset_day(i, block):
        day = int(schedule[block, i])
        envs[i].initial = states[day]
        envs[i].start_index = day * 1440
        return envs[i].reset()[0]

    for i in range(c.n_env):
        obs[i] = reset_day(i, 0)
    shapes = (('obs', (OBS_DIM,)), ('obs_next', (OBS_DIM,)), ('act', (ACT_DIM,)), ('act_exec', (ACT_DIM,)),
              ('logp', ()), ('rew', ()), ('cost', ()), ('flow', ()), ('violation', ()), ('vr', ()), ('vc', ()),
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
            assert not trunc and term == ((t + 1) % STEPS_PER_DAY == 0)
            b['rew'][t, i] = reward
            b['cost'][t, i] = cost_modes.step_cost(mode, info)
            b['flow'][t, i] = info['flow_ratio']
            b['violation'][t, i] = info['violation']
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
        if (t + 1) % STEPS_PER_DAY == 0:
            lo = t + 1 - STEPS_PER_DAY
            b['cost'][lo:t + 1] = cost_modes.finalize_day(mode, b['cost'][lo:t + 1], b['flow'][lo:t + 1])
            if t + 1 < c.horizon:
                for i in range(c.n_env):
                    obs[i] = reset_day(i, (t + 1) // STEPS_PER_DAY)
        if bar is not None:
            bar.update(c.n_env)
    b['adv_r'], b['ret_r'] = gae(b['rew'], b['vr'], b['nvr'], b['end'], c.gamma, c.gae_lambda)
    b['adv_c'], b['ret_c'] = gae(b['cost'], b['vc'], b['nvc'], b['end'], c.gamma, c.gae_lambda)
    r = RepresentationConfig()
    b['feasi'] = feasibility_targets(b['cost'], b['next_feasi'], b['end'], b['terminated'],
                                     r.feasibility_discount)
    daily_cost = b['cost'].reshape(-1, STEPS_PER_DAY, c.n_env).sum(axis=1).ravel()
    daily_violations = b['violation'].reshape(-1, STEPS_PER_DAY, c.n_env).sum(axis=1).ravel()
    diagnostics = {
        'daily_cost_mean': float(daily_cost.mean()),
        'daily_cost_max': float(daily_cost.max()),
        'daily_violating_steps_mean': float(daily_violations.mean()),
        'train_step_violation_percent': float(b['violation'].mean() * 100),
        'train_daily_nh4_mean': float(np.mean(b['cost'].reshape(-1, STEPS_PER_DAY, c.n_env).sum(axis=1)) / 288 * 4)
        if mode == 'mean' else None,
        'stage_cost_max': float(b['cost'].max()),
        'feasibility_symlog_outside_fraction': float((np.log1p(b['feasi']) > r.bucket_high).mean()),
        'feasibility_target_mean': float(b['feasi'].mean()),
        'raw_action_clip_fraction': float((np.abs(b['act']) > 1).mean()),
        'observation_clip_fraction': float((np.abs(b['obs']) >= 10).mean()),
        'mean_kla5_per_day': float((b['act_exec'] + 1).mean() * 120),
        'training_days': schedule.tolist()}
    diagnostics = {k: v for k, v in diagnostics.items() if v is not None}
    flat = {k: v.reshape(c.horizon * c.n_env, *v.shape[2:]) for k, v in b.items()}
    return flat, diagnostics


def checked_update(agent, buf):
    """Yoo/e17 train_pilot.checked_update 와 같은 유한성 및 실제 갱신 점검이다."""
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
        info['full_update_probe_kl'] = float(
            torch.distributions.kl_divergence(before, agent.dist(probe)).sum(-1).mean())
        pi_after = torch.cat([p.detach().flatten() for p in agent.pi.parameters()])
        info['policy_parameter_delta_l2'] = float(torch.linalg.vector_norm(pi_after - pi_before))
    info['gradient_norm_max'] = max(norms)
    info['gradient_clip_fraction'] = float(np.mean(np.array(norms) > agent.config.max_grad_norm))
    assert all(np.isfinite(v) for v in info.values())
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=cost_modes.MODES, required=True)
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--updates', type=int, default=10)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    warnings.filterwarnings('error', category=ODEintWarning)
    torch.set_num_threads(1)
    DeviceConfig().select_device(-1)

    agent, c = make_agent(a.method, a.mode, a.seed, np.load(a.prepared / 'train_observations.npy'), a.updates)
    initial = dict(np.load(a.prepared / 'initial.npz'))
    save(a.out / 'config.json', {
        'mode': a.mode, 'method': a.method, 'seed': a.seed, 'updates': a.updates,
        'config': asdict(c), 'representation': asdict(RepresentationConfig()),
        'daily_cost_limit': cost_modes.daily_limit(a.mode),
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
    bar = tqdm(total=c.actual_steps, desc=f'{a.mode}/{a.method} seed={a.seed}', mininterval=10)
    for update in range(1, c.n_updates + 1):
        tick = time.monotonic()
        buf, diag = collect(agent, c, a.mode, a.prepared, rng, bar)
        info = checked_update(agent, buf)
        agent.pid.update(diag['daily_cost_mean'], c.cost_limit)
        agent.lagrg = agent.pid.get_lagrg()
        record = {'update': update, 'seconds': time.monotonic() - tick, 'lambda': agent.lagrg, **diag, **info}
        records.append(record)
        save(a.out / 'training.json', records)
        save(a.out / 'progress.json', {
            'phase': 'training', 'update': update, 'updates': c.n_updates,
            'steps': update * c.n_env * c.horizon, 'total_steps': c.actual_steps,
            'elapsed_seconds': time.monotonic() - start,
            'remaining_training_seconds': float(np.mean([r['seconds'] for r in records]) * (c.n_updates - update)),
            'latest': {k: v for k, v in record.items() if k != 'training_days'}})
        bar.set_postfix(lam=f'{agent.lagrg:.3f}', cost=f'{diag["daily_cost_mean"]:.1f}')
    bar.close()

    save(a.out / 'progress.json', {'phase': 'final_evaluation', 'steps': c.actual_steps,
                                   'total_steps': c.actual_steps})
    final = evaluate(initial, lambda o: agent.act_det(o))
    agent.save(a.out / 'model_final')
    result = {'mode': a.mode, 'method': a.method, 'seed': a.seed, 'steps': c.actual_steps,
              'updates': c.n_updates, 'daily_cost_limit': c.cost_limit,
              'initial': baseline, 'final': final, 'training': records,
              'final_within_own_limit': {w: satisfied(a.mode, final[w]['week']) for w in final},
              'elapsed_seconds': time.monotonic() - start,
              'finished_at': datetime.now(timezone.utc).isoformat()}
    save(a.out / 'metrics.json', result)
    save(a.out / 'progress.json', {'phase': 'complete', 'steps': c.actual_steps,
                                   'total_steps': c.actual_steps, 'elapsed_seconds': result['elapsed_seconds']})
    print('COMPLETE', a.mode, a.method, a.seed, round(result['elapsed_seconds'], 1), flush=True)


if __name__ == '__main__':
    main()

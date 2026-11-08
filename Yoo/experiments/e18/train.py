"""e18 학습: BSM1 DO 설정값 제어에서 PPO-Lag 와 FCSRL + PPO-Lag

e17 train_pilot.py 와 같은 수집, 갱신, 점검 흐름을 쓰고 다음을 바꿨다.
- 환경: setpoint_env.SetpointEnv (행동 = DO 설정값, PI 가 추종). 관측 12차원
- 하루 시작 상태: DO PI 로 운전한 건기 0 ~ 6일 snapshot (prepare.py)
- 비용 한도: 학습 날짜에서 DO PI 의 하루 평균 비용 (같은 평균 NH4 에서 에너지를 줄이는 문제)
- 같은 출발점: 정책 평균의 마지막 층 가중치를 0, 편향을 설정값 2.0 에 해당하는 값으로 둔다.
  따라서 모든 방법의 학습 전 결정적 정책은 DO PI 와 같다 (e17 의 초기 정책 차이 제거)
- 가치망 척도: value_scale = 100 (하루 반환값 규모). e17 의 경사 clipping 대부분이 가치망에서 났는지 보려고
  정책과 가치망의 clipping 비율을 따로 기록한다
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
sys.path.insert(0, str(ROOT / 'Yoo/code/wwtp'))
from setpoint_env import OBS_DIM, SetpointEnv, evaluate, normalized_cost, setpoint_to_action, training_days  # noqa: E402
from fcsrl.utils import DeviceConfig  # noqa: E402
from ppo_config import PPOConfig  # noqa: E402
from ppo_lag import PPOLagAgent, gae  # noqa: E402
from ppo_fcsrl import PPOFCSRLAgent, RepresentationConfig, feasibility_targets  # noqa: E402

METHODS = {'ppo': PPOLagAgent, 'fcsrl': PPOFCSRLAgent}
FLOORS = np.array([1., .5, 1., 1., .5, .2, 5., .1, .5, .5, .25, .1])
SP2 = setpoint_to_action(2.0)


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def config(updates, cost_limit, init_log_std=-1.2, value_scale=100., pid_mult=1.):
    # e17 과 같은 PID 단위 환산(같은 상대 제약 오차에 같은 응답)에 pid_mult 를 곱한다. e18 은 1, e19 는 3
    k = 15 / cost_limit * pid_mult
    return PPOConfig(total_steps=updates * 5760, n_env=4, horizon=1440, n_chunks=1, cost_limit=cost_limit,
                     init_log_std=init_log_std, pid_kp=.02 * k, pid_ki=.005 * k, pid_kd=.01 * k,
                     value_scale=value_scale)


def make_agent(method, seed, observations, c):
    torch.manual_seed(seed)
    np.random.seed(seed)
    agent = METHODS[method](OBS_DIM, 1, config=c)
    agent.obs_normalizer.update(observations)
    agent.obs_normalizer.m2 = np.maximum(agent.obs_normalizer.m2, agent.obs_normalizer.n * FLOORS ** 2)
    last = agent.pi[-1]
    with torch.no_grad():
        last.weight.zero_()
        last.bias.fill_(SP2)
    return agent


def collect(agent, c, states, rng, bar=None, tanks=(5,)):
    envs = [SetpointEnv(states[0], tanks=tanks) for _ in range(c.n_env)]
    schedule = rng.integers(0, 7, size=(c.horizon // 288, c.n_env))
    obs = np.empty((c.n_env, OBS_DIM), np.float32)

    def reset_day(i, block):
        day = int(schedule[block, i])
        envs[i].initial = states[day]
        envs[i].start_index = day * 1440
        return envs[i].reset()[0]

    for i in range(c.n_env):
        obs[i] = reset_day(i, 0)
    shapes = (('obs', (OBS_DIM,)), ('obs_next', (OBS_DIM,)), ('act', (1,)), ('act_exec', (1,)),
              ('logp', ()), ('rew', ()), ('cost', ()), ('flow', ()), ('vr', ()), ('vc', ()),
              ('nvr', ()), ('nvc', ()), ('end', ()), ('terminated', ()), ('next_feasi', ()), ('energy', ()))
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
            b['energy'][t, i] = info['energy_kwh']
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
    daily_cost = b['cost'].reshape(-1, 288, c.n_env).sum(axis=1).ravel()
    daily_energy = b['energy'].reshape(-1, 288, c.n_env).sum(axis=1).ravel()
    sp = 0.5 + (b['act_exec'] + 1) / 2 * 2.5
    diagnostics = {'daily_nh4_mean': float(daily_cost.mean() / 288 * 4), 'daily_cost_mean': float(daily_cost.mean()),
                   'daily_aeration_kwh': float(daily_energy.mean()),
                   'setpoint_mean': float(sp.mean()), 'setpoint_std': float(sp.std()),
                   'feasibility_symlog_outside_fraction': float((np.log1p(b['feasi']) > r.bucket_high).mean()),
                   'raw_action_clip_fraction': float((np.abs(b['act']) > 1).mean()),
                   'observation_clip_fraction': float((np.abs(b['obs']) >= 10).mean()),
                   'value_r_explained_var': float(1 - np.var(b['ret_r'] - b['vr']) / max(np.var(b['ret_r']), 1e-8)),
                   'value_c_explained_var': float(1 - np.var(b['ret_c'] - b['vc']) / max(np.var(b['ret_c']), 1e-8))}
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
    torch.nn.utils.clip_grad_norm_ = capture
    try:
        info = agent.update(buf)
    finally:
        torch.nn.utils.clip_grad_norm_ = original
    modules = [agent.pi, agent.vr, agent.vc]
    if isinstance(agent, PPOFCSRLAgent):
        modules += [agent.encoder, agent.fixed_encoder, agent.feasi_head, agent.proj_layer, agent.post_proj]
    assert all(torch.isfinite(p).all() for m in modules for p in m.parameters())
    with torch.no_grad():
        info['full_update_probe_kl'] = float(torch.distributions.kl_divergence(before, agent.dist(probe)).sum(-1).mean())
    pi_n, v_n = np.array(norms[0::2]), np.array(norms[1::2])  # 미니배치마다 정책, 가치망 순서로 호출된다
    mg = agent.config.max_grad_norm
    info.update(policy_grad_norm_median=float(np.median(pi_n)), value_grad_norm_median=float(np.median(v_n)),
                policy_clip_fraction=float((pi_n > mg).mean()), value_clip_fraction=float((v_n > mg).mean()))
    assert all(np.isfinite(v) for v in info.values())
    return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--updates', type=int, default=40)
    p.add_argument('--value_scale', type=float, default=100.)
    p.add_argument('--tanks', nargs='+', type=int, default=[5])
    p.add_argument('--pid_mult', type=float, default=1.)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    warnings.filterwarnings('error', category=ODEintWarning)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    DeviceConfig().select_device(-1)
    meta = json.loads((a.prepared / 'metadata.json').read_text())
    tanks = tuple(a.tanks)
    assert tuple(meta.get('tanks', [5])) == tanks, '준비 폴더와 제어 구조가 다르다'
    c = config(a.updates, float(meta['cost_limit']), value_scale=a.value_scale, pid_mult=a.pid_mult)
    agent = make_agent(a.method, a.seed, np.load(a.prepared / 'train_observations.npy'), c)
    states = [dict(np.load(a.prepared / f'day_{d}.npz')) for d in range(7)]
    initial = dict(np.load(a.prepared / 'initial.npz'))
    save(a.out / 'config.json', {'method': a.method, 'seed': a.seed, 'config': asdict(c), 'tanks': list(tanks),
        'pid_mult': a.pid_mult,
        'representation': asdict(RepresentationConfig()), 'started_at': datetime.now(timezone.utc).isoformat(),
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'source_hashes': {x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in
                          [*HERE.glob('*.py'), *(ROOT / 'Yoo/code/wwtp').glob('ppo_*.py')]},
        'prepared_metadata_sha256': hashlib.sha256((a.prepared / 'metadata.json').read_bytes()).hexdigest(),
        'normalizer': agent.obs_normalizer.state()})
    start = time.monotonic()
    pol = lambda o, e: agent.act_det(o)[0]  # noqa: E731
    save(a.out / 'progress.json', {'phase': 'initial_evaluation', 'steps': 0, 'total_steps': c.actual_steps})
    init_train = training_days(states, pol, tanks=tanks)
    assert abs(init_train['mean']['cost'] - c.cost_limit) < 1e-3 * c.cost_limit, '학습 전 정책이 DO PI 와 다르다'
    rng = np.random.default_rng(10000 + a.seed)
    records = []
    bar = tqdm(total=c.actual_steps, desc=f'{a.method} seed={a.seed}', mininterval=5)
    for update in range(1, c.n_updates + 1):
        tick = time.monotonic()
        buf, diag = collect(agent, c, states, rng, bar, tanks)
        info = checked_update(agent, buf)
        agent.pid.update(diag['daily_cost_mean'], c.cost_limit)
        agent.lagrg = agent.pid.get_lagrg()
        record = {'update': update, 'seconds': time.monotonic() - tick, 'lambda': agent.lagrg, **diag, **info}
        records.append(record)
        save(a.out / 'training.json', records)
        save(a.out / 'progress.json', {'phase': 'training', 'update': update, 'updates': c.n_updates,
            'steps': update * c.n_env * c.horizon, 'total_steps': c.actual_steps,
            'elapsed_seconds': time.monotonic() - start, 'latest': record})
        bar.set_postfix(lam=f'{agent.lagrg:.3f}', nh4=f'{diag["daily_nh4_mean"]:.3f}',
                        kwh=f'{diag["daily_aeration_kwh"]:.0f}', sp=f'{diag["setpoint_mean"]:.2f}')
        if update % 10 == 0:
            agent.save(a.out / f'model_u{update}')
    bar.close()
    save(a.out / 'progress.json', {'phase': 'final_evaluation', 'steps': c.actual_steps, 'total_steps': c.actual_steps})
    final_train = training_days(states, pol, tanks=tanks)
    final = evaluate(initial, pol, tanks=tanks)
    agent.save(a.out / 'model_final')
    result = {'method': a.method, 'seed': a.seed, 'steps': c.actual_steps, 'updates': c.n_updates,
              'cost_limit': c.cost_limit, 'initial_training_days': init_train, 'final_training_days': final_train,
              'final': final, 'training': records, 'elapsed_seconds': time.monotonic() - start,
              'finished_at': datetime.now(timezone.utc).isoformat()}
    save(a.out / 'metrics.json', result)
    save(a.out / 'progress.json', {'phase': 'complete', 'steps': c.actual_steps, 'total_steps': c.actual_steps,
                                   'elapsed_seconds': result['elapsed_seconds']})
    print('COMPLETE', a.method, a.seed, result['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()

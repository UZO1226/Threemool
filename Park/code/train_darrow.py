"""DARROW 환경에서 PPO-Lag 또는 FCSRL + PPO-Lag 학습 (Park 작성, Park/e08).

학습 설정은 Lee/e07 (Yoo/e05 의 PPO 설정을 DARROW 로 옮긴 것) 과 같게 맞췄다.
환경 8개 x 760 스텝(8일) = 6,080 전이를 66번 모은다 (401,280 전이). 갱신마다 대리 모델 구성원 하나를 골라
모든 환경이 같이 쓴다 (Lee 와 같음, GBM 호출이 느려서). 하루가 끝나면 학습 구간 188일 중 하나를 무작위로 골라
다시 시작한다. 하루 끝은 시간 제한이라 가치를 이어 붙인다. 관측 정규화 통계는 학습 구간 실제 운전 재생으로
한 번 정하고 고정한다 (Lee, Yoo 는 학습 중 갱신).

--brake-theta 를 주면 학습 중에도 NH4 규칙 차단(Park/e08) 을 켠다 (Park/e09). 정책 입장에서 차단은 환경의 일부다.
PPO 의 확률비는 정책이 뽑은 행동으로 계산하고, 보상, 비용, FCSRL 표현 학습은 실제로 넣은 행동을 쓴다.
주지 않으면 Park/e08 과 완전히 같다.
--min-brake-target 을 주면 대신 최소 개입 차단(min_brake.py, 1스텝, 다른 구성원) 을 학습 중에 켠다 (Park/e12).
--cost-limit 을 주면 학습의 하루 비용 한도(라그랑주 갱신 기준) 를 바꾼다. 평가의 한도 만족 % 는 언제나 12 기준이다.
--intervention-cost 를 주면 학습 중 차단이 개입한 스텝마다 그 값을 학습 비용에 더한다 (Park/e13). 평가 비용은 바꾸지 않는다.
--brake-trim 을 주면 최소 개입 차단이 DO 가 넘을 때 공기량을 내리기도 한다 (min_brake.py 의 trim, Park/e14).
--adaptive-margin 을 주면 고정 target 대신 예측 불확실성에 맞춘 여유(min_brake.E15_MARGINS) 를 쓴다 (Park/e15).

    python train_darrow.py --method fcsrl --seed 0 --data ../data/darrow/darrow_15min.pkl \
        --surrogate ../data/darrow/darrow_surrogate_gbm.pkl --out <결과>/dw_fcsrl_s0
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from tqdm import tqdm

from agent import FCSRL, METHODS, Config, ObsNormalizer, feasibility_targets, gae
from darrow_env import BatchEnv, Dynamics, action_to_air, air_to_action, all_episodes, evaluate, load_days
from min_brake import E15_MARGINS, MinAirBrake

HERE = Path(__file__).resolve().parent
OBS_DIM, ACT_DIM = 11, 1
BRAKE_AIR = 5800.0   # Park/e08 과 같은 차단 공기량 (학습 구간 기록 총 공기량 90% 분위수 반올림)


def darrow_config():
    return Config(n_env=8, horizon=760, updates=66, init_log_std=-0.5, cost_limit=12.0,
                  max_lambda=20.0, pid=(0.02, 0.005, 0.01))


def replay_observations(env, days):
    obs = env.reset(all_episodes(days, env.dyn.n_members))
    out, done = [obs], False
    while not done:
        obs, _, _, done, _ = env.step(env.historical_action())
        out.append(obs)
    return np.concatenate(out)


def nh4_brake(raw_obs, action, theta):
    """NH4 측정값(관측 0번) >= theta 이면 공기량을 최소 BRAKE_AIR 로 올린 행동과 개입 여부. theta 가 None 이면 그대로."""
    used = np.zeros(len(action), bool)
    if theta is None:
        return action, used
    policy_air = action_to_air(action)
    air = np.maximum(policy_air, np.where(raw_obs[:, 0] >= theta, BRAKE_AIR, 0.0))
    used = air > policy_air + 1e-9
    return np.where(used, air_to_action(air), action), used


def brake_shield(theta):
    return lambda obs, a, env: nh4_brake(obs, a, theta)


@torch.no_grad()
def agent_act(agent, obs):
    o = agent.tensor(agent.normalizer(obs))
    return agent.pi(agent.features(o)).clamp(-1, 1).numpy()[:, 0]


def collect(agent, c, env, days, rng, shield=None, intervention_cost=0.0):
    T, N, D = c.horizon, c.n_env, env.T
    member = int(rng.integers(env.dyn.n_members))
    keys = ('logp', 'rew', 'cost', 'vr', 'vc', 'nvr', 'nvc', 'end', 'terminated', 'next_feasi')
    b = {k: np.zeros((T, N), np.float32) for k in keys}
    b['obs'], b['obs_next'] = np.zeros((T, N, OBS_DIM), np.float32), np.zeros((T, N, OBS_DIM), np.float32)
    b['act'], b['act_exec'] = np.zeros((T, N, ACT_DIM), np.float32), np.zeros((T, N, ACT_DIM), np.float32)
    new_days = lambda: [(days[i], member) for i in rng.integers(len(days), size=N)]  # noqa: E731
    raw = env.reset(new_days())
    day_costs, env_day_costs, used_steps = [], [], 0
    env_cost = np.zeros((T, N), np.float32)
    for t in range(T):
        o = agent.normalizer(raw)
        with torch.no_grad():
            ot = agent.tensor(o)
            d = agent.dist(ot)
            act = d.sample()
            b['logp'][t] = d.log_prob(act).sum(-1).numpy()
            vr, vc = agent.values(ot)
            b['vr'][t], b['vc'][t] = vr.numpy(), vc.numpy()
        b['obs'][t], b['act'][t] = o, act.numpy()
        exec_a = np.clip(b['act'][t, :, 0], -1, 1)
        used = np.zeros(N, bool)
        if shield is not None:
            exec_a, used = shield(raw, exec_a, env)
        b['act_exec'][t, :, 0] = exec_a
        used_steps += int(used.sum())
        nxt, r, cost, trunc, _ = env.step(b['act_exec'][t, :, 0])
        env_cost[t] = cost
        b['rew'][t], b['cost'][t] = r, cost + intervention_cost * used
        no = agent.normalizer(nxt)
        b['obs_next'][t] = no
        b['end'][t] = float(trunc)
        with torch.no_grad():
            vr, vc = agent.values(agent.tensor(no))
            b['nvr'][t], b['nvc'][t] = vr.numpy(), vc.numpy()
        if isinstance(agent, FCSRL):
            b['next_feasi'][t] = agent.feasibility_state(no)
        raw = nxt
        if trunc:
            day_costs += b['cost'][t + 1 - D:t + 1].sum(0).tolist()
            env_day_costs += env_cost[t + 1 - D:t + 1].sum(0).tolist()
            if t + 1 < T:
                raw = env.reset(new_days())
    b['adv_r'], b['ret_r'] = gae(b['rew'], b['vr'], b['nvr'], b['end'], c.gamma, c.gae_lambda)
    b['adv_c'], b['ret_c'] = gae(b['cost'], b['vc'], b['nvc'], b['end'], c.gamma, c.gae_lambda)
    b['feasi'] = feasibility_targets(b['cost'], b['end'], b['terminated'], b['next_feasi'], c.feasibility_discount)
    diag = {'member': member, 'daily_cost_mean': float(np.mean(day_costs)), 'env_daily_cost_mean': float(np.mean(env_day_costs)),
            'raw_action_clip_percent': float((np.abs(b['act']) > 1).mean() * 100),
            'feasi_target_mean': float(b['feasi'].mean()), 'intervention_percent': 100 * used_steps / (T * N)}
    return {k: v.reshape(T * N, *v.shape[2:]) for k, v in b.items()}, diag


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--seed', type=int, required=True)
    for k in ('data', 'surrogate', 'out'):
        p.add_argument(f'--{k}', type=Path, required=True)
    p.add_argument('--updates', type=int, default=66)
    p.add_argument('--brake-theta', type=float, default=None, help='학습 중 NH4 규칙 차단 기준 (Park/e09)')
    p.add_argument('--min-brake-target', type=float, default=None, help='학습 중 최소 개입 차단 target (Park/e12)')
    p.add_argument('--cost-limit', type=float, default=None, help='학습 하루 비용 한도 (기본 12)')
    p.add_argument('--intervention-cost', type=float, default=0.0, help='개입 스텝마다 학습 비용에 더할 값 (Park/e13)')
    p.add_argument('--brake-trim', action='store_true', help='최소 개입 차단이 공기량을 내리기도 한다 (Park/e14)')
    p.add_argument('--adaptive-margin', action='store_true', help='불확실성에 맞춘 여유를 쓴다 (Park/e15)')
    a = p.parse_args()
    assert a.brake_theta is None or a.min_brake_target is None, '차단은 하나만 고른다'
    assert not a.brake_trim or a.min_brake_target is not None, 'trim 은 최소 개입 차단과 함께 쓴다'
    assert not a.adaptive_margin or a.min_brake_target is not None, '여유 표는 최소 개입 차단과 함께 쓴다'
    assert a.intervention_cost == 0 or a.brake_theta is not None or a.min_brake_target is not None, '개입 비용은 차단과 함께 쓴다'
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    normalizer = ObsNormalizer(replay_observations(env, splits['train']), floors=1e-3)
    c = darrow_config()
    if a.updates != c.updates:
        c = Config(**{**asdict(c), 'updates': a.updates})
    if a.cost_limit is not None:
        c = Config(**{**asdict(c), 'cost_limit': a.cost_limit})
    agent = METHODS[a.method](OBS_DIM, ACT_DIM, c, normalizer)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()
    except Exception:
        commit = None
    (a.out / 'config.json').write_text(json.dumps({'method': a.method, 'seed': a.seed, 'brake_theta': a.brake_theta,
                                                   'min_brake_target': a.min_brake_target,
                                                   'intervention_cost': a.intervention_cost, 'brake_trim': a.brake_trim,
                                                   'adaptive_margin': a.adaptive_margin, 'config': asdict(c),
                                                   'code_commit': commit, 'normalizer': normalizer.state()}, indent=2))
    val_env = BatchEnv(day_data, dyn)
    act = lambda obs: agent_act(agent, obs)  # noqa: E731
    start = time.monotonic()
    initial = evaluate(val_env, splits['val'], act)
    rng = np.random.default_rng(a.seed)
    shield = None
    if a.brake_theta is not None:
        shield = brake_shield(a.brake_theta)
    elif a.min_brake_target is not None:
        shield = MinAirBrake(dyn, a.min_brake_target, 1, 'others', trim=a.brake_trim,
                             margins=E15_MARGINS if a.adaptive_margin else None)
    records = []
    for update in tqdm(range(1, c.updates + 1), desc=f'darrow {a.method} seed {a.seed}', mininterval=30):
        buf, diag = collect(agent, c, env, splits['train'], rng, shield, a.intervention_cost)
        info = agent.update(buf)
        agent.update_lagrangian(diag['daily_cost_mean'])
        records.append({'update': update, 'lambda': agent.lagrangian, **diag, **info})
        (a.out / 'training.json').write_text(json.dumps(records, indent=1))
    agent.save(a.out / 'model_final')
    final = evaluate(val_env, splits['val'], act)
    braked = None if shield is None else evaluate(val_env, splits['val'], act, shield=shield)
    (a.out / 'metrics.json').write_text(json.dumps({'method': a.method, 'seed': a.seed, 'initial_val': initial,
                                                    'final_val': final, 'final_val_brake': braked, 'elapsed_seconds': time.monotonic() - start},
                                                   indent=2))
    print('완료', a.out, json.dumps(final), flush=True)


if __name__ == '__main__':
    main()

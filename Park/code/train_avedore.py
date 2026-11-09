"""Avedøre 환경에서 PPO-Lag 또는 FCSRL + PPO-Lag 학습 (Park 작성, Park/e07).

학습 설정은 Yoo/e05, e06 및 Lee/e06 의 FCSRL + PPO-Lag 와 같게 맞췄다.
환경 8개 x 1,148 스텝(4일) = 9,184 전이를 44번 모은다 (404,096 전이). 하루가 끝나면 학습 구간 330일 x
대리 모델 구성원 5개 중 하나를 무작위로 골라 다시 시작한다. 하루 끝은 시간 제한이라 가치를 이어 붙인다.

다른 점: Yoo 는 관측 정규화 통계를 학습 중에 계속 갱신한다. 여기서는 학습 구간 실제 운전 재생 관측으로
한 번 정하고 고정한다.

    python train_avedore.py --method fcsrl --seed 0 --data ../data/avedore/aved_5min.pkl --out <결과>/av_fcsrl_s0
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
from avedore_env import BatchEnv, Surrogate, all_episodes, evaluate, load_days

HERE = Path(__file__).resolve().parent
SURROGATE = HERE.parents[1] / 'Yoo/experiments/e02/surrogate.json'
OBS_DIM, ACT_DIM = 11, 1


def avedore_config():
    return Config(n_env=8, horizon=1148, updates=44, init_log_std=-0.5, cost_limit=15.0,
                  max_lambda=20.0, pid=(0.02, 0.005, 0.01))


def replay_observations(env, days):
    """실제 운전 공기량을 재생한 관측. 관측 정규화 통계용."""
    obs = env.reset(all_episodes(days, env.sur.n_members))
    out, done = [obs], False
    while not done:
        obs, _, _, done, _ = env.step(env.historical_action())
        out.append(obs)
    return np.concatenate(out)


def collect(agent, c, env, pool, rng):
    T, N, D = c.horizon, c.n_env, env.T
    keys = ('logp', 'rew', 'cost', 'vr', 'vc', 'nvr', 'nvc', 'end', 'terminated', 'next_feasi')
    b = {k: np.zeros((T, N), np.float32) for k in keys}
    b['obs'], b['obs_next'] = np.zeros((T, N, OBS_DIM), np.float32), np.zeros((T, N, OBS_DIM), np.float32)
    b['act'], b['act_exec'] = np.zeros((T, N, ACT_DIM), np.float32), np.zeros((T, N, ACT_DIM), np.float32)
    raw = env.reset([pool[i] for i in rng.integers(len(pool), size=N)])
    day_costs = []
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
        b['act_exec'][t] = np.clip(b['act'][t], -1, 1)
        nxt, r, cost, trunc, _ = env.step(b['act_exec'][t, :, 0])
        b['rew'][t], b['cost'][t] = r, cost
        no = agent.normalizer(nxt)
        b['obs_next'][t] = no
        b['end'][t] = float(trunc)          # 시간 제한. terminated 는 0 으로 두고 가치를 이어 붙인다
        with torch.no_grad():
            vr, vc = agent.values(agent.tensor(no))
            b['nvr'][t], b['nvc'][t] = vr.numpy(), vc.numpy()
        if isinstance(agent, FCSRL):
            b['next_feasi'][t] = agent.feasibility_state(no)
        raw = nxt
        if trunc:
            day_costs += b['cost'][t + 1 - D:t + 1].sum(0).tolist()
            if t + 1 < T:
                raw = env.reset([pool[i] for i in rng.integers(len(pool), size=N)])
    b['adv_r'], b['ret_r'] = gae(b['rew'], b['vr'], b['nvr'], b['end'], c.gamma, c.gae_lambda)
    b['adv_c'], b['ret_c'] = gae(b['cost'], b['vc'], b['nvc'], b['end'], c.gamma, c.gae_lambda)
    b['feasi'] = feasibility_targets(b['cost'], b['end'], b['terminated'], b['next_feasi'], c.feasibility_discount)
    diag = {'daily_cost_mean': float(np.mean(day_costs)), 'daily_air_mean_m3': None,
            'raw_action_clip_percent': float((np.abs(b['act']) > 1).mean() * 100),
            'feasi_target_mean': float(b['feasi'].mean())}
    flat = {k: v.reshape(T * N, *v.shape[2:]) for k, v in b.items()}
    return flat, diag


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--updates', type=int, default=44)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    splits, day_data = load_days(a.data)
    sur = Surrogate(SURROGATE)
    env = BatchEnv(day_data, sur)
    normalizer = ObsNormalizer(replay_observations(env, splits['train']), floors=1e-3)
    c = avedore_config()
    if a.updates != c.updates:
        c = Config(**{**asdict(c), 'updates': a.updates})
    agent = METHODS[a.method](OBS_DIM, ACT_DIM, c, normalizer)
    pool = all_episodes(splits['train'], sur.n_members)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()
    except Exception:
        commit = None
    (a.out / 'config.json').write_text(json.dumps({'method': a.method, 'seed': a.seed, 'config': asdict(c),
                                                   'code_commit': commit, 'normalizer': normalizer.state()}, indent=2))
    val_env = BatchEnv(day_data, sur)
    act = lambda obs: agent_act(agent, obs)  # noqa: E731
    start = time.monotonic()
    initial = evaluate(val_env, splits['val'], act)
    rng = np.random.default_rng(a.seed)
    records = []
    for update in tqdm(range(1, c.updates + 1), desc=f'avedore {a.method} seed {a.seed}', mininterval=30):
        buf, diag = collect(agent, c, env, pool, rng)
        info = agent.update(buf)
        agent.update_lagrangian(diag['daily_cost_mean'])
        records.append({'update': update, 'lambda': agent.lagrangian, **diag, **info})
        (a.out / 'training.json').write_text(json.dumps(records, indent=1))
    agent.save(a.out / 'model_final')
    final = evaluate(val_env, splits['val'], act)
    (a.out / 'metrics.json').write_text(json.dumps({'method': a.method, 'seed': a.seed, 'initial_val': initial,
                                                    'final_val': final, 'elapsed_seconds': time.monotonic() - start},
                                                   indent=2))
    print('완료', a.out, json.dumps(final), flush=True)


@torch.no_grad()
def agent_act(agent, obs):
    """여러 관측을 한꺼번에 받는 결정적 행동."""
    o = agent.tensor(agent.normalizer(obs))
    return agent.pi(agent.features(o)).clamp(-1, 1).numpy()[:, 0]


if __name__ == '__main__':
    main()

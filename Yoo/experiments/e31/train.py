"""e31 2단계 학습: 계절 혼합 분할과 공기 배분 방식(기록 비율, 고정 비율)에서 PPO-Lag 와 FCSRL + PPO-Lag

e30 의 수집, 갱신, 에이전트(기록 운전 모방 초기 정책 + 잔차 정책)를 그대로 불러 쓴다. 바꾼 것:
- 준비 폴더: e31 prepare.py (7일 블록 계절 혼합 분할, 새로 학습한 GBM 과 초기 정책)
- --frac recorded: e30 과 같이 5개 구간 배분을 그 시각 기록 비율로 한다
- --frac fixed: 모든 시각에 학습 구간 중앙값 비율로 배분한다 (기존 제어기의 배분 판단을 정책에 주지 않음)
- 검증 평가를 (날짜, 구성원) 에피소드별로 저장한다. 같은 배분 방식의 기록 운전 재생과 날짜별로 비교하기 위해서다
"""
import argparse
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e30'))
from common import config, make_agent, save  # noqa: E402
from fcsrl.utils import DeviceConfig  # noqa: E402
from ppo_fcsrl import PPOFCSRLAgent  # noqa: E402
from darrow_env import BatchEnv, Dynamics, STEP_H, load_days, all_episodes  # noqa: E402

_spec = importlib.util.spec_from_file_location('e30_train', HERE.parent / 'e30' / 'train.py')
e30_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e30_train)
collect = e30_train.collect

METHODS = ('ppo', 'fcsrl')


def make_env(prepared, frac):
    import pandas as pd
    splits, day_data = load_days(prepared / 'data.pkl')
    if frac == 'fixed':
        med = pd.read_pickle(prepared / 'data.pkl')['zone_fractions']
        f = np.array([med[f'air{i}'] for i in range(1, 6)])
        for d in day_data.values():
            d['frac'] = np.broadcast_to(f, d['frac'].shape).copy()
    return splits, BatchEnv(day_data, Dynamics(prepared / 'surrogate.pkl'))


def rollout(env, days, act):
    """모든 (날짜, 구성원) 에피소드별 공기량(Nm3, 23.75시간 합), 비용, NH4 / DO 위반 스텝 수"""
    obs = env.reset(all_episodes(days, env.dyn.n_members))
    N = len(env.episodes)
    air, cost, vn, vd = (np.zeros(N) for _ in range(4))
    done = False
    while not done:
        obs, _, c, done, info = env.step(np.asarray(act(obs, env), dtype=float).reshape(N))
        air, cost, vn, vd = air + info['air'] * STEP_H, cost + c, vn + info['v_nh4'], vd + info['v_do']
    return [{'day': d, 'member': int(m), 'air_m3': float(air[i]), 'cost': float(cost[i]),
             'nh4_steps': float(vn[i]), 'do_steps': float(vd[i])} for i, (d, m) in enumerate(env.episodes)]


def train(a):
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    DeviceConfig().select_device(-1)
    torch.set_num_threads(1)  # 원본 DeviceConfig 가 스레드 수를 바꾸므로 다시 1 로 둔다
    c = config(a.updates)
    agent = make_agent(a.method, a.seed, a.prepared, c)
    splits, env = make_env(a.prepared, a.frac)
    rng = np.random.default_rng(a.seed)
    save(a.out / 'effective_config.json', {'method': a.method, 'seed': a.seed, 'frac': a.frac, 'config': c.to_dict(),
         'prepared': str(a.prepared), 'evaluation_split': 'val', 'model_selection': 'last'})
    start, records = time.monotonic(), []
    for u in tqdm(range(1, c.n_updates + 1), desc=f'{a.method} {a.frac} s{a.seed}', mininterval=10):
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
             'steps': u * c.n_env * c.horizon, 'total_steps': c.actual_steps, 'elapsed_seconds': elapsed,
             'latest': records[-1]})
    agent.save(a.out / 'model_final')
    save(a.out / 'progress.json', {'phase': 'validation', 'steps': c.actual_steps, 'total_steps': c.actual_steps})
    rows = rollout(env, splits['val'], lambda x, e: agent.act_det(x)[:, 0])
    save(a.out / 'val_episodes.json', rows)
    summ = {k: float(np.mean([r[k] for r in rows])) for k in ('air_m3', 'cost', 'nh4_steps', 'do_steps')}
    save(a.out / 'metrics.json', {'method': a.method, 'seed': a.seed, 'frac': a.frac, 'val_final': summ,
         'steps': c.actual_steps, 'elapsed_seconds': time.monotonic() - start, 'final_lambda': agent.lagrg})
    save(a.out / 'progress.json', {'phase': 'complete', 'steps': c.actual_steps, 'total_steps': c.actual_steps})
    print('DONE', summ, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--frac', choices=('recorded', 'fixed'), required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--updates', type=int, default=66)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    train(p.parse_args())

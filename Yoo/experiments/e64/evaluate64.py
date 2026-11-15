"""e64: e63 정책을 실행 상한 (구간 공기 train 99% 분위수) 과 연속 날짜 평가로 다시 잰다. 새 학습 없음."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e63'))
import alloc as al  # noqa: E402
se, base = al.se, al.base
from darrow_env import action_to_air, air_to_action, all_episodes  # noqa: E402

PREPARED = Path('/home/jio/fcsrl_src/results/e63/prepared')
MODELS = {'corrected': PREPARED / 'surrogate.pkl', 'common': se.PREPARED / 'surrogate.pkl'}
E63 = Path('/home/jio/fcsrl_src/results/e63/main/runs')


def caps(prepared=PREPARED):
    d = pd.read_pickle(prepared / 'data.pkl')
    zones = [f'air{i}' for i in range(1, 6)]
    rows = np.concatenate([d['data'].loc[day][zones].to_numpy(float) for day in d['days']['train']])
    return np.percentile(rows, 99, axis=0)


def load_policy(condition, method, seed):
    c = base.common.config(66)
    if condition == 'alloc':
        agent = al.make_agent(method, seed, PREPARED, c)
    else:
        agent = base.common.make_agent(method, seed, PREPARED, c)
    path = E63 / condition / method / f'seed_{seed}' / 'model_final'
    agent.load(path)
    return agent, path


def make_act(policy, agent, env):
    if policy == 'record':
        return lambda o: env.historical_action()
    if policy == 'zero':
        return lambda o: np.full(len(o), -1.)
    return lambda o: agent.act_det(o)


def one_step(env, obs, act, shield, record_frac, cap):
    N = len(obs)
    a = np.asarray(act(obs), dtype=float).reshape(N, -1)
    w = al.apply_allocation(env, record_frac, a[:, 1]) if a.shape[1] == 2 else np.zeros(N)
    total = np.clip(a[:, 0], -1, 1)
    used = np.zeros(N, bool)
    if shield is not None:
        total, used = shield(obs, total, env)
    t = env.t
    frac = env.x['frac'][:, t].copy()
    clipped = np.zeros(N, bool)
    if cap is not None:
        zone = action_to_air(total)[:, None] * frac
        capped = np.minimum(zone, cap[None, :])
        clipped = (zone > cap[None, :] + 1e-9).any(1)
        air = capped.sum(1)
        # 잘린 행만 배분을 다시 쓴다. 잘리지 않은 행은 원래 배분을 그대로 둬 상한 없는 평가와 비트 단위로 같게 한다.
        frac = np.where(clipped[:, None] & (air[:, None] > 0), capped / np.maximum(air[:, None], 1e-12), frac)
        env.x['frac'][:, t] = frac
        total = np.where(clipped, air_to_action(air), total)
    obs, _, cost, done, info = env.step(total)
    assert np.array_equal(cost, info['v_nh4'] + info['v_do'])
    zone_used = info['air'][:, None] * frac
    if cap is not None:
        assert (zone_used <= cap[None, :] + 1e-6).all()
    return obs, done, info['air'].copy(), zone_used, w, used, clipped


def evaluate(env, dates, policy, agent, shield, cap, mode):
    """daily: 기존과 같은 일별 리셋. continuous: 구성원별로 날짜를 잇는다."""
    act = None
    S, A, Z, W, U, C = [], [], [], [], [], []
    if mode == 'daily':
        obs = env.reset(all_episodes(dates, env.dyn.n_members))
        act = make_act(policy, agent, env)
        rec = env.x['frac'].copy()
        for _ in range(env.T):
            obs, done, air, zone, w, used, cl = one_step(env, obs, act, shield, rec, cap)
            S.append(env.state.copy()); A.append(air); Z.append(zone); W.append(w); U.append(used); C.append(cl)
        assert done
        out = [np.stack(x, 1) for x in (S, A, Z, W, U, C)]
    else:
        carry = None
        days_out = []
        for d in dates:
            obs = env.reset([(d, m) for m in range(env.dyn.n_members)])
            if carry is not None:
                env.state, env.prev_air, env.prev_zone = (x.copy() for x in carry)
                obs = env.obs()
            act = make_act(policy, agent, env)
            rec = env.x['frac'].copy()
            s, a_, z, w_, u_, c_ = [], [], [], [], [], []
            for _ in range(env.T):
                obs, done, air, zone, w, used, cl = one_step(env, obs, act, shield, rec, cap)
                s.append(env.state.copy()); a_.append(air); z.append(zone); w_.append(w); u_.append(used); c_.append(cl)
            assert done
            carry = (env.state, env.prev_air, env.prev_zone)
            days_out.append([np.stack(x, 1) for x in (s, a_, z, w_, u_, c_)])
        out = [np.concatenate([day[k] for day in days_out]) for k in range(6)]
    states, airs, zones, ws, useds, clips = out
    episodes = all_episodes(dates, env.dyn.n_members)
    result, rows, members = base.metrics(states, airs, episodes)
    result.pop('both_mean_limits')
    result.pop('both_limits_episode_pct')
    n = (states[:, :, 0] > 4).sum(1)
    d_ = (states[:, :, 2:].max(2) > 2).sum(1)
    v = states[:, :, 0] > 4
    carry_n = np.where(v.all(1), v.shape[1], np.argmin(v, axis=1))
    result.update(cost=float((n + d_).mean()), nh4_viol_pct=float((100 * n / 95).mean()), do_viol_pct=float((100 * d_ / 95).mean()),
                  days_within_limit_pct=float(100 * np.mean(n + d_ <= 12)), n_episodes=len(episodes),
                  intervention_pct=float(100 * useds.mean()) if shield is not None else 0.0, clipped_pct=float(100 * clips.mean()),
                  nh4_carryover=float(carry_n.mean()), nh4_controllable=float((n - carry_n).mean()),
                  w_mean=float(ws.mean()), w_positive_pct=float(100 * (ws > 0).mean()))
    return result, rows, states, airs, zones


def main(a):
    a.out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    splits, days = base.load_selected(PREPARED / 'data.pkl')
    dates = splits['val'] if a.days is None else splits['val'][:a.days]
    dyn = base.Dynamics(MODELS[a.model])
    env = base.BatchEnv(days, dyn)
    agent, path, fp = None, None, None
    if a.policy == 'policy':
        agent, path = load_policy(a.condition, a.method, a.seed)
        fp = base.fingerprint(agent)
    shield = None if a.shield == 'none' else se.make_shield(a.shield, dyn)
    cap = caps() if a.cap else None
    result, rows, states, airs, zones = evaluate(env, dates, a.policy, agent, shield, cap, a.mode)
    if agent is not None:
        assert base.fingerprint(agent) == fp
    base.save(a.out / 'result.json', dict(policy=a.policy, condition=a.condition, method=a.method, seed=a.seed, model=a.model, cap=bool(a.cap),
              mode=a.mode, shield=a.shield, days=len(dates), checkpoint=None if path is None else str(path), policy_fingerprint=fp,
              cap_values=None if cap is None else cap.tolist(), val=result, model_sha256=base.sha(MODELS[a.model]),
              elapsed_seconds=time.monotonic() - t0, test_evaluated=False))
    base.save(a.out / 'episodes.json', rows)
    np.savez_compressed(a.out / 'rollout.npz', states=states, air=airs, zones=zones)
    print('DONE', a.policy, a.condition, a.method, a.seed, a.model, a.cap, a.mode, round(result['air_m3']), round(result['cost'], 3), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--policy', choices=['policy', 'record', 'zero'], default='policy')
    p.add_argument('--condition', choices=['total', 'alloc', 'none'], default='none')
    p.add_argument('--method', choices=['ppo', 'fcsrl', 'none'], default='none')
    p.add_argument('--seed', type=int, default=-1)
    p.add_argument('--model', choices=list(MODELS), required=True)
    p.add_argument('--cap', type=int, choices=[0, 1], required=True)
    p.add_argument('--mode', choices=['daily', 'continuous'], required=True)
    p.add_argument('--shield', choices=['none', 'park_e10', 'park_e19'], default='park_e19')
    p.add_argument('--days', type=int, default=None, help='검사용: 앞 날짜 수만 평가')
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())

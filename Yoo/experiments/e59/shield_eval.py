"""e59: Park의 모델 예측 최소 개입 차단을 Yoo 정책에 실행 시점에만 걸어 평가한다.

차단 코드 min_brake.py는 Park/code/min_brake.py (main bbaefe2) 를 수정 없이 복사한 것이다.
Yoo/experiments/e30/darrow_env.py 는 Park/code/darrow_env.py 와 바이트 단위로 같으므로 그대로 동작한다.
정책은 e50 (PPO-Lag, FCSRL, FCSRL+Actor 보정, 모두 모방 prior on, CPU4) 과 e58 severity (GPU1) 의 마지막 checkpoint다.
새 학습은 없다. 검증 40일 x 구성원 5개, 결정적 행동. test 행은 추출하거나 평가하지 않는다.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e58'))
import severity as s  # noqa: E402  e58 모듈. base = e56 train (common, darrow_env, e45 metrics 포함)
base = s.base
sys.path.insert(0, str(HERE))
from min_brake import E15_MARGINS, MinAirBrake  # noqa: E402
from darrow_env import BatchEnv, Dynamics, all_episodes, action_to_air  # noqa: E402

PREPARED = Path('/home/jio/fcsrl_src/results/e50/prepared')
RUNS = {'e50': Path('/home/jio/fcsrl_src/results/e50/main/runs'), 'e58': Path('/home/jio/fcsrl_src/results/e58/main/runs')}
POLICIES = {'ppo': ('e50', 'ppo'), 'fcsrl': ('e50', 'fcsrl'), 'fcsrl_actor': ('e50', 'fcsrl_actor'),
            'severity': ('e58', 'severity')}
FIXED = ('record', 'zero')
SEEDS = (0, 1, 2, 3, 4)
P1 = [(u, m + 0.1) for u, m in E15_MARGINS]   # Park/e19 run_e19.py 의 P1
SHIELDS = {
    'none': None,
    # Park/e19 H1 = Park/e22 BRAKE: 1스텝, 다른 구성원, 양방향, 여유 P1, DO 1.8, 회복 (3, 3.5), 충돌 0.1
    'park_e19': dict(target=3.5, horizon=1, members='others', trim=True, margins=P1, do_limit=1.8,
                     recover=(3, 3.5), conflict=0.1),
    # Park/e10, e12 학습에 쓴 가장 단순한 형태: 1스텝, 다른 구성원, target 3.5, 올리기만
    'park_e10': dict(target=3.5, horizon=1, members='others'),
}


def jsonable(kw):
    """여유 표의 inf 상한을 문자열로 바꿔 JSON 에 남긴다."""
    if kw is None:
        return None
    return {k: ([[('inf' if u == float('inf') else u), m] for u, m in v] if k == 'margins' else v) for k, v in kw.items()}


def make_shield(name, dyn):
    kw = SHIELDS[name]
    if kw is None:
        return None
    kw = dict(kw)
    return MinAirBrake(dyn, kw.pop('target'), kw.pop('horizon'), kw.pop('members'), **kw)


def load_policy(name, seed, c):
    src, method = POLICIES[name]
    if src == 'e50':
        agent = base.common.make_agent(method, seed, PREPARED, c)
    else:
        agent = s.make_agent(method, seed, PREPARED, c)
    path = RUNS[src] / method / f'seed_{seed}' / 'model_final'
    agent.load(path)
    recorded = json.loads((path.parent / 'metrics.json').read_text())['val_final']
    return agent, path, recorded


def make_act(name, agent, env):
    if name == 'record':
        return lambda obs: env.historical_action()
    if name == 'zero':
        return lambda obs: np.full(len(obs), -1.0)   # 공기 0, 차단이 전부 정한다
    return lambda obs: agent.act_det(obs)[:, 0]


def rollout(env, days, act, shield):
    episodes = all_episodes(days, env.dyn.n_members)
    obs = env.reset(episodes)
    N = len(episodes)
    states, airs, policy_airs, used_steps = [], [], [], 0
    done = False
    for _ in range(env.T):
        a = np.asarray(act(obs), dtype=float).reshape(N)
        a_exec = a
        if shield is not None:
            a_exec, used = shield(obs, a, env)
            used_steps += int(used.sum())
        obs, reward, cost, done, info = env.step(a_exec)
        assert np.allclose(reward, -info['air'] / 9000.)
        assert np.array_equal(cost, info['v_nh4'] + info['v_do'])
        states.append(env.state.copy())
        airs.append(info['air'].copy())
        policy_airs.append(action_to_air(a))
    assert done
    states, airs, policy_airs = np.stack(states, 1), np.stack(airs, 1), np.stack(policy_airs, 1)
    result, rows, members = base.metrics(states, airs, episodes)
    result.pop('both_mean_limits')
    result.pop('both_limits_episode_pct')
    n = (states[:, :, 0] > 4).sum(1)
    d = (states[:, :, 2:].max(2) > 2).sum(1)
    result.update(cost=float((n + d).mean()), nh4_viol_pct=float((100 * n / 95).mean()),
                  do_viol_pct=float((100 * d / 95).mean()), days_within_limit_pct=float(100 * np.mean(n + d <= 12)),
                  n_episodes=N, policy_air_m3=float((policy_airs.sum(1) * .25).mean()),
                  intervention_pct=0.0 if shield is None else 100 * used_steps / (N * env.T))
    if shield is not None:
        result.update(lowered_pct=100 * shield.lowered_steps / max(shield.steps, 1),
                      conflict_pct=100 * shield.conflict_steps / max(shield.steps, 1),
                      futile_steps=int(shield.futile_steps))
    return result, rows, members, states, airs


def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    c = base.common.config(66)
    splits, days = base.load_selected(PREPARED / 'data.pkl')
    dyn = Dynamics(PREPARED / 'surrogate.pkl')
    env = BatchEnv(days, dyn)
    agent, path, recorded, fp = None, None, None, None
    if a.policy not in FIXED:
        agent, path, recorded = load_policy(a.policy, a.seed, c)
        fp = base.fingerprint(agent)
    start = time.monotonic()
    result, rows, members, states, airs = rollout(env, splits['val'], make_act(a.policy, agent, env), make_shield(a.shield, dyn))
    elapsed = time.monotonic() - start
    reproduced = None
    if agent is not None:
        assert base.fingerprint(agent) == fp
        if a.shield == 'none':
            # 차단 없는 평가는 원 실험의 기록값을 정확히 재현해야 한다.
            reproduced = bool(abs(result['air_m3'] - recorded['air_m3']) < 1e-8 * recorded['air_m3'] and abs(result['cost'] - recorded['cost']) < 1e-9)
            assert reproduced, (result['air_m3'], recorded['air_m3'], result['cost'], recorded['cost'])
    out = dict(policy=a.policy, seed=a.seed, shield=a.shield, shield_config=jsonable(SHIELDS[a.shield]), checkpoint=None if path is None else str(path),
               policy_fingerprint=fp, reproduced_recorded_metrics=reproduced, val=result, members=members,
               elapsed_seconds=elapsed, data_sha256=base.sha(PREPARED / 'data.pkl'), model_sha256=base.sha(PREPARED / 'surrogate.pkl'),
               min_brake_sha256=base.sha(HERE / 'min_brake.py'), test_evaluated=False)
    base.save(a.out / 'result.json', out)
    base.save(a.out / 'episodes.json', rows)
    np.savez_compressed(a.out / 'rollout.npz', states=states, air=airs)
    print('DONE', a.policy, a.seed, a.shield, {k: round(result[k], 3) for k in ('air_m3', 'cost', 'nh4_cost', 'do_cost', 'intervention_pct')}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--policy', choices=list(POLICIES) + list(FIXED), required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--shield', choices=list(SHIELDS), required=True)
    p.add_argument('--out', type=Path, required=True)
    run(p.parse_args())

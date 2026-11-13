"""e59 사전 검사: 복사한 차단 코드의 동일성, 정답 모델 차단의 동작, 정책 재현.

본 평가 전에 통과해야 한다. 결과는 checks.json 에 남긴다.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np

import shield_eval as se
from min_brake import MinAirBrake
from darrow_env import BatchEnv, Dynamics, all_episodes, AIR_MAX, NH4_MAX

base = se.base
ROOT = se.HERE.parents[2]


def oracle_check(env, days, dyn, target=3.5):
    """members='same' 이면 차단의 1스텝 예측은 환경 전이와 같다. 그러면 필요 공기량 < 9000 인 스텝에서
    실제 다음 NH4 <= target 이어야 한다 (Park/e10 설계 점검의 0% 와 같은 검사)."""
    brake = MinAirBrake(dyn, target, 1, 'same')
    obs = env.reset(all_episodes(days, dyn.n_members))
    N = len(env.episodes)
    feasible = violated = 0
    done = False
    while not done:
        a = np.full(N, -1.0)
        required = brake.required_air(obs, env)
        a_exec, used = brake(obs, a, env)
        obs, _, _, done, _ = env.step(a_exec)
        ok = required < AIR_MAX - 1e-9
        feasible += int(ok.sum())
        violated += int((env.state[ok, 0] > target + 1e-9).sum())
    return dict(feasible_steps=feasible, violated_after_feasible=violated, passed=violated == 0)


def no_intervention_check(env, days, dyn):
    brake = MinAirBrake(dyn, 1e9, 1, 'same')
    obs = env.reset(all_episodes(days[:4], dyn.n_members))
    N = len(env.episodes)
    used_total = 0
    done = False
    rng = np.random.default_rng(0)
    while not done:
        a = rng.uniform(-1, 1, N)
        a_exec, used = brake(obs, a, env)
        assert np.array_equal(a_exec, a)
        used_total += int(used.sum())
        obs, _, _, done, _ = env.step(a_exec)
    return dict(interventions=used_total, passed=used_total == 0)


def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    park = ROOT / 'Park/code/min_brake.py'
    checks = dict(min_brake_identical=base.sha(park) == base.sha(se.HERE / 'min_brake.py'),
                  min_brake_sha256=base.sha(park),
                  park_commit=subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', str(park)], cwd=ROOT, text=True).strip(),
                  env_identical=base.sha(ROOT / 'Park/code/darrow_env.py') == base.sha(se.HERE.parent / 'e30/darrow_env.py'))
    splits, days = base.load_selected(se.PREPARED / 'data.pkl')
    dyn = Dynamics(se.PREPARED / 'surrogate.pkl')
    env = BatchEnv(days, dyn)
    t = time.monotonic()
    checks['oracle_target_3.5_val'] = oracle_check(env, splits['val'], dyn)
    checks['no_intervention_when_target_inf'] = no_intervention_check(env, splits['val'], dyn)
    checks['oracle_seconds'] = time.monotonic() - t
    c = base.common.config(66)
    rep = {}
    for name in se.POLICIES:
        agent, path, recorded = se.load_policy(name, 0, c)
        result, *_ = se.rollout(env, splits['val'], se.make_act(name, agent, env), None)
        rep[name] = dict(air_m3=result['air_m3'], recorded_air_m3=recorded['air_m3'], cost=result['cost'], recorded_cost=recorded['cost'],
                         passed=abs(result['air_m3'] - recorded['air_m3']) < 1e-8 * recorded['air_m3'] and abs(result['cost'] - recorded['cost']) < 1e-9)
    checks['reproduce_seed0_without_shield'] = rep
    t = time.monotonic()
    brake = se.make_shield('park_e19', dyn)
    result, *_ = se.rollout(env, splits['val'][:2], se.make_act('zero', None, env), brake)
    checks['shield_timing_2_days'] = dict(seconds=time.monotonic() - t, intervention_pct=result['intervention_pct'])
    checks['passed'] = bool(checks['min_brake_identical'] and checks['env_identical'] and checks['oracle_target_3.5_val']['passed']
                            and checks['no_intervention_when_target_inf']['passed'] and all(v['passed'] for v in rep.values()))
    checks['hashes'] = {str(p): base.sha(p) for p in [se.PREPARED / 'data.pkl', se.PREPARED / 'surrogate.pkl', se.HERE / 'min_brake.py',
                                                        se.HERE / 'shield_eval.py', park]}
    base.save(a.out / 'checks.json', checks)
    print(json.dumps({k: v for k, v in checks.items() if k != 'hashes'}, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    run(p.parse_args())

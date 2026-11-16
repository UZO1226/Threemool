"""2단계 진단: 학습 날짜에서 막을 수 없는 NH4 위반이 FCSRL feasibility 목표를 얼마나 차지하는가.

막을 수 없는 스텝: 그 시각 배분으로 총 공기 9,000 을 넣어도 다른 구성원 4개 평균의 다음 스텝 NH4 예측이 4 를 넘는 스텝
(Park/e28 과 같은 정의). 이 스텝 직후의 NH4 위반을 막을 수 없는 위반으로 센다.
정책은 e63 배분 FCSRL seed 0 마지막 정책, 학습 중 차단 park_e10, DO 수정 모델, 학습 188일 x 구성원 5, 결정적 행동.
feasibility 목표는 원 FCSRL 과 같은 역방향 max 재귀 (할인 0.9) 로, 원래 비용과 막을 수 없는 NH4 위반을 뺀 비용 두 가지를 계산한다.
새 학습 없음, test 없음.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e64'))
import evaluate64 as ev  # noqa: E402
base, se, al = ev.base, ev.se, ev.al
from ppo_fcsrl import feasibility_targets  # noqa: E402
from darrow_env import all_episodes  # noqa: E402


def main(a):
    a.out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    splits, days = base.load_selected(ev.PREPARED / 'data.pkl')
    dyn = base.Dynamics(ev.MODELS['corrected'])
    env = base.BatchEnv(days, dyn)
    agent, _ = ev.load_policy(a.condition, a.method, a.seed)
    shield = se.make_shield('park_e10', dyn)
    probe = se.make_shield('park_e10', dyn)
    obs = env.reset(all_episodes(splits['train'], 5))
    rec = env.x['frac'].copy()
    N = len(obs)
    U, VN, VD = [], [], []
    for _ in range(env.T):
        act = np.asarray(agent.act_det(obs), dtype=float).reshape(N, -1)
        if act.shape[1] == 2:
            al.apply_allocation(env, rec, act[:, 1])
        nh4_max_air, _ = probe._rollout(obs, env, np.full((N, 1), 9000.), 1)
        U.append(nh4_max_air[:, 0] > 4)
        total, _ = shield(obs, np.clip(act[:, 0], -1, 1), env)
        obs, _, cost, done, info = env.step(total)
        VN.append(info['v_nh4'].astype(bool))
        VD.append(info['v_do'].astype(bool))
    U, VN, VD = (np.stack(x, 0) for x in (U, VN, VD))  # (T, N)
    cost = VN.astype(float) + VD
    ctrl = (VN & ~U).astype(float) + VD
    end = np.zeros_like(cost)
    end[-1] = 1
    zeros = np.zeros_like(cost)
    f_all = feasibility_targets(cost, zeros, end, zeros, 0.9)
    f_ctrl = feasibility_targets(ctrl, zeros, end, zeros, 0.9)
    out = dict(policy=f'{a.condition} {a.method} seed {a.seed}', episodes=int(N),
               unavoidable_step_pct=float(100 * U.mean()), nh4_violations_per_day=float(VN.sum(0).mean()),
               unavoidable_nh4_violations_per_day=float((VN & U).sum(0).mean()),
               unavoidable_share_of_nh4_violations=float((VN & U).sum() / max(VN.sum(), 1)),
               do_violations_per_day=float(VD.sum(0).mean()),
               feasi_target_mean_all=float(f_all.mean()), feasi_target_mean_controllable=float(f_ctrl.mean()),
               feasi_positive_pct_all=float(100 * (f_all > 1e-9).mean()), feasi_positive_pct_controllable=float(100 * (f_ctrl > 1e-9).mean()),
               share_of_positive_states_removed=float(((f_all > 1e-9) & (f_ctrl <= 1e-9)).sum() / max((f_all > 1e-9).sum(), 1)),
               seconds=time.monotonic() - t0, test_evaluated=False)
    (a.out / 'result.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--condition', default='alloc')
    p.add_argument('--method', default='fcsrl')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())

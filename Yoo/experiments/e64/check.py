"""e64 사전 검사.

C1 상한 없는 일별 평가가 e63 저장 에피소드 값을 정확히 재현 (alloc PPO seed 0, total PPO seed 0, 검증 40일 전체).
   앞 몇 일만 자르면 정책 신경망의 묶음 크기가 달라져 행동이 끝자리에서 달라지므로 전체 날짜로 비교한다.
C2 상한 평가에서 모든 구간 공기가 상한 이하이고, 잘린 스텝이 실제로 있다 (alloc PPO seed 0, 앞 3일).
C3 연속 평가 첫날은 일별 평가 첫날과 같고, 둘째 날은 달라진다 (기록 운전, 앞 2일).
C4 상한이 느슨하면 (무한대) 상한 없는 평가와 같다.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate64 as ev  # noqa: E402
base, se = ev.base, ev.se


def rows_equal(rows, saved, n):
    keys = ('air_m3', 'nh4_cost', 'do_cost', 'nh4_excess_mg_h_L', 'do_excess_mg_h_L')
    return all(r['day'] == s['day'] and r['member'] == s['member'] and all(r[k] == s[k] for k in keys) for r, s in zip(rows[:n], saved[:n]))


def main(a):
    a.out.mkdir(parents=True, exist_ok=False)
    splits, days = base.load_selected(ev.PREPARED / 'data.pkl')
    dyn = base.Dynamics(ev.MODELS['corrected'])
    env = base.BatchEnv(days, dyn)
    cap = ev.caps()
    C = {'cap_values': cap.tolist()}
    c1 = {}
    for cond in ('alloc', 'total'):
        agent, path = ev.load_policy(cond, 'ppo', 0)
        res, rows, *_ = ev.evaluate(env, splits['val'], 'policy', agent, se.make_shield('park_e19', dyn), None, 'daily')
        saved = json.loads((path.parent / 'val_park_e19_episodes.json').read_text())
        c1[cond] = rows_equal(rows, saved, len(saved)) and len(rows) == len(saved) == 200
    C['C1'] = dict(**c1, passed=all(c1.values()))
    agent, _ = ev.load_policy('alloc', 'ppo', 0)
    res, rows, states, airs, zones = ev.evaluate(env, splits['val'][:3], 'policy', agent, se.make_shield('park_e19', dyn), cap, 'daily')
    C['C2'] = dict(clipped_pct=res['clipped_pct'], max_ratio=float((zones / cap).max()), passed=bool((zones <= cap + 1e-6).all() and res['clipped_pct'] > 0))
    rd, rows_d, sd, *_ = ev.evaluate(env, splits['val'][:2], 'record', None, None, None, 'daily')
    rc, rows_c, sc, *_ = ev.evaluate(env, splits['val'][:2], 'record', None, None, None, 'continuous')
    C['C3'] = dict(first_day_equal=bool(np.array_equal(sd[:5], sc[:5])), second_day_differs=bool(not np.array_equal(sd[5:], sc[5:])),
                   passed=bool(np.array_equal(sd[:5], sc[:5]) and not np.array_equal(sd[5:], sc[5:])))
    r_inf, _, s_inf, a_inf, _ = ev.evaluate(env, splits['val'][:3], 'policy', agent, se.make_shield('park_e19', dyn), np.full(5, np.inf), 'daily')
    r_no, _, s_no, a_no, _ = ev.evaluate(env, splits['val'][:3], 'policy', agent, se.make_shield('park_e19', dyn), None, 'daily')
    C['C4'] = dict(passed=bool(np.array_equal(s_inf, s_no) and np.array_equal(a_inf, a_no)))
    C['passed'] = all(C[k]['passed'] for k in ('C1', 'C2', 'C3', 'C4'))
    C['hashes'] = {str(p): base.sha(p) for p in [HERE / 'evaluate64.py', ev.PREPARED / 'data.pkl', ev.MODELS['corrected'], ev.MODELS['common'],
                                                  HERE.parent / 'e63/alloc.py', se.HERE / 'min_brake.py', se.HERE / 'shield_eval.py']}
    base.save(a.out / 'checks.json', json.loads(json.dumps(C, default=lambda o: o.item())))
    print(json.dumps({k: v for k, v in C.items() if k != 'hashes'}, indent=1, default=lambda o: o.item()))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())

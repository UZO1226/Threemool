"""Park/e75 차단 위 예측 제어의 후보 공기량 간격 (Park 작성). 규칙은 README.md 를 따른다.
대표 차단 위의 초과량 예측 제어 (NH4 여유 1.2, 공기 가중치 1.5, 1시간) 에서 후보 공기량 간격 만 바꿔 검증 구간에서 돌리고,
Park/e58, e59 의 기본 설정 곡선과 같은 공기량에서 비교한다 (Park/e72 와 같은 틀).

    python run_e75.py val --part <i> --nparts 2 --data .. --surrogate .. --out <결과>/e75
    python run_e75.py report --out <결과>/e75 --e58 <결과>/e58 --e59 <결과>/e59 --md results.md
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
sys.path.insert(0, str(HERE.parent / 'e38'))
sys.path.insert(0, str(HERE.parent / 'e59'))
from darrow_env import BatchEnv, Dynamics, evaluate, load_days  # noqa: E402
from min_brake import MinAirBrake  # noqa: E402
from mpc_darrow import ShortMPC  # noqa: E402
from run_e38 import BRAKE  # noqa: E402
from run_e59 import curve_cost, frontier, gather  # noqa: E402

BASE = dict(margin=1.2, lam=1.5, objective='excess')
TOL = 0.05           # 곡선과의 차이가 이보다 커야 "낮음" 또는 "높음"
# 이름: (표 이름, 예측 제어 인자 바꾸기, 차단 NH4 여유 더하기)
CONFIGS = {'fine': ('간격 250 / 750 (후보 481개)', dict(step0=250.0, step1=750.0), 0.0),
           'coarse': ('간격 1000 / 3000 (후보 40개)', dict(step0=1000.0, step1=3000.0), 0.0)}
H1 = (('fine',), -1)            # (설정 이름들, 방향): 그 가운데 하나라도 곡선보다 TOL 넘게 낮으면 지지
H2 = (('coarse',), +1)


def make(env, dyn, name):
    _, kw, delta = CONFIGS[name]
    mpc = ShortMPC(env, dyn, **dict(BASE, **kw))
    brake = dict(BRAKE, margins=[(u, m + delta) for u, m in BRAKE['margins']]) if delta else BRAKE
    return mpc, MinAirBrake(dyn, 3.5, 1, 'others', **brake)


def val(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    out = {}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for name in list(CONFIGS)[a.part::a.nparts]:
        mpc, br = make(env, dyn, name)
        out[name] = evaluate(env, splits['val'], mpc, shield=br)
        print(name, out[name], flush=True)
        (Path(a.out) / f'val_part{a.part}.json').write_text(json.dumps(out, indent=2))


def verdict(gap, names, sign):
    g = [gap[n] for n in names if n in gap and not np.isnan(gap[n])]
    if not g:
        return '판정 안 함 (범위 밖)'
    hit = any(x < -TOL for x in g) if sign < 0 else any(x > TOL for x in g)
    return '지지' if hit else '지지 안 됨'


def report(a):
    v = gather(a.out, 'val_part*.json')
    mp = gather(a.e58, 'val_part*.json')
    mp.update(gather(a.e59, 'val_part*.json'))
    mf = frontier([(r['air_m3'], r['cost']) for r in mp.values()])
    b = mp['1.5']
    L = [f'검증 40일 x 구성원 5개 = 200 에피소드. 대표 차단 위 초과량 예측 제어, 기본 설정 NH4 여유 1.2, 공기 가중치 1.5, 1시간. 기본 설정 곡선 범위 {mf[0, 0]:,.0f} ~ {mf[-1, 0]:,.0f} (Park/e58, e59).', '',
         '| 설정 | 공기량 | 하루 비용 | NH4 위반 % | DO 위반 % | 차단 개입 % | 같은 공기량의 기본 곡선 비용 | 차이 |', '| --- | --- | --- | --- | --- | --- | --- | --- |',
         f"| 기본 (Park/e58) | {b['air_m3']:,.0f} | {b['cost']:.3f} | {b['nh4_viol_pct']:.2f} | {b['do_viol_pct']:.2f} | {b['intervention_pct']:.1f} | | |"]
    gap = {}
    for name in CONFIGS:
        if name not in v:
            continue
        r = v[name]
        c = curve_cost(mf, r['air_m3'])
        gap[name] = r['cost'] - c
        L.append(f"| {CONFIGS[name][0]} | {r['air_m3']:,.0f} | {r['cost']:.3f} | {r['nh4_viol_pct']:.2f} | {r['do_viol_pct']:.2f} | {r['intervention_pct']:.1f} | {'범위 밖' if np.isnan(c) else f'{c:.3f}'} | {'' if np.isnan(c) else f'{gap[name]:+.3f}'} |")
    L += ['', '## 판정', '']
    for tag, (names, sign) in (('H1', H1), ('H2', H2)):
        what = '낮은가' if sign < 0 else '높은가'
        ds = ', '.join(f'{gap[n]:+.3f}' if n in gap else '없음' for n in names)
        ns = ', '.join(CONFIGS[n][0] for n in names)
        L.append(f"- {tag} ({ns} 가운데 하나라도 같은 공기량의 기본 곡선보다 {TOL} 넘게 {what}): 차이 {ds}: {verdict(gap, names, sign)}")
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    v = sub.add_parser('val')
    v.add_argument('--part', type=int, required=True)
    v.add_argument('--nparts', type=int, default=2)
    for k in ('data', 'surrogate', 'out'):
        v.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'e58', 'e59', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    {'val': val, 'report': report}[a.cmd](a)


if __name__ == '__main__':
    main()

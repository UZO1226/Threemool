"""Park/e72 차단 위 예측 제어의 예측 구간 길이 (Park 작성). 규칙은 README.md 를 따른다.
대표 차단 위의 초과량 예측 제어 (NH4 여유 1.2, 공기 가중치 1.5) 를 예측 구간 2 스텝 (30분) 과 8 스텝 (2시간) 으로 바꿔 검증 구간에서 돌리고,
Park/e58, e59 의 4 스텝 곡선과 같은 공기량에서 비교한다.

    python run_e72.py val --part <i> --nparts 2 --data .. --surrogate .. --out <결과>/e72
    python run_e72.py report --out <결과>/e72 --e58 <결과>/e58 --e59 <결과>/e59 --md results.md
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

MARGIN, LAM = 1.2, 1.5
HS = (8, 2)          # 8 을 먼저 돈다 (더 오래 걸림)
TOL = 0.05           # 곡선과의 차이가 이보다 커야 "낮음" 또는 "높음"


def val(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    out = {}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for h in HS[a.part::a.nparts]:
        out[str(h)] = evaluate(env, splits['val'], ShortMPC(env, dyn, H=h, margin=MARGIN, lam=LAM, objective='excess'),
                               shield=MinAirBrake(dyn, 3.5, 1, 'others', **BRAKE))
        print(h, out[str(h)], flush=True)
        (Path(a.out) / f'val_part{a.part}.json').write_text(json.dumps(out, indent=2))


def report(a):
    v = gather(a.out, 'val_part*.json')
    mp = gather(a.e58, 'val_part*.json')
    mp.update(gather(a.e59, 'val_part*.json'))
    mf = frontier([(r['air_m3'], r['cost']) for r in mp.values()])
    m4 = mp[str(LAM)]
    L = [f'검증 40일 x 구성원 5개 = 200 에피소드. 대표 차단 위 초과량 예측 제어, NH4 여유 {MARGIN}, 공기 가중치 {LAM}. 4 스텝 곡선 범위 {mf[0, 0]:,.0f} ~ {mf[-1, 0]:,.0f} (Park/e58, e59).', '',
         '| 예측 구간 | 공기량 | 하루 비용 | NH4 위반 % | DO 위반 % | 차단 개입 % | 같은 공기량의 4 스텝 곡선 비용 | 차이 |', '| --- | --- | --- | --- | --- | --- | --- | --- |',
         f"| 4 스텝 (1시간, Park/e58) | {m4['air_m3']:,.0f} | {m4['cost']:.3f} | {m4['nh4_viol_pct']:.2f} | {m4['do_viol_pct']:.2f} | {m4['intervention_pct']:.1f} | | |"]
    gap = {}
    for h in sorted(v, key=int):
        r = v[h]
        c = curve_cost(mf, r['air_m3'])
        gap[h] = r['cost'] - c
        L.append(f"| {h} 스텝 ({int(h) * 15}분) | {r['air_m3']:,.0f} | {r['cost']:.3f} | {r['nh4_viol_pct']:.2f} | {r['do_viol_pct']:.2f} | {r['intervention_pct']:.1f} | {'범위 밖' if np.isnan(c) else f'{c:.3f}'} | {'' if np.isnan(c) else f'{gap[h]:+.3f}'} |")
    g8, g2 = gap.get('8', float('nan')), gap.get('2', float('nan'))
    L += ['', '## 판정', '',
          f"- H1 (2시간 예측 구간이 같은 공기량의 1시간 곡선보다 {TOL} 넘게 낮은가): 차이 {g8:+.3f}: {'지지' if g8 < -TOL else ('판정 안 함 (범위 밖)' if np.isnan(g8) else '지지 안 됨')}",
          f"- H2 (30분 예측 구간이 같은 공기량의 1시간 곡선보다 {TOL} 넘게 높은가): 차이 {g2:+.3f}: {'지지' if g2 > TOL else ('판정 안 함 (범위 밖)' if np.isnan(g2) else '지지 안 됨')}"]
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

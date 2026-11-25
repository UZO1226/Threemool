"""Park/e74 2시간 예측 구간의 공기량 곡선과 1시간 곡선의 곡선 대 곡선 비교 (Park 작성). 규칙은 README.md 를 따른다.
대표 차단 위 초과량 예측 제어 (NH4 여유 1.2) 를 예측 구간 8 스텝 (2시간) 으로 두고 공기 가중치 1.0, 2.0, 2.5 를 더 돌려,
Park/e72 의 가중치 1.5 점과 합친 4개 점 곡선을 Park/e58, e59 의 1시간 곡선과 겹치는 공기량 범위에서 비교한다.

    python run_e74.py val --part <i> --nparts 3 --data .. --surrogate .. --out <결과>/e74
    python run_e74.py report --out <결과>/e74 --e72 <결과>/e72 --e58 <결과>/e58 --e59 <결과>/e59 --md results.md
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
sys.path.insert(0, str(HERE.parent / 'e67'))
from darrow_env import BatchEnv, Dynamics, evaluate, load_days  # noqa: E402
from min_brake import MinAirBrake  # noqa: E402
from mpc_darrow import ShortMPC  # noqa: E402
from run_e38 import BRAKE  # noqa: E402
from run_e59 import curve_cost, frontier, gather  # noqa: E402
from run_e67 import mean_gap  # noqa: E402

H, MARGIN = 8, 1.2
LAMS = (1.0, 2.0, 2.5)   # 가중치 1.5 는 Park/e72
TOL = 0.05


def val(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    out = {}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for lam in LAMS[a.part::a.nparts]:
        out[str(lam)] = evaluate(env, splits['val'], ShortMPC(env, dyn, H=H, margin=MARGIN, lam=lam, objective='excess'),
                                 shield=MinAirBrake(dyn, 3.5, 1, 'others', **BRAKE))
        print(lam, out[str(lam)], flush=True)
        (Path(a.out) / f'val_part{a.part}.json').write_text(json.dumps(out, indent=2))


def report(a):
    v = gather(a.out, 'val_part*.json')
    v['1.5'] = gather(a.e72, 'val_part*.json')['8']
    mp = gather(a.e58, 'val_part*.json')
    mp.update(gather(a.e59, 'val_part*.json'))
    f4 = frontier([(r['air_m3'], r['cost']) for r in mp.values()])
    f8 = frontier([(r['air_m3'], r['cost']) for r in v.values()])
    L = [f'검증 40일 x 구성원 5개 = 200 에피소드. 대표 차단 위 초과량 예측 제어, NH4 여유 {MARGIN}. 2시간 (8 스텝) 곡선과 1시간 곡선 (Park/e58, e59, {f4[0, 0]:,.0f} ~ {f4[-1, 0]:,.0f}).', '',
         '| 공기 가중치 (2시간) | 공기량 | 하루 비용 | NH4 위반 % | DO 위반 % | 차단 개입 % | 곡선 점 | 같은 공기량의 1시간 곡선 비용 | 차이 |', '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    below = 0
    for lam in sorted(v, key=float):
        r = v[lam]
        on = any(abs(r['air_m3'] - f[0]) < 1e-6 and abs(r['cost'] - f[1]) < 1e-9 for f in f8)
        c = curve_cost(f4, r['air_m3'])
        d = r['cost'] - c
        below += int(not np.isnan(c) and d < 0)
        L.append(f"| {lam}{' (Park/e72)' if lam == '1.5' else ''} | {r['air_m3']:,.0f} | {r['cost']:.3f} | {r['nh4_viol_pct']:.2f} | {r['do_viol_pct']:.2f} | {r['intervention_pct']:.1f} | {'예' if on else '아니오 (지배됨)'} | {'범위 밖' if np.isnan(c) else f'{c:.3f}'} | {'' if np.isnan(c) else f'{d:+.3f}'} |")
    g, lo, hi = mean_gap(f8, f4)
    L += ['', f'2시간 곡선 범위 {f8[0, 0]:,.0f} ~ {f8[-1, 0]:,.0f}, 겹치는 범위 ' + ('없음' if np.isnan(g) else f'{lo:,.0f} ~ {hi:,.0f}'), '', '## 판정', '',
          f"- H1 (겹치는 범위에서 2시간 곡선이 1시간 곡선보다 평균 {TOL} 넘게 낮은가): 평균 차이 (2시간 빼기 1시간) {g:+.3f}: {'판정 안 함 (겹침 없음)' if np.isnan(g) else ('지지' if g < -TOL else '지지 안 됨')}",
          f"- H2 (2시간 점 4개 가운데 3개 이상이 같은 공기량의 1시간 곡선보다 낮은가): {below}/4: {'지지' if below >= 3 else '지지 안 됨'}"]
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    v = sub.add_parser('val')
    v.add_argument('--part', type=int, required=True)
    v.add_argument('--nparts', type=int, default=3)
    for k in ('data', 'surrogate', 'out'):
        v.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'e72', 'e58', 'e59', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    {'val': val, 'report': report}[a.cmd](a)


if __name__ == '__main__':
    main()

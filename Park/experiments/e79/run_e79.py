"""Park/e79 예측 제어 NH4 여유 1.8 의 공기량 곡선과 기본 여유 1.2 곡선의 비교 (Park 작성). 규칙은 README.md 를 따른다.
대표 차단 위 초과량 예측 제어 (1시간) 를 NH4 여유 1.8 로 두고 공기 가중치 1.0, 2.0, 2.5, 3.0 을 더 돌려,
Park/e76 의 가중치 1.5 점과 합친 5개 점을 Park/e58, e59 의 기본 설정 (여유 1.2) 곡선과 같은 공기량에서 비교한다.

    python run_e79.py val --part <i> --nparts 4 --data .. --surrogate .. --out <결과>/e79
    python run_e79.py report --out <결과>/e79 --e76 <결과>/e76 --e58 <결과>/e58 --e59 <결과>/e59 --md results.md
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

MARGIN = 1.8
LAMS = (1.0, 2.0, 2.5, 3.0)   # 가중치 1.5 는 Park/e76 의 m18
TOL = 0.05
REP = (82104, 4.658)          # 대표, Park/e35 묶음 5개 평균 (공기량, 하루 비용)


def val(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    out = {}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for lam in LAMS[a.part::a.nparts]:
        out[str(lam)] = evaluate(env, splits['val'], ShortMPC(env, dyn, H=4, margin=MARGIN, lam=lam, objective='excess'),
                                 shield=MinAirBrake(dyn, 3.5, 1, 'others', **BRAKE))
        print(lam, out[str(lam)], flush=True)
        (Path(a.out) / f'val_part{a.part}.json').write_text(json.dumps(out, indent=2))


def report(a):
    v = gather(a.out, 'val_part*.json')
    v['1.5'] = gather(a.e76, 'val_part*.json')['m18']
    mp = gather(a.e58, 'val_part*.json')
    mp.update(gather(a.e59, 'val_part*.json'))
    f12 = frontier([(r['air_m3'], r['cost']) for r in mp.values()])
    f18 = frontier([(r['air_m3'], r['cost']) for r in v.values()])
    L = [f'검증 40일 x 구성원 5개 = 200 에피소드. 대표 차단 위 초과량 예측 제어, 1시간. NH4 여유 {MARGIN} 의 점과 기본 여유 1.2 곡선 (Park/e58, e59, {f12[0, 0]:,.0f} ~ {f12[-1, 0]:,.0f}). 대표 {REP[0]:,} / {REP[1]:.3f} (Park/e35).', '',
         '| 공기 가중치 (여유 1.8) | 공기량 | 하루 비용 | NH4 위반 % | DO 위반 % | 차단 개입 % | 같은 공기량의 기본 곡선 비용 | 차이 | 대표보다 공기 적고 비용 0.05 넘게 낮음 |',
         '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    gaps, below, both = [], 0, 0
    for lam in sorted(v, key=float):
        r = v[lam]
        c = curve_cost(f12, r['air_m3'])
        d = r['cost'] - c
        if not np.isnan(c):
            gaps.append(d)
            below += int(d < 0)
        win = r['air_m3'] <= REP[0] and r['cost'] < REP[1] - TOL
        both += int(win)
        L.append(f"| {lam}{' (Park/e76)' if lam == '1.5' else ''} | {r['air_m3']:,.0f} | {r['cost']:.3f} | {r['nh4_viol_pct']:.2f} | {r['do_viol_pct']:.2f} | {r['intervention_pct']:.1f} | {'범위 밖' if np.isnan(c) else f'{c:.3f}'} | {'' if np.isnan(c) else f'{d:+.3f}'} | {'예' if win else '아니오'} |")
    m = float(np.mean(gaps)) if gaps else float('nan')
    g, lo, hi = mean_gap(f18, f12)
    L += ['', f'참고 (판정 없음): 여유 1.8 의 지배되지 않는 점 곡선 {f18[0, 0]:,.0f} ~ {f18[-1, 0]:,.0f}, 기본 곡선과의 곡선 대 곡선 평균 차이 ' + ('겹침 없음' if np.isnan(g) else f'{g:+.3f} ({lo:,.0f} ~ {hi:,.0f})'),
          '', '## 판정', '',
          f"- H1 (범위 안의 여유 1.8 점들이 같은 공기량의 기본 곡선보다 평균 {TOL} 넘게 낮은가): 점 {len(gaps)}개 평균 차이 {m:+.3f}: {'판정 안 함 (범위 안 점 없음)' if not gaps else ('지지' if m < -TOL else '지지 안 됨')}",
          f"- H2 (여유 1.8 점 5개 가운데 4개 이상이 같은 공기량의 기본 곡선보다 낮은가): {below}/5: {'지지' if below >= 4 else '지지 안 됨'}",
          f"- H3 (여유 1.8 점 가운데 하나라도 공기량이 대표 이하이고 비용이 대표보다 {TOL} 넘게 낮은가): {both}개: {'지지' if both else '지지 안 됨'}"]
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    v = sub.add_parser('val')
    v.add_argument('--part', type=int, required=True)
    v.add_argument('--nparts', type=int, default=4)
    for k in ('data', 'surrogate', 'out'):
        v.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'e76', 'e58', 'e59', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    {'val': val, 'report': report}[a.cmd](a)


if __name__ == '__main__':
    main()

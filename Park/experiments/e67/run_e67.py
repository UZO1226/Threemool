"""Park/e67 대표의 공기량 곡선과 차단 위 예측 제어 곡선의 곡선 대 곡선 비교 (Park 작성). 규칙은 README.md 를 따른다.
대표 묶음마다 정책 공기량에 배율을 곱한 대표 (ScaledPolicy, Park/e33, 모든 스텝) 를 검증 구간에서 돌려 묶음별 공기량, 비용 곡선을 만들고,
Park/e58, e59 의 차단 위 예측 제어 곡선과 겹치는 공기량 범위에서 평균 비용 차이를 낸다.

    python run_e67.py val --part <i> --nparts <n> --data .. --surrogate .. --runs <결과> --out <결과>/e67
    python run_e67.py report --out <결과>/e67 --e35 <결과>/e35 --e58 <결과>/e58 --e59 <결과>/e59 --md results.md
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
from min_brake import EnsemblePolicy, MinAirBrake, ScaledPolicy  # noqa: E402
from run_e38 import BRAKE, COMBOS, ci, key, load  # noqa: E402
from run_e59 import frontier, gather  # noqa: E402
from train_darrow import agent_act  # noqa: E402

FACTORS = (0.8, 0.9, 1.1, 1.25)   # 배율 1.0 은 Park/e35 orig
NGRID = 21                        # 겹치는 범위의 비교 점 수
JOBS = [(f, c) for f in FACTORS for c in COMBOS]


def val(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    out = {}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for f, c in JOBS[a.part::a.nparts]:
        ens = EnsemblePolicy([lambda obs, g=load(a.runs, 'fcsrl', s): agent_act(g, obs) for s in c])
        r = evaluate(env, splits['val'], ScaledPolicy(ens, f), shield=MinAirBrake(dyn, 3.5, 1, 'others', **BRAKE))
        out[f'{f}|{key(c)}'] = r
        print(f, c, r, flush=True)
        (Path(a.out) / f'val_part{a.part}.json').write_text(json.dumps(out, indent=2))


def mean_gap(rep_front, mpc_front):
    """겹치는 공기량 범위에서 (대표 곡선 빼기 예측 제어 곡선) 의 평균. 겹치지 않으면 nan."""
    lo, hi = max(rep_front[0, 0], mpc_front[0, 0]), min(rep_front[-1, 0], mpc_front[-1, 0])
    if hi <= lo:
        return float('nan'), lo, hi
    x = np.linspace(lo, hi, NGRID)
    return float((np.interp(x, rep_front[:, 0], rep_front[:, 1]) - np.interp(x, mpc_front[:, 0], mpc_front[:, 1])).mean()), lo, hi


def report(a):
    v = gather(a.out, 'val_part*.json')
    e35 = gather(a.e35, 'e35_part*.json')
    mp = gather(a.e58, 'val_part*.json')
    mp.update(gather(a.e59, 'val_part*.json'))
    mf = frontier([(r['air_m3'], r['cost']) for r in mp.values()])
    L = ['검증 40일 x 구성원 5개 = 200 에피소드. 대표 묶음마다 정책 공기량 배율을 바꿔 곡선을 만들고 (배율 1.0 은 Park/e35), 차단 위 예측 제어 곡선 (Park/e58, e59, '
         f'{mf[0, 0]:,.0f} ~ {mf[-1, 0]:,.0f}) 과 겹치는 범위에서 {NGRID} 점 평균 비용 차이를 낸다.', '',
         '| 묶음 | 배율 | 공기량 | 하루 비용 | NH4 위반 % | DO 위반 % | 차단 개입 % |', '| --- | --- | --- | --- | --- | --- | --- |']
    gaps, rows = [], []
    for c in COMBOS:
        k = key(c)
        pts = {1.0: e35[f'orig_{k}']}
        pts.update({f: v[f'{f}|{k}'] for f in FACTORS if f'{f}|{k}' in v})
        for f in sorted(pts):
            r = pts[f]
            iv = r.get('intervention_pct', float('nan'))
            L.append(f"| {k} | {f} | {r['air_m3']:,.0f} | {r['cost']:.3f} | {r['nh4_viol_pct']:.2f} | {r['do_viol_pct']:.2f} | {iv:.1f} |")
        rf = frontier([(r['air_m3'], r['cost']) for r in pts.values()])
        g, lo, hi = mean_gap(rf, mf)
        gaps.append(g)
        rows.append((k, len(rf), lo, hi, g))
    L += ['', '## 판정', '', '| 묶음 | 대표 곡선 점 | 겹치는 범위 | 평균 비용 차이 (대표 빼기 예측 제어) |', '| --- | --- | --- | --- |']
    for k, n, lo, hi, g in rows:
        L.append(f"| {k} | {n} | {'없음' if np.isnan(g) else f'{lo:,.0f} ~ {hi:,.0f}'} | {'겹침 없음' if np.isnan(g) else f'{g:+.3f}'} |")
    g = np.array(gaps)
    n1, n2 = int(np.nansum(g < 0)), int(np.nansum(g > 0))
    m = np.nanmean(g)
    L += ['', f"- H1 (대표 곡선이 낮은가): 평균 차이가 0 아래인 묶음 {n1}/5 (겹침 없음 {int(np.isnan(g).sum())} 개), 평균 {m:+.3f}: {'지지' if n1 >= 4 and m < 0 else '지지 안 됨'}",
          f"- H2 (예측 제어 곡선이 낮은가): 0 위인 묶음 {n2}/5: {'지지' if n2 >= 4 and m > 0 else '지지 안 됨'}"]
    if not np.isnan(g).any():
        L.append(f"- 참고 묶음 평균 차이 {ci(g, '{:+.3f}')}")
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    v = sub.add_parser('val')
    v.add_argument('--part', type=int, required=True)
    v.add_argument('--nparts', type=int, default=4)
    for k in ('data', 'surrogate', 'runs', 'out'):
        v.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'e35', 'e58', 'e59', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    {'val': val, 'report': report}[a.cmd](a)


if __name__ == '__main__':
    main()

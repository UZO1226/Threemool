"""Park/e80 예측 제어 NH4 여유 1.8 을 고를 때 쓰지 않은 날 (학습 구간 마지막 40일) 에서 다시 보기 (Park 작성). 규칙은 README.md 를 따른다.
대표 차단 위 초과량 예측 제어 (1시간, NH4 여유 1.8) 를 공기 가중치 1.0, 1.5, 2.0, 2.5 로 학습 구간 마지막 40일에서 돌리고,
같은 날의 대표 (Park/e43) 와 여유 1.2 곡선 (Park/e60 격자) 과 비교한다.

    python run_e80.py train --part <i> --nparts 4 --data .. --surrogate .. --out <결과>/e80
    python run_e80.py report --out <결과>/e80 --e43 <결과>/e43 --e60 <결과>/e60 --md results.md
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
from run_e38 import BRAKE, COMBOS, key  # noqa: E402
from run_e59 import curve_cost, frontier, gather  # noqa: E402

MARGIN = 1.8
LAMS = (1.0, 1.5, 2.0, 2.5)
TOL = 0.05


def train(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    out = {}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for lam in LAMS[a.part::a.nparts]:
        out[str(lam)] = evaluate(env, splits['train'][-40:], ShortMPC(env, dyn, H=4, margin=MARGIN, lam=lam, objective='excess'),
                                 shield=MinAirBrake(dyn, 3.5, 1, 'others', **BRAKE))
        print(lam, out[str(lam)], flush=True)
        (Path(a.out) / f'train_part{a.part}.json').write_text(json.dumps(out, indent=2))


def report(a):
    v = gather(a.out, 'train_part*.json')
    rep = gather(a.e43, 'rep_train_part*.json')
    grp = [rep[key(c)] for c in COMBOS]
    ra = float(np.mean([g['air_m3'] for g in grp]))
    rc = float(np.mean([g['cost'] for g in grp]))
    grid = gather(a.e60, 'grid_part*.json')
    f12 = frontier([(r['air_m3'], r['cost']) for k, r in grid.items() if k.startswith('1.2|')])
    L = [f'학습 구간 마지막 40일 x 구성원 5개 = 200 에피소드 (여유 1.8 을 고를 때 쓰지 않은 날). 대표 차단 위 초과량 예측 제어, 1시간, NH4 여유 {MARGIN}.',
         f'대표 (Park/e43, 묶음 5개 평균) {ra:,.0f} / {rc:.3f}. 여유 1.2 곡선 (Park/e60 격자, 공기 가중치 0.7 ~ 2.0) 범위 {f12[0, 0]:,.0f} ~ {f12[-1, 0]:,.0f}.', '',
         '| 공기 가중치 (여유 1.8) | 공기량 | 하루 비용 | NH4 위반 % | DO 위반 % | 차단 개입 % | 같은 공기량의 여유 1.2 곡선 비용 | 차이 | 대표보다 공기 적고 비용 0.05 넘게 낮음 | 비용이 대표보다 낮은 묶음 |',
         '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    gaps, wins = [], []
    for lam in sorted(v, key=float):
        r = v[lam]
        c = curve_cost(f12, r['air_m3'])
        d = r['cost'] - c
        if not np.isnan(c):
            gaps.append(d)
        nlow = sum(r['cost'] < g['cost'] for g in grp)
        win = r['air_m3'] <= ra and r['cost'] < rc - TOL and nlow >= 4
        if win:
            wins.append(lam)
        L.append(f"| {lam} | {r['air_m3']:,.0f} | {r['cost']:.3f} | {r['nh4_viol_pct']:.2f} | {r['do_viol_pct']:.2f} | {r['intervention_pct']:.1f} | {'범위 밖' if np.isnan(c) else f'{c:.3f}'} | {'' if np.isnan(c) else f'{d:+.3f}'} | {'예' if win else '아니오'} | {nlow}/5 |")
    L += ['', '대표 묶음별 (Park/e43):', '', '| 묶음 | 공기량 | 하루 비용 |', '| --- | --- | --- |']
    L += [f"| {key(c)} | {g['air_m3']:,.0f} | {g['cost']:.3f} |" for c, g in zip(COMBOS, grp)]
    m = float(np.mean(gaps)) if gaps else float('nan')
    L += ['', '## 판정', '',
          f"- H1 (여유 1.8 점 가운데 하나라도 공기량이 대표 평균 이하이고, 비용이 대표 평균보다 {TOL} 넘게 낮고, 대표 묶음 5개 가운데 4개 이상보다 낮은가): {', '.join(wins) if wins else '없음'}: {'지지' if wins else '지지 안 됨'}",
          f"- H2 (여유 1.2 곡선 범위 안의 여유 1.8 점들이 같은 공기량의 여유 1.2 곡선보다 평균 {TOL} 넘게 낮은가): 점 {len(gaps)}개 평균 차이 {m:+.3f}: {'판정 안 함 (범위 안 점 없음)' if not gaps else ('지지' if m < -TOL else '지지 안 됨')}"]
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('--part', type=int, required=True)
    t.add_argument('--nparts', type=int, default=4)
    for k in ('data', 'surrogate', 'out'):
        t.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'e43', 'e60', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    {'train': train, 'report': report}[a.cmd](a)


if __name__ == '__main__':
    main()

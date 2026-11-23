"""Park/e66 공기량을 묶음마다 직접 맞춘 차단 위 예측 제어와의 짝 비교 (Park 작성). 규칙은 README.md 를 따른다.
대표 묶음마다 그 묶음의 검증 공기량에 맞는 공기 가중치를 Park/e58, e59 의 차단 위 예측 제어 (공기 가중치, 공기량) 점에서 보간으로 정하고
(비용은 읽지 않음), 대표 차단을 씌운 초과량 예측 제어 (NH4 여유 1.2) 를 검증 구간에서 돌려 같은 묶음의 대표와 비교한다. Park/e57 의 차단 판이다.

    python run_e66.py plan --round 1 --out <결과>/e66 --e58 <결과>/e58 --e59 <결과>/e59 --e35 <결과>/e35
    python run_e66.py val --round 1 --part <i> --nparts <n> --data .. --surrogate .. --out <결과>/e66
    python run_e66.py plan --round 2 ...   (1% 밖인 묶음만)
    python run_e66.py val --round 2 ...
    python run_e66.py report --out <결과>/e66 --e35 <결과>/e35 --md results.md
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
sys.path.insert(0, str(HERE.parent / 'e38'))
sys.path.insert(0, str(HERE.parent / 'e57'))
from darrow_env import BatchEnv, Dynamics, evaluate, load_days  # noqa: E402
from min_brake import MinAirBrake  # noqa: E402
from mpc_darrow import ShortMPC  # noqa: E402
from run_e38 import BRAKE, COMBOS, ci, key  # noqa: E402
from run_e57 import FIX_TOL, VAL_TOL, gather, lam_for, targets  # noqa: E402

MARGIN = 1.2


def air_points(a):
    """(공기 가중치, 공기량) 만 읽는다. 비용은 읽지 않는다."""
    v = gather(a.e58, 'val_part*.json')
    v.update(gather(a.e59, 'val_part*.json'))
    pts = [(float(k), r['air_m3']) for k, r in v.items()]
    for rnd in (1, 2):
        for k, r in gather(a.out, f'r{rnd}_part*.json').items():
            pts.append((r['lam'], r['air_m3']))
    return pts


def plan(a):
    tg = targets(gather(a.e35, 'e35_part*.json'))
    pts = air_points(a)
    if a.round == 1:
        todo = list(tg)
    else:
        r1 = gather(a.out, 'r1_part*.json')
        todo = [k for k in tg if abs(r1[k]['air_m3'] / tg[k] - 1) > FIX_TOL]
    p = {k: {'target': tg[k], 'lam': lam_for(pts, tg[k])} for k in todo}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f'plan_r{a.round}.json').write_text(json.dumps(p, indent=2))
    print(json.dumps(p, indent=2))


def val(a):
    p = json.loads((Path(a.out) / f'plan_r{a.round}.json').read_text())
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    out = {}
    for k in sorted(p)[a.part::a.nparts]:
        lam = p[k]['lam']
        r = evaluate(env, splits['val'], ShortMPC(env, dyn, margin=MARGIN, lam=lam, objective='excess'),
                     shield=MinAirBrake(dyn, 3.5, 1, 'others', **BRAKE))
        r['lam'] = lam
        out[k] = r
        print(k, r, flush=True)
        (Path(a.out) / f'r{a.round}_part{a.part}.json').write_text(json.dumps(out, indent=2))


def final(a):
    """묶음마다 마지막 판 (2판이 있으면 2판) 의 값."""
    r = gather(a.out, 'r1_part*.json')
    r.update(gather(a.out, 'r2_part*.json'))
    return r


def report(a):
    e35 = gather(a.e35, 'e35_part*.json')
    m = final(a)
    ks = [key(c) for c in COMBOS]
    ra = np.array([e35[f'orig_{k}']['air_m3'] for k in ks])
    rc = np.array([e35[f'orig_{k}']['cost'] for k in ks])
    ma = np.array([m[k]['air_m3'] for k in ks])
    mc = np.array([m[k]['cost'] for k in ks])
    gap = ma / ra - 1
    ok = np.abs(gap) <= VAL_TOL
    L = ['검증 40일 x 구성원 5개 = 200 에피소드. 묶음마다 공기량을 맞춘 차단 위 예측 제어 (NH4 여유 1.2, 대표 차단) 와 짝지어 비교한다.', '',
         '| 묶음 | 공기 가중치 | 대표 공기량 | 예측 제어 공기량 | 차이 | 대표 비용 | 예측 제어 비용 | 차단 개입 % | 대표가 낮음 |',
         '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for i, k in enumerate(ks):
        L.append(f"| {k} | {m[k]['lam']:.3f} | {ra[i]:,.0f} | {ma[i]:,.0f} | {100 * gap[i]:+.1f}% | {rc[i]:.3f} | {mc[i]:.3f} | {m[k]['intervention_pct']:.1f} | {'예' if rc[i] < mc[i] else '아니오'} |")
    L += ['', '## 판정', '']
    if not ok.all():
        L.append(f'- 공기량 차이가 ±{100 * VAL_TOL:.0f}% 밖인 묶음 {(~ok).sum()} 개: 사전 등록 규칙대로 H1, H2 를 판정하지 않는다')
    else:
        n1, n2 = int((rc < mc).sum()), int((mc < rc).sum())
        L.append(f"- H1 (대표가 낮은가): 낮은 묶음 {n1}/5, 평균 {rc.mean():.3f} 대 {mc.mean():.3f}: {'지지' if n1 >= 4 and rc.mean() < mc.mean() else '지지 안 됨'}")
        L.append(f"- H2 (차단 위 예측 제어가 낮은가): 낮은 묶음 {n2}/5: {'지지' if n2 >= 4 and mc.mean() < rc.mean() else '지지 안 됨'}")
    L.append(f"- 참고 비용 차이 (대표 빼기 예측 제어) {ci(rc - mc, '{:+.3f}')}, 공기량 차이 평균 {100 * gap.mean():+.2f}%")
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('plan')
    o.add_argument('--round', type=int, required=True)
    for k in ('out', 'e58', 'e59', 'e35'):
        o.add_argument(f'--{k}', required=True)
    v = sub.add_parser('val')
    v.add_argument('--round', type=int, required=True)
    v.add_argument('--part', type=int, required=True)
    v.add_argument('--nparts', type=int, default=5)
    for k in ('data', 'surrogate', 'out'):
        v.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'e35', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    {'plan': plan, 'val': val, 'report': report}[a.cmd](a)


if __name__ == '__main__':
    main()

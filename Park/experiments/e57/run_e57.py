"""Park/e57 공기량을 묶음마다 직접 맞춘 예측 제어와의 짝 비교 (Park 작성). 규칙은 README.md 를 따른다.
대표 묶음마다 그 묶음의 검증 공기량에 맞는 공기 가중치를 Park/e56 의 (공기 가중치, 공기량) 점에서 보간으로 정하고 (비용은 읽지 않음),
초과량 예측 제어 (NH4 여유 1.2) 를 검증 구간에서 돌려 같은 묶음의 대표와 비교한다. 공기량 차이가 1% 를 넘으면 한 번 고친다.

    python run_e57.py plan --round 1 --out <결과>/e57 --e56 <결과>/e56 --e52 <결과>/e52 --e35 <결과>/e35
    python run_e57.py val --round 1 --part <i> --nparts <n> --data .. --surrogate .. --out <결과>/e57
    python run_e57.py plan --round 2 ...   (1% 밖인 묶음만)
    python run_e57.py val --round 2 ...
    python run_e57.py report --out <결과>/e57 --e35 <결과>/e35 --md results.md
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
sys.path.insert(0, str(HERE.parent / 'e38'))
from darrow_env import BatchEnv, Dynamics, evaluate, load_days  # noqa: E402
from mpc_darrow import ShortMPC  # noqa: E402
from run_e38 import COMBOS, ci, key  # noqa: E402

MARGIN = 1.2      # Park/e52 가 고른 NH4 여유
FIX_TOL = 0.01    # 이보다 공기량 차이가 크면 한 번 고친다
VAL_TOL = 0.03    # 짝마다 판정에 필요한 공기량 차이 (Park/e43 과 같음)


def gather(d, pat):
    res = {}
    for p in sorted(Path(d).glob(pat)):
        res.update(json.loads(p.read_text()))
    return res


def targets(e35):
    return {key(c): e35[f'orig_{key(c)}']['air_m3'] for c in COMBOS}


def air_points(a):
    """(공기 가중치, 공기량) 만 읽는다. 비용은 읽지 않는다."""
    v = gather(a.e56, 'val_part*.json')
    v['1.0'] = json.loads((Path(a.e52) / 'val.json').read_text())['mpc']
    pts = [(float(k), r['air_m3']) for k, r in v.items()]
    for rnd in (1, 2):
        for k, r in gather(a.out, f'r{rnd}_part*.json').items():
            pts.append((r['lam'], r['air_m3']))
    return pts


def lam_for(pts, target):
    """공기량에 대한 log(공기 가중치) 의 선형 보간. 목표 양옆의 가장 가까운 두 점을 쓴다."""
    pts = sorted(pts, key=lambda p: p[1])
    lo = [p for p in pts if p[1] <= target]
    hi = [p for p in pts if p[1] >= target]
    if not lo or not hi:
        p = min(pts, key=lambda p: abs(p[1] - target))
        return p[0]
    (l1, a1), (l2, a2) = lo[-1], hi[0]
    if a2 == a1:
        return l1
    w = (target - a1) / (a2 - a1)
    return float(np.exp((1 - w) * np.log(l1) + w * np.log(l2)))


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
    ks = sorted(p)[a.part::a.nparts]
    if not ks:
        return
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    out = {}
    for k in ks:
        r = evaluate(env, splits['val'], ShortMPC(env, dyn, margin=MARGIN, lam=p[k]['lam'], objective='excess'))
        r.update(p[k])
        out[k] = r
        print(k, r, flush=True)
        (Path(a.out) / f'r{a.round}_part{a.part}.json').write_text(json.dumps(out, indent=2))


def report(a):
    e35 = gather(a.e35, 'e35_part*.json')
    v = gather(a.out, 'r1_part*.json')
    r2 = gather(a.out, 'r2_part*.json')
    v.update(r2)
    ks = [key(c) for c in COMBOS]
    ra, rc = np.array([e35[f'orig_{k}']['air_m3'] for k in ks]), np.array([e35[f'orig_{k}']['cost'] for k in ks])
    ma, mc = np.array([v[k]['air_m3'] for k in ks]), np.array([v[k]['cost'] for k in ks])
    gap = ma / ra - 1
    L = ['검증 40일 x 구성원 5개 = 200 에피소드. 예측 제어의 공기 가중치는 짝이 되는 대표 묶음의 검증 공기량에 맞춰 보간으로 정했다.', '',
         '| 묶음 | 대표 공기량 | 대표 비용 | 공기 가중치 | 고친 횟수 | 예측 제어 공기량 | 공기량 차이 | 예측 제어 비용 | NH4 위반 % | DO 위반 % | 대표가 낮음 |',
         '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    inside = np.abs(gap) <= VAL_TOL
    ok = (rc < mc) & inside
    for i, k in enumerate(ks):
        L.append(f"| {k} | {ra[i]:,.0f} | {rc[i]:.3f} | {v[k]['lam']:.3f} | {1 if k in r2 else 0} | {ma[i]:,.0f} | {100 * gap[i]:+.1f}% | {mc[i]:.3f} | {v[k]['nh4_viol_pct']:.2f} | {v[k]['do_viol_pct']:.2f} | {'예' if ok[i] else '아니오'} |")
    if inside.sum() < 4:
        verdict = f'판정 안 함 (공기량이 허용폭 안인 짝 {inside.sum()}/5)'
    else:
        verdict = '지지' if ok.sum() >= 4 and rc.mean() < mc.mean() else '지지 안 됨'
    L += ['', f"- H1: 공기량이 ±{100 * VAL_TOL:.0f}% 안인 짝 {inside.sum()}/5, 대표가 낮은 짝 {ok.sum()}/5, 평균 {rc.mean():.3f} 대 {mc.mean():.3f}: {verdict}",
          f"- 참고 비용 차이 (대표 빼기 공기량을 맞춘 예측 제어, 짝) {ci(rc - mc, '{:+.3f}')}",
          f"- 참고 공기량 차이 평균 {100 * gap.mean():+.2f}% (절댓값 최대 {100 * np.abs(gap).max():.2f}%)"]
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    pl = sub.add_parser('plan')
    pl.add_argument('--round', type=int, required=True)
    for x in ('out', 'e56', 'e52', 'e35'):
        pl.add_argument(f'--{x}', required=True)
    v = sub.add_parser('val')
    v.add_argument('--round', type=int, required=True)
    v.add_argument('--part', type=int, required=True)
    v.add_argument('--nparts', type=int, default=5)
    for k in ('data', 'surrogate', 'out'):
        v.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for x in ('out', 'e35', 'md'):
        r.add_argument(f'--{x}', required=True)
    a = p.parse_args()
    {'plan': plan, 'val': val, 'report': report}[a.cmd](a)


if __name__ == '__main__':
    main()

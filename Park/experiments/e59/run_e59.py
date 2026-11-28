"""Park/e59 같은 차단 위의 예측 제어 곡선을 대표 공기량 아래로 늘린 비교 (Park 작성). 규칙은 README.md 를 따른다.
Park/e58 의 7개 점에 공기 가중치가 큰 점 4개를 더해 곡선을 다시 만든다. 나머지는 Park/e58 과 같다.
대표와 같은 차단 (MinAirBrake, BRAKE) 을 씌운 초과량 예측 제어 (NH4 여유 1.2) 의 공기 가중치를 바꿔 검증 구간에서 공기량, 비용 곡선을 만들고, 대표 묶음마다 그 공기량에서의
곡선 비용 (지배되지 않는 점들의 선형 보간) 과 비교한다.

    python run_e59.py val --part <i> --nparts <n> --data .. --surrogate .. --out <결과>/e59
    python run_e59.py report --out <결과>/e59 --e58 <결과>/e58 --e35 <결과>/e35 --e43 <결과>/e43 --md results.md
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
from min_brake import MinAirBrake  # noqa: E402
from mpc_darrow import ShortMPC  # noqa: E402
from run_e38 import BRAKE, COMBOS, ci, key  # noqa: E402

MARGIN = 1.2                                  # Park/e52 가 고른 NH4 여유
LAMS = (2.0, 2.5, 3.0, 4.0)   # 새로 돌리는 공기 가중치 (Park/e58 의 7개 점과 합침)


def make(env, dyn, lam):
    return ShortMPC(env, dyn, margin=MARGIN, lam=lam, objective='excess')


def gather(d, pat):
    res = {}
    for p in sorted(Path(d).glob(pat)):
        res.update(json.loads(p.read_text()))
    return res


def val(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    out = {}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for lam in LAMS[a.part::a.nparts]:
        out[str(lam)] = evaluate(env, splits['val'], make(env, dyn, lam), **extra(dyn))
        print(lam, out[str(lam)], flush=True)
        (Path(a.out) / f'val_part{a.part}.json').write_text(json.dumps(out, indent=2))


def extra(dyn):
    return {'shield': MinAirBrake(dyn, 3.5, 1, 'others', **BRAKE)}


def frontier(points):
    """(공기량, 비용) 점에서 지배되지 않는 점만 남겨 공기량 오름차순으로 돌려준다."""
    pts = sorted(points)
    keep = [p for p in pts if not any(q[0] <= p[0] and q[1] < p[1] for q in pts if q is not p)]
    return np.array(keep)


def curve_cost(front, air):
    """곡선 범위 밖이면 nan (그 묶음은 대표가 낮지 않은 것으로 센다)."""
    if air < front[0, 0] or air > front[-1, 0]:
        return float('nan')
    return float(np.interp(air, front[:, 0], front[:, 1]))


def points(a):
    v = gather(a.e58, 'val_part*.json')
    v.update(gather(a.out, 'val_part*.json'))
    return v


def report(a):
    v = points(a)
    e35 = gather(a.e35, 'e35_part*.json')
    v43 = gather(a.e43, 'val_part*.json')
    p43 = json.loads((Path(a.e43) / 'pick.json').read_text())
    sp, kp, ki = p43['abac']
    abac, brake = v43[f'abac|{sp}|{kp}|{ki}'], v43[f"brake|{p43['delta']}"]
    ra = np.array([e35[f'orig_{key(c)}']['air_m3'] for c in COMBOS])
    rc = np.array([e35[f'orig_{key(c)}']['cost'] for c in COMBOS])
    front = frontier([(r['air_m3'], r['cost']) for r in v.values()])
    mc = np.array([curve_cost(front, x) for x in ra])
    L = [f'검증 40일 x 구성원 5개 = 200 에피소드. 대표와 같은 차단을 씌운 예측 제어의 공기 가중치별 값 (2.0 ~ 4.0 은 이 실험, 나머지는 Park/e58).', '',
         '| 공기 가중치 | 공기량 | 하루 비용 | NH4 위반 % | DO 위반 % | 차단 개입 % | 곡선 점 |', '| --- | --- | --- | --- | --- | --- | --- |']
    for lam in sorted(v, key=lambda k: -float(k)):
        r = v[lam]
        on = any(abs(r['air_m3'] - f[0]) < 1e-6 and abs(r['cost'] - f[1]) < 1e-9 for f in front)
        L.append(f"| {lam} | {r['air_m3']:,.0f} | {r['cost']:.3f} | {r['nh4_viol_pct']:.2f} | {r['do_viol_pct']:.2f} | {r['intervention_pct']:.1f} | {'예' if on else '아니오 (지배됨)'} |")
    L += ['', f"곡선 범위: 공기량 {front[0, 0]:,.0f} ~ {front[-1, 0]:,.0f}", '', '## 판정', '',
          '| 묶음 | 대표 공기량 | 대표 비용 | 같은 공기량의 예측 제어 곡선 비용 | 대표가 낮음 |', '| --- | --- | --- | --- | --- |']
    ok = np.where(np.isnan(mc), False, rc < np.nan_to_num(mc, nan=-1))
    for i, c in enumerate(COMBOS):
        L.append(f"| {key(c)} | {ra[i]:,.0f} | {rc[i]:.3f} | {'범위 밖' if np.isnan(mc[i]) else f'{mc[i]:.3f}'} | {'예' if ok[i] else '아니오'} |")
    inside = ~np.isnan(mc)
    verdict = '지지' if ok.sum() >= 4 and inside.all() and rc.mean() < mc.mean() else '지지 안 됨'
    L += ['', f"- H1: 대표가 낮은 묶음 {ok.sum()}/5 (곡선 범위 밖 {(~inside).sum()} 개), 평균 {rc.mean():.3f} 대 {np.nanmean(mc):.3f}: {verdict}"]
    lo = np.where(np.isnan(mc), False, np.nan_to_num(mc, nan=1e9) < rc)
    v2 = '지지' if inside.all() and lo.sum() >= 4 and np.nanmean(mc) < rc.mean() else '지지 안 됨'
    L.append(f"- H2 (반대 방향): 곡선이 대표보다 낮은 묶음 {lo.sum()}/5: {v2}")
    if inside.all():
        L.append(f"- 참고 비용 차이 (대표 빼기 같은 공기량의 예측 제어) {ci(rc - mc, '{:+.3f}')}")
    for n, m in (('차단만 (Park/e43)', brake), ('ABAC (Park/e43)', abac)):
        c = curve_cost(front, m['air_m3'])
        L.append(f"- 참고 {n} {m['air_m3']:,.0f} / {m['cost']:.3f}, 같은 공기량의 예측 제어 곡선 비용 {'범위 밖' if np.isnan(c) else f'{c:.3f}'}")
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
    for x in ('out', 'e58', 'e35', 'e43', 'md'):
        r.add_argument(f'--{x}', required=True)
    a = p.parse_args()
    {'val': val, 'report': report}[a.cmd](a)


if __name__ == '__main__':
    main()

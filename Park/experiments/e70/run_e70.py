"""Park/e70 대표 차단 위의 ABAC 와의 같은 공기량 비교 (Park 작성). 규칙은 README.md 를 따른다.
다시 고른 ABAC (Park/e54, Kp 4,000, Ki 1,600) 의 설정값을 바꿔 대표 차단을 씌우고 검증 구간에서 공기량, 비용 곡선을 만든 뒤,
대표 묶음마다 그 공기량에서의 곡선 비용 (지배되지 않는 점의 선형 보간, Park/e58 과 같은 규칙) 과 비교한다.

    python run_e70.py val --part <i> --nparts <n> --data .. --surrogate .. --out <결과>/e70
    python run_e70.py report --out <결과>/e70 --e35 <결과>/e35 --e54 <결과>/e54 --e58 <결과>/e58 --e59 <결과>/e59 --md results.md
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
from baselines_darrow import ABAC  # noqa: E402
from darrow_env import BatchEnv, Dynamics, evaluate, load_days  # noqa: E402
from min_brake import MinAirBrake  # noqa: E402
from run_e38 import BRAKE, COMBOS, ci, key  # noqa: E402
from run_e59 import curve_cost, frontier, gather  # noqa: E402

KP, KI = 4000.0, 1600.0                 # Park/e54 가 고른 이득
SPS = (2.6, 3.2, 3.8, 4.4, 5.0)          # 설정값 (3.2 는 Park/e54 고른 값)


def val(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    out = {}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for sp in SPS[a.part::a.nparts]:
        out[str(sp)] = evaluate(env, splits['val'], ABAC(env, sp, KP, KI), shield=MinAirBrake(dyn, 3.5, 1, 'others', **BRAKE))
        print(sp, out[str(sp)], flush=True)
        (Path(a.out) / f'val_part{a.part}.json').write_text(json.dumps(out, indent=2))


def report(a):
    v = gather(a.out, 'val_part*.json')
    e35 = gather(a.e35, 'e35_part*.json')
    mp = gather(a.e58, 'val_part*.json')
    mp.update(gather(a.e59, 'val_part*.json'))
    mf = frontier([(r['air_m3'], r['cost']) for r in mp.values()])
    v54 = gather(a.e54, 'val*.json')
    front = frontier([(r['air_m3'], r['cost']) for r in v.values()])
    ks = [key(c) for c in COMBOS]
    ra = np.array([e35[f'orig_{k}']['air_m3'] for k in ks])
    rc = np.array([e35[f'orig_{k}']['cost'] for k in ks])
    bc = np.array([curve_cost(front, x) for x in ra])
    L = [f'검증 40일 x 구성원 5개 = 200 에피소드. ABAC (Kp {KP:,.0f}, Ki {KI:,.0f}) 의 설정값을 바꾸고 대표 차단을 씌움.', '',
         '| 설정값 | 공기량 | 하루 비용 | NH4 위반 % | DO 위반 % | 차단 개입 % | 곡선 점 | 같은 공기량의 차단 위 예측 제어 곡선 (e58, e59) |', '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for sp in sorted(v, key=float):
        r = v[sp]
        on = any(abs(r['air_m3'] - f[0]) < 1e-6 and abs(r['cost'] - f[1]) < 1e-9 for f in front)
        m = curve_cost(mf, r['air_m3'])
        L.append(f"| {sp} | {r['air_m3']:,.0f} | {r['cost']:.3f} | {r['nh4_viol_pct']:.2f} | {r['do_viol_pct']:.2f} | {r['intervention_pct']:.1f} | {'예' if on else '아니오 (지배됨)'} | {'범위 밖' if np.isnan(m) else f'{m:.3f}'} |")
    L += ['', f'곡선 범위: 공기량 {front[0, 0]:,.0f} ~ {front[-1, 0]:,.0f}', '', '## 판정', '',
          '| 묶음 | 대표 공기량 | 대표 비용 | 같은 공기량의 "차단 + ABAC" 곡선 비용 | 대표가 낮음 |', '| --- | --- | --- | --- | --- |']
    ok = np.where(np.isnan(bc), False, rc < np.nan_to_num(bc, nan=-1))
    for i, k in enumerate(ks):
        L.append(f"| {k} | {ra[i]:,.0f} | {rc[i]:.3f} | {'범위 밖' if np.isnan(bc[i]) else f'{bc[i]:.3f}'} | {'예' if ok[i] else '아니오'} |")
    inside = ~np.isnan(bc)
    n2 = int(np.where(inside, bc < rc, False).sum())
    L += ['', f"- H1 (대표가 낮은가): 낮은 묶음 {ok.sum()}/5 (범위 밖 {(~inside).sum()} 개), 평균 {rc.mean():.3f} 대 {np.nanmean(bc):.3f}: {'지지' if ok.sum() >= 4 and inside.all() and rc.mean() < bc.mean() else '지지 안 됨'}",
          f"- H2 (차단 + ABAC 가 낮은가): 낮은 묶음 {n2}/5: {'지지' if n2 >= 4 and inside.all() and bc.mean() < rc.mean() else '지지 안 됨'}"]
    if inside.all():
        L.append(f"- 참고 비용 차이 (대표 빼기 차단 + ABAC) {ci(rc - bc, '{:+.3f}')}")
    r54 = v54.get(f'abac|3.2|{KP:g}|{KI:g}') or next((r for k, r in v54.items() if k.startswith('abac')), None)
    if r54:
        L.append(f"- 참고 차단 없는 ABAC (Park/e54) {r54['air_m3']:,.0f} / {r54['cost']:.3f}")
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    v = sub.add_parser('val')
    v.add_argument('--part', type=int, required=True)
    v.add_argument('--nparts', type=int, default=5)
    for k in ('data', 'surrogate', 'out'):
        v.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'e35', 'e54', 'e58', 'e59', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    {'val': val, 'report': report}[a.cmd](a)


if __name__ == '__main__':
    main()

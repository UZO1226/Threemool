"""Park/e54 넓힌 ABAC 격자로 다시 고른 같은 공기량 기준 (Park 작성). 규칙은 README.md 를 따른다.
Park/e43 이 고른 ABAC 의 Kp 2000 은 격자 끝이었다. Kp 를 4000 ~ 32000 으로 넓힌 격자를 학습 구간에서 돌리고, Park/e43 격자와 합쳐
같은 규칙 (대표 학습 구간 공기량 ±2% 안에서 비용 최소) 으로 다시 고른 뒤 검증에서 대표와 비교한다.

    python run_e54.py grid --part <i> --nparts <n> --data .. --surrogate .. --out <결과>/e54
    python run_e54.py pick --out <결과>/e54 --e43 <결과>/e43 --md select.md
    python run_e54.py val --data .. --surrogate .. --out <결과>/e54
    python run_e54.py report --out <결과>/e54 --e35 <결과>/e35 --e43 <결과>/e43 --md results.md
"""
import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
sys.path.insert(0, str(HERE.parent / 'e38'))
from baselines_darrow import ABAC  # noqa: E402
from darrow_env import BatchEnv, Dynamics, evaluate, load_days  # noqa: E402
from run_e38 import COMBOS, ci, key  # noqa: E402

SP = tuple(round(2.6 + 0.1 * i, 1) for i in range(9))      # 2.6 ~ 3.4 (Park/e43 이 고른 3.0 주변)
KP = (4000.0, 8000.0, 16000.0, 32000.0)                     # Park/e43 격자 끝 (2000) 너머
KI = (400.0, 800.0, 1600.0, 3200.0, 6400.0)
GRID = list(itertools.product(SP, KP, KI))
TOL, VAL_TOL = 0.02, 0.03   # Park/e43 과 같음


def setup(a):
    splits, day_data = load_days(a.data)
    return splits, BatchEnv(day_data, Dynamics(a.surrogate))


def gather(d, pat):
    res = {}
    for p in sorted(Path(d).glob(pat)):
        res.update(json.loads(p.read_text()))
    return res


def grid(a):
    splits, env = setup(a)
    out = {}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for sp, kp, ki in GRID[a.part::a.nparts]:
        out[f'{sp}_{kp}_{ki}'] = evaluate(env, splits['train'][-40:], ABAC(env, sp, kp, ki))
        print(sp, kp, ki, out[f'{sp}_{kp}_{ki}'], flush=True)
        (Path(a.out) / f'grid_part{a.part}.json').write_text(json.dumps(out, indent=2))


def pick(a):
    target = json.loads((Path(a.e43) / 'pick.json').read_text())['target_train_air']
    new, old = gather(a.out, 'grid_part*.json'), gather(a.e43, 'abac_train_part*.json')
    assert len(new) == len(GRID)
    ab = {**old, **new}
    near = {k: v for k, v in ab.items() if abs(v['air_m3'] / target - 1) <= TOL}
    ka = min(near, key=lambda k: (near[k]['cost'], near[k]['air_m3'])) if near else min(ab, key=lambda k: abs(ab[k]['air_m3'] - target))
    sp, kp, ki = map(float, ka.split('_'))
    allkp = sorted({float(k.split('_')[1]) for k in ab})
    sel = {'target_train_air': target, 'abac': [sp, kp, ki], 'within_tol': bool(near), 'from_new_grid': ka in new, 'kp_edge': kp == max(allkp)}
    (Path(a.out) / 'pick.json').write_text(json.dumps(sel, indent=2))
    L = [f'학습 구간 마지막 40일 x 구성원 5개. 맞출 공기량 {target:,.0f} (Park/e43). Park/e43 격자 {len(old)} 개와 넓힌 격자 {len(new)} 개를 합친 {len(ab)} 개에서 고른다.', '',
         f'±{100 * TOL:.0f}% 안의 조합 {len(near)} 개 가운데 하루 비용이 낮은 10개:', '',
         '| 설정값 | Kp | Ki | 공기량 | 하루 비용 | 격자 |', '| --- | --- | --- | --- | --- | --- |']
    for k in sorted(near, key=lambda k: (near[k]['cost'], near[k]['air_m3']))[:10]:
        s_, p_, i_ = k.split('_')
        L.append(f"| {s_} | {float(p_):g} | {float(i_):g} | {near[k]['air_m3']:,.0f} | {near[k]['cost']:.3f} | {'넓힌 격자' if k in new else 'Park/e43'} |")
    po = json.loads((Path(a.e43) / 'pick.json').read_text())['abac']
    L += ['', f"- Park/e43 이 고른 ABAC: 설정값 {po[0]}, Kp {po[1]:g}, Ki {po[2]:g}",
          f"- 다시 고른 ABAC: 설정값 {sp}, Kp {kp:g}, Ki {ki:g} ({'넓힌 격자' if sel['from_new_grid'] else 'Park/e43 격자'}). 허용폭 안: {'예' if near else '아니오'}. Kp 격자 끝: {'예' if sel['kp_edge'] else '아니오'}"]
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def val(a):
    sel = json.loads((Path(a.out) / 'pick.json').read_text())
    splits, env = setup(a)
    r = evaluate(env, splits['val'], ABAC(env, *sel['abac']))
    print(r, flush=True)
    (Path(a.out) / 'val.json').write_text(json.dumps({'abac': r}, indent=2))


def report(a):
    sel = json.loads((Path(a.out) / 'pick.json').read_text())
    ab = json.loads((Path(a.out) / 'val.json').read_text())['abac']
    e35, v43 = gather(a.e35, 'e35_part*.json'), gather(a.e43, 'val_part*.json')
    p43 = json.loads((Path(a.e43) / 'pick.json').read_text())
    old = v43['abac|{}|{}|{}'.format(*p43['abac'])]
    keys = ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct', 'days_within_limit_pct')
    ra, rc = (np.array([e35[f'orig_{key(c)}'][k] for c in COMBOS]) for k in ('air_m3', 'cost'))
    rep = {k: np.mean([e35[f'orig_{key(c)}'][k] for c in COMBOS]) for k in keys}
    row = lambda n, m: f"| {n} | {m['air_m3']:,.0f} | {100 * (m['air_m3'] / ra.mean() - 1):+.1f}% | {m['cost']:.3f} | {m['nh4_viol_pct']:.2f} | {m['do_viol_pct']:.2f} | {m['days_within_limit_pct']:.1f} |"  # noqa: E731
    L = ['검증 40일 x 구성원 5개 = 200 에피소드.', '', '| 방법 | 공기량 Nm3/일 | 대표 대비 | 하루 비용 | NH4 위반 % | DO 위반 % | 한도 12 만족 % |', '| --- | --- | --- | --- | --- | --- | --- |',
         row('대표 (묶음 5개 평균, Park/e35)', rep),
         row('ABAC 다시 고름 (설정값 {}, Kp {:g}, Ki {:g})'.format(*sel['abac']), ab),
         row('ABAC Park/e43 (설정값 {}, Kp {:g}, Ki {:g})'.format(*p43['abac']), old)]
    gap = ab['air_m3'] / ra.mean() - 1
    ok = rc < ab['cost']
    L += ['', '## 판정', '', '| 묶음 | 대표 공기량 | 대표 비용 | 다시 고른 ABAC 보다 낮음 |', '| --- | --- | --- | --- |']
    for i, c in enumerate(COMBOS):
        L.append(f"| {key(c)} | {ra[i]:,.0f} | {rc[i]:.3f} | {'예' if ok[i] else '아니오'} |")
    v = f'판정 안 함 (맞춤 실패, 공기량 차이 {100 * gap:+.1f}%)' if abs(gap) > VAL_TOL else ('지지' if ok.sum() >= 4 and rc.mean() < ab['cost'] else '지지 안 됨')
    L += ['', f"- H1: 공기량 차이 {100 * gap:+.1f}%, 대표가 낮은 묶음 {ok.sum()}/5, 평균 {rc.mean():.3f} 대 {ab['cost']:.3f}: {v}",
          f"- 참고 비용 차이 (대표 빼기 다시 고른 ABAC) {ci(rc - ab['cost'], '{:+.3f}')}",
          f"- 참고 다시 고른 ABAC 빼기 Park/e43 ABAC: 공기량 {ab['air_m3'] - old['air_m3']:+,.0f}, 비용 {ab['cost'] - old['cost']:+.3f}"]
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('grid')
    g.add_argument('--part', type=int, required=True)
    g.add_argument('--nparts', type=int, default=4)
    v = sub.add_parser('val')
    for q in (g, v):
        for k in ('data', 'surrogate', 'out'):
            q.add_argument(f'--{k}', required=True)
    k = sub.add_parser('pick')
    for x in ('out', 'e43', 'md'):
        k.add_argument(f'--{x}', required=True)
    r = sub.add_parser('report')
    for x in ('out', 'e35', 'e43', 'md'):
        r.add_argument(f'--{x}', required=True)
    a = p.parse_args()
    {'grid': grid, 'pick': pick, 'val': val, 'report': report}[a.cmd](a)


if __name__ == '__main__':
    main()

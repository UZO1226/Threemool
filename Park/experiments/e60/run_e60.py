"""Park/e60 학습 구간에서 고른 차단 위 예측 제어와의 같은 공기량 비교 (Park 작성). 규칙은 README.md 를 따른다.
대표와 같은 차단을 씌운 초과량 예측 제어의 손잡이 (NH4 여유, 공기 가중치) 를 학습 구간에서 대표의 공기량에 맞춰 고르고 (Park/e43 규칙),
검증 구간에서 한 번 돌려 대표와 비교한다.

    python run_e60.py grid --part <i> --nparts <n> --data .. --surrogate .. --out <결과>/e60
    python run_e60.py pick --out <결과>/e60 --e43 <결과>/e43 --md select.md
    python run_e60.py val --data .. --surrogate .. --out <결과>/e60
    python run_e60.py report --out <결과>/e60 --e35 <결과>/e35 --e43 <결과>/e43 --e54 <결과>/e54 --md results.md
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
from darrow_env import BatchEnv, Dynamics, evaluate, load_days  # noqa: E402
from min_brake import MinAirBrake  # noqa: E402
from mpc_darrow import ShortMPC  # noqa: E402
from run_e38 import BRAKE, COMBOS, ci, key  # noqa: E402

MARGINS = (0.8, 1.2, 1.6)
LAMS = (0.7, 1.0, 1.5, 2.0)
GRID = list(itertools.product(MARGINS, LAMS))
TOL, VAL_TOL = 0.02, 0.03   # Park/e43 과 같음


def controller(env, dyn, m, lam):
    return ShortMPC(env, dyn, margin=m, lam=lam, objective='excess'), MinAirBrake(dyn, 3.5, 1, 'others', **BRAKE)


def setup(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    return splits, BatchEnv(day_data, dyn), dyn


def gather(d, pat):
    res = {}
    for p in sorted(Path(d).glob(pat)):
        res.update(json.loads(p.read_text()))
    return res


def grid(a):
    splits, env, dyn = setup(a)
    out = {}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for m, lam in GRID[a.part::a.nparts]:
        act, br = controller(env, dyn, m, lam)
        out[f'{m}|{lam}'] = evaluate(env, splits['train'][-40:], act, shield=br)
        print(m, lam, out[f'{m}|{lam}'], flush=True)
        (Path(a.out) / f'grid_part{a.part}.json').write_text(json.dumps(out, indent=2))


def pick(a):
    target = json.loads((Path(a.e43) / 'pick.json').read_text())['target_train_air']
    res = gather(a.out, 'grid_part*.json')
    rows = [(float(k.split('|')[0]), float(k.split('|')[1]), v) for k, v in res.items()]
    inside = [r for r in rows if abs(r[2]['air_m3'] / target - 1) <= TOL]
    best = min(inside, key=lambda r: (r[2]['cost'], r[2]['air_m3'])) if inside else min(rows, key=lambda r: abs(r[2]['air_m3'] - target))
    sel = {'target_train_air': target, 'margin': best[0], 'lam': best[1], 'within_tol': bool(inside)}
    (Path(a.out) / 'pick.json').write_text(json.dumps(sel, indent=2))
    L = [f'학습 구간 마지막 40일 x 구성원 5개, 대표와 같은 차단을 씌움. 대표 학습 구간 공기량 {target:,.0f} (Park/e43) 의 ±{100 * TOL:.0f}% 안에서 하루 비용이 가장 낮은 설정을 고른다.', '',
         '| NH4 여유 | 공기 가중치 | 공기량 | 목표 대비 | 하루 비용 | NH4 위반 % | DO 위반 % | 차단 개입 % |', '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for m, lam, v in sorted(rows):
        L.append(f"| {m} | {lam} | {v['air_m3']:,.0f} | {100 * (v['air_m3'] / target - 1):+.1f}% | {v['cost']:.3f} | {v['nh4_viol_pct']:.2f} | {v['do_viol_pct']:.2f} | {v['intervention_pct']:.1f} |")
    L += ['', f"- 고른 설정: NH4 여유 {sel['margin']}, 공기 가중치 {sel['lam']}. 허용폭 안: {'예' if inside else '아니오 (가장 가까운 공기량)'}"]
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def val(a):
    sel = json.loads((Path(a.out) / 'pick.json').read_text())
    splits, env, dyn = setup(a)
    act, br = controller(env, dyn, sel['margin'], sel['lam'])
    r = evaluate(env, splits['val'], act, shield=br)
    print(r, flush=True)
    (Path(a.out) / 'val.json').write_text(json.dumps({'mpc_brake': r}, indent=2))


def report(a):
    sel = json.loads((Path(a.out) / 'pick.json').read_text())
    mpc = json.loads((Path(a.out) / 'val.json').read_text())['mpc_brake']
    e35 = gather(a.e35, 'e35_part*.json')
    v43 = gather(a.e43, 'val_part*.json')
    p43 = json.loads((Path(a.e43) / 'pick.json').read_text())
    brake = v43[f"brake|{p43['delta']}"]
    ab54 = json.loads((Path(a.e54) / 'val.json').read_text())['abac']
    ra = np.array([e35[f'orig_{key(c)}']['air_m3'] for c in COMBOS])
    rc = np.array([e35[f'orig_{key(c)}']['cost'] for c in COMBOS])
    gap = mpc['air_m3'] / ra.mean() - 1
    rep = {k: np.mean([e35[f'orig_{key(c)}'][k] for c in COMBOS]) for k in ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct', 'days_within_limit_pct')}
    row = lambda n, m: f"| {n} | {m['air_m3']:,.0f} | {100 * (m['air_m3'] / ra.mean() - 1):+.1f}% | {m['cost']:.3f} | {m['nh4_viol_pct']:.2f} | {m['do_viol_pct']:.2f} | {m['days_within_limit_pct']:.1f} |"  # noqa: E731
    L = ['검증 40일 x 구성원 5개 = 200 에피소드. 차단 위 예측 제어의 설정은 학습 구간에서 대표의 공기량에 맞춰 골랐다 ([select.md](select.md)).', '',
         '| 방법 | 공기량 Nm3/일 | 대표 대비 | 하루 비용 | NH4 위반 % | DO 위반 % | 한도 12 만족 % |', '| --- | --- | --- | --- | --- | --- | --- |',
         row('대표 (FCSRL 4 x 16 + 대표 차단, 묶음 5개 평균, Park/e35)', rep),
         row(f"차단 위 예측 제어 (NH4 여유 {sel['margin']}, 공기 가중치 {sel['lam']}), 학습 정책 없음", mpc),
         row('참고: 차단만 (Park/e43)', brake), row('참고: 다시 고른 ABAC (Park/e54)', ab54), '', '## 판정', '',
         '| 묶음 | 대표 공기량 | 대표 비용 | 대표가 낮음 |', '| --- | --- | --- | --- |']
    ok = rc < mpc['cost']
    for i, c in enumerate(COMBOS):
        L.append(f"| {key(c)} | {ra[i]:,.0f} | {rc[i]:.3f} | {'예' if ok[i] else '아니오'} |")
    fit = abs(gap) <= VAL_TOL
    v1 = ('지지' if ok.sum() >= 4 and rc.mean() < mpc['cost'] else '지지 안 됨') if fit else f'판정 안 함 (맞춤 실패, 공기량 차이 {100 * gap:+.1f}%)'
    v2 = ('지지' if (~ok).sum() >= 4 and rc.mean() > mpc['cost'] else '지지 안 됨') if fit else '판정 안 함'
    L += ['', f"- H1: 공기량 차이 {100 * gap:+.1f}%, 대표가 낮은 묶음 {ok.sum()}/5, 평균 {rc.mean():.3f} 대 {mpc['cost']:.3f}: {v1}",
          f"- H2 (반대 방향): 차단 위 예측 제어가 낮은 묶음 {(~ok).sum()}/5: {v2}",
          f"- 참고 비용 차이 (대표 빼기 차단 위 예측 제어) {ci(rc - mpc['cost'], '{:+.3f}')}",
          f"- 참고 차단 개입률: 차단 위 예측 제어 {mpc['intervention_pct']:.1f}%"]
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
    for x in ('out', 'e35', 'e43', 'e54', 'md'):
        r.add_argument(f'--{x}', required=True)
    a = p.parse_args()
    {'grid': grid, 'pick': pick, 'val': val, 'report': report}[a.cmd](a)


if __name__ == '__main__':
    main()

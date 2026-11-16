"""Park/e32 회복 계획 재현 평가 (Park 작성). 규칙은 README.md 를 따른다.

    python run_e32.py run --part <0~3> --data <darrow_15min.pkl> --surrogate <darrow_surrogate_gbm.pkl> --runs <결과> --out <결과>/e32
    python run_e32.py report --out <결과>/e32 --md results.md
"""
import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
from agent import FCSRL, ObsNormalizer  # noqa: E402
from darrow_env import BatchEnv, Dynamics, evaluate, load_days  # noqa: E402
from min_brake import E15_MARGINS, EnsemblePolicy, MinAirBrake  # noqa: E402
from train_darrow import agent_act, darrow_config  # noqa: E402

P1 = [(u, m + 0.1) for u, m in E15_MARGINS]
BRAKE = dict(trim=True, margins=P1, do_limit=1.8, recover=(3, 3.5), conflict=0.1)   # e19 H1 차단
PLAN = (4, 1.8, 500.0)   # Park/e28 H2 회복 계획
COMBOS = list(itertools.combinations(range(10, 15), 4)) + list(itertools.combinations(range(15, 20), 4))
CONDS = {'base': None, 'plan': PLAN}
name = lambda c: '_'.join(map(str, c))  # noqa: E731


def load(path):
    ag = FCSRL(11, 1, darrow_config(), ObsNormalizer(np.zeros((2, 11))))
    ag.load(path)
    return ag


def run(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    out = {}
    for c in COMBOS[a.part::4]:
        ags = [load(Path(a.runs) / 'dw_rep16_runs' / f'A_s{s}' / 'model_final') for s in c]
        pol = EnsemblePolicy([lambda obs, g=g: agent_act(g, obs) for g in ags])
        for k, plan in CONDS.items():
            br = MinAirBrake(dyn, 3.5, 1, 'others', plan=plan, **BRAKE)
            r = evaluate(env, splits['val'], pol, shield=br)
            out[f'{k}_{name(c)}'] = r
            print(k, c, r, flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f'e32_part{a.part}.json').write_text(json.dumps(out, indent=2))


def report(a):
    res = {}
    for p in sorted(Path(a.out).glob('e32_part*.json')):
        res.update(json.loads(p.read_text()))
    base = {c: res[f'base_{name(c)}'] for c in COMBOS}
    plan = {c: res[f'plan_{name(c)}'] for c in COMBOS}
    keys = ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct', 'days_within_limit_pct', 'intervention_pct')
    L = ['검증 40일 x 구성원 5개 = 200 에피소드. 새 seed 10 ~ 19 의 4 x 16 앙상블 조합 10개 (seed 10 ~ 14, 15 ~ 19).', '',
         '| 조건 (조합 10개 평균) | 공기량 Nm3/일 | 하루 비용 | NH4 위반 % | DO 위반 % | 한도 12 만족 % | 개입 % |',
         '| --- | --- | --- | --- | --- | --- | --- |']
    for label, d in (('기준 e19 차단', base), ('H1 e19 차단 + 4 스텝 회복 계획', plan)):
        m = {k: np.mean([d[c][k] for c in COMBOS]) for k in keys}
        L.append(f"| {label} | {m['air_m3']:.0f} | {m['cost']:.3f} | {m['nh4_viol_pct']:.2f} | {m['do_viol_pct']:.2f} | {m['days_within_limit_pct']:.1f} | {m['intervention_pct']:.1f} |")
    L += ['', '| 조합 | 기준 공기량 | 기준 비용 | 공기량 | 비용 | 둘 다 낮음 |', '| --- | --- | --- | --- | --- | --- |']
    ok = []
    for c in COMBOS:
        b, n = base[c], plan[c]
        ok.append(n['air_m3'] < b['air_m3'] and n['cost'] < b['cost'])
        L.append(f"| {name(c)} | {b['air_m3']:.0f} | {b['cost']:.3f} | {n['air_m3']:.0f} | {n['cost']:.3f} | {'예' if ok[-1] else '아니오'} |")
    ma, mc = np.mean([plan[c]['air_m3'] for c in COMBOS]), np.mean([plan[c]['cost'] for c in COMBOS])
    ba, bc = np.mean([base[c]['air_m3'] for c in COMBOS]), np.mean([base[c]['cost'] for c in COMBOS])
    sup = sum(ok) >= 6 and ma < ba and mc < bc
    L += ['', f"- H1 판정: 둘 다 낮은 조합 {sum(ok)}/10, 공기량 {ma:.0f} 대 {ba:.0f}, 비용 {mc:.3f} 대 {bc:.3f}: {'지지' if sup else '지지 안 됨'}"]
    for k, f in (('air_m3', '{:+,.0f}'), ('cost', '{:+.3f}')):
        diff = np.array([plan[c][k] - base[c][k] for c in COMBOS])
        hw = stats.t.ppf(0.975, 9) * diff.std(ddof=1) / np.sqrt(10)
        L.append(f"- 참고 {k} 짝 차이 평균 {f.format(diff.mean())} (95% 신뢰구간 {f.format(diff.mean() - hw)} ~ {f.format(diff.mean() + hw)}, 조합끼리 독립 아님)")
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('run')
    o.add_argument('--part', type=int, required=True)
    for k in ('data', 'surrogate', 'runs', 'out'):
        o.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run(a) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()

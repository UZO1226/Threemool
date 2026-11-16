"""Park/e33 정책 공기량 조절 층 검증 평가 (Park 작성). 규칙은 README.md 를 따른다.

    python run_e33.py run --part <i> --nparts <n> --data <darrow_15min.pkl> --surrogate <darrow_surrogate_gbm.pkl> --runs <결과> --out <결과>/e33
    python run_e33.py report --out <결과>/e33 --e28 <결과>/e28 --e32 <결과>/e32 --md results.md
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
from min_brake import E15_MARGINS, EnsemblePolicy, MinAirBrake, ScaledPolicy  # noqa: E402
from train_darrow import agent_act, darrow_config  # noqa: E402

P1 = [(u, m + 0.1) for u, m in E15_MARGINS]
BRAKE = dict(trim=True, margins=P1, do_limit=1.8, recover=(3, 3.5), conflict=0.1, plan=(4, 1.8, 500.0))   # 새 대표 차단
CONDS = {'H1': (0.7, 2.5), 'H2': (0.85, 3.0)}   # (factor, below): S25_70, S3_85
COMBOS = [c for lo in (0, 5, 10, 15) for c in itertools.combinations(range(lo, lo + 5), 4)]
key = lambda c: '_'.join(map(str, c))  # noqa: E731


def member_dir(runs, s):
    sub = 'dw_u16_runs' if s < 5 else ('dw_new16_runs' if s < 10 else 'dw_rep16_runs')
    return Path(runs) / sub / f'A_s{s}' / 'model_final'


def load(path):
    ag = FCSRL(11, 1, darrow_config(), ObsNormalizer(np.zeros((2, 11))))
    ag.load(path)
    return ag


def run(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    out = {}
    for c in COMBOS[a.part::a.nparts]:
        ens = EnsemblePolicy([lambda obs, g=load(member_dir(a.runs, s)): agent_act(g, obs) for s in c])
        for h, (factor, below) in CONDS.items():
            r = evaluate(env, splits['val'], ScaledPolicy(ens, factor, below), shield=MinAirBrake(dyn, 3.5, 1, 'others', **BRAKE))
            out[f'{h}_{key(c)}'] = r
            print(h, c, r, flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f'e33_part{a.part}.json').write_text(json.dumps(out, indent=2))


def base_values(a):
    """같은 조합의 새 대표 값: seed 0 ~ 9 는 Park/e28 H2, seed 10 ~ 19 는 Park/e32 plan."""
    e28, e32 = {}, {}
    for p in Path(a.e28).glob('e28_part*.json'):
        e28.update(json.loads(p.read_text()))
    for p in Path(a.e32).glob('e32_part*.json'):
        e32.update(json.loads(p.read_text()))
    return {c: (e28['H2_' + ''.join(map(str, c))] if c[0] < 10 else e32['plan_' + key(c)]) for c in COMBOS}


def report(a):
    res = {}
    for p in sorted(Path(a.out).glob('e33_part*.json')):
        res.update(json.loads(p.read_text()))
    base = base_values(a)
    keys = ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct', 'days_within_limit_pct', 'intervention_pct')
    conds = {'기준: 새 대표 (e19 차단 + 회복 계획)': base,
             'H1 S25_70 (NH4 < 2.5 에서 0.7 배)': {c: res[f'H1_{key(c)}'] for c in COMBOS},
             'H2 S3_85 (NH4 < 3.0 에서 0.85 배)': {c: res[f'H2_{key(c)}'] for c in COMBOS}}
    L = ['검증 40일 x 구성원 5개 = 200 에피소드. 4 x 16 앙상블 조합 20개 (seed 0 ~ 4, 5 ~ 9, 10 ~ 14, 15 ~ 19). 기준은 같은 조합의 새 대표 (Park/e28 H2, e32 결과).', '',
         '| 조건 (조합 20개 평균) | 공기량 Nm3/일 | 하루 비용 | NH4 위반 % | DO 위반 % | 한도 12 만족 % | 개입 % |',
         '| --- | --- | --- | --- | --- | --- | --- |']
    for label, d in conds.items():
        m = {k: np.mean([d[c][k] for c in COMBOS]) for k in keys}
        L.append(f"| {label} | {m['air_m3']:.0f} | {m['cost']:.3f} | {m['nh4_viol_pct']:.2f} | {m['do_viol_pct']:.2f} | {m['days_within_limit_pct']:.1f} | {m['intervention_pct']:.1f} |")
    for label in list(conds)[1:]:
        d = conds[label]
        L += ['', f'## {label} 조합별', '', '| 조합 | 기준 공기량 | 기준 비용 | 공기량 | 비용 | 둘 다 낮음 |', '| --- | --- | --- | --- | --- | --- |']
        ok = []
        for c in COMBOS:
            b, n = base[c], d[c]
            ok.append(n['air_m3'] < b['air_m3'] and n['cost'] < b['cost'])
            L.append(f"| {key(c)} | {b['air_m3']:.0f} | {b['cost']:.3f} | {n['air_m3']:.0f} | {n['cost']:.3f} | {'예' if ok[-1] else '아니오'} |")
        ma, mc = np.mean([d[c]['air_m3'] for c in COMBOS]), np.mean([d[c]['cost'] for c in COMBOS])
        ba, bc = np.mean([base[c]['air_m3'] for c in COMBOS]), np.mean([base[c]['cost'] for c in COMBOS])
        sup = sum(ok) >= 12 and ma < ba and mc < bc
        L += ['', f"- 판정: 둘 다 낮은 조합 {sum(ok)}/20, 공기량 {ma:.0f} 대 {ba:.0f}, 비용 {mc:.3f} 대 {bc:.3f}: {'지지' if sup else '지지 안 됨'}"]
        for k, f in (('air_m3', '{:+,.0f}'), ('cost', '{:+.3f}')):
            diff = np.array([d[c][k] - base[c][k] for c in COMBOS])
            hw = stats.t.ppf(0.975, 19) * diff.std(ddof=1) / np.sqrt(20)
            L.append(f"- 참고 {k} 짝 차이 평균 {f.format(diff.mean())} (95% 신뢰구간 {f.format(diff.mean() - hw)} ~ {f.format(diff.mean() + hw)}, 조합끼리 독립 아님)")
        r0 = d[(0, 1, 2, 3)]
        L.append(f"- seed 0 ~ 3 조합: 공기량 {r0['air_m3']:.0f}, 비용 {r0['cost']:.3f}, NH4 {r0['nh4_viol_pct']:.2f}, DO {r0['do_viol_pct']:.2f}, 한도 만족 {r0['days_within_limit_pct']:.1f}")
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('run')
    o.add_argument('--part', type=int, required=True)
    o.add_argument('--nparts', type=int, default=4)
    for k in ('data', 'surrogate', 'runs', 'out'):
        o.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'e28', 'e32', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run(a) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()

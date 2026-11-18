"""Park/e36 정책 기여 분리 검증 평가 (Park 작성). 규칙은 README.md 를 따른다.
모든 조건에 대표 차단 (e19 차단 + e28 원판 회복 계획) 을 똑같이 붙이고 정책만 바꾼다.

    python run_e36.py run --part <i> --nparts <n> --data <darrow_15min.pkl> --surrogate <darrow_surrogate_gbm.pkl> --runs <결과> --out <결과>/e36
    python run_e36.py report --out <결과>/e36 --md results.md

--runs 아래에 dw_runs (Park/e08 기본 PPO-Lag, FCSRL), dw_u16_runs (Park/e22 대표 구성원) 가 있어야 한다.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
from agent import METHODS, ObsNormalizer  # noqa: E402
from darrow_env import BatchEnv, Dynamics, air_to_action, evaluate, load_days  # noqa: E402
from min_brake import E15_MARGINS, EnsemblePolicy, MinAirBrake  # noqa: E402
from train_darrow import agent_act, darrow_config  # noqa: E402

P1 = [(u, m + 0.1) for u, m in E15_MARGINS]
BRAKE = dict(trim=True, margins=P1, do_limit=1.8, recover=(3, 3.5), conflict=0.1, plan=(4, 1.8, 500.0))   # 대표 차단
SEEDS = range(5)
# 조건 이름 -> (종류, 인자)
CONDS = {'zero': ('const', None), 'rec': ('const', 'rec')}
CONDS.update({f'ppo_s{s}': ('agents', ('dw_runs', 'ppo', [s])) for s in SEEDS})
CONDS.update({f'fcsrl_s{s}': ('agents', ('dw_runs', 'fcsrl', [s])) for s in SEEDS})
CONDS.update({'ppo_ens': ('agents', ('dw_runs', 'ppo', [0, 1, 2, 3])), 'fcsrl_ens': ('agents', ('dw_runs', 'fcsrl', [0, 1, 2, 3]))})


def load(runs, sub, method, s):
    ag = METHODS[method](11, 1, darrow_config(), ObsNormalizer(np.zeros((2, 11))))
    ag.load(Path(runs) / sub / f'{method}_s{s}' / 'model_final')
    return ag


def rec_air(splits, day_data):
    """학습 구간 기록 총 공기량 평균 (Nm3/h)."""
    return float(np.concatenate([day_data[d]['air'][:-1] for d in splits['train']]).mean())


def policy(name, a, splits, day_data):
    kind, arg = CONDS[name]
    if kind == 'const':
        air = 0.0 if arg is None else rec_air(splits, day_data)
        return lambda obs: air_to_action(np.full(len(obs), air))
    sub, method, seeds = arg
    return EnsemblePolicy([lambda obs, g=load(a.runs, sub, method, s): agent_act(g, obs) for s in seeds])


def run(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    out = {}
    for name in list(CONDS)[a.part::a.nparts]:
        br = MinAirBrake(dyn, 3.5, 1, 'others', **BRAKE)
        r = evaluate(env, splits['val'], policy(name, a, splits, day_data), shield=br)
        r['plan_steps'] = br.plan_steps
        out[name] = r
        print(name, r, flush=True)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / f'e36_part{a.part}.json').write_text(json.dumps(out, indent=2))


def report(a):
    res = {}
    for p in sorted(Path(a.out).glob('e36_part*.json')):
        res.update(json.loads(p.read_text()))
    keys = ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct', 'days_within_limit_pct', 'intervention_pct')
    row = lambda label, r: (f"| {label} | {r['air_m3']:.0f} | {r['cost']:.3f} | {r['nh4_viol_pct']:.2f} | {r['do_viol_pct']:.2f} "  # noqa: E731
                            f"| {r['days_within_limit_pct']:.1f} | {r['intervention_pct']:.1f} |")
    mean = lambda names: {k: np.mean([res[n][k] for n in names]) for k in keys}  # noqa: E731
    L = ['검증 40일 x 구성원 5개 = 200 에피소드. 모든 조건에 대표 차단 (e19 차단 + e28 원판 회복 계획) 을 붙였다.', '',
         '| 정책 (+ 대표 차단) | 공기량 Nm3/일 | 하루 비용 | NH4 위반 % | DO 위반 % | 한도 12 만족 % | 개입 % |',
         '| --- | --- | --- | --- | --- | --- | --- |',
         row('정책 없음 (공기 0 제안, 차단만)', res['zero']),
         row('고정 규칙 (학습 구간 기록 평균 공기량)', res['rec']),
         row('PPO-Lag 단일, seed 0 ~ 4 평균', mean([f'ppo_s{s}' for s in SEEDS])),
         row('FCSRL 단일, seed 0 ~ 4 평균', mean([f'fcsrl_s{s}' for s in SEEDS])),
         row('PPO-Lag 앙상블 seed 0 ~ 3', res['ppo_ens']),
         row('FCSRL 앙상블 seed 0 ~ 3', res['fcsrl_ens'])]
    L += ['', '## H1 FCSRL 대 PPO-Lag (단일, 같은 seed 짝)', '',
          '| seed | PPO 공기량 | PPO 비용 | FCSRL 공기량 | FCSRL 비용 | FCSRL 둘 다 낮음 |', '| --- | --- | --- | --- | --- | --- |']
    ok = []
    for s in SEEDS:
        p, f = res[f'ppo_s{s}'], res[f'fcsrl_s{s}']
        ok.append(f['air_m3'] < p['air_m3'] and f['cost'] < p['cost'])
        L.append(f"| {s} | {p['air_m3']:.0f} | {p['cost']:.3f} | {f['air_m3']:.0f} | {f['cost']:.3f} | {'예' if ok[-1] else '아니오'} |")
    mp, mf = mean([f'ppo_s{s}' for s in SEEDS]), mean([f'fcsrl_s{s}' for s in SEEDS])
    sup = sum(ok) >= 4 and mf['air_m3'] < mp['air_m3'] and mf['cost'] < mp['cost']
    L += ['', f"- H1 판정: 둘 다 낮은 seed {sum(ok)}/5, 공기량 {mf['air_m3']:.0f} 대 {mp['air_m3']:.0f}, 비용 {mf['cost']:.3f} 대 {mp['cost']:.3f}: {'지지' if sup else '지지 안 됨'}"]
    for k, fmt in (('air_m3', '{:+,.0f}'), ('cost', '{:+.3f}'), ('nh4_viol_pct', '{:+.2f}'), ('do_viol_pct', '{:+.2f}')):
        d = np.array([res[f'fcsrl_s{s}'][k] - res[f'ppo_s{s}'][k] for s in SEEDS])
        L.append(f"- 참고 {k} 짝 차이 평균 {fmt.format(d.mean())} (seed 별 {', '.join(fmt.format(x) for x in d)})")
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
    for k in ('out', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run(a) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()

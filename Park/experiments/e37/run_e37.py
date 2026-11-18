"""Park/e37 같은 학습 설정의 PPO-Lag 와 FCSRL 비교 검증 평가 (Park 작성). 규칙은 README.md 를 따른다.
PPO-Lag 를 대표와 같은 설정 (학습 중 e12 차단 target 3.5, 갱신 16번) 으로 seed 0 ~ 19 학습하고, 겹치지 않는 4 x 16 앙상블
묶음 5개에 대표 차단을 붙여 평가한다. FCSRL 쪽은 Park/e35 orig 값을 쓴다.

    python run_e37.py run --part <i> --nparts <n> --data <darrow_15min.pkl> --surrogate <darrow_surrogate_gbm.pkl> --runs <결과> --out <결과>/e37
    python run_e37.py report --out <결과>/e37 --e35 <결과>/e35 --md results.md

--runs 아래 dw_ppo16_runs/P_s<seed> 에 PPO-Lag 정책이 있어야 한다.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
from agent import METHODS, ObsNormalizer  # noqa: E402
from darrow_env import BatchEnv, Dynamics, evaluate, load_days  # noqa: E402
from min_brake import E15_MARGINS, EnsemblePolicy, MinAirBrake  # noqa: E402
from train_darrow import agent_act, darrow_config  # noqa: E402

P1 = [(u, m + 0.1) for u, m in E15_MARGINS]
BRAKE = dict(trim=True, margins=P1, do_limit=1.8, recover=(3, 3.5), conflict=0.1, plan=(4, 1.8, 500.0))   # 대표 차단 (원판)
COMBOS = [tuple(range(lo, lo + 4)) for lo in (0, 4, 8, 12, 16)]   # 구성원이 겹치지 않는 묶음 5개
key = lambda c: '_'.join(map(str, c))  # noqa: E731


def load(runs, s):
    ag = METHODS['ppo'](11, 1, darrow_config(), ObsNormalizer(np.zeros((2, 11))))
    ag.load(Path(runs) / 'dw_ppo16_runs' / f'P_s{s}' / 'model_final')
    return ag


def run(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    out = {}
    for c in COMBOS[a.part::a.nparts]:
        ens = EnsemblePolicy([lambda obs, g=load(a.runs, s): agent_act(g, obs) for s in c])
        r = evaluate(env, splits['val'], ens, shield=MinAirBrake(dyn, 3.5, 1, 'others', **BRAKE))
        out[f'ppo_{key(c)}'] = r
        print('ppo', c, r, flush=True)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        (Path(a.out) / f'e37_part{a.part}.json').write_text(json.dumps(out, indent=2))


def pair(L, res, new, base, need):
    """new 와 base 조건의 묶음별 짝 비교 줄과 판정."""
    L += ['', f'## {new} 대 {base} 묶음별', '', '| 묶음 | 기준 공기량 | 기준 비용 | 공기량 | 비용 | NH4 위반 % | DO 위반 % | 둘 다 낮음 |', '| --- | --- | --- | --- | --- | --- | --- | --- |']
    ok = []
    for c in COMBOS:
        b, n = res[f'{base}_{key(c)}'], res[f'{new}_{key(c)}']
        ok.append(n['air_m3'] < b['air_m3'] and n['cost'] < b['cost'])
        L.append(f"| {key(c)} | {b['air_m3']:.0f} | {b['cost']:.3f} | {n['air_m3']:.0f} | {n['cost']:.3f} | {n['nh4_viol_pct']:.2f} | {n['do_viol_pct']:.2f} | {'예' if ok[-1] else '아니오'} |")
    m = lambda h, k: np.mean([res[f'{h}_{key(c)}'][k] for c in COMBOS])  # noqa: E731
    sup = sum(ok) >= need and m(new, 'air_m3') < m(base, 'air_m3') and m(new, 'cost') < m(base, 'cost')
    L += ['', f"- 판정: 둘 다 낮은 묶음 {sum(ok)}/{len(COMBOS)}, 공기량 {m(new, 'air_m3'):.0f} 대 {m(base, 'air_m3'):.0f}, 비용 {m(new, 'cost'):.3f} 대 {m(base, 'cost'):.3f}: {'지지' if sup else '지지 안 됨'}"]
    n = len(COMBOS)
    for k, f in (('air_m3', '{:+,.0f}'), ('cost', '{:+.3f}'), ('nh4_viol_pct', '{:+.2f}'), ('do_viol_pct', '{:+.2f}')):
        diff = np.array([res[f'{new}_{key(c)}'][k] - res[f'{base}_{key(c)}'][k] for c in COMBOS])
        hw = stats.t.ppf(0.975, n - 1) * diff.std(ddof=1) / np.sqrt(n)
        L.append(f"- 참고 {k} 짝 차이 평균 {f.format(diff.mean())} (95% 신뢰구간 {f.format(diff.mean() - hw)} ~ {f.format(diff.mean() + hw)}, 묶음 5개는 구성원이 겹치지 않음)")


def report(a):
    res = {}
    for p in sorted(Path(a.out).glob('e37_part*.json')):
        res.update(json.loads(p.read_text()))
    for p in sorted(Path(a.e35).glob('e35_part*.json')):
        res.update({k.replace('orig_', 'fcsrl_'): v for k, v in json.loads(p.read_text()).items() if k.startswith('orig_')})
    keys = ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct', 'days_within_limit_pct', 'intervention_pct')
    label = {'fcsrl': 'FCSRL 대표 (e22 4 x 16 앙상블, Park/e35 orig 값)', 'ppo': 'PPO-Lag 같은 설정 (4 x 16 앙상블, 이 실험)'}
    L = ['검증 40일 x 구성원 5개 = 200 에피소드. 구성원이 겹치지 않는 4 x 16 앙상블 묶음 5개 (seed 0 ~ 3, 4 ~ 7, 8 ~ 11, 12 ~ 15, 16 ~ 19). 모두 대표 차단 (e19 차단 + e28 원판 회복 계획).', '',
         '| 정책 (묶음 5개 평균) | 공기량 Nm3/일 | 하루 비용 | NH4 위반 % | DO 위반 % | 한도 12 만족 % | 개입 % |',
         '| --- | --- | --- | --- | --- | --- | --- |']
    for h in ('fcsrl', 'ppo'):
        mm = {k: np.mean([res[f'{h}_{key(c)}'][k] for c in COMBOS]) for k in keys}
        L.append(f"| {label[h]} | {mm['air_m3']:.0f} | {mm['cost']:.3f} | {mm['nh4_viol_pct']:.2f} | {mm['do_viol_pct']:.2f} | {mm['days_within_limit_pct']:.1f} | {mm['intervention_pct']:.1f} |")
    pair(L, res, 'fcsrl', 'ppo', 4)
    pair(L, res, 'ppo', 'fcsrl', 4)
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('run')
    o.add_argument('--part', type=int, required=True)
    o.add_argument('--nparts', type=int, default=5)
    for k in ('data', 'surrogate', 'runs', 'out'):
        o.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'e35', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run(a) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()

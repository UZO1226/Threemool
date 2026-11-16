"""Park/e23 새 seed 로 e22 앙상블 재현 평가 (Park 작성). 규칙은 README.md 를 따른다.

    python run_e23.py run --data <darrow_15min.pkl> --surrogate <darrow_surrogate_gbm.pkl> --runs <결과> --out <결과>/e23
    python run_e23.py report --out <결과>/e23 --e19 <결과>/e19 --md results.md
"""
import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
from agent import FCSRL, ObsNormalizer  # noqa: E402
from darrow_env import BatchEnv, Dynamics, evaluate, load_days  # noqa: E402
from min_brake import E15_MARGINS, MinAirBrake  # noqa: E402
from train_darrow import agent_act, darrow_config  # noqa: E402

P1 = [(u, m + 0.1) for u, m in E15_MARGINS]
BRAKE = dict(trim=True, margins=P1, do_limit=1.8, recover=(3, 3.5), conflict=0.1)   # e19 H1 차단
NEW = [5, 6, 7, 8, 9]
ENSEMBLES = {'4x16_5678': [5, 6, 7, 8]}   # H1, H2 (README 에 미리 적음)
for _c in itertools.combinations(NEW, 4):   # 참고: 다른 seed 조합 (판정에 쓰지 않음)
    ENSEMBLES.setdefault('4x16_' + ''.join(map(str, _c)), list(_c))


def load(path):
    ag = FCSRL(11, 1, darrow_config(), ObsNormalizer(np.zeros((2, 11))))
    ag.load(path)
    return ag


def run(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    out = {'single': {}, 'ensemble': {}}
    for s in NEW:   # 기준: 새 seed 정책 하나 (e12 A 설정, 업데이트 66번) + e19 H1 차단
        ag = load(Path(a.runs) / 'dw_new66_runs' / f'A_s{s}' / 'model_final')
        out['single'][s] = evaluate(env, splits['val'], lambda obs: agent_act(ag, obs), shield=MinAirBrake(dyn, 3.5, 1, 'others', **BRAKE))
        print('single', s, out['single'][s], flush=True)
    u16 = {s: load(Path(a.runs) / 'dw_new16_runs' / f'A_s{s}' / 'model_final') for s in NEW}
    for name, seeds in ENSEMBLES.items():
        act = lambda obs, seeds=seeds: np.mean([np.asarray(agent_act(u16[s], obs), float).reshape(len(obs)) for s in seeds], axis=0)  # noqa: E731
        out['ensemble'][name] = evaluate(env, splits['val'], act, shield=MinAirBrake(dyn, 3.5, 1, 'others', **BRAKE))
        print(name, out['ensemble'][name], flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / 'e23.json').write_text(json.dumps(out, indent=2))


def report(a):
    res = json.loads((Path(a.out) / 'e23.json').read_text())
    single = {int(s): r for s, r in res['single'].items()}
    e19 = {r['seed']: r['conf_l3.5_m0.1_rec'] for r in (json.loads(p.read_text()) for p in Path(a.e19).glob('seed_*.json'))}
    mean = lambda d, k: float(np.mean([v[k] for v in d.values()]))  # noqa: E731
    sd = lambda d, k: float(np.std([v[k] for v in d.values()]))  # noqa: E731
    L = ['검증 40일 x 구성원 5개 = 200 에피소드. 앙상블은 제어기 하나라 값이 하나다.', '',
         '| 조건 | 공기량 Nm3/일 | 하루 비용 | NH4 위반 % | DO 위반 % | 한도 12 만족 % | 개입 % |',
         '| --- | --- | --- | --- | --- | --- | --- |']
    for label, d in (('새 seed 5 ~ 9 정책 하나 + e19 차단 (평균 ± 표준편차)', single), ('참고: e19 H1 seed 0 ~ 4 (평균 ± 표준편차)', e19)):
        L.append(f"| {label} | {mean(d, 'air_m3'):.0f} ± {sd(d, 'air_m3'):.0f} | {mean(d, 'cost'):.2f} ± {sd(d, 'cost'):.2f} | "
                 f"{mean(d, 'nh4_viol_pct'):.2f} | {mean(d, 'do_viol_pct'):.2f} | {mean(d, 'days_within_limit_pct'):.1f} | {mean(d, 'intervention_pct'):.1f} |")
    for n, r in res['ensemble'].items():
        L.append(f"| 앙상블 {n} | {r['air_m3']:.0f} | {r['cost']:.2f} | {r['nh4_viol_pct']:.2f} | {r['do_viol_pct']:.2f} | "
                 f"{r['days_within_limit_pct']:.1f} | {r['intervention_pct']:.1f} |")
    h = res['ensemble']['4x16_5678']
    L += ['', '## 새 seed 정책 하나 + e19 차단', '', '| seed | 공기량 | 하루 비용 | 앙상블 4x16_5678 이 둘 다 낮음 |', '| --- | --- | --- | --- |']
    ok = []
    for s in NEW:
        r = single[s]
        ok.append(h['air_m3'] < r['air_m3'] and h['cost'] < r['cost'])
        L.append(f"| {s} | {r['air_m3']:.0f} | {r['cost']:.2f} | {'예' if ok[-1] else '아니오'} |")
    sup1 = sum(ok) >= 3 and h['air_m3'] < mean(single, 'air_m3') and h['cost'] < mean(single, 'cost')
    L += ['', f"- H1 판정: 공기량 {h['air_m3']:.0f} 대 평균 {mean(single, 'air_m3'):.0f}, 비용 {h['cost']:.2f} 대 평균 {mean(single, 'cost'):.2f}, "
              f"이긴 seed {sum(ok)}/5: {'지지' if sup1 else '지지 안 됨'}"]
    sup2 = h['air_m3'] < mean(e19, 'air_m3') and h['cost'] < mean(e19, 'cost')
    L += [f"- H2 판정: 공기량 {h['air_m3']:.0f} 대 e19 H1 평균 {mean(e19, 'air_m3'):.0f}, 비용 {h['cost']:.2f} 대 {mean(e19, 'cost'):.2f}: {'지지' if sup2 else '지지 안 됨'}"]
    L += ['', '## 참고: 새 seed 4개 조합 5개 (판정에 쓰지 않음)', '']
    E = res['ensemble']
    for n, r in E.items():
        w = sum(r['air_m3'] < single[s]['air_m3'] and r['cost'] < single[s]['cost'] for s in NEW)
        L.append(f"- {n}: 공기량 {r['air_m3']:.0f}, 비용 {r['cost']:.2f}, 둘 다 이긴 새 seed 정책 {w}/5")
    L.append(f"- 조합 5개 평균 ± 표준편차: 공기량 {np.mean([r['air_m3'] for r in E.values()]):.0f} ± {np.std([r['air_m3'] for r in E.values()]):.0f}, "
             f"비용 {np.mean([r['cost'] for r in E.values()]):.2f} ± {np.std([r['cost'] for r in E.values()]):.2f}")
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('run')
    for k in ('data', 'surrogate', 'runs', 'out'):
        o.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'e19', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run(a) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()

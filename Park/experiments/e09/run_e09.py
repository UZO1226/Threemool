"""Park/e09 학습 중에도 NH4 규칙 차단을 켠 정책 평가 (Park 작성). 규칙은 README.md 를 따른다.

    python run_e09.py one --seed 0 --data <darrow_15min.pkl> --surrogate <darrow_surrogate_gbm.pkl> --runs <결과>/dw_brake_runs --out <결과>/e09
    python run_e09.py report --out <결과>/e09 --runs <결과>/dw_brake_runs --e08 <결과>/e08 --md results.md
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
sys.path.insert(0, str(HERE.parent / 'e07'))
from agent import FCSRL, ObsNormalizer  # noqa: E402
from darrow_env import BatchEnv, Dynamics, evaluate, load_days  # noqa: E402
from run_e07 import gaps, ms  # noqa: E402
from train_darrow import agent_act, brake_shield, darrow_config  # noqa: E402

THETAS = (2.0, 2.5, 3.0, 3.5, 4.0)    # Park/e08 의 NH4 규칙 곡선과 같다
H1_E08_AIR = 91279                     # e08 정책 + theta 3.0 의 seed 평균 공기량 (README 에 미리 적은 값)


def run_one(a):
    splits, day_data = load_days(a.data)
    env = BatchEnv(day_data, Dynamics(a.surrogate))
    agent = FCSRL(11, 1, darrow_config(), ObsNormalizer(np.zeros((2, 11))))
    agent.load(Path(a.runs) / f'fcsrl_b3_s{a.seed}' / 'model_final')
    act = lambda obs: agent_act(agent, obs)  # noqa: E731
    out = {'seed': a.seed, 'none': evaluate(env, splits['val'], act),
           'nh4_rule': {str(v): evaluate(env, splits['val'], act, shield=brake_shield(v)) for v in THETAS}}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f'seed_{a.seed}.json').write_text(json.dumps(out, indent=2))
    print('완료', a.seed, flush=True)


def load(d):
    return {r['seed']: r for r in (json.loads(p.read_text()) for p in Path(d).glob('seed_*.json'))}


def report(a):
    new, old = load(a.out), load(a.e08)
    seeds = sorted(new)
    rn, ro = [new[s] for s in seeds], [old[s] for s in sorted(old)]
    keys = ('air_m3', 'cost')
    ref = {'none': {k: float(np.mean([r['none'][k] for r in ro])) for k in keys}}
    for v in THETAS:
        ref[str(v)] = {k: float(np.mean([r['nh4_rule'][str(v)][k] for r in ro])) for k in keys}

    def row(label, ps):
        return (f"| {label} | {ms([p['air_m3'] for p in ps], 0)} | {ms([p['cost'] for p in ps])} | "
                f"{ms([p['nh4_viol_pct'] for p in ps])} | {ms([p['do_viol_pct'] for p in ps])} | "
                f"{ms([p['days_within_limit_pct'] for p in ps], 1)} | {ms([p.get('intervention_pct', 0) for p in ps], 1)} |")
    head = ['| 조건 | 공기량 Nm3/일 | 하루 비용 | NH4 위반 % | DO 위반 % | 한도 만족 % | 개입 % |',
            '| --- | --- | --- | --- | --- | --- | --- |']
    L = [f'FCSRL + PPO-Lag 정책 seed {len(seeds)}개씩, 평균 ± 표준편차(ddof=0). 검증 40일 x 구성원 5개 = 200 에피소드.', '',
         '## 학습 마지막 상태 (학습 차단 정책)', '',
         '| seed | 마지막 갱신 라그랑주 승수 | 마지막 갱신 학습 하루 비용 | 마지막 갱신 개입 % |', '| --- | --- | --- | --- |']
    for s in seeds:
        t = json.loads((Path(a.runs) / f'fcsrl_b3_s{s}' / 'training.json').read_text())[-1]
        L.append(f"| {s} | {t['lambda']:.3f} | {t['daily_cost_mean']:.2f} | {t['intervention_percent']:.1f} |")
    L += ['', '## 학습 차단 정책', ''] + head + [row('차단 없음', [r['none'] for r in rn])]
    L += [row(f'NH4 규칙 theta {v}', [r['nh4_rule'][str(v)] for r in rn]) for v in THETAS]
    L += ['', '## e08 정책 (학습 중 차단 없음, Park/e08 결과 파일)', ''] + head + [row('차단 없음', [r['none'] for r in ro])]
    L += [row(f'NH4 규칙 theta {v}', [r['nh4_rule'][str(v)] for r in ro]) for v in THETAS]
    L += ['', '## 판정용 seed 별 값', '',
          '| seed | H1 theta 3.0 공기량 빼기 91,279 | H2 같은 공기량 차이 (점) | H3 차단 없음 비용 빼기 e08 평균 |',
          '| --- | --- | --- | --- |']
    table = {}
    for s in seeds:
        r = new[s]
        table[('H1', s)] = r['nh4_rule']['3.0']['air_m3'] - H1_E08_AIR
        g = gaps(r['nh4_rule'], ref)
        table[('H2', s)] = float(np.mean(g)) if g else None
        table[('H3', s)] = r['none']['cost'] - ref['none']['cost']
        h2 = '없음 (0)' if not g else f'{np.mean(g):.2f} ({len(g)})'
        L.append(f"| {s} | {table[('H1', s)]:.0f} | {h2} | {table[('H3', s)]:.2f} |")
    L += ['', f'## 사전 등록 판정 (seed {len(seeds)}개 중 3개 이상 성립, 평균 성립)', '']
    for hid, label, sign in (('H1', '공기량이 e08 + theta 3.0 보다 낮다', -1), ('H2', '같은 공기량에서 비용이 낮다', -1),
                             ('H3', '차단 없이 쓰면 e08 정책보다 비용이 높다', 1)):
        vals = [table[(hid, s)] for s in seeds]
        known = [v for v in vals if v is not None]
        hit = sum(v is not None and sign * v > 0 for v in vals)
        ok = hit >= 3 and bool(known) and sign * np.mean(known) > 0
        mean_txt = f'{np.mean(known):.2f}' if known else '없음'
        L.append(f'- {hid} ({label}): 평균 {mean_txt}, 성립 seed {hit}/{len(seeds)}: {"지지" if ok else "지지 안 됨"}')
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('one')
    o.add_argument('--seed', type=int, required=True)
    for k in ('data', 'surrogate', 'runs', 'out'):
        o.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'runs', 'e08', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run_one(a) if a.cmd == 'one' else report(a)


if __name__ == '__main__':
    main()

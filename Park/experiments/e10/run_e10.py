"""Park/e10 최소 개입 차단 평가 (Park 작성). 규칙은 README.md 를 따른다.

    python run_e10.py one --seed 0 --data <darrow_15min.pkl> --surrogate <darrow_surrogate_gbm.pkl> --runs <결과>/dw_runs --out <결과>/e10
    python run_e10.py report --out <결과>/e10 --e08 <결과>/e08 --md results.md
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
from min_brake import MinAirBrake  # noqa: E402
from run_e07 import gaps, ms  # noqa: E402
from train_darrow import agent_act, darrow_config  # noqa: E402

TARGETS = (2.0, 2.5, 3.0, 3.5, 4.0)
THETAS = (2.0, 2.5, 3.0, 3.5, 4.0)
KINDS = {'min1': (1, 'others'), 'min4': (4, 'others'), 'min1_same': (1, 'same')}
LABELS = {'min1': '최소 개입 1스텝 target', 'min4': '최소 개입 4스텝 target', 'min1_same': '정답 모델 1스텝 target (참고)'}


def run_one(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    agent = FCSRL(11, 1, darrow_config(), ObsNormalizer(np.zeros((2, 11))))
    agent.load(Path(a.runs) / f'fcsrl_s{a.seed}' / 'model_final')
    act = lambda obs: agent_act(agent, obs)  # noqa: E731
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'seed_{a.seed}.json'
    out = json.loads(path.read_text()) if path.exists() else {'seed': a.seed}
    if 'none' not in out:
        out['none'] = evaluate(env, splits['val'], act)
    for kind, (h, members) in KINDS.items():
        out.setdefault(kind, {})
        for target in TARGETS:
            if str(target) in out[kind]:
                continue
            out[kind][str(target)] = evaluate(env, splits['val'], act, shield=MinAirBrake(dyn, target, h, members))
            path.write_text(json.dumps(out, indent=2))   # 중간 저장 (긴 평가가 끊겨도 이어서 돈다)
            print(a.seed, kind, target, flush=True)
    path.write_text(json.dumps(out, indent=2))
    print('완료', a.seed, flush=True)


def load(d):
    return {r['seed']: r for r in (json.loads(p.read_text()) for p in Path(d).glob('seed_*.json'))}


def air_at_cost(curve, cost):
    """곡선을 비용 순으로 이어 주어진 비용을 내는 공기량. 범위 밖이면 None."""
    pts = sorted(curve.values(), key=lambda p: p['cost'])
    xs, ys = [p['cost'] for p in pts], [p['air_m3'] for p in pts]
    return float(np.interp(cost, xs, ys)) if xs[0] <= cost <= xs[-1] else None


def report(a):
    new, old = load(a.out), load(a.e08)
    seeds = sorted(new)
    rn = [new[s] for s in seeds]
    head = ['| 조건 | 공기량 Nm3/일 | 하루 비용 | NH4 위반 % | DO 위반 % | 한도 만족 % | 개입 % |',
            '| --- | --- | --- | --- | --- | --- | --- |']

    def row(label, ps):
        return (f"| {label} | {ms([p['air_m3'] for p in ps], 0)} | {ms([p['cost'] for p in ps])} | "
                f"{ms([p['nh4_viol_pct'] for p in ps])} | {ms([p['do_viol_pct'] for p in ps])} | "
                f"{ms([p['days_within_limit_pct'] for p in ps], 1)} | {ms([p.get('intervention_pct', 0) for p in ps], 1)} |")
    same_none = all(abs(new[s]['none']['cost'] - old[s]['none']['cost']) < 1e-9 and
                    abs(new[s]['none']['air_m3'] - old[s]['none']['air_m3']) < 1e-6 for s in seeds)
    L = [f'e08 의 FCSRL + PPO-Lag 정책 seed {len(seeds)}개, 평균 ± 표준편차(ddof=0). 검증 40일 x 구성원 5개 = 200 에피소드.', '',
         f'차단 없음 재평가가 e08 결과와 같은가: {"같다" if same_none else "다르다"}', '', '## 차단 곡선 (정책 평균)', ''] + head
    L.append(row('차단 없음', [r['none'] for r in rn]))
    L += [row(f'e08 NH4 규칙 theta {v}', [old[s]['nh4_rule'][str(v)] for s in seeds]) for v in THETAS]
    for kind in KINDS:
        L += [row(f'{LABELS[kind]} {t}', [r[kind][str(t)] for r in rn]) for t in TARGETS]
    L += ['', '## 판정용 seed 별 값', '',
          '| seed | H1 1스텝 같은 공기량 차이 (점) | H2 4스텝 같은 공기량 차이 (점) | H3 theta 3.0 비용에서 공기량 감소율 % | 참고: 정답 모델 같은 공기량 차이 |',
          '| --- | --- | --- | --- | --- |']
    table = {}
    for s in seeds:
        ref = {'none': old[s]['none'], **{str(v): old[s]['nh4_rule'][str(v)] for v in THETAS}}
        cells = []
        for hid, kind in (('H1', 'min1'), ('H2', 'min4'), ('ref', 'min1_same')):
            g = gaps(new[s][kind], ref)
            table[(hid, s)] = float(np.mean(g)) if g else None
            cells.append('없음 (0)' if not g else f'{np.mean(g):.2f} ({len(g)})')
        t3 = old[s]['nh4_rule']['3.0']
        need = air_at_cost(new[s]['min1'], t3['cost'])
        table[('H3', s)] = None if need is None else 100 * (t3['air_m3'] - need) / t3['air_m3']
        h3 = '범위 밖' if need is None else f"{table[('H3', s)]:.1f}"
        L.append(f'| {s} | {cells[0]} | {cells[1]} | {h3} | {cells[2]} |')
    L += ['', f'## 사전 등록 판정 (seed {len(seeds)}개 중 3개 이상 성립, 평균 성립)', '']
    rules = (('H1', '1스텝 같은 공기량 차이 < 0', lambda v: v < 0), ('H2', '4스텝 같은 공기량 차이 < 0', lambda v: v < 0),
             ('H3', 'theta 3.0 비용에서 공기량 10% 이상 감소', lambda v: v >= 10))
    for hid, label, holds in rules:
        vals = [table[(hid, s)] for s in seeds]
        known = [v for v in vals if v is not None]
        hit = sum(v is not None and holds(v) for v in vals)
        ok = hit >= 3 and bool(known) and holds(float(np.mean(known)))
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
    for k in ('out', 'e08', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run_one(a) if a.cmd == 'one' else report(a)


if __name__ == '__main__':
    main()

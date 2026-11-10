"""Park/e11 DO 상한도 함께 보는 최소 개입 차단 평가 (Park 작성). 규칙은 README.md 를 따른다.

    python run_e11.py one --seed 0 --data <darrow_15min.pkl> --surrogate <darrow_surrogate_gbm.pkl> --runs <결과>/dw_runs --out <결과>/e11
    python run_e11.py report --out <결과>/e11 --e10 <결과>/e10 --e08 <결과>/e08 --md results.md
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
KINDS = {'do1': 1, 'do4': 4}
LABELS = {'do1': 'DO 함께 1스텝 target', 'do4': 'DO 함께 4스텝 target'}


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
    for kind, h in KINDS.items():
        out.setdefault(kind, {})
        for target in TARGETS:
            if str(target) in out[kind]:
                continue
            shield = MinAirBrake(dyn, target, h, 'others', do_cap=True)
            out[kind][str(target)] = evaluate(env, splits['val'], act, shield=shield)
            path.write_text(json.dumps(out, indent=2))   # 중간 저장 (긴 평가가 끊겨도 이어서 돈다)
            print(a.seed, kind, target, flush=True)
    print('완료', a.seed, flush=True)


def load(d):
    return {r['seed']: r for r in (json.loads(p.read_text()) for p in Path(d).glob('seed_*.json'))}


def report(a):
    new, e10, e08 = load(a.out), load(a.e10), load(a.e08)
    seeds = sorted(new)
    head = ['| 조건 | 공기량 Nm3/일 | 하루 비용 | NH4 위반 % | DO 위반 % | 한도 만족 % | 개입 % |',
            '| --- | --- | --- | --- | --- | --- | --- |']

    def row(label, ps):
        return (f"| {label} | {ms([p['air_m3'] for p in ps], 0)} | {ms([p['cost'] for p in ps])} | "
                f"{ms([p['nh4_viol_pct'] for p in ps])} | {ms([p['do_viol_pct'] for p in ps])} | "
                f"{ms([p['days_within_limit_pct'] for p in ps], 1)} | {ms([p.get('intervention_pct', 0) for p in ps], 1)} |")
    L = [f'e08 의 FCSRL + PPO-Lag 정책 seed {len(seeds)}개, 평균 ± 표준편차(ddof=0). 검증 40일 x 구성원 5개 = 200 에피소드.', '',
         '## 차단 곡선 (정책 평균)', ''] + head
    L.append(row('차단 없음 (e10 파일)', [e10[s]['none'] for s in seeds]))
    L.append(row('e08 NH4 규칙 theta 3.0', [e08[s]['nh4_rule']['3.0'] for s in seeds]))
    for kind, label in (('min1', 'e10 1스텝 target'), ('do1', LABELS['do1']), ('min4', 'e10 4스텝 target'), ('do4', LABELS['do4'])):
        src = new if kind in KINDS else e10
        L += [row(f'{label} {t}', [src[s][kind][str(t)] for s in seeds]) for t in TARGETS]
    L += ['', '## 판정용 seed 별 값', '',
          '| seed | H1 1스텝 같은 공기량 차이 (점) | H2 target 3.5 DO 위반 차이 %p | H3 4스텝 같은 공기량 차이 (점) |',
          '| --- | --- | --- | --- |']
    table = {}
    for s in seeds:
        cells = []
        for hid, kind, ref in (('H1', 'do1', 'min1'), ('H3', 'do4', 'min4')):
            g = gaps(new[s][kind], e10[s][ref])
            table[(hid, s)] = float(np.mean(g)) if g else None
            cells.append('없음 (0)' if not g else f'{np.mean(g):.2f} ({len(g)})')
        table[('H2', s)] = new[s]['do1']['3.5']['do_viol_pct'] - e10[s]['min1']['3.5']['do_viol_pct']
        L.append(f"| {s} | {cells[0]} | {table[('H2', s)]:.2f} | {cells[1]} |")
    L += ['', f'## 사전 등록 판정 (seed {len(seeds)}개 중 3개 이상 음수, 평균 음수)', '']
    for hid, label in (('H1', '1스텝 같은 공기량 차이'), ('H2', 'target 3.5 DO 위반 차이'), ('H3', '4스텝 같은 공기량 차이')):
        vals = [table[(hid, s)] for s in seeds]
        known = [v for v in vals if v is not None]
        neg = sum(v is not None and v < 0 for v in vals)
        ok = neg >= 3 and bool(known) and np.mean(known) < 0
        mean_txt = f'{np.mean(known):.2f}' if known else '없음'
        L.append(f'- {hid} ({label}): 평균 {mean_txt}, 음수 seed {neg}/{len(seeds)}: {"지지" if ok else "지지 안 됨"}')
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
    for k in ('out', 'e10', 'e08', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run_one(a) if a.cmd == 'one' else report(a)


if __name__ == '__main__':
    main()

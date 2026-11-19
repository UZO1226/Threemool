"""Park/e44 대표 차단의 결정 감사 (Park 작성). 규칙은 README.md 를 따른다.
대표 (FCSRL 4 x 16 앙상블 + 대표 차단) 를 검증 구간에서 다시 돌리며 스텝마다 차단 단계별 공기량, 실행한 공기량에서의 차단 자신의
NH4, DO 검사 통과 여부, 실제 위반을 기록한다. 스텝 기록은 Park/e45 에서도 쓴다.

    python run_e44.py run --part <i> --nparts <n> --data <darrow_15min.pkl> --surrogate <gbm.pkl> --runs <결과> --out <결과>/e44
    python run_e44.py report --out <결과>/e44 --e35 <결과>/e35 --md results.md
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
sys.path.insert(0, str(HERE.parent / 'e38'))
from audit_brake import STAGES, AuditBrake, rollout_log  # noqa: E402
from darrow_env import BatchEnv, Dynamics, load_days  # noqa: E402
from min_brake import EnsemblePolicy  # noqa: E402
from run_e38 import BRAKE, COMBOS, key, load  # noqa: E402
from train_darrow import agent_act  # noqa: E402

NAME = {'policy': '정책 그대로', 'nh4_floor': 'NH4 하한 (1 스텝 여유 검사)', 'recover': '회복 검사 (45분)', 'do_trim': 'DO 줄이기',
        'conflict': '충돌 조정 (여유 0.1 로 줄임)', 'plan': '회복 계획 (4 스텝)'}


def run(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for c in COMBOS[a.part::a.nparts]:
        br = AuditBrake(dyn, 3.5, 1, 'others', **BRAKE)
        ens = EnsemblePolicy([lambda obs, g=load(a.runs, 'fcsrl', s): agent_act(g, obs) for s in c])
        out, L = rollout_log(env, splits['val'], ens, shield=br)
        A = br.arrays()
        out['final_vs_env_air_max_abs'] = float(np.abs(A['plan'] - L['air']).max())
        out['plan_steps'], out['conflict_steps'] = br.plan_steps, br.conflict_steps
        np.savez_compressed(Path(a.out) / f'log_{key(c)}.npz', **L, **{f'b_{k}': v for k, v in A.items()})
        (Path(a.out) / f'e44_{key(c)}.json').write_text(json.dumps(out, indent=2))
        print(c, out, flush=True)


def report(a):
    e35 = {}
    for p in sorted(Path(a.e35).glob('e35_part*.json')):
        e35.update(json.loads(p.read_text()))
    keys = ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct', 'days_within_limit_pct', 'intervention_pct')
    L = ['검증 40일 x 구성원 5개 = 200 에피소드 x 95 스텝. 대표 (FCSRL 4 x 16 + 대표 차단), 구성원이 겹치지 않는 묶음 5개.', '',
         '## 재현 확인 (Park/e35 orig 와 같은가)', '', '| 묶음 | 공기량 | 하루 비용 | e35 와 최대 차이 | 실행 공기량과 차단 최종값 최대 차이 |', '| --- | --- | --- | --- | --- |']
    G = {}
    for c in COMBOS:
        k = key(c)
        r = json.loads((Path(a.out) / f'e44_{k}.json').read_text())
        diff = max(abs(r[x] - e35[f'orig_{k}'][x]) for x in keys)
        L.append(f"| {k} | {r['air_m3']:,.0f} | {r['cost']:.3f} | {diff:.2e} | {r['final_vs_env_air_max_abs']:.2e} |")
        G[k] = dict(np.load(Path(a.out) / f'log_{k}.npz'))
    # 단계별
    L += ['', '## 마지막으로 공기량을 정한 단계 (묶음 5개 평균)', '',
          '| 단계 | 스텝 비율 % | 실행 공기량이 자신의 NH4 검사 통과 % | 자신의 DO 검사 통과 % | 다음 스텝 NH4 위반 % | 다음 스텝 DO 위반 % | 전체 NH4 위반 중 % | 전체 DO 위반 중 % |',
          '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for i, s in enumerate(STAGES):
        cells = []
        for g in G.values():
            m = g['b_last'] == i
            n = max(m.sum(), 1)
            cells.append([100 * m.mean(), 100 * g['b_nh4_ok'][m].sum() / n, 100 * g['b_do_ok'][m].sum() / n,
                          100 * g['v_nh4'][m].sum() / n, 100 * g['v_do'][m].sum() / n,
                          100 * g['v_nh4'][m].sum() / max(g['v_nh4'].sum(), 1), 100 * g['v_do'][m].sum() / max(g['v_do'].sum(), 1)])
        c = np.mean(cells, axis=0)
        L.append(f'| {NAME[s]} | {c[0]:.1f} | {c[1]:.1f} | {c[2]:.1f} | {c[3]:.2f} | {c[4]:.2f} | {c[5]:.1f} | {c[6]:.1f} |')
    # 가설
    L += ['', '## 판정', '', '| 묶음 | NH4 위반 스텝 | 그중 자신의 NH4 검사 실패 % | 검사 통과 스텝의 NH4 위반 % | DO 위반 중 자신의 DO 검사 실패 % | 검사 통과 스텝의 DO 위반 % |',
          '| --- | --- | --- | --- | --- | --- |']
    h1, h2, s1, s2 = [], [], [], []
    for k, g in G.items():
        v, ok = g['v_nh4'] > 0, g['b_nh4_ok']
        vd, okd = g['v_do'] > 0, g['b_do_ok']
        f1 = 100 * (v & ~ok).sum() / max(v.sum(), 1)
        f2 = 100 * (v & ok).sum() / max(ok.sum(), 1)
        f3 = 100 * (vd & ~okd).sum() / max(vd.sum(), 1)
        f4 = 100 * (vd & okd).sum() / max(okd.sum(), 1)
        s1.append(f1); s2.append(f2)
        h1.append(f1 >= 50); h2.append(f2 <= 1.0)
        L.append(f'| {k} | {int(v.sum())} | {f1:.1f} | {f2:.2f} | {f3:.1f} | {f4:.2f} |')
    yn = lambda x: '지지' if x else '지지 안 됨'  # noqa: E731
    L += ['', f'- H1 판정: NH4 위반의 절반 이상이 자신의 검사를 통과하지 못한 행동에서 나온 묶음 {sum(h1)}/5, 평균 {np.mean(s1):.1f}%: {yn(sum(h1) >= 4 and np.mean(s1) >= 50)}',
          f'- H2 판정: 검사를 통과한 행동의 다음 스텝 NH4 위반이 1% 이하인 묶음 {sum(h2)}/5, 평균 {np.mean(s2):.2f}%: {yn(sum(h2) >= 4 and np.mean(s2) <= 1.0)}']
    # 참고: 검사 실패 행동의 이유
    L += ['', '## 참고: 자신의 NH4 검사를 통과하지 못한 실행 행동의 내역 (묶음 5개 합)', '',
          '| 내역 | 스텝 수 | 다음 스텝 NH4 위반 % |', '| --- | --- | --- |']
    fail = {s: [0, 0] for s in STAGES}
    for g in G.values():
        bad = ~g['b_nh4_ok']
        for i, s in enumerate(STAGES):
            m = bad & (g['b_last'] == i)
            fail[s][0] += int(m.sum()); fail[s][1] += int((m & (g['v_nh4'] > 0)).sum())
    for s in STAGES:
        n, v = fail[s]
        L.append(f'| 마지막 단계 {NAME[s]} | {n} | {100 * v / max(n, 1):.1f} |')
    sat = sum(int(((~g['b_nh4_ok']) & (g['b_r0'] >= 9000 - 1e-6)).sum()) for g in G.values())
    L.append(f'| (그중 NH4 하한이 최대 공기 9,000 인 스텝, 막을 수 없다고 본 스텝) | {sat} | |')
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

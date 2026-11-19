"""Park/e45 사건 단위 평가와 실패 날짜 (Park 작성). 규칙은 README.md 를 따른다.
대표의 스텝 기록은 Park/e44 를 쓰고, 기준 (Park/e43 이 고른 ABAC 와 차단만, 기본 PPO-Lag, FCSRL) 은 여기서 스텝 기록과 함께 다시 돈다.

    python run_e45.py run --part <i> --nparts <n> --data .. --surrogate .. --runs <결과> --e43 <결과>/e43 --out <결과>/e45
    python run_e45.py report --out <결과>/e45 --e44 <결과>/e44 --e43 <결과>/e43 --md results.md
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
sys.path.insert(0, str(HERE.parent / 'e38'))
sys.path.insert(0, str(HERE.parent / 'e43'))
from audit_brake import rollout_log  # noqa: E402
from baselines_darrow import ABAC  # noqa: E402
from darrow_env import NH4_MAX, STEP_H, BatchEnv, Dynamics, load_days  # noqa: E402
from run_e38 import COMBOS, ci, key  # noqa: E402
from run_e43 import brake_only, ppo_default, zero  # noqa: E402
from agent import METHODS, ObsNormalizer  # noqa: E402
from train_darrow import agent_act, darrow_config  # noqa: E402

WORST = 4   # 실패 날짜: 검증 40일 가운데 하루 비용 (구성원 평균) 이 가장 큰 4일 (10%)


def fcsrl_default(runs, s):
    ag = METHODS['fcsrl'](11, 1, darrow_config(), ObsNormalizer(np.zeros((2, 11))))
    ag.load(Path(runs) / 'dw_runs' / f'fcsrl_s{s}' / 'model_final')
    return lambda obs: agent_act(ag, obs)


def jobs():
    return [('brake',)] + [('abac',)] + [('ppo', s) for s in range(5)] + [('fcsrl', s) for s in range(5)]


def run(a):
    sel = json.loads((Path(a.e43) / 'pick.json').read_text())
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for j in jobs()[a.part::a.nparts]:
        if j[0] == 'brake':
            out, L = rollout_log(env, splits['val'], zero, shield=brake_only(dyn, sel['delta']))
        elif j[0] == 'abac':
            out, L = rollout_log(env, splits['val'], ABAC(env, *sel['abac']))
        else:
            out, L = rollout_log(env, splits['val'], (ppo_default if j[0] == 'ppo' else fcsrl_default)(a.runs, j[1]))
        name = '_'.join(map(str, j))
        np.savez_compressed(Path(a.out) / f'log_{name}.npz', **L)
        (Path(a.out) / f'e45_{name}.json').write_text(json.dumps(out, indent=2))
        print(j, out, flush=True)


def metrics(g):
    """스텝 기록 하나 (에피소드 N x 95) 의 사건 지표."""
    v, nh4, air = g['v_nh4'] > 0, g['nh4'], g['air']
    start = v & ~np.concatenate([np.zeros((len(v), 1), bool), v[:, :-1]], axis=1)
    n_ev = start.sum(axis=1)
    vd = g['v_do'] > 0
    dstart = vd & ~np.concatenate([np.zeros((len(vd), 1), bool), vd[:, :-1]], axis=1)
    return {
        'air_m3': float((air.sum(axis=1) * STEP_H).mean()),
        'cost': float((g['v_nh4'] + g['v_do']).sum(axis=1).mean()),
        'nh4_events': float(n_ev.mean()),
        'nh4_event_h': float(v.sum() / max(n_ev.sum(), 1) * STEP_H),
        'nh4_exceed': float((np.maximum(nh4 - NH4_MAX, 0) * STEP_H).sum(axis=1).mean()),
        'nh4_p95': float(np.percentile(nh4, 95)),
        'nh4_p99': float(np.percentile(nh4, 99)),
        'nh4_daymax': float(nh4.max(axis=1).mean()),
        'do_events': float(dstart.sum(axis=1).mean()),
        'air_move': float(np.abs(np.diff(air, axis=1)).sum(axis=1).mean() / 1000.0),
        'nh4_viol_days_pct': float(100 * (v.sum(axis=1) > 0).mean()),
    }


def day_cost(g):
    days = np.unique(g['day'])
    c = (g['v_nh4'] + g['v_do']).sum(axis=1)
    return {d: float(c[g['day'] == d].mean()) for d in days}


def worst(dc):
    return set(sorted(dc, key=lambda d: (-dc[d], d))[:WORST])


def jac(x, y):
    return len(x & y) / max(len(x | y), 1)


COLS = [('air_m3', '공기량 Nm3/일', '{:,.0f}'), ('cost', '하루 비용', '{:.3f}'), ('nh4_events', 'NH4 위반 사건/일', '{:.2f}'),
        ('nh4_event_h', '사건 평균 길이 h', '{:.2f}'), ('nh4_exceed', 'NH4 초과 적분 mg/L h/일', '{:.3f}'), ('nh4_p95', 'NH4 95%', '{:.2f}'),
        ('nh4_p99', 'NH4 99%', '{:.2f}'), ('nh4_daymax', '하루 최대 NH4 평균', '{:.2f}'), ('nh4_viol_days_pct', 'NH4 위반이 있는 날 %', '{:.1f}'),
        ('do_events', 'DO 위반 사건/일', '{:.2f}'), ('air_move', '공기 변화량 합 천 Nm3/h/일', '{:.1f}')]


def report(a):
    sel = json.loads((Path(a.e43) / 'pick.json').read_text())
    load_ = lambda p: dict(np.load(p))  # noqa: E731
    R = [load_(Path(a.e44) / f'log_{key(c)}.npz') for c in COMBOS]
    B, A = load_(Path(a.out) / 'log_brake.npz'), load_(Path(a.out) / 'log_abac.npz')
    P = [load_(Path(a.out) / f'log_ppo_{s}.npz') for s in range(5)]
    F = [load_(Path(a.out) / f'log_fcsrl_{s}.npz') for s in range(5)]
    mR, mB, mA = [metrics(g) for g in R], metrics(B), metrics(A)
    mP, mF = [metrics(g) for g in P], [metrics(g) for g in F]
    avg = lambda ms: {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}  # noqa: E731
    rows = [('대표 (묶음 5개 평균)', avg(mR)), (f"ABAC (e43 고름: 설정값 {sel['abac'][0]}, Kp {sel['abac'][1]:g}, Ki {sel['abac'][2]:g})", mA),
            (f"차단만 (e43 고름: 여유 + {sel['delta']:.1f})", mB), ('기본 PPO-Lag, 차단 없음 (seed 5개 평균)', avg(mP)), ('기본 FCSRL, 차단 없음 (seed 5개 평균)', avg(mF))]
    L = ['검증 40일 x 구성원 5개 = 200 에피소드. 사건 = NH4 > 4 (DO > 2) 가 이어진 스텝 묶음. 초과 적분 = 4 를 넘은 NH4 x 0.25 h 의 하루 합. '
         '공기 변화량 = 스텝 사이 총 공기량 변화의 절댓값 하루 합 (BSM1 6절의 조작량 변화에 해당).', '',
         '| 방법 | ' + ' | '.join(c[1] for c in COLS) + ' |', '| --- |' + ' --- |' * len(COLS)]
    for n, m in rows:
        L.append(f'| {n} | ' + ' | '.join(f.format(m[k]) for k, _, f in COLS) + ' |')
    # H1
    ev, ex = np.array([m['nh4_events'] for m in mR]), np.array([m['nh4_exceed'] for m in mR])
    ok1 = (ev < mA['nh4_events']) & (ex < mA['nh4_exceed'])
    L += ['', '## H1 대표 대 ABAC: NH4 사건 수와 초과 적분', '', '| 묶음 | 사건/일 | 초과 적분 | 둘 다 ABAC 보다 작음 |', '| --- | --- | --- | --- |']
    for i, c in enumerate(COMBOS):
        L.append(f'| {key(c)} | {ev[i]:.2f} | {ex[i]:.3f} | {"예" if ok1[i] else "아니오"} |')
    s1 = ok1.sum() >= 4 and ev.mean() < mA['nh4_events'] and ex.mean() < mA['nh4_exceed']
    L += [f"| ABAC | {mA['nh4_events']:.2f} | {mA['nh4_exceed']:.3f} | |", '',
          f"- H1 판정: 둘 다 작은 묶음 {ok1.sum()}/5, 평균 사건 {ev.mean():.2f} 대 {mA['nh4_events']:.2f}, 초과 적분 {ex.mean():.3f} 대 {mA['nh4_exceed']:.3f}: {'지지' if s1 else '지지 안 됨'}",
          f"- 참고 같은 비교를 차단만과: 둘 다 작은 묶음 {((ev < mB['nh4_events']) & (ex < mB['nh4_exceed'])).sum()}/5 (차단만 사건 {mB['nh4_events']:.2f}, 초과 적분 {mB['nh4_exceed']:.3f})"]
    # H2 실패 날짜
    dR = [day_cost(g) for g in R]
    dRm = {d: float(np.mean([x[d] for x in dR])) for d in dR[0]}
    wR, wA, wB = worst(dRm), worst(day_cost(A)), worst(day_cost(B))
    dP = [day_cost(g) for g in P]
    wP = worst({d: float(np.mean([x[d] for x in dP])) for d in dP[0]})
    dF = [day_cost(g) for g in F]
    wF = worst({d: float(np.mean([x[d] for x in dF])) for d in dF[0]})
    per = [jac(worst(x), wA) for x in dR]
    jRA = jac(wR, wA)
    L += ['', f'## H2 실패 날짜 (하루 비용이 가장 큰 {WORST}일) 가 겹치는가', '', '| 방법 | 실패 날짜 | 대표와 겹침 (Jaccard) |', '| --- | --- | --- |']
    for n, w in (('대표 (묶음 평균)', wR), ('ABAC', wA), ('차단만', wB), ('기본 PPO-Lag', wP), ('기본 FCSRL', wF)):
        L.append(f"| {n} | {', '.join(sorted(w))} | {jac(wR, w):.2f} |")
    L += ['', f"- H2 판정: 대표와 ABAC 의 실패 날짜 Jaccard {jRA:.2f} (묶음별 {', '.join(f'{x:.2f}' for x in per)}): {'지지' if jRA >= 0.5 else '지지 안 됨'}"]
    # 참고: 실패 날짜의 비용
    L += ['', '## 참고: 대표 실패 날짜의 하루 비용', '', '| 날짜 | 대표 | ABAC | 차단만 | 기본 PPO-Lag | 기본 FCSRL |', '| --- | --- | --- | --- | --- | --- |']
    dA, dB = day_cost(A), day_cost(B)
    for d in sorted(wR, key=lambda d: -dRm[d]):
        L.append(f"| {d} | {dRm[d]:.1f} | {dA[d]:.1f} | {dB[d]:.1f} | {np.mean([x[d] for x in dP]):.1f} | {np.mean([x[d] for x in dF]):.1f} |")
    share = sum(dRm[d] for d in wR) / max(sum(dRm.values()), 1e-9)
    L.append(f'- 대표 비용의 {share * 100:.0f}% 가 실패 날짜 {WORST}일 (10%) 에 몰려 있다')
    am = np.array([m['air_move'] for m in mR])
    L.append(f"- 참고 공기 변화량 (대표 빼기 ABAC) {ci(am - mA['air_move'], '{:+.1f}')} 천 Nm3/h/일")
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    o = sub.add_parser('run')
    o.add_argument('--part', type=int, required=True)
    o.add_argument('--nparts', type=int, default=4)
    for k in ('data', 'surrogate', 'runs', 'e43', 'out'):
        o.add_argument(f'--{k}', required=True)
    r = sub.add_parser('report')
    for k in ('out', 'e44', 'e43', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    run(a) if a.cmd == 'run' else report(a)


if __name__ == '__main__':
    main()

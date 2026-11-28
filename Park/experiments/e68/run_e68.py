"""Park/e68 30분 지연에서 대표가 무너지는 곳 나누기: 정책 대 차단 (Park 작성). 규칙은 README.md 를 따른다.
대표 묶음마다 NH4 30분 지연 (d = 2, 보정 없음) 을 정책에만 주거나 차단에만 주어 검증 구간에서 돌린다. 둘 다 지연 (Park/e46) 과 지연 없음 (Park/e35) 은 기존 값을 쓴다.

    python run_e68.py run --part <i> --nparts <n> --data .. --surrogate .. --runs <결과> --out <결과>/e68
    python run_e68.py report --out <결과>/e68 --e35 <결과>/e35 --e46 <결과>/e46 --e62 <결과>/e62 --md results.md
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
sys.path.insert(0, str(HERE.parent / 'e38'))
sys.path.insert(0, str(HERE.parent / 'e59'))
from darrow_env import COST_LIMIT, STEP_H, BatchEnv, Dynamics, all_episodes, load_days  # noqa: E402
from min_brake import EnsemblePolicy, MinAirBrake  # noqa: E402
from run_e38 import BRAKE, COMBOS, ci, key, load  # noqa: E402
from run_e59 import gather  # noqa: E402
from train_darrow import agent_act  # noqa: E402

D = 2
VARIANTS = ('policy', 'brake')   # 지연을 받는 쪽
JOBS = [(w, c) for w in VARIANTS for c in COMBOS]


def rollout(env, days, act, shield, d_pol, d_brk):
    """audit_brake.rollout_log 와 같은 지표. 정책과 차단이 보는 관측 NH4 의 지연을 따로 둔다."""
    obs = env.reset(all_episodes(days, env.dyn.n_members))
    N, T = len(env.episodes), env.T
    hist = [np.asarray(obs[:, 0]).copy()]
    air = np.zeros((N, T))
    vn = np.zeros((N, T))
    vd = np.zeros((N, T))
    used = np.zeros((N, T))
    for t in range(T):
        op, ob = obs.copy(), obs.copy()
        op[:, 0] = hist[max(t - d_pol, 0)]
        ob[:, 0] = hist[max(t - d_brk, 0)]
        a = np.asarray(act(op), dtype=float).reshape(N)
        a, u = shield(ob, a, env)
        used[:, t] = u
        obs, _, c, done, info = env.step(a)
        hist.append(np.asarray(obs[:, 0]).copy())
        air[:, t], vn[:, t], vd[:, t] = info['air'], info['v_nh4'], info['v_do']
    assert done
    cost = (vn + vd).sum(axis=1)
    return {'air_m3': float((air.sum(axis=1) * STEP_H).mean()), 'cost': float(cost.mean()),
            'nh4_viol_pct': float((100 * vn.sum(axis=1) / T).mean()), 'do_viol_pct': float((100 * vd.sum(axis=1) / T).mean()),
            'days_within_limit_pct': float(100 * np.mean(cost <= COST_LIMIT)), 'intervention_pct': 100 * float(used.mean()), 'n_episodes': N}


def run(a):
    splits, day_data = load_days(a.data)
    dyn = Dynamics(a.surrogate)
    env = BatchEnv(day_data, dyn)
    out = {}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for w, c in JOBS[a.part::a.nparts]:
        ens = EnsemblePolicy([lambda obs, g=load(a.runs, 'fcsrl', s): agent_act(g, obs) for s in c])
        r = rollout(env, splits['val'], ens, MinAirBrake(dyn, 3.5, 1, 'others', **BRAKE), D if w == 'policy' else 0, D if w == 'brake' else 0)
        out[f'{w}_{key(c)}'] = r
        print(w, c, r, flush=True)
        (Path(a.out) / f'e68_part{a.part}.json').write_text(json.dumps(out, indent=2))


def report(a):
    v = gather(a.out, 'e68_part*.json')
    e35 = gather(a.e35, 'e35_part*.json')
    e46 = gather(a.e46, 'e46_part*.json')
    e62 = gather(a.e62, 'e62_part*.json')
    ks = [key(c) for c in COMBOS]
    get = lambda f: np.array([f(k) for k in ks])  # noqa: E731
    c0 = get(lambda k: e35[f'orig_{k}']['cost'])
    cb = get(lambda k: e46[f'fcsrl_d2_{k}']['cost'])
    cp = get(lambda k: v[f'policy_{k}']['cost'])
    cr = get(lambda k: v[f'brake_{k}']['cost'])
    L = ['검증 40일 x 구성원 5개 = 200 에피소드. 지연 30분 (d = 2), 보정 없음. "정책만" 은 정책이 늦은 NH4 를 보고 차단은 지금 NH4 를 본다. "차단만" 은 그 반대.', '',
         '| 묶음 | 지연 없음 (e35) | 정책만 지연 | 차단만 지연 | 둘 다 지연 (e46) | 정책만 증가 | 차단만 증가 | 둘 다 증가 |', '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for i, k in enumerate(ks):
        L.append(f'| {k} | {c0[i]:.3f} | {cp[i]:.3f} | {cr[i]:.3f} | {cb[i]:.3f} | {cp[i] - c0[i]:+.3f} | {cr[i] - c0[i]:+.3f} | {cb[i] - c0[i]:+.3f} |')
    ap = np.mean([v[f'policy_{k}']['air_m3'] for k in ks])
    ar = np.mean([v[f'brake_{k}']['air_m3'] for k in ks])
    L.append(f'| 평균 | {c0.mean():.3f} | {cp.mean():.3f} | {cr.mean():.3f} | {cb.mean():.3f} | {(cp - c0).mean():+.3f} | {(cr - c0).mean():+.3f} | {(cb - c0).mean():+.3f} |')
    L += ['', f'공기량 평균: 정책만 지연 {ap:,.0f}, 차단만 지연 {ar:,.0f}', '', '## 판정', '']
    dp, dr = cp - c0, cr - c0
    n1 = int((dp > dr).sum())
    L.append(f"- H1 (정책이 지연에 더 민감한가): 정책만 증가가 차단만 증가보다 큰 묶음 {n1}/5, 평균 {dp.mean():+.3f} 대 {dr.mean():+.3f}: {'지지' if n1 >= 4 and dp.mean() > dr.mean() else '지지 안 됨'}")
    share = dp / (cb - c0)
    n2 = int((share >= 0.5).sum())
    L.append(f"- H2 (정책만 지연이 둘 다 지연 증가의 절반 이상인가): 묶음 {n2}/5, 평균 비율 {share.mean():.2f}: {'지지' if n2 >= 4 else '지지 안 됨'}")
    L.append(f"- 참고 증가 차이 (정책만 빼기 차단만) {ci(dp - dr, '{:+.3f}')}")
    m = e62.get('mpc_raw_d2')
    if m:
        L.append(f"- 참고 차단 위 예측 제어 (Park/e62, 둘 다 지연, 보정 없음) {m['air_m3']:,.0f} / {m['cost']:.3f}")
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
    for k in ('out', 'e35', 'e46', 'e62', 'md'):
        r.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    {'run': run, 'report': report}[a.cmd](a)


if __name__ == '__main__':
    main()

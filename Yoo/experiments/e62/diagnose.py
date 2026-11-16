"""e62 진단: (A) 차단 아래 NH4 비용의 시작 상태 몫 분해, (B) 구간 배분의 NH4 효과와 근거.

새 학습 없음. test 행은 추출하지 않는다. 결과는 --out 의 result.json 에 남기고 report.py 가 표로 만든다.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e59'))
import shield_eval as se  # noqa: E402
from darrow_env import BatchEnv, Dynamics, all_episodes, air_to_action  # noqa: E402

base = se.base
MODELS = {'common_v1_1': se.PREPARED / 'surrogate.pkl', 'do_corrected': Path('/home/jio/fcsrl_src/results/e51/main/corrected_surrogate.pkl')}
R = Path('/home/jio/fcsrl_src/results')
TARGETS = {'uniform': np.full(5, .2), 'back': np.array([0., 0., .2, .3, .5])}
WEIGHTS = (0.25, 0.5, 1.0)


def plain(x):
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    return x.item() if isinstance(x, np.generic) else x


# ---------- A ----------
def carryover(viol):
    """(N, 95) 위반 배열에서 첫 스텝부터 이어진 위반 길이."""
    first_ok = np.where(viol.all(1), viol.shape[1], np.argmin(viol, axis=1))
    return first_ok


def decompose(states, floor_cost, init):
    v = states[:, :, 0] > 4
    cost = v.sum(1)
    carry = carryover(v)
    start_hi = init > 4
    return dict(nh4_cost=float(cost.mean()), carryover=float(carry.mean()), controllable=float((cost - carry).mean()),
                floor=float(floor_cost.mean()), above_floor=float((cost - floor_cost).mean()),
                nh4_cost_start_above4=float(cost[start_hi].mean()), nh4_cost_start_below4=float(cost[~start_hi].mean()),
                share_episodes_start_above4=float(start_hi.mean()))


def part_a(splits, days, env):
    eps = all_episodes(splits['val'], 5)
    init = np.array([days[d]['state'][0, 0] for d, _ in eps])
    floor_states = rollout(env, splits['val'], lambda e: np.full(len(e.episodes), 9000.), None)
    floor_cost = (floor_states[:, :, 0] > 4).sum(1)
    sources = {'기록 운전': R / 'e59/main/runs/none/record_s0/rollout.npz', '공기 0 + park_e19': R / 'e59/main/runs/park_e19/zero_s0/rollout.npz'}
    groups = {}
    for pol in ('ppo', 'fcsrl'):
        groups[f'e50 {pol} 차단 없음'] = [R / f'e59/main/runs/none/{pol}_s{s}/rollout.npz' for s in range(5)]
    for pol in ('ppo', 'fcsrl', 'fcsrl_actor', 'severity'):
        groups[f'e59 {pol} + park_e19'] = [R / f'e59/main/runs/park_e19/{pol}_s{s}/rollout.npz' for s in range(5)]
    for pol in ('ppo', 'fcsrl'):
        groups[f'e60 {pol} + park_e19'] = [next(p for p in (R / f'e60/main/runs/{pol}/seed_{s}/val_park_e19_rollout.npz',
                                                            R / f'e60/main_rest/runs/{pol}/seed_{s}/val_park_e19_rollout.npz') if p.exists()) for s in range(5)]
        groups[f'e61 {pol} + park_e19'] = [R / f'e61/main/runs/{pol}/seed_{s}/val_park_e19_rollout.npz' for s in range(5)]
    out = {'floor_max_air': decompose(floor_states, floor_cost, init)}
    for name, path in sources.items():
        out[name] = decompose(np.load(path)['states'], floor_cost, init)
    for name, paths in groups.items():
        rows = [decompose(np.load(p)['states'], floor_cost, init) for p in paths]
        out[name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        out[name]['seeds'] = len(rows)
    return out


# ---------- B ----------
def rollout(env, dates, total_fn, alloc):
    """total_fn(env) -> 총 공기 (N,). alloc 이 None 이면 기록 배분, (w, target) 이면 기록과 목표의 결합."""
    eps = all_episodes(dates, env.dyn.n_members)
    env.reset(eps)
    if alloc is not None:
        w, target = alloc
        env.x['frac'] = (1 - w) * env.x['frac'] + w * target[None, None, :]
    states = []
    for _ in range(env.T):
        total = np.clip(total_fn(env), 0, 9000)
        env.step(air_to_action(total))
        states.append(env.state.copy())
    return np.stack(states, 1)


def zone_airs(env, dates, total_fn, alloc):
    """rollout 과 같은 배분으로 각 스텝의 구간 공기 (N, 95, 5)."""
    eps = all_episodes(dates, env.dyn.n_members)
    env.reset(eps)
    frac = env.x['frac'] if alloc is None else (1 - alloc[0]) * env.x['frac'] + alloc[0] * alloc[1][None, None, :]
    totals = np.stack([np.clip(env.x['air'][:, t], 0, 9000) if total_fn == 'record' else np.full(len(eps), 9000.) for t in range(env.T)], 1)
    return totals[:, :, None] * frac[:, :env.T, :]


def part_b(splits, days):
    train_zone = np.concatenate([days[d]['zone'] for d in splits['train']])
    p99, mx = np.percentile(train_zone, 99, axis=0), train_zone.max(0)
    tot = train_zone.sum(1)
    frac = train_zone[tot > 500] / tot[tot > 500, None]
    support = dict(zone_air_p99=p99.tolist(), zone_air_max=mx.tolist(),
                   fraction_quantiles={q: np.percentile(frac, q, axis=0).tolist() for q in (1, 5, 50, 95, 99)},
                   back_share_quantiles={q: float(np.percentile(frac[:, 3] + frac[:, 4], q)) for q in (1, 5, 50, 95, 99)})
    allocs = {'record': None}
    for name, target in TARGETS.items():
        for w in WEIGHTS:
            allocs[f'{name}_w{w}'] = (w, target)
    schedules = {'max9000': lambda e: np.full(len(e.episodes), 9000.), 'record': lambda e: e.x['air'][:, e.t]}
    res = {}
    for mname, mpath in MODELS.items():
        dyn = Dynamics(mpath)
        env = BatchEnv(days, dyn)
        for split in ('val', 'train'):
            for sname, fn in schedules.items():
                for aname, alloc in allocs.items():
                    st = rollout(env, splits[split], fn, alloc)
                    za = zone_airs(env, splits[split], 'record' if sname == 'record' else 'max', alloc)
                    n = (st[:, :, 0] > 4).sum(1)
                    d = (st[:, :, 2:].max(2) > 2).sum(1)
                    res[f'{mname}|{split}|{sname}|{aname}'] = dict(
                        nh4_cost=float(n.mean()), do_cost=float(d.mean()), cost=float((n + d).mean()),
                        air_m3=float(za.sum(2).sum(1).mean() * .25),
                        coverage_p99=float((za <= p99[None, None, :] + 1e-9).all(2).mean()),
                        coverage_max=float((za <= mx[None, None, :] + 1e-9).all(2).mean()))
    return support, res


def part_b_data(splits, days, n_boot=500, seed=0):
    """기록 자료와 공통 모델 예측에서 뒤 구간 비율 계수."""
    dyn = Dynamics(MODELS['common_v1_1'])
    X, Y, M, G = [], [], [], []
    for gi, d in enumerate(splits['train']):
        g = days[d]
        s, exo, z = g['state'], g['exo'], g['zone']
        zp = np.vstack([z[:1], z[:-2]])
        x_model = np.column_stack([s[:-1], exo[:-1], z[:-1] / 1000., zp / 1000.])
        pred = np.mean([dyn.models[m][0].predict(x_model) for m in range(dyn.n_members)], 0)
        tot, ptot = z[:-1].sum(1), zp.sum(1)
        keep = tot > 500
        share = (z[:-1, 3] + z[:-1, 4]) / np.maximum(tot, 1)
        X.append(np.column_stack([s[:-1], exo[:-1], tot / 1000., ptot / 1000., share])[keep])
        Y.append((s[1:, 0] - s[:-1, 0])[keep])
        M.append(pred[keep])
        G.append(np.full(keep.sum(), gi))
    X, Y, M, G = map(np.concatenate, (X, Y, M, G))
    Xs = (X - X.mean(0)) / X.std(0)
    A = np.column_stack([np.ones(len(Xs)), Xs])
    coef = lambda y, idx: np.linalg.lstsq(A[idx], y[idx], rcond=None)[0][-1] / X[:, -1].std()  # noqa: E731  비율 1 단위당 변화량
    all_idx = np.arange(len(Y))
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(G == g) for g in np.unique(G)]
    boots_d, boots_m = [], []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        boots_d.append(coef(Y, idx))
        boots_m.append(coef(M, idx))
    return dict(rows=int(len(Y)), days=len(groups), data_coef=float(coef(Y, all_idx)), data_ci95=np.percentile(boots_d, [2.5, 97.5]).tolist(),
                model_coef=float(coef(M, all_idx)), model_ci95=np.percentile(boots_m, [2.5, 97.5]).tolist(),
                share_mean=float(X[:, -1].mean()), share_sd=float(X[:, -1].std()))


def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    t = time.monotonic()
    splits, days = base.load_selected(se.PREPARED / 'data.pkl')
    env = BatchEnv(days, Dynamics(MODELS['common_v1_1']))
    out = dict(A=part_a(splits, days, env))
    out['B_support'], out['B_rollouts'] = part_b(splits, days)
    out['B_data'] = part_b_data(splits, days)
    out['hashes'] = {str(p): base.sha(p) for p in list(MODELS.values()) + [se.PREPARED / 'data.pkl', Path(__file__)]}
    out['seconds'] = time.monotonic() - t
    out['test_evaluated'] = False
    base.save(a.out / 'result.json', plain(out))
    print('DONE', round(out['seconds'], 1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    run(p.parse_args())

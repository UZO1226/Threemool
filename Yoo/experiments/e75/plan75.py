"""Privileged full-day planning diagnostic, not an optimality bound or deployed MPC."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e74'))
sys.path.insert(0, str(HERE.parent / 'e68'))
import env74 as V
import baselines as B
E = V.E
BASE = Path('/home/jio/fcsrl_src/results/e74/baselines/baselines.json')
KEYS = ('air_m3', 'step_cost', 'nh4_excess_mg_h_L', 'nh4_steps', 'over_budget_steps', 'do_steps')
TOL = 1e-8
POP, ELITES, ITERS = 32, 8, 8
BLOCK = np.arange(95) // 4


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def context():
    splits, days = E.load()
    cap = E.caps(days, splits['dev'])
    allowed = set(splits['dev'] + splits['val'])
    days = {d: v for d, v in days.items() if d in allowed}
    dyn = E.Dynamics(E.PREPARED / 'surrogate.pkl')
    ref = json.loads(BASE.read_text())['chosen']['abac']
    return splits, days, dyn, cap, ref


def env_for(ctx):
    return E.DailyEnv(ctx[1], ctx[2], ctx[3])


def abac_actions(ctx, eps):
    env = env_for(ctx)
    obs = env.reset(eps)
    policy = B.ABAC(env, **ctx[4]['params'])
    acts, states = [], []
    for _ in range(95):
        action = policy(obs)
        acts.append(action.copy())
        obs, _, _, done, _ = env.step(action)
        states.append(env.state.copy())
    assert done
    return np.stack(acts, 1), np.stack(states, 1)


def simulate(ctx, eps, acts, trace=False):
    """Episode-major candidate layout; the unmodified DailyEnv74 executes each step."""
    n, k, t = acts.shape
    assert t == 95 and n == len(eps) and np.isfinite(acts).all()
    assert (np.abs(acts) <= 1 + TOL).all()
    env = env_for(ctx)
    env.reset([ep for ep in eps for _ in range(k)])
    out = {key: np.zeros(n*k) for key in KEYS}
    out['air_tv_m3h'] = np.zeros(n*k)
    prev = env.prev_air.copy()
    hist = {key: [] for key in ('states', 'airs', 'costs', 'zones')} if trace else None
    for t in range(95):
        _, _, c, done, info = env.step(acts[:, :, t].reshape(-1))
        nh = env.state[:, 0]
        out['air_m3'] += info['air'] * E.STEP_H
        out['step_cost'] += c
        out['nh4_excess_mg_h_L'] += np.maximum(nh - 4, 0) * E.STEP_H
        out['nh4_steps'] += nh > 4
        out['over_budget_steps'] += env.nh4_sum > E.BUDGET
        out['do_steps'] += env.state[:, 2:].max(1) > 2
        out['air_tv_m3h'] += np.abs(info['air'] - prev)
        prev = info['air'].copy()
        if trace:
            for key, value in (('states', env.state), ('airs', info['air']), ('costs', c), ('zones', info['zone'])):
                hist[key].append(value.copy())
    assert done
    out['day_violation'] = (env.nh4_sum > E.BUDGET).astype(float)
    out = {key: value.reshape(n, k) for key, value in out.items()}
    assert all(np.isfinite(value).all() for value in out.values())
    if trace:
        hist = {key: np.stack(value, 1) for key, value in hist.items()}
        assert np.array_equal(hist['costs'], V.step_cost(hist['states'][:, :, 0], np.cumsum(hist['states'][:, :, 0], 1)))
        assert np.allclose(hist['airs'], hist['zones'].sum(-1), rtol=0, atol=TOL)
    return out, hist


def rank(metrics, baseline, mode):
    excess = np.stack([(metrics[key] - baseline[key][:, None]) / np.maximum(np.abs(baseline[key][:, None]), 1.) for key in KEYS], -1)
    feasible = np.all(np.stack([metrics[key] <= baseline[key][:, None] + TOL for key in KEYS], -1), -1)
    violation = np.maximum(excess, 0).sum(-1)
    violation = np.where(feasible, 0, violation)
    if mode == 'air':
        order = np.lexsort((metrics['step_cost'], metrics['nh4_excess_mg_h_L'], metrics['air_m3'], violation, ~feasible), axis=1)
    else:
        order = np.lexsort((metrics['air_m3'], metrics['nh4_excess_mg_h_L'], metrics['step_cost'], violation, ~feasible), axis=1)
    return order, feasible


def summary(metrics):
    res = {key: float(value.mean()) for key, value in metrics.items()}
    res['within_4_75_pct'] = float(100 * (metrics['step_cost'] <= V.EVAL_LIMIT).mean())
    return res


def select_dates(dates):
    return [dates[i] for i in np.linspace(0, len(dates)-1, 12).astype(int)]


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    ctx = context()
    dates = select_dates(ctx[0][args.split])
    eps = E.episodes(dates, ctx[2].n_members)
    base, _ = abac_actions(ctx, eps)
    bm, _ = simulate(ctx, eps, base[:, None, :])
    baseline = {key: value[:, 0] for key, value in bm.items()}
    incumbent = base.copy()
    incumbent_metrics = baseline
    mean = np.zeros((len(eps), 24))
    std = np.full_like(mean, .25)
    rng = np.random.default_rng(args.seed)
    trace_log, snapshots = [], {}
    t0 = time.monotonic()
    query_count = 0
    for iteration in range(1, ITERS+1):
        residual = mean[:, None, :] + std[:, None, :] * rng.standard_normal((len(eps), POP, 24))
        acts = np.clip(base[:, None, :] + residual[:, :, BLOCK], -1, 1)
        acts[:, 0], acts[:, 1] = base, incumbent
        for i, scale in enumerate((.8, .9, .95, 1.05), 2):
            acts[:, i] = np.clip((base + 1) * scale - 1, -1, 1)
        metrics, _ = simulate(ctx, eps, acts)
        query_count += len(eps) * POP * 95
        order, feasible = rank(metrics, baseline, args.mode)
        chosen = order[:, 0]
        rows = np.arange(len(eps))
        assert feasible[rows, chosen].all()
        next_metrics = {key: value[rows, chosen] for key, value in metrics.items()}
        primary = 'air_m3' if args.mode == 'air' else 'step_cost'
        assert np.all(next_metrics[primary] <= incumbent_metrics[primary] + TOL)
        incumbent = acts[rows, chosen].copy()
        incumbent_metrics = next_metrics
        if args.algorithm == 'cem':
            elites = acts[rows[:, None], order[:, :ELITES]] - base[:, None, :]
            er = np.stack([elites[:, :, BLOCK == block].mean(2) for block in range(24)], -1)
            mean = .25 * mean + .75 * er.mean(1)
            variance = .25 * std**2 + .75 * er.var(1)
            std = np.maximum(np.sqrt(variance), .03)
        row = dict(iteration=iteration, seconds=time.monotonic()-t0,
                   feasible_candidate_fraction=float(feasible.mean()), **summary(incumbent_metrics))
        trace_log.append(row)
        if iteration in (2, 8):
            snapshots[str(iteration)] = summary(incumbent_metrics)
            np.savez_compressed(args.out / f'plan_{iteration}.npz', commands=incumbent,
                                **{key: value for key, value in incumbent_metrics.items()})
        write_json(args.out / 'progress.json', dict(iteration=iteration, total=ITERS, elapsed_seconds=time.monotonic()-t0))
        print(json.dumps(row), flush=True)
    check, hist = simulate(ctx, eps, incumbent[:, None, :], trace=True)
    for key in incumbent_metrics:
        np.testing.assert_allclose(check[key][:, 0], incumbent_metrics[key], rtol=0, atol=TOL)
    np.savez_compressed(args.out / 'final_rollout.npz', commands=incumbent, **hist)
    target_eps = [(day, (member+1) % ctx[2].n_members) for day, member in eps]
    ta, _ = abac_actions(ctx, target_eps)
    tbm, _ = simulate(ctx, target_eps, ta[:, None, :])
    tm, _ = simulate(ctx, target_eps, incumbent[:, None, :])
    tb = {key: value[:, 0] for key, value in tbm.items()}
    transfer = {key: value[:, 0] for key, value in tm.items()}
    _, ok = rank(tm, tb, args.mode)
    np.savez_compressed(args.out / 'transfer.npz', **{f'candidate_{k}':v for k,v in transfer.items()},
                        **{f'baseline_{k}':v for k,v in tb.items()})
    result = dict(split=args.split, mode=args.mode, algorithm=args.algorithm, search_seed=args.seed,
                  dates=[str(d) for d in dates], episodes=[[str(d),m] for d,m in eps], baseline=summary(baseline),
                  snapshots=snapshots, final=summary(incumbent_metrics), trajectory_replay_pass=True,
                  transfer_baseline=summary(tb), transfer=summary(transfer),
                  transfer_weak_dominance_pct=float(100*ok.mean()), trace=trace_log,
                  optimization_transition_queries=query_count, elapsed_seconds=time.monotonic()-t0)
    write_json(args.out / 'result.json', result)
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['dev','val'], required=True)
    parser.add_argument('--mode', choices=['air','cost'], required=True)
    parser.add_argument('--algorithm', choices=['cem','random'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args())

"""e74 체크포인트의 같은 공기량 진단. 설계는 REFERENCES_AND_DESIGN.md에 고정했다."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e74'))
sys.path.insert(0, str(HERE.parent / 'e68'))
import env74 as V
import env68 as E
import train68 as T

RUNS = Path('/home/jio/fcsrl_src/results/e74/main/runs')
BASELINES = Path('/home/jio/fcsrl_src/results/e74/baselines/baselines.json')
TARGET = json.loads(BASELINES.read_text())['chosen']['abac']['dev']['air_m3']
COARSE = tuple(round(.5 + .1 * i, 3) for i in range(11))


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scale_action(action, alpha):
    action = np.asarray(action)
    assert np.isfinite(action).all() and np.max(np.abs(action)) <= 1.0
    assert .5 <= alpha <= 1.5
    if alpha == 1.0:
        return action.copy()
    return np.clip(alpha * (action.astype(float) + 1.0) - 1.0, -1.0, 1.0)


def load_agent(method, seed):
    run = RUNS / f'{method}_s{seed}'
    cfg = json.loads((run / 'effective_config.json').read_text())
    assert cfg['method'] == method and cfg['seed'] == seed
    agent = T.Agent68(cfg['config'], learn_repr=method == 'fcsrl')
    agent.load_state(torch.load(run / 'model_final.pt', map_location='cpu', weights_only=False))
    agent.train(False)
    return agent


def make_env():
    splits, days = E.load()
    env = V.DailyEnv74(days, E.Dynamics(E.PREPARED / 'surrogate.pkl'), E.caps(days, splits['dev']))
    return splits, env


def episode_metrics(states, air, zones):
    """궤적에서 직접 재계산하고 원 평균 집계와 대조한다."""
    assert np.allclose(air, zones.sum(-1), atol=1e-9, rtol=0)
    nh4 = states[:, :, 0]
    cumulative = np.cumsum(nh4, axis=1)
    cost = np.logical_or(nh4 > 4, cumulative > 380).sum(1).astype(float)
    return dict(air_m3=air.sum(1) * .25, step_cost=cost,
                nh4_excess_mg_h_L=np.maximum(nh4 - 4, 0).sum(1) * .25,
                do_step_viol_pct=100 * (states[:, :, 2:].max(-1) > 2).mean(1),
                day_violation_pct=100 * (nh4.sum(1) > 380).astype(float),
                cost_limit_violation_pct=100 * (cost > 4.75).astype(float))


def evaluate(agent, env, dates, seed, alpha, save=None, reference=None):
    runs, episode_rows, traces = [], [], []
    t0 = time.monotonic()
    for k in (0, 1, 2):
        torch.manual_seed(100_000 + 1000 * seed + k)
        res, states, air, zones = V.rollout74(env, dates, lambda o: scale_action(agent.act_sample(o), alpha))
        ep = episode_metrics(states, air, zones)
        for key, vals in ep.items():
            if key in res:
                assert np.isclose(np.mean(vals), res[key], atol=1e-9, rtol=0), key
        cost = ep['step_cost']
        res.update(cost_limit_violation_pct=float(ep['cost_limit_violation_pct'].mean()),
                   normalized_cost_deviation=float((cost.mean() - 4.75) / 4.75),
                   normalized_violation_magnitude=float((cost[cost > 4.75].mean() - 4.75) / 4.75)
                   if (cost > 4.75).any() else 0.0)
        if reference is not None:
            old = reference['metrics'][k]
            for key, val in old.items():
                if val is not None:
                    assert np.isclose(res[key], val, atol=1e-9, rtol=0), (key, res[key], val)
            if k == 0:
                with np.load(reference['trace']) as original:
                    for key, arr in [('states', states), ('air', air), ('zones', zones)]:
                        assert np.allclose(original[key], arr, atol=1e-9, rtol=0), key
        runs.append(res)
        episode_rows.append(ep)
        if save is not None:
            traces.append((states, air, zones))
    avg = {key: None if any(r[key] is None for r in runs) else float(np.mean([r[key] for r in runs])) for key in runs[0]}
    if save is not None:
        np.savez_compressed(save,
                            dates=np.array(dates),
                            states=np.stack([x[0] for x in traces]),
                            air=np.stack([x[1] for x in traces]),
                            zones=np.stack([x[2] for x in traces]),
                            **{key: np.stack([ep[key] for ep in episode_rows]) for key in episode_rows[0]})
    return dict(alpha=alpha, mean=avg, runs=runs, seconds=time.monotonic() - t0)


def choose(rows):
    return min(rows, key=lambda r: (abs(r['mean']['air_m3'] - TARGET), abs(r['alpha'] - 1), r['alpha']))


def run_job(method, seed, out, smoke=False):
    torch.set_num_threads(1)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    run = RUNS / f'{method}_s{seed}'
    hashes = {str(p): sha(p) for p in (run / 'model_final.pt', run / 'effective_config.json', run / 'metrics.json')}
    original = json.loads((run / 'metrics.json').read_text())
    agent = load_agent(method, seed)
    normalizer_before = {k: np.array(v, copy=True) for k, v in agent.state()['normalizer'].items()}
    splits, env = make_env()
    t0 = time.monotonic()
    if smoke:
        # Only development dates, no selection and no new inferential result.
        native = evaluate(agent, env, splits['dev'][:2], seed, 1, out / 'smoke.npz')
        assert np.array_equal(scale_action(np.array([-1., 0., 1.]), 1), [-1, 0, 1])
        assert np.allclose(scale_action(np.array([-1., 0., 1.]), .5), [-1, -.5, 0])
        assert np.allclose(scale_action(np.array([-1., 0., 1.]), 1.5), [-1, .5, 1])
        dump(out / 'smoke.json', dict(pass_checks=True, seconds=time.monotonic() - t0, episodes=native['mean']['n_episodes']))
        return
    # Full alpha=1 reproduction is a hard gate before any calibration candidate.
    native = {}
    for split in ('dev', 'val'):
        native[split] = evaluate(agent, env, splits[split], seed, 1,
                                 out / f'{split}_native.npz',
                                 dict(metrics=original[f'{split}_stochastic_runs'], trace=run / f'{split}_stochastic_rollout.npz'))
    dump(out / 'reproduction.json', dict(passed=True, native=native))
    rows = []
    for alpha in COARSE:
        row = native['dev'] if alpha == 1 else evaluate(agent, env, splits['dev'], seed, alpha)
        rows.append(row)
        dump(out / 'progress.json', dict(phase='coarse', candidates=len(rows), elapsed=time.monotonic() - t0))
    coarse_best = choose(rows)
    refine = sorted({round(coarse_best['alpha'] + delta, 3) for delta in (-.05, -.025, .025, .05)
                     if .5 <= round(coarse_best['alpha'] + delta, 3) <= 1.5} - set(COARSE))
    for alpha in refine:
        rows.append(evaluate(agent, env, splits['dev'], seed, alpha))
        dump(out / 'progress.json', dict(phase='refine', candidates=len(rows), elapsed=time.monotonic() - t0))
    best = choose(rows)
    selection = dict(method=method, seed=seed, alpha=best['alpha'], target_air=TARGET,
                     target_error_pct=100 * (best['mean']['air_m3'] / TARGET - 1),
                     dev_target_matched=abs(best['mean']['air_m3'] / TARGET - 1) <= .01,
                     coarse_best_alpha=coarse_best['alpha'], candidates=rows)
    dump(out / 'selection.json', selection)
    selection_sha = sha(out / 'selection.json')
    selected = {}
    for split in ('dev', 'val'):
        selected[split] = evaluate(agent, env, splits[split], seed, best['alpha'], out / f'{split}_selected.npz')
    for key, value in best['mean'].items():
        if value is not None:
            assert np.isclose(value, selected['dev']['mean'][key], rtol=0, atol=1e-9), key
    assert sha(out / 'selection.json') == selection_sha
    assert all(sha(p) == v for p, v in hashes.items())
    assert all(np.array_equal(agent.state()['normalizer'][k], v) for k, v in normalizer_before.items())
    result = dict(method=method, seed=seed, target_air=TARGET, alpha=best['alpha'],
                  target_error_pct=selection['target_error_pct'], dev_target_matched=selection['dev_target_matched'],
                  native=native, selected=selected, hashes=hashes, selection_sha=selection_sha,
                  seconds=time.monotonic() - t0, holdout_evaluated=False, test_evaluated=False)
    dump(out / 'result.json', result)
    dump(out / 'progress.json', dict(phase='complete', elapsed=result['seconds']))
    print(f"DONE {method} {seed} alpha={best['alpha']} match={result['dev_target_matched']} {result['seconds']:.1f}s", flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['fcsrl', 'ppolag'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    run_job(a.method, a.seed, a.out, a.smoke)

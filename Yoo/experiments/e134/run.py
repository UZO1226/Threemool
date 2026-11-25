"""Replay e133 policies with air-only calibration and a separate winter pass."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e133"))
import trial

from darrow_env import action_to_air, air_to_action, all_episodes, AIR_MAX, STEP_H

base = trial.base
source = trial.REF.parents[1] / "e133/main"
coarse = tuple(range(500, 1501, 100))  # multiplier in thousandths


def scale(action, alpha):
    if alpha == 1000:
        return action
    air = np.clip(action_to_air(action) * (alpha / 1000), 0, AIR_MAX)
    return air_to_action(air)


def setup():
    global splits, env, target
    torch.set_num_threads(1)
    base.DeviceConfig().select_device(-1)
    splits, days = base.load_selected(trial.PREP / "data.pkl")
    env = trial.fast.BatchEnv(days, trial.fast.FastDynamics(trial.PREP / "surrogate.pkl"))
    target = np.mean([days[d]["air"][:95].sum() * STEP_H for d in splits["train"]])
    expected = trial.read(trial.REF / "replay_check.json")["train"]["value"]["air_m3"]
    np.testing.assert_allclose(target, expected, rtol=0, atol=1e-8)


def rollout(agent, dates, alpha, *, keep=False):
    episodes = all_episodes(dates, 5)
    obs = env.reset(episodes)
    airs, states = [], []
    for _ in range(95):
        obs, _, _, done, info = env.step(scale(agent.act_det(obs)[:, 0], alpha))
        airs.append(info["air"])
        if keep:
            states.append(env.state.copy())
    assert done
    air = np.stack(airs, axis=1)
    assert np.isfinite(air).all() and air.min() >= 0 and air.max() <= AIR_MAX
    if keep:
        return air, np.stack(states, axis=1)
    return float((air.sum(axis=1) * STEP_H).mean())


def load_agent(method, seed):
    agent = trial.factory(method, seed, trial.PREP, base.common.config(66))
    agent.load(source / method / f"seed_{seed}/model_final")
    return agent


def restore(job):
    method, seed = job
    agent = load_agent(method, seed)
    air, states = rollout(agent, splits["val"], 1000, keep=True)
    with np.load(source / method / f"seed_{seed}/winter_rollout.npz") as ref:
        np.testing.assert_array_equal(air, ref["air"])
        np.testing.assert_array_equal(states, ref["states"])
    print("reproduced", method, seed, flush=True)


def fit(job):
    method, seed = job
    agent = load_agent(method, seed)
    values = {}
    def select():
        return min(values, key=lambda a: (abs(values[a] - target), abs(a - 1000), a))
    for alpha in coarse:
        values[alpha] = rollout(agent, splits["train"], alpha)
    best = select()
    for offset in (-50, -25, 25, 50):
        alpha = best + offset
        if coarse[0] <= alpha <= coarse[-1]:
            values[alpha] = rollout(agent, splits["train"], alpha)
    chosen = select()
    print("selected", method, seed, chosen, flush=True)
    return dict(method=method, seed=seed, alpha_milli=chosen, candidates=values,
                train_air=values[chosen], target=float(target))


def evaluate(job):
    row, out = job
    agent = load_agent(row["method"], row["seed"])
    air, states = rollout(agent, splits["val"], row["alpha_milli"], keep=True)
    episodes = all_episodes(splits["val"], 5)
    _, rows, _ = base.metrics(states, air, episodes)
    name = f"{row['method']}_{row['seed']}"
    base.save(out / f"{name}.json", rows)
    np.savez_compressed(out / f"{name}.npz", air=air, states=states)


def run(out, jobs):
    out.mkdir(parents=True, exist_ok=False)
    for name, expected in trial.read(trial.REF / "manifest.json")["input_sha256"].items():
        assert base.sha(trial.PREP / name) == expected
    manifest = source / "tables/SHA256SUMS"
    assert base.sha(manifest) == "297268876814ab83d214f6866d5ae725eaa9b792ae485a1e0960682eb88122c4"
    subprocess.run(["sha256sum", "--quiet", "-c", str(manifest)], cwd=source, check=True)
    (out / "commit").write_bytes(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE))
    a = np.array([-1., 0., 1.])
    np.testing.assert_array_equal(scale(a, 500), [-1., -.5, 0.])
    np.testing.assert_array_equal(scale(a, 1500), [-1., .5, 1.])
    tasks = [(m, s) for s in trial.SEEDS for m in trial.METHODS]
    with ProcessPoolExecutor(max_workers=jobs, initializer=setup) as pool:
        list(pool.map(restore, tasks))
        rows = list(pool.map(fit, tasks))
        base.save(out / "selection.json", rows)
        frozen = trial.read(out / "selection.json")
        list(pool.map(evaluate, [(row, out) for row in frozen]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    run(args.out, args.jobs)

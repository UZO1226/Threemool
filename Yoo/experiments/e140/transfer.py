"""Replay the e139 policies at later DARROW dates with no refitting."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e139"))
import folds
from replay import water
from darrow_env import air_to_action

SOURCE = Path("/home/jio/fcsrl_src/results/e139/main")
CELLS = [(108, 148), (108, 188), (148, 188)]
ARMS = {"base_risk": "fcsrl", "combined": "fcsrl_actor"}


def environment(prefix, offset):
    fold, prep = folds.paths(SOURCE, prefix)
    splits, days = folds.base.load_selected(folds.trial.PREP / "data.pkl")
    dates = (splits["train"] + splits["val"])[offset:offset + 40]
    meta = folds.trial.read(prep / "manifest.json")
    assert len(dates) == 40 and meta["train"][-1] < dates[0]
    if prefix != offset:
        assert meta["val"][-1] < dates[0]
    else:
        assert dates == meta["val"]
    # The common loader's fallback proportions are unused on every evaluated date.
    assert all((days[d]["zone"].sum(axis=1) > 1).all() for d in dates)
    env = folds.trial.fast.BatchEnv({d: days[d] for d in dates}, folds.trial.fast.FastDynamics(prep / "surrogate.pkl"))
    return fold, prep, dates, env


def replay_and_verify(out):
    assert folds.base.sha(SOURCE / "SHA256SUMS") == "1c07a2da951ffb0729d0cd4ac80d203b7ce5faabe35f9980eb22f6eec1698ad1"
    subprocess.run(["sha256sum", "--quiet", "-c", "SHA256SUMS"], cwd=SOURCE, check=True)
    path = folds.trial.PREP / "data.pkl"
    assert folds.base.sha(path) == folds.trial.read(folds.trial.REF / "manifest.json")["input_sha256"][path.name]
    rows = []
    for prefix, offset in CELLS:
        _, _, dates, env = environment(prefix, offset)
        env.reset(folds.old.all_episodes(dates, 5))
        predictions = []
        for t in range(95):
            env.step(air_to_action(env.x["air"][:, t]))
            predictions.append(env.state.copy())
        obs = env.x["state"][:, 1:]
        sim = np.stack(predictions, axis=1)
        real, model = (water(x).reshape(40, 5, 3).mean(axis=1) for x in (obs, sim))
        for i, day in enumerate(dates):
            row = dict(prefix=prefix, offset=offset, date=day)
            for j, name in enumerate(["nh4_min", "nh4_excess", "do_min"]):
                row["observed_" + name] = real[i, j]
                row["replay_" + name] = model[i, j]
            rows.append(row)
    pd.DataFrame(rows).to_csv(out / "replay.csv", index=False)


def evaluate(job):
    out, prefix, offset, seed, smoke = job
    torch.set_num_threads(1)
    folds.base.DeviceConfig().select_device(-1)
    fold, prep, dates, folds.old.env = environment(prefix, offset)
    archived = fold / "evaluated" / f"seed_{seed}"
    with (archived / "risk.pkl").open("rb") as stream:
        risk = folds.study.FastRate(pickle.load(stream))
    dest = out / f"p{prefix}_o{offset}_s{seed}"
    if not smoke:
        dest.mkdir()
    for arm, method in ARMS.items():
        agent = folds.trial.factory(method, seed, prep, folds.base.common.config(66))
        agent.load(fold / method / f"seed_{seed}" / "model_final")
        policy = folds.study.LocalPolicy(agent, risk, 0)
        for scale in ([1000] if smoke else folds.report.SCALES):
            air, states = folds.old.rollout(policy, dates, scale, keep=True)
            name = f"{arm}_{scale}.npz"
            if smoke:
                with np.load(archived / name) as saved:
                    np.testing.assert_array_equal(air, saved["air"])
                    np.testing.assert_array_equal(states, saved["states"])
            else:
                np.savez_compressed(dest / name, air=air, states=states)
    return prefix, offset, seed


def run(out, jobs):
    out.mkdir(parents=True, exist_ok=False)
    replay_and_verify(out)
    evaluate((out, 108, 108, 10, True))
    print("Recorded-air replay and adjacent-window trajectory check complete.", flush=True)
    tasks = [(out, n, offset, s, False) for n, offset in CELLS for s in folds.report.SEEDS]
    with ProcessPoolExecutor(jobs) as pool:
        for cell in pool.map(evaluate, tasks):
            print(*cell, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--jobs", type=int, default=5)
    args = parser.parse_args()
    run(args.out, args.jobs)

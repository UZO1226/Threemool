"""Check frozen intervention decisions against logged one-hour NH4 responses."""
import argparse
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e135"))
import study
from darrow_env import action_to_air

old = study.old
trial = old.trial
SOURCE = Path("/home/jio/fcsrl_src/results")
SEEDS = range(10, 15)


def crossing(paths, initial):
    changed = (paths > 4) != (initial[:, None] > 4)
    return np.where(changed.any(axis=1), changed.argmax(axis=1) + 1, 0)


def audit(out):
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    old.base.DeviceConfig().select_device(-1)
    assert old.base.sha(SOURCE / "e135/main/SHA256SUMS") == "3a5cbbbfdbca2ab487a1cc6a50138d31e4b6a2b063c3234e7b503e8782cc474c"
    subprocess.run(["sha256sum", "--quiet", "-c", "SHA256SUMS"], cwd=SOURCE / "e135/main", check=True)
    manifest = trial.read(trial.REF / "manifest.json")["input_sha256"]
    for name in ("data.pkl", "surrogate.pkl"):
        assert old.base.sha(trial.PREP / name) == manifest[name]
    splits, days = old.base.load_selected(trial.PREP / "data.pkl")
    dyn = trial.fast.FastDynamics(trial.PREP / "surrogate.pkl")
    env = trial.fast.BatchEnv(days, dyn)
    with np.load(SOURCE / "e104/main/replay.npz") as source:
        dates, times, dq = [source[k] for k in ("dates", "t", "dq")]
        truth = source["truth"][:, :5]
        archived = source["rec"][:, :2]
    assert set(dates) <= set(splits["val"]) and (times >= 1).all()
    def at(key, shift):
        return np.stack([days[d][key][t + shift] for d, t in zip(dates, times)])

    np.testing.assert_allclose(at("air", 0)-at("air", -1), dq, rtol=0, atol=1e-8)
    obs = np.empty((len(times), 11), np.float32)
    for t in np.unique(times):
        idx = np.flatnonzero(times == t)
        env.reset([(dates[i], 0) for i in idx])
        env.t = t
        env.state = env.x["state"][:, t]
        env.prev_air = env.x["air"][:, t-1]
        obs[idx] = env.obs()
    votes, supported = [], []
    seed_rows = []
    for seed in SEEDS:
        agent = old.load_agent("fcsrl", seed)
        with (SOURCE / f"e135/main/seed_{seed}/rate.pkl").open("rb") as stream:
            models = pickle.load(stream)
        policy = study.LocalPolicy(agent, study.FastRate(models), 0)
        base_air = action_to_air(agent.act_det(obs)[:, 0])
        selected = action_to_air(policy.act_det(obs)[:, 0])
        changed = abs(base_air - selected) > .05
        close = abs(selected - at("air", 0)) <= 281.25
        votes.append(changed)
        supported.append(close)
        seed_rows.extend(dict(seed=seed, day=d, t=t, base_air=b, choice=a, logged_air=g)
                         for d,t,b,a,g in zip(dates,times,base_air,selected,at("air", 0)))
    pd.DataFrame(seed_rows).to_csv(out / "actions.csv", index=False)

    members = np.tile(np.arange(5), len(times))
    state = np.repeat(at("state", 0), 5, axis=0)
    predictions = []
    for t in range(4):
        z = [np.repeat(at(key, shift), 5, axis=0) for key,shift in [("exo", t), ("zone", t), ("zone", t-1)]]
        state = dyn.step(state, *z, members)
        all_states = state.reshape(len(times), 5, 5)
        if t in (0, 3):
            np.testing.assert_allclose(all_states, archived[:, int(t == 3)], atol=1e-10, rtol=0)
        np.testing.assert_array_equal(at("state", t+1), truth[:, t+1])
        predictions.append(all_states[:, :, 0].mean(axis=1))
    predicted = np.column_stack(predictions)
    actual = truth[:, 1:, 0]
    initial = truth[:, 0, 0]
    obs_delta, pred_delta = actual[:, -1]-initial, predicted[:, -1]-initial
    observed, forecast = crossing(actual, initial), crossing(predicted, initial)
    frame = pd.DataFrame(dict(day=dates, t=times, direction=np.where(dq>0, "up", "down"),
        dq=dq, nh4=initial, temp=at("exo", 0)[:, 1], inflow=at("exo", 0)[:, 0],
        votes=np.sum(votes, axis=0), close_votes=np.sum(supported, axis=0),
        observed_delta=obs_delta, model_delta=pred_delta, mae=abs(pred_delta-obs_delta),
        persistence_mae=abs(obs_delta), direction_hit=np.sign(obs_delta)==np.sign(pred_delta),
        observed_cross=observed, model_cross=forecast))
    frame["majority"] = frame.votes >= 3
    frame["mae_gap"] = frame.mae - frame.persistence_mae
    frame.to_csv(out / "windows.csv", index=False)
    rows = []
    for (direction, majority), group in frame.groupby(["direction", "majority"]):
        water = group.groupby("day")[["observed_delta","model_delta","mae","persistence_mae","mae_gap","direction_hit"]].mean().mean()
        real, sim = group.observed_cross, group.model_cross
        rows.append(dict(direction=direction, majority=majority, windows=len(group),
            days=group.day.nunique(), close=int((group.close_votes >= 3).sum()),
            actual_cross=int((real>0).sum()), hits=int(((real>0)&(sim>0)).sum()),
            false_alarms=int(((real==0)&(sim>0)).sum()), exact_hits=int(((real>0)&(sim==real)).sum()),
            **water.to_dict()))
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    strata = frame.assign(high=frame.nh4>4, nh_bin=np.floor(frame.nh4), temp_bin=np.floor(frame.temp/2),
                          q_bin=np.floor(frame.inflow/1000), air_bin=np.floor(frame.dq.abs()/500))
    cols = ["direction", "high", "nh_bin", "temp_bin", "q_bin", "air_bin"]
    overlap = strata.groupby(cols+["majority"]).size().unstack(fill_value=0).reindex(columns=[False,True], fill_value=0)
    overlap.columns = ["minority", "majority"]
    overlap.to_csv(out / "strata.csv")
    matched = overlap[(overlap>0).all(axis=1)]
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("Shared strata", len(matched), "windows", matched.sum().to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    audit(ap.parse_args().out)

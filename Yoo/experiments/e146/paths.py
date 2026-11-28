"""Replay the single e145 cut and record information available before that cut."""
import argparse
from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import torch

spec = importlib.util.spec_from_file_location("e145_paths", Path(__file__).resolve().parents[1] / "e145/evaluate.py")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)

OBS = ["nh4", "no3", "do1", "do2", "do3", "inflow", "temperature", "sin_time",
       "cos_time", "inlet_nh4", "previous_air"]
FEATURES = OBS + ["native_command", "proposed_saving"]


def extract(prefix, seed, source, out):
    torch.set_num_threads(1)
    previous.choices.base.DeviceConfig().select_device(-1)
    env, _, episodes, policy, air, states = previous.choices.load(prefix, seed)
    path = previous.rollout(env, episodes, policy, "once")
    cut_air, _, first, count = path
    daily = previous.compare(path, (air, states), "once", "current")
    archive = pd.read_csv(source / f"p{prefix}_s{seed}/daily.csv.gz")
    once = archive.loc[(archive["mode"] == "once") & (archive.version == "current")].sort_values("episode")
    for col in previous.METRICS + ["episode", "first", "switches", "initial_m3", "hour_m3"]:
        np.testing.assert_allclose(daily[col], once[col], atol=1e-8, rtol=0)
    assert list(once.day) == [str(day) for day, member in episodes]
    np.testing.assert_array_equal(once.member, [member for day, member in episodes])
    delta = (cut_air - air) / 4
    features = np.full((len(episodes), len(FEATURES)), np.nan)
    cumulative = np.full((len(episodes), 5), np.nan)
    obs = env.reset(episodes)
    for t in range(env.T):
        if t in previous.HOURS:
            actions, win, mask, cheapest = previous.options(policy, obs, .025 + 1e-12)
            ids = np.where((first == t) & (count > 0))[0]
            native = previous.choices.old.action_to_air(actions[np.arange(len(win)), win])
            alternative = previous.choices.old.action_to_air(actions[ids, cheapest[ids]])
            assert mask[ids, cheapest[ids]].all()
            np.testing.assert_array_equal(native, air[:, t])
            np.testing.assert_array_equal(alternative, cut_air[ids, t])
            features[ids] = np.column_stack((obs[ids], native[ids], (native[ids] - alternative) / 4))
            if t + 24 <= env.T:
                for i, h in enumerate(range(2, 7)):
                    cumulative[ids, i] = delta[ids, t:t+4*h].sum(1)
        action = policy.act_det(obs)[:, 0]
        obs, _, _, _, info = env.step(action)
        np.testing.assert_array_equal(info["air"], air[:, t])
        np.testing.assert_array_equal(env.state, states[:, t])
    eligible = (count == 1) & (first + 24 <= env.T)
    np.testing.assert_array_equal(np.isfinite(cumulative).all(1), eligible)
    np.testing.assert_allclose(features[count == 1, -1], -daily.loc[count == 1, "initial_m3"])
    table = pd.concat([once.reset_index(drop=True), pd.DataFrame(features, columns=FEATURES),
                       pd.DataFrame(cumulative, columns=[f"air_{h}h" for h in range(2, 7)])], axis=1)
    out.mkdir(parents=True, exist_ok=True)
    table.assign(prefix=prefix, seed=seed, eligible=eligible).to_csv(out / f"p{prefix}_s{seed}.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prefix", type=int)
    p.add_argument("seed", type=int)
    p.add_argument("source", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    extract(a.prefix, a.seed, a.source, a.out)

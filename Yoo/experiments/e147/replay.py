"""Replay each frozen first cut and native command from an identical state on all five models."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e144"))
import choices

METRICS = ["air_m3", "nh4_min", "excess", "do_min"]


def starting_points(env, episodes, policy, table, air, states):
    env.reset(episodes)
    first = table["first"].to_numpy(int)
    rows = np.arange(len(first))
    last = np.maximum(first - 1, 0)
    state = np.where(first[:, None] > 0, states[rows, last], env.x["state"][:, 0])
    prev = np.where(first > 0, air[rows, last], env.x["air"][:, 0])
    zone = np.where(first[:, None] > 0, prev[:, None] * env.x["frac"][rows, last], env.x["zone"][:, 0])
    native, cut = np.zeros(len(first)), np.zeros(len(first))
    for t in np.unique(first):
        env.t = t
        env.state = states[:, t-1].copy() if t else env.x["state"][:, 0].copy()
        env.prev_air = air[:, t-1].copy() if t else env.x["air"][:, 0].copy()
        actions, win, gap, unique = choices.proposals(policy, env.obs())
        rates = choices.old.action_to_air(actions)
        eligible = unique & (gap <= .025 + 1e-12) & (rates < rates[rows, win, None] - 1e-8)
        hit = rows[first == t]
        assert eligible[hit].any(axis=1).all()
        cheap = np.where(eligible, actions, np.inf).argmin(axis=1)
        native[hit] = actions[hit, win[hit]]
        cut[hit] = actions[hit, cheap[hit]]
    native_air = choices.old.action_to_air(native)
    np.testing.assert_array_equal(native_air, table.native_command)
    np.testing.assert_array_equal(native_air, air[rows, first])
    np.testing.assert_array_equal((native_air - choices.old.action_to_air(cut)) / 4, table.proposed_saving)
    # Exact equality across origins within a single policy seed, date and decision time.
    keys = [(d, t, *s, p, *z, a, b) for (d, _), t, s, p, z, a, b in
            zip(episodes, first, state, prev, zone, native, cut)]
    group = pd.factorize(pd.Series(keys))[0]
    return first, state, prev, zone, native, cut, group


def simulate(env, episodes, policy, member, start, command):
    first, state, prev, zone = start[:4]
    env.reset(episodes)
    env.member[:] = member
    airs, states = [], []
    for t in range(env.T):
        hit = first == t
        env.state[hit] = state[hit]
        env.prev_air[hit] = prev[hit]
        env.prev_zone[hit] = zone[hit]
        action = policy.act_det(env.obs())[:, 0].copy()
        action[hit] = command[hit]
        _, _, _, _, info = env.step(action)
        airs.append(info["air"])
        states.append(env.state.copy())
    # Only post-decision outcomes count. Pre-decision air and both violation indicators become zero.
    before = np.arange(env.T) < first[:, None]
    air = np.stack(airs, axis=1)
    state_path = np.stack(states, axis=1)
    air[before] = 0.
    state_path[before] = 0.
    return choices.metrics(air, state_path)


def run(prefix, seed, source, dest):
    torch.set_num_threads(1)
    choices.base.DeviceConfig().select_device(-1)
    env, older, episodes, policy, air, states = choices.load(prefix, seed)
    table = pd.read_csv(source / f"p{prefix}_s{seed}.csv", float_precision="round_trip").sort_values("episode")
    assert table.eligible.all() and len(table) == 200
    assert list(table.day) == [str(d) for d, _ in episodes]
    np.testing.assert_array_equal(table.member, [m for _, m in episodes])
    start = starting_points(env, episodes, policy, table, air, states)
    base_air = air.sum(1) / 4
    results = []
    for version, dynamics in [("current", env.dyn), ("previous", older)]:
        if dynamics is None:
            continue
        env.dyn = dynamics
        for member in range(5):
            initial = simulate(env, episodes, policy, member, start, start[4])
            alternative = simulate(env, episodes, policy, member, start, start[5])
            delta = alternative - initial
            if version == "current":
                match = table.member == member
                np.testing.assert_allclose(delta[match], table.loc[match, METRICS], rtol=0, atol=1e-7)
            part = pd.DataFrame(delta, columns=METRICS)
            results.append(part.assign(prefix=prefix, seed=seed, day=table.day.values, episode=table.episode.values,
                                       origin=table.member.values, first=start[0], decision=start[6], version=version,
                                       evaluator=member, native_daily_air=base_air))
    dest.mkdir(parents=True, exist_ok=True)
    pd.concat(results, ignore_index=True).to_csv(dest / f"p{prefix}_s{seed}.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prefix", type=int)
    p.add_argument("seed", type=int)
    p.add_argument("source", type=Path)
    p.add_argument("dest", type=Path)
    args = p.parse_args()
    run(args.prefix, args.seed, args.source, args.dest)

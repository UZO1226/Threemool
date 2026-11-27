"""Compare lower-air choices to the frozen FCSRL risk selector on DARROW."""
import argparse
import copy
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e144"))
import choices

HOURS = range(0, 92, 4)
METRICS = ["air_m3", "nh4_min", "excess", "do_min"]
EPS = 1e-8


def options(policy, obs, tol):
    actions, win, gap, unique = choices.proposals(policy, obs)
    row = np.arange(len(win))
    rates = choices.old.action_to_air(actions)
    mask = unique & (gap <= tol) & (rates < rates[row, win, None] - EPS)
    cheapest = np.where(mask, actions, np.inf).argmin(axis=1)
    return actions, win, mask, np.where(mask.any(axis=1), cheapest, win)


def hour(env, indices, action, policy):
    fork = copy.copy(env)
    fork.member = env.member[indices]
    for key in ("state", "prev_air", "prev_zone"):
        setattr(fork, key, getattr(env, key)[indices].copy())
    fork.x = {key: env.x[key][indices, env.t:env.t+5] for key in ("exo", "frac")}
    fork.t = 0
    air, states = [], []
    for j in range(4):
        obs, _, _, _, info = fork.step(action)
        air.append(info["air"])
        states.append(fork.state.copy())
        if j < 3:
            action = policy.act_det(obs)[:, 0]
    return choices.metrics(np.stack(air, axis=1), np.stack(states, axis=1))


def inspect(env, policy, air, states):
    anchors, pairs = [], []
    for t in HOURS:
        env.t = t
        env.state = states[:, t-1].copy() if t else env.x["state"][:, 0].copy()
        env.prev_air = air[:, t-1].copy() if t else env.x["air"][:, 0].copy()
        env.prev_zone = env.prev_air[:, None] * env.x["frac"][:, t-1] if t else env.x["zone"][:, 0].copy()
        actions, win, mask, cheapest = options(policy, env.obs(), .025 + 1e-12)
        ids = np.arange(len(win))
        np.testing.assert_array_equal(choices.old.action_to_air(actions[ids, win]), air[:, t])
        anc = pd.DataFrame(dict(episode=ids, t=t, high=env.state[:, 0]>4, eligible=mask.any(1),
                                favorable=False, cheapest_favorable=False))
        rows, cols = np.nonzero(mask)
        if len(rows):
            origin, inv = np.unique(rows, return_inverse=True)
            moves = np.r_[actions[origin, win[origin]], actions[rows, cols]]
            values = hour(env, np.r_[origin, rows], moves, policy)
            k = len(origin)
            archive = choices.metrics(air[origin, t:t+4], states[origin, t:t+4])
            # Fused actor inference can round differently for a smaller batch.
            np.testing.assert_allclose(values[:k], archive, rtol=0, atol=.005)
            diff = values[k:] - values[:k][inv]
            good = (diff[:, 0] < -EPS) & (diff[:, 1:] <= EPS).all(1)
            cheap = cols == cheapest[rows]
            anc.loc[rows[good], "favorable"] = True
            anc.loc[rows[good & cheap], "cheapest_favorable"] = True
            pair = pd.DataFrame(diff, columns=METRICS)
            pair = pair.assign(episode=rows, t=t, candidate=cols, action=actions[rows, cols],
                               cheapest=cheap, favorable=good)
            pairs.append(pair)
        anchors.append(anc)
    return pd.concat(anchors, ignore_index=True), pd.concat(pairs, ignore_index=True)


def rollout(env, episodes, policy, mode, plan=None):
    obs = env.reset(episodes)
    rows = np.arange(len(episodes))
    first = np.full(len(rows), -1)
    count = np.zeros(len(rows), int)
    air, states = [], []
    for t in range(env.T):
        if t in HOURS and mode in ("repeat", "once", "exact"):
            tol = 1e-12 if mode == "exact" else .025 + 1e-12
            actions, win, _, cheapest = options(policy, obs, tol)
            hit = (cheapest != win) & ((first < 0) if mode == "once" else True)
            action = actions[rows, np.where(hit, cheapest, win)]
        else:
            action = policy.act_det(obs)[:, 0].copy()
            hit = (plan[0] == t) if mode == "oracle" else np.zeros(len(rows), bool)
            if hit.any():
                action[hit] = plan[1][hit]
        first[(first < 0) & hit] = t
        count += hit
        obs, _, _, _, info = env.step(action)
        air.append(info["air"])
        states.append(env.state.copy())
    return np.stack(air, axis=1), np.stack(states, axis=1), first, count


def compare(path, reference, mode, version):
    air, states, first, count = path
    ref_air, ref_states = reference
    before = (first[:, None] < 0) | (np.arange(air.shape[1]) < first[:, None])
    np.testing.assert_array_equal(air[before], ref_air[before])
    np.testing.assert_array_equal(states[before], ref_states[before])
    delta = choices.metrics(air, states) - choices.metrics(ref_air, ref_states)
    table = pd.DataFrame(delta, columns=METRICS)
    change = (air - ref_air) / 4
    rows = np.flatnonzero(count)
    initial = np.zeros(len(first))
    within_hour = initial.copy()
    initial[rows] = change[rows, first[rows]]
    within_hour[rows] = change[rows[:, None], first[rows, None] + np.arange(4)].sum(1)
    assert (initial[rows] < 0).all()
    return table.assign(episode=np.arange(len(first)), version=version, mode=mode, switches=count,
                        first=first, initial_m3=initial, hour_m3=within_hour)


def run(n, seed, out):
    torch.set_num_threads(1)
    choices.base.DeviceConfig().select_device(-1)
    env, older, episodes, policy, air, states = choices.load(n, seed)
    anchors, pairs = inspect(env, policy, air, states)
    selected = (pairs[pairs.favorable].sort_values(["episode", "t", "action"])
                .drop_duplicates("episode").set_index("episode"))
    start = np.full(len(episodes), -1)
    action = np.zeros(len(episodes))
    start[selected.index] = selected.t
    action[selected.index] = selected.action
    base_air, base_states, _, _ = rollout(env, episodes, policy, "base")
    np.testing.assert_array_equal(base_air, air)
    np.testing.assert_array_equal(base_states, states)
    daily = []
    for mode in ("repeat", "once", "exact", "oracle"):
        path = rollout(env, episodes, policy, mode, (start, action))
        daily.append(compare(path, (air, states), mode, "current"))
    if older is not None:
        env.dyn = older
        reference = rollout(env, episodes, policy, "base")[:2]
        path = rollout(env, episodes, policy, "repeat")
        daily.append(compare(path, reference, "repeat", "previous"))
    folder = out / f"p{n}_s{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    for name, table in (("anchors", anchors), ("pairs", pairs), ("daily", pd.concat(daily))):
        choices.save_table(folder, name, table, episodes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=int)
    parser.add_argument("seed", type=int)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    run(args.prefix, args.seed, args.out)

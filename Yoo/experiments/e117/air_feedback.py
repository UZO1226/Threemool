"""Replay the four frozen previous-air feedback conditions in DARROW."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e115"))
import replay
from darrow_env import AIR_MAX, air_to_action, action_to_air
from runpy import run_path

daily = run_path(str(HERE.parent / "e115/report.py"))["daily"]

ARMS = {"A": (False, False), "B": (True, False),
        "C": (True, True), "D": (False, True)}
METRICS = ["air", "nh", "do", "excess"]
PRIOR = replay.ROOT.parents[1] / "e115/main"


def window(trigger, t):
    return (trigger >= 0) & (trigger < t) & (t <= trigger + 4)


def trajectory(env, episodes, agents, arm, forced=None):
    pulse, damp = ARMS[arm]
    obs = env.reset(episodes)
    previous = obs[:, 0].copy()
    n = len(obs)
    trigger = np.full(n, -1)
    anchor = np.zeros(n)
    trace = np.empty((n, 95, 3))
    for t in range(95):
        native = replay.action(agents["weighted"], obs)
        other = replay.action(agents["uniform"], obs)
        flow = action_to_air(native)
        if forced is None:
            use = ((trigger < 0) & (t > 0) & (obs[:, 0] >= 3)
                   & (obs[:, 0] > previous) & (action_to_air(other) - flow >= 250))
        else:
            use = forced == t
        trigger[use] = t
        anchor[use] = flow[use]
        executed = native.astype(float)
        if pulse:
            executed[use] = other[use]
        active = window(trigger, t)
        if damp and active.any():
            alternative = obs.copy()
            alternative[active, 10] = anchor[active] / AIR_MAX
            alt_flow = action_to_air(replay.action(agents["weighted"], alternative))
            mixed = (flow[active] + alt_flow[active]) / 2
            executed[active] = air_to_action(mixed)
            np.testing.assert_allclose(action_to_air(executed[active]), mixed, rtol=0, atol=1e-9)
        previous = obs[:, 0].copy()
        obs, _, cost, done, info = env.step(executed)
        nh = env.state[:, 0]
        do = env.state[:, 2:].max(1)
        np.testing.assert_array_equal(cost, (nh > 4).astype(int) + (do > 2))
        np.testing.assert_allclose(obs[:, 10], info["air"] / AIR_MAX, atol=1e-7, rtol=0)
        trace[:, t] = np.column_stack((info["air"], nh, do))
    assert done and np.isfinite(trace).all()
    return trace, trigger


def compare_traces(actual, expected):
    np.testing.assert_allclose(actual[..., 0], expected[..., 0], rtol=0, atol=.05)
    np.testing.assert_allclose(actual[..., 1:], expected[..., 1:], rtol=0, atol=2e-6)
    np.testing.assert_array_equal(actual[..., 1:] > [4, 2], expected[..., 1:] > [4, 2])


def experiment(env, episodes, agents, forced=None):
    traces = {}
    for arm in ARMS:
        traces[arm], trigger = trajectory(env, episodes, agents, arm, forced)
        if arm == "A":
            first = trigger.copy()
        np.testing.assert_array_equal(trigger, first)
    time = np.arange(95)
    for arm, control in [("B", "A"), ("C", "B"), ("D", "A")]:
        same = (first[:, None] < 0) | (time < first[:, None])
        if arm in ("C", "D"):
            same |= time == first[:, None]
        np.testing.assert_array_equal(traces[arm][same], traces[control][same])
    return traces, first


def rows(episodes, seed, traces, trigger):
    data = []
    for arm, trace in traces.items():
        metrics = daily(*trace.transpose(2, 0, 1))
        for (day, member), when, values in zip(episodes, trigger, metrics):
            data.append(dict(day=day, member=member, seed=seed, arm=arm,
                             trigger=int(when), **dict(zip(METRICS, values))))
    return data


def run(out, smoke, sklearn):
    e = replay.e113
    plan = replay.read(replay.ROOT / "plan.json")
    for name, digest in plan["input_sha256"].items():
        replay.verify(e.prepared / name, digest)
    hashes = replay.read(replay.ROOT / "checksums.json")
    splits, days = e.base.load_selected(e.prepared / "data.pkl")
    dates = plan["winter"]
    assert splits["val"] == dates == sorted(dates) and len(dates) == 40
    e.base.DeviceConfig().select_device(-1)
    env = e.BatchEnv(days, (e.Dynamics if sklearn else e.FastDynamics)(e.prepared / "surrogate.pkl"))
    episodes = [(d, m) for d in dates for m in range(5)]
    if smoke:
        _, agents = replay.load(1, hashes)
        forced = np.array([-1, 0, 93, 94])
        traces, _ = experiment(env, episodes[:4], agents, forced)
        counts = sum(window(forced, t).astype(int) for t in range(95))
        np.testing.assert_array_equal(counts, [0, 4, 1, 0])
        np.testing.assert_array_equal(traces["C"][3], traces["B"][3])
        return
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for seed in range(5):
        archives, agents = replay.load(seed, hashes)
        traces, trigger = experiment(env, episodes, agents)
        archived = archives["weighted"]
        expected = np.stack((archived["air"], archived["states"][:, :, 0],
                             archived["states"][:, :, 2:].max(2)), axis=2)
        compare_traces(traces["A"], expected)
        with np.load(PRIOR / f"online{seed}.npz") as z:
            np.testing.assert_array_equal(trigger, z["trigger"])
            compare_traces(traces["B"], z["trace"][:, :, :3])
        np.savez_compressed(out / f"seed{seed}.npz", trigger=trigger, **traces)
        records.extend(rows(episodes, seed, traces, trigger))
        if seed == 1:
            focal = [("2021-12-06", m) for m in range(5)]
            selected, when = experiment(env, focal, agents, np.zeros(5, dtype=int))
            frame = pd.DataFrame(rows(focal, seed, selected, when))
            prior = pd.read_csv(HERE.parent / "e116/blocks.csv").query("pulse == 0 and block == 'day'")
            for arm in ("A", "B"):
                cols = [arm.lower() + "_" + suffix for suffix in ("air", "nh_steps", "do_steps", "excess")]
                np.testing.assert_allclose(frame[frame.arm == arm][METRICS],
                                           prior.sort_values("member")[cols], atol=.05, rtol=0)
            frame.to_csv(out / "focal.csv", index=False)
        print(f"seed {seed} archived controls verified", flush=True)
    pd.DataFrame(records).to_csv(out / "episodes.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sklearn", action="store_true")
    args = parser.parse_args()
    run(args.out, args.smoke, args.sklearn)

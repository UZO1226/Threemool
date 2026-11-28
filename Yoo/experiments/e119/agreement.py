"""Cross each e118 decision state with five fixed transition models."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e118"))
import timing as prev
from darrow_env import action_to_air, air_to_action

r = prev.r
KEY = ["day", "member", "seed", "lag"]
SOURCE = r.ROOT.parents[1] / "e118/main"


def prefix(env, episodes, flows, saved):
    obs = env.reset(episodes)
    states, zones = [], []
    for t in range(95):
        states.append(env.state.copy())
        zones.append(env.prev_zone.copy())
        obs, _, _, _, info = env.step(air_to_action(flows[:, t]))
        trace = np.column_stack((info["air"], env.state[:, 0], env.state[:, 2:].max(1)))
        prev.prior.compare_traces(trace, saved[:, t])
    return np.stack(states), np.stack(zones)


def branch(env, episodes, agent, ref, flows, states, zones, when, shift, cut):
    obs = env.reset(episodes)
    n = len(episodes)
    trace = np.empty((n, 95, 3))
    ids = ref.index.to_numpy()
    commanded = ref.proposal.to_numpy() - (ref.reduction.to_numpy() if cut else 0.)
    for t in range(95):
        # Replay the prefix on its original model. The first action is shared across scoring models.
        act = air_to_action(flows[:, t])
        at = when[ids] == t
        entering = ids[at]
        if len(entering):
            np.testing.assert_array_equal(env.state[entering], states[t, entering])
            np.testing.assert_array_equal(env.prev_zone[entering], zones[t, entering])
            np.testing.assert_allclose(obs[entering], ref.loc[entering, prev.INPUTS[:11]], atol=1e-7, rtol=0)
            np.testing.assert_allclose(env.prev_air[entering], flows[entering, t-1], atol=1e-8, rtol=0)
            np.testing.assert_allclose(ref.loc[entering, "proposal"], flows[entering, t], atol=.01, rtol=0)
            env.member[entering] = (env.member[entering] + shift) % 5
            act[entering] = air_to_action(commanded[at])
        following = (when >= 0) & (t > when)
        if following.any():
            act[following] = r.action(agent, obs)[following]
        obs, _, _, done, info = env.step(act)
        trace[:, t] = np.column_stack((info["air"], env.state[:, 0], env.state[:, 2:].max(1)))
    assert done and np.isfinite(trace).all()
    np.testing.assert_array_equal(env.member[ids], (ref.member.to_numpy() + shift) % 5)
    return trace


def run(out, smoke=False, original=False):
    e = r.e113
    plan = r.read(r.ROOT / "plan.json")
    for name, digest in plan["input_sha256"].items():
        r.verify(e.prepared / name, digest)
    hashes = r.read(r.ROOT / "checksums.json")
    splits, days = e.base.load_selected(e.prepared / "data.pkl")
    dates = plan["winter"]
    assert dates == splits["val"] == sorted(dates) and len(dates) == 40
    if smoke:
        dates = dates[:2]
    episodes = [(d, m) for d in dates for m in range(5)]
    positions = {dm: i for i, dm in enumerate(episodes)}
    samples = pd.read_csv(SOURCE / "winter_options.csv", float_precision="round_trip")
    assert len(samples) == 3054 and not samples.duplicated(KEY).any()
    samples = samples[samples.day.isin(dates)]
    e.base.DeviceConfig().select_device(-1)
    env = e.BatchEnv(days, (e.Dynamics if original else e.FastDynamics)(e.prepared / "surrogate.pkl"))
    out.mkdir(parents=True, exist_ok=False)
    for seed in ([0] if smoke else range(5)):
        agent = r.load(seed, hashes)[1]["weighted"]
        with np.load(r.ROOT.parents[1] / "e117/main" / f"seed{seed}.npz") as z:
            saved = z["B"][:len(episodes)]
            trigger = z["trigger"][:len(episodes)]
        flows = saved[:, :, 0]
        states, zones = prefix(env, episodes, flows, saved)
        pieces = []
        for lag in ([1] if smoke else range(1, 5)):
            ref = samples[(samples.seed == seed) & (samples.lag == lag)].copy()
            ref.index = [positions[(row.day, row.member)] for row in ref.itertuples()]
            ids = ref.index.to_numpy()
            when = np.full(len(episodes), -1)
            when[ids] = trigger[ids] + lag
            assert np.all(trigger[ids] > 0) and np.all(when[ids] < 95)
            for shift in ([0, 1] if smoke else range(5)):
                args = (env, episodes, agent, ref, flows, states, zones, when, shift)
                base = branch(*args, cut=False)
                cut = branch(*args, cut=True)
                before = (when[:, None] < 0) | (np.arange(95) < when[:, None])
                np.testing.assert_array_equal(cut[before], base[before])
                t = when[ids]
                np.testing.assert_allclose(base[ids, t, 0] - cut[ids, t, 0], ref.reduction, atol=1e-7, rtol=0)
                delta = prev.prior.daily(*cut[ids].transpose(2, 0, 1)) - prev.prior.daily(*base[ids].transpose(2, 0, 1))
                if shift == 0:
                    prev.prior.compare_traces(base, saved)
                    np.testing.assert_allclose(delta, ref[prev.METRICS], atol=1e-6, rtol=0)
                rows = ref[KEY + ["score", "good"]].rename(columns={"good": "source_good"}).copy()
                rows["model"] = (rows.member + shift) % 5
                rows["t"] = t
                rows["nh4_first"] = cut[ids, t, 1] - base[ids, t, 1]
                rows[prev.METRICS] = delta
                pieces.append(rows)
        pd.concat(pieces).to_csv(out / f"seed{seed}.csv", index=False)
        print("seed", seed, "completed", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--original", action="store_true")
    args = p.parse_args()
    run(args.out, args.smoke, args.original)

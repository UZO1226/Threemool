"""Replay equal-volume aeration transfers on frozen e135 command sequences."""
import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e135"))
import report
import study

old = study.old
SOURCE = old.source.parents[1] / "e135/main"
MAX_RATE = 562.5
HORIZON = 8
LAG = 4


def anchors(nh4, basic):
    joint = (nh4 > 4) & (basic > 4)
    edges = np.zeros_like(joint)
    edges[:, 1:] = joint[:, :-1] & (nh4[:, 1:] > 4) & (basic[:, 1:] <= 4)
    usable = edges.copy()
    usable[:, :LAG] = False
    usable[:, nh4.shape[1]-HORIZON+1:] = False
    starts = np.where(usable.any(axis=1), usable.argmax(axis=1), -1)
    return starts, edges.any(axis=1)


def transfer(air, starts, direction):
    shifted = air.copy()
    rows = np.flatnonzero(starts >= 0)
    later = starts[rows]
    earlier = later - LAG
    early, late = air[rows, earlier], air[rows, later]
    rates = np.minimum.reduce([np.full(len(rows), MAX_RATE), early, late,
                               old.AIR_MAX - early, old.AIR_MAX - late])
    shifted[rows, earlier] += direction * rates
    shifted[rows, later] -= direction * rates
    assert np.all((shifted >= 0) & (shifted <= old.AIR_MAX))
    np.testing.assert_allclose(shifted.sum(axis=1)/4, air.sum(axis=1)/4, atol=1e-8, rtol=0)
    return shifted, rates


def simulate(env, episodes, air):
    # Use the physical flow recorded by e135 directly to preserve every input bit.
    env.reset(episodes)
    state, zone_prev = env.state, env.prev_zone
    states = []
    for t, rate in enumerate(air.T):
        zone = rate[:, None] * env.x["frac"][:, t]
        state = env.dyn.step(state, env.x["exo"][:, t], zone, zone_prev, env.member)
        states.append(state)
        zone_prev = zone
    return np.stack(states, axis=1)


def outcomes(state):
    nh4 = state[:, :, 0]
    nh = (nh4 > 4).sum(axis=1)
    do = (state[:, :, 2:].max(axis=2) > 2).sum(axis=1)
    return np.column_stack([nh * 15, np.maximum(nh4-4, 0).sum(axis=1)/4, do * 15, nh+do])


def smoke():
    air = np.full((3, 16), 3000.)
    air[1, 0] = old.AIR_MAX
    coupled = np.full_like(air, 5)
    base = coupled.copy()
    base[0, 4:] = 4
    base[1, 4:] = 4
    base[2, 9:] = 4
    starts, edges = anchors(coupled, base)
    np.testing.assert_array_equal(starts, [4, 4, -1])
    assert edges.all()
    for direction in (-1, 1):
        plan, rates = transfer(air, starts, direction)
        np.testing.assert_array_equal(rates, [562.5, 0])
        expected = air.copy()
        expected[0, [0, 4]] += np.array([1,-1]) * direction * MAX_RATE
        np.testing.assert_array_equal(plan, expected)


def run(out):
    smoke()
    manifest = SOURCE / "SHA256SUMS"
    assert old.base.sha(manifest) == "3a5cbbbfdbca2ab487a1cc6a50138d31e4b6a2b063c3234e7b503e8782cc474c"
    subprocess.run(["sha256sum", "--quiet", "-c", str(manifest)], cwd=SOURCE, check=True)
    for filename, digest in old.trial.read(old.trial.REF / "manifest.json")["input_sha256"].items():
        assert old.base.sha(old.trial.PREP / filename) == digest
    old.setup()
    episodes = old.all_episodes(old.splits["val"], 5)
    assert len(episodes) == 200
    out.mkdir(parents=True, exist_ok=False)
    (out / "commit").write_bytes(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE))
    counts, events, daily = [], [], []
    for seed in report.SEEDS:
        source = SOURCE / f"seed_{seed}"
        with np.load(source / "combined_1000.npz") as ref, np.load(source / "base_1000.npz") as base:
            air, original = ref["air"], ref["states"]
            starts, edges = anchors(original[:, :, 0], base["states"][:, :, 0])
        rows = np.flatnonzero(starts >= 0)
        for name, direction in [("original", 0), ("early", 1), ("late", -1)]:
            plan, rates = transfer(air, starts, direction)
            states = simulate(old.env, episodes, plan)
            if name == "original":
                np.testing.assert_array_equal(states, original)
                counts.append(dict(seed=seed, paths=len(episodes), no_edge=int((~edges).sum()),
                                   boundary_only=int(edges.sum()-len(rows)), eligible=len(rows),
                                   zero=int((rates==0).sum()), moved=int((rates>0).sum())))
            same_prefix = (starts[:,None] < 0) | (np.arange(95) < starts[:,None]-LAG)
            np.testing.assert_array_equal(states[same_prefix], original[same_prefix])
            np.savez_compressed(out / f"{seed}_{name}.npz", states=states, air=plan)
            measures = np.column_stack([plan.sum(axis=1)/4, outcomes(states)])
            aggregated = measures.reshape(40,5,-1).mean(axis=1)
            if name == "original":
                np.testing.assert_allclose(aggregated, report.day_metrics(source/"combined_1000.npz"), atol=1e-9)
            for day, values in zip(old.splits["val"], aggregated):
                daily.append(dict(seed=seed, day=day, arm=name, **dict(zip(report.METRICS, values))))
            for i, rate in zip(rows, rates):
                k = starts[i]
                nh = states[i, k:k+HORIZON, 0]
                recovery = np.flatnonzero(nh <= 4)
                time = int(recovery[0]) if len(recovery) else HORIZON
                day, member = episodes[i]
                events.append(dict(seed=seed, day=day, member=member, k=k, arm=name, movable_m3=rate/4,
                                   recovered=int(time < HORIZON), restricted_min=time*15,
                                   excess_2h=np.maximum(nh-4,0).sum()/4))
        print(seed, counts[-1], flush=True)
    days = pd.DataFrame(daily)
    indexed = days.set_index(["arm","seed","day"])[report.METRICS]
    pairs, intervals = [], []
    for reference in ["original","late"]:
        diff = indexed.loc["early"].subtract(indexed.loc[reference]).sort_index()
        means = diff.groupby(level="seed").mean()
        def passed(v):
            return abs(v.air_m3)<=1e-8 and v.nh4_excess < -1e-8 and v.nh4_min<=1e-8 and v.do_min<=1e-8
        for seed, v in means.iterrows():
            pairs.append(dict(seed=seed, reference=reference, passed=passed(v), **v.to_dict()))
        lo, hi = report.block_interval(diff.to_numpy().reshape(5,40,-1))
        for m, x, l, h in zip(report.METRICS, means.mean(), lo, hi):
            intervals.append(dict(reference=reference,metric=m,mean=x,low=l,high=h))
    for name, frame in [("counts",counts), ("events",events), ("daily",days),
                        ("pairs",pairs), ("intervals",intervals)]:
        pd.DataFrame(frame).to_csv(out/f"{name}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    run(args.out)

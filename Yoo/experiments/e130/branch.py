"""Remove one archived increment and let both branches follow the same policy."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e127"))
import replicate as prior

p, e, root = prior.p, prior.e, prior.ROOT
KEYS = ["seed", "day", "member", "lag", "issued"]


class Branch(prior.late.LateCutEnv):
    def step(self, action):
        at = self.when == self.clock
        if at.any():
            assert (self.issued[at] >= 0).all() and (self.issued[at] < self.clock).all()
            self.observed[at] = self.current[at]
            self.proposed[at] = p.action_to_air(action[at])
            if self.remove:
                action = action.copy()
                action[at] = p.air_to_action(self.reference[at])
        return super().step(action)


def replay(seed, out):
    plan = p.r.read(p.r.ROOT / "plan.json")
    for name, digest in plan["input_sha256"].items():
        p.r.verify(e.prepared / name, digest)
    splits, days = e.base.load_selected(e.prepared / "data.pkl")
    assert splits["val"] == plan["winter"] and len(splits["val"]) == 40
    episodes = [(day, member) for day in splits["val"] for member in range(5)]
    index = {key: i for i, key in enumerate(episodes)}
    events = pd.read_csv(HERE.parent / "e129/cases.csv").query("seed == @seed")
    if seed < 5:
        frame = pd.read_csv(root / "e126/main/cases.csv").query("mode == 'matched' and seed == @seed")
        source, archive = p.r.ROOT, root / f"e126/main/traces{seed}.npz"
    else:
        source = root / "e127/main"
        frame = pd.read_csv(source / f"cases{seed}.csv").query("mode == 'matched'").reset_index(drop=True)
        frame["trace_index"] = frame.index
        archive = source / f"paired{seed}.npz"
    rows = events.merge(frame[[*KEYS, "first", "trace_index"]], on=KEYS, validate="one_to_one")
    assert len(rows) == len(events) and not rows.duplicated(["day", "member"]).any()
    assert (rows.extra_command_min % 15 == 0).all()
    rows["step"] = (rows.issued + rows.extra_command_min / 15).astype(int)
    with np.load(archive) as z:
        old, original = [(z[k] if seed < 5 else z[k][0])[rows.trace_index] for k in ("old", "new")]
    keep, remove = np.empty_like(original), np.empty_like(original)
    e.base.DeviceConfig().select_device(-1)
    agents = prior.load_pair(source, seed)
    env = Branch(days, e.FastDynamics(e.prepared / "surrogate.pkl"))
    for lag, group in rows.groupby("lag"):
        ix = group.index.to_numpy()
        ids = np.array([index[(row.day, row.member)] for row in group.itertuples()])
        t = group.step.to_numpy()
        assert ((t > group.issued) & (t < 95)).all()
        gap = original[ix, t, 0] - old[ix, t, 0]
        np.testing.assert_allclose(gap, group.gap, rtol=0, atol=1e-8)
        env.first = np.full(len(episodes), -1)
        env.first[ids] = group["first"]
        env.when = np.full(len(episodes), -1)
        env.when[ids] = t
        env.reference = np.zeros(len(episodes))
        env.reference[ids] = old[ix, t, 0]
        observations = []
        for env.remove, target in [(False, keep), (True, remove)]:
            env.observed = np.full((len(episodes), 11), np.nan)
            env.proposed = np.full(len(episodes), np.nan)
            arr, _, stamp, chosen = p.simulate(env, episodes, agents, slot=int(lag))
            target[ix] = arr[ids]
            np.testing.assert_array_equal(stamp[ids] + lag, group["first"])
            np.testing.assert_array_equal(chosen[ids], lag)
            np.testing.assert_array_equal(env.issued[ids], group.issued)
            np.testing.assert_allclose(env.proposed[ids], original[ix, t, 0], atol=.05, rtol=0)
            observations.append(env.observed[ids])
        np.testing.assert_array_equal(*observations)
        prefix = np.arange(95) < t[:, None]
        np.testing.assert_array_equal(keep[ix][prefix], remove[ix][prefix])
        np.testing.assert_allclose(remove[ix, t, 0] - keep[ix, t, 0], -gap, rtol=0, atol=1e-7)
    p.prior.compare_traces(keep, original)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"trace{seed}.npz", keep=keep, remove=remove)
    rows[[*KEYS, "seed_set", "period", "step", "gap"]].to_csv(out / f"keys{seed}.csv", index=False)
    print(f"seed {seed} verified {len(rows)} pairs", flush=True)


def measure(keep, remove):
    values = (p.prior.daily(*remove.T[:, None, :]) - p.prior.daily(*keep.T[:, None, :]))[0]
    deltas = dict(zip(prior.METRICS, values))
    a, b = keep[:, 1] > 4, remove[:, 1] > 4
    deltas.update(remove_only=int((b & ~a).sum()), keep_only=int((a & ~b).sum()))
    assert deltas["nh"] == deltas["remove_only"] - deltas["keep_only"]
    return deltas


def report(out, dest):
    entries = []
    for seed in range(10):
        rows = pd.read_csv(out / f"keys{seed}.csv")
        with np.load(out / f"trace{seed}.npz") as z:
            keep, remove = z["keep"], z["remove"]
        for i, row in rows.iterrows():
            t = int(row.step)
            for label, end in [("15min", t+1), ("60min", t+4), ("day", 95)]:
                if end > 95:
                    continue
                x = measure(keep[i, t:end], remove[i, t:end])
                x["later_air"] = x["air"] + row.gap / 4
                x["joint"] = x["air"] < -.05 and x["nh"] <= 0 and x["do"] <= 0 and x["excess"] <= 1e-8
                entries.append(dict(**row, window=label, **x))
    cases = pd.DataFrame(entries)
    summary = []
    for (batch, window), cohort in cases.groupby(["seed_set", "window"]):
        for period, part in [("all", cohort), *cohort.groupby("period")]:
            for seed, group in [("all", part), *part.groupby("seed")]:
                summary.append(dict(seed_set=batch, window=window, period=period, seed=seed, n=len(group),
                    **group[[*prior.METRICS, "later_air"]].mean().to_dict(),
                    **group[["remove_only", "keep_only", "joint"]].sum().to_dict(),
                    nh_worse=(group.nh > 0).sum(), nh_equal=(group.nh == 0).sum(), nh_better=(group.nh < 0).sum(),
                    air_saved=(group.air < -.05).sum()))
    dest.mkdir(parents=True, exist_ok=True)
    cases.to_csv(dest / "cases.csv", index=False)
    pd.DataFrame(summary).to_csv(dest / "summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--seed", type=int, choices=range(10))
    parser.add_argument("--tables", type=Path, default=HERE)
    args = parser.parse_args()
    if args.seed is None:
        report(args.out, args.tables)
    else:
        replay(args.seed, args.out)

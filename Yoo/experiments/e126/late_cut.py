"""Apply a single prospective late flow reduction to frozen e121 paths."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e118"))
import timing

E = timing.r.e113
ROOT = timing.r.ROOT.parents[1]
KEY = ["day", "member", "seed", "lag"]
METRICS = timing.METRICS


def eligible(t, first, issued, nh4, flow):
    return (first >= 0) & (t >= first + 24) & (issued < 0) & (nh4 <= 3.) & (flow >= 250.)


class LateCutEnv(E.BatchEnv):
    """Use the action passed by the policy, changing one interval per selected path."""

    def reset(self, episodes):
        self.clock = 0
        self.issued = np.full(len(episodes), -1)
        self.current = super().reset(episodes)
        return self.current

    def step(self, action):
        flow = timing.action_to_air(action)
        take = eligible(self.clock, self.first, self.issued, self.current[:, 0], flow)
        if take.any():
            action = action.copy()
            action[take] = timing.air_to_action(flow[take] - 250.)
            self.issued[take] = self.clock
        result = super().step(action)
        self.current = result[0]
        self.clock += 1
        return result


def replay(out, seed):
    plan = timing.r.read(timing.r.ROOT / "plan.json")
    for name, digest in plan["input_sha256"].items():
        timing.r.verify(E.prepared / name, digest)
    splits, days = E.base.load_selected(E.prepared / "data.pkl")
    assert plan["winter"] == splits["val"] == sorted(plan["winter"])
    episodes = [(day, member) for day in plan["winter"] for member in range(5)]
    index = {pair: i for i, pair in enumerate(episodes)}
    agents = timing.r.load(seed, timing.r.read(timing.r.ROOT / "checksums.json"))[1]
    dynamics = E.FastDynamics(E.prepared / "surrogate.pkl")
    native_env = E.BatchEnv(days, dynamics)
    late_env = LateCutEnv(days, dynamics)
    choices = pd.read_csv(ROOT / "e121/main/choices.csv").query("arm == 'risk_air' and seed == @seed")
    offsets = pd.read_csv(ROOT / "e122/main/cases.csv").query("seed == @seed")
    cases = offsets.merge(choices[["mode", *KEY, *METRICS]], on=["mode", *KEY], validate="one_to_one")
    assert len(cases) == len(offsets)
    with np.load(ROOT / "e117/main" / f"seed{seed}.npz") as data:
        boost, trigger = data["B"], data["trigger"]
    records, old_traces, new_traces = [], [], []
    for lag, rows in cases.drop_duplicates(KEY).groupby("lag"):
        original, _, old_stamp, old_slot = timing.simulate(native_env, episodes, agents, slot=lag)
        late_env.first = np.full(len(episodes), -1)
        ids = [index[(row.day, row.member)] for row in rows.itertuples()]
        late_env.first[ids] = rows.t
        updated, _, new_stamp, new_slot = timing.simulate(late_env, episodes, agents, slot=lag)
        np.testing.assert_array_equal(old_stamp, trigger)
        np.testing.assert_array_equal(new_stamp, old_stamp)
        np.testing.assert_array_equal(new_slot, old_slot)
        for row, i in zip(rows.itertuples(), ids):
            assert row.t == old_stamp[i] + lag and old_slot[i] == lag
            first = row.t
            old, new = original[i], updated[i]
            # For each decision, NH4 comes from the end of the preceding interval.
            candidates = np.flatnonzero((old[first+23:-1, 1].astype(np.float32) <= 3.)
                                         & (old[first+24:, 0] >= 250.))
            expected = int(first + 24 + candidates[0]) if len(candidates) else -1
            t = late_env.issued[i]
            assert t == expected
            stop = t if t >= 0 else len(old)
            np.testing.assert_array_equal(new[:stop], old[:stop])
            if t >= 0:
                np.testing.assert_allclose(new[t, 0] - old[t, 0], -250., atol=1e-7, rtol=0)
            daily = timing.prior.daily(*np.stack([boost[i], old, new]).transpose(2, 0, 1))
            np.testing.assert_allclose(daily[1] - daily[0], [getattr(row, k) for k in METRICS], atol=1e-6, rtol=0)
            record = {k: getattr(row, k) for k in KEY}
            record.update(first=first, issued=t, delay_min=(t-first)*15 if t >= 0 else np.nan,
                          initial_air=-62.5 if t >= 0 else 0., trace_index=len(records))
            for name, values in zip(["boost", "old", "new"], daily):
                record.update({f"{name}_{k}": v for k, v in zip(METRICS, values)})
            records.append(record)
            old_traces.append(old)
            new_traces.append(new)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"traces{seed}.npz", old=old_traces, new=new_traces)
    pd.DataFrame(records).to_csv(out / f"seed{seed}.csv", index=False)
    print("verified", seed, flush=True)


def summarize(out, dest):
    source = pd.read_csv(ROOT / "e122/main/cases.csv")
    records = pd.concat(pd.read_csv(out / f"seed{i}.csv") for i in range(5))
    assert len(records) == 254 and not records.duplicated(KEY).any()
    cases = source[["mode", "period", "group", *KEY]].merge(records, on=KEY, how="left", validate="many_to_one")
    assert len(cases) == 318 and cases.issued.notna().all()
    for k in METRICS:
        cases[k] = cases["new_" + k] - cases["old_" + k]
        cases["vs_boost_" + k] = cases["new_" + k] - cases["boost_" + k]
    cases["later_air"] = cases.air - cases.initial_air
    cases["fired"] = cases.issued >= 0
    cases["quality_worse"] = (cases.nh > 0) | (cases["do"] > 0) | (cases.excess > 1e-8)
    cases["joint"] = (cases.air < -.05) & ~cases.quality_worse
    cases.to_csv(out / "cases.csv", index=False)
    summaries = []
    means = [*METRICS, "initial_air", "later_air", *["vs_boost_" + k for k in METRICS]]
    for mode, cohort in cases.groupby("mode"):
        for period in ("all", "early", "late"):
            part = cohort if period == "all" else cohort[cohort.period == period]
            for seed in ("all", *range(5)):
                x = part if seed == "all" else part[part.seed == seed]
                groups = {"all": x, "fired": x[x.fired], "old_large": x[x.group == "large"],
                          "other": x[x.group != "large"]}
                for name, sub in groups.items():
                    if sub.empty:
                        continue
                    avg = sub[means].mean()
                    summaries.append(dict(mode=mode, period=period, seed=seed, cohort=name, n=len(sub),
                        fired=sub.fired.sum(), joint=sub.joint.sum(), quality_worse=sub.quality_worse.sum(),
                        nh_worse=(sub.nh > 0).sum(), air_saved=(sub.air < -.05).sum(),
                        candidate=avg.air < -.05 and avg.nh <= 0 and avg["do"] <= 0 and avg.excess <= 1e-8,
                        **avg.to_dict()))
    dest.mkdir(parents=True, exist_ok=True)
    perf = pd.DataFrame(summaries)
    perf.to_csv(dest / "summary.csv", index=False)
    halves = perf.query("mode == 'matched' and period != 'all' and seed == 'all' and cohort == 'all'")
    assert len(halves) == 2
    print("Candidate:", bool(halves.candidate.all()))


def smoke():
    first = np.array([0, 1, 0, 0, 0, -1])
    nh4 = np.array([3., 3., 3.01, 2., 1., 2.])
    air = np.array([250., 300., 250., 249.9, 300., 300.])
    issued = np.array([-1, -1, -1, -1, 23, -1])
    np.testing.assert_array_equal(eligible(24, first, issued, nh4, air), [1, 0, 0, 0, 0, 0])
    assert not eligible(23, first, issued, nh4, air).any()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "replay", "report"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--seed", type=int, choices=range(5))
    parser.add_argument("--tables", type=Path, default=HERE)
    args = parser.parse_args()
    if args.stage == "smoke":
        smoke()
    elif args.stage == "report":
        summarize(args.out, args.tables)
    else:
        if args.seed is None:
            parser.error("replay requires --seed")
        E.base.DeviceConfig().select_device(-1)
        replay(args.out, args.seed)

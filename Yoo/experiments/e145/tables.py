"""Aggregate paired episode differences and verify decisions from raw branches."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "e135"))
from report import block_interval, improves

METRICS = ["air_m3", "nh4_min", "excess", "do_min"]


def summarize(source, dest):
    days, outputs = [], []
    for prefix in (108, 148, 188):
        for seed in range(10, 15):
            folder = source / f"p{prefix}_s{seed}"
            anchors, pairs, daily = [pd.read_csv(folder / f"{name}.csv.gz")
                                     for name in ("anchors", "pairs", "daily")]
            assert len(anchors) == 4600 and not anchors.duplicated(["episode", "t"]).any()
            np.testing.assert_array_equal(pairs.favorable,
                                          (pairs.air_m3 < -1e-8) & (pairs[METRICS[1:]] <= 1e-8).all(axis=1))
            for key, sub in (("eligible", pairs), ("favorable", pairs[pairs.favorable]),
                             ("cheapest_favorable", pairs[pairs.cheapest & pairs.favorable])):
                expected = sub[["episode", "t"]].drop_duplicates().sort_values(["episode", "t"])
                actual = anchors.loc[anchors[key], ["episode", "t"]].sort_values(["episode", "t"])
                np.testing.assert_array_equal(expected.values, actual.values)
            for mode, flag in (("once", "eligible"), ("oracle", "favorable"), ("repeat", "eligible")):
                first = anchors[anchors[flag]].groupby("episode").t.min().reindex(range(200), fill_value=-1)
                check = daily[(daily["mode"] == mode) & (daily.version == "current")].sort_values("episode")
                np.testing.assert_array_equal(check["first"], first)
            oracle = daily[daily["mode"] == "oracle"].set_index("episode")
            selected = pairs[pairs.favorable].sort_values(["episode", "t", "action"]).drop_duplicates("episode")
            np.testing.assert_allclose(oracle.loc[selected.episode, "hour_m3"], selected.air_m3, atol=.005, rtol=0)
            assert daily[METRICS].notna().all().all()
            assert (daily.loc[daily.switches == 0, METRICS] == 0).all().all()
            assert (daily.groupby(["mode", "version"]).size() == 200).all()
            days.append(daily.assign(prefix=prefix, seed=seed))
            by_day = anchors.groupby(["day", "high"], as_index=False).agg(
                decisions=("t", "size"), eligible=("eligible", "sum"), favorable=("favorable", "sum"),
                cheapest_favorable=("cheapest_favorable", "sum"))
            outputs.append(by_day.assign(prefix=prefix, seed=seed))
    dest.mkdir(parents=True, exist_ok=True)
    denom = pd.concat(outputs)
    counts = ["decisions", "eligible", "favorable", "cheapest_favorable"]
    both = denom.groupby("prefix")[counts].sum().reset_index().assign(high="all")
    detail = denom.groupby(["prefix", "high"])[counts].sum().reset_index()
    pd.concat([both, detail]).to_csv(dest / "opportunities.csv", index=False)
    denom.to_csv(dest / "opportunities_by_day_seed.csv", index=False)
    daily = pd.concat(days, ignore_index=True)
    keys = ["prefix", "version", "mode"]
    seeds = daily.groupby(keys + ["seed"])[METRICS].mean().reset_index()
    seeds["joint"] = improves(seeds[METRICS].values)
    seeds.to_csv(dest / "seeds.csv", index=False)
    summary, intervals = [], []
    for key, group in daily.groupby(keys):
        hit = group[group.switches > 0]
        mean = group[METRICS].mean().values
        per_seed = group.groupby("seed")[METRICS].mean()
        result = dict(zip(keys, key))
        summary.append(dict(**result, **dict(zip(METRICS, mean)), seeds_joint=improves(per_seed.values).sum(),
                            mean_joint=improves(mean), episodes=len(group), changed=len(hit),
                            episodes_joint=improves(group[METRICS].values).sum(), switches=group.switches.sum(),
                            offset=(hit.air_m3 >= -1e-8).sum(), first_when_changed=hit.initial_m3.mean(),
                            hour_when_changed=hit.hour_m3.mean(), day_when_changed=hit.air_m3.mean()))
        if key[2] == "repeat":
            by_day = group.groupby(["seed", "day"])[METRICS].mean()
            assert by_day.shape == (200, 4)
            low, high = block_interval(by_day.values.reshape(5, 40, 4))
            for metric, lo, hi in zip(METRICS, low, high):
                intervals.append(dict(**result, metric=metric, low=lo, high=hi))
    pd.DataFrame(summary).to_csv(dest / "daily.csv", index=False)
    pd.DataFrame(intervals).to_csv(dest / "intervals.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("dest", type=Path)
    a = p.parse_args()
    summarize(a.source, a.dest)

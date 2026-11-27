"""Compare two frozen controllers at the risk-only controller's native water target."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from transfer import ARMS, CELLS, HERE, folds
from curves import minimum_air

METRICS = folds.report.METRICS
WATER = METRICS[1:4]


def choose(points, arm, target):
    block = points.loc[points.arm == arm].sort_values("scale")
    np.testing.assert_array_equal(block.scale.to_numpy()*1000, folds.report.SCALES)
    scale, _ = minimum_air(block[METRICS].to_numpy(), target)
    return scale


def native_target(table):
    native = table[(table.arm == "base_risk") & (table.scale == 1)]
    assert len(native) == 1
    return native.iloc[0][WATER].to_numpy(float)


def compare(raw, dest):
    source = pd.read_csv(HERE.parent / "e139/operating_points.csv")
    points, pairs = [], []
    for prefix, offset in CELLS:
        for seed in folds.report.SEEDS:
            key = dict(prefix=prefix, offset=offset, seed=seed)
            dev = source[(source.prefix == prefix) & (source.seed == seed)]
            records = []
            for arm in ARMS:
                for scale in folds.report.SCALES:
                    path = raw / f"p{prefix}_o{offset}_s{seed}/{arm}_{scale}.npz"
                    values = folds.report.day_metrics(path).mean(axis=0)
                    records.append(dict(**key, arm=arm, scale=scale/1000, **dict(zip(METRICS, values))))
            later = pd.DataFrame(records)
            points.extend(records)
            target = native_target(later)
            for mode, selection in [("fixed", dev), ("retrospective", later)]:
                row = dict(**key, mode=mode, **dict(zip(["target_"+m for m in WATER], target)))
                for arm in ARMS:
                    scale = choose(selection, arm, native_target(selection))
                    selected = later[(later.arm == arm) & (later.scale == scale)]
                    ok = len(selected) == 1 and (selected.iloc[0][WATER].to_numpy(float) <= target+1e-8).all()
                    row[arm+"_scale"] = scale
                    row[arm+"_air"] = selected.iloc[0].air_m3 if len(selected) else np.nan
                    row[arm+"_feasible"] = bool(ok)
                feasible = row["base_risk_feasible"] and row["combined_feasible"]
                row["paired"] = feasible
                row["delta_air"] = row["combined_air"]-row["base_risk_air"] if feasible else np.nan
                row["saved"] = feasible and row["delta_air"] < -.05
                pairs.append(row)
    pd.DataFrame(points).to_csv(dest / "operating_points.csv", index=False)
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(dest / "paired.csv", index=False)
    groups = pairs.groupby(["prefix", "offset", "mode"])
    summary = groups.agg(seeds=("seed", "size"), both_feasible=("paired", "sum"),
                         saved=("saved", "sum"), mean_paired_delta_air=("delta_air", "mean"))
    summary["passed"] = (summary.saved >= 4) & (summary.mean_paired_delta_air < -.05)
    summary.to_csv(dest / "summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raw", type=Path)
    ap.add_argument("dest", type=Path)
    args = ap.parse_args()
    compare(args.raw, args.dest)

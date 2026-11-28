"""Report paired recovery differences and check the event-level accounting."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(out, dest):
    counts = pd.read_csv(out / "counts.csv").set_index("seed")
    pairs = pd.read_csv(out / "pairs.csv")
    dates = pd.read_csv(out / "daily.csv")
    events = pd.read_csv(out / "events.csv")
    key = ["seed", "day", "member"]
    by_arm = events.set_index(["arm", *key]).sort_index()
    original = by_arm.loc["original"]
    assert not original.index.has_duplicates
    rows = []
    for arm in ("original", "early", "late"):
        seen = by_arm.loc[arm]
        pd.testing.assert_frame_equal(original[["k", "movable_m3"]], seen[["k", "movable_m3"]])
        diff = seen.restricted_min - original.restricted_min
        summary = seen.assign(faster=diff<0, same=diff==0, slower=diff>0)
        for seed, grp in summary.groupby(level="seed"):
            assert len(grp) == counts.loc[seed, "eligible"]
            rows.append(dict(arm=arm, seed=seed, events=len(grp), moved_m3=grp.movable_m3.mean(),
                             recovered=grp.recovered.sum(), restricted_min=grp.restricted_min.mean(),
                             excess_2h=grp.excess_2h.mean(), faster=grp.faster.sum(),
                             same=grp.same.sum(), slower=grp.slower.sum()))
    outcomes = ["air_m3","nh4_min","nh4_excess","do_min","cost"]
    means = dates.groupby("arm")[outcomes].mean()
    for ref in ("original","late"):
        np.testing.assert_allclose(means.loc["early"]-means.loc[ref],
                                   pairs.query("reference==@ref")[outcomes].mean(), atol=1e-8)
    dest.mkdir(parents=True, exist_ok=True)
    # Post hoc: locate the daily excess difference relative to the transfer.
    phases = []
    for seed in counts.index:
        with np.load(out / f"{seed}_original.npz") as orig, np.load(out / f"{seed}_early.npz") as early:
            delta = (np.maximum(early["states"][:, :, 0]-4, 0)
                     - np.maximum(orig["states"][:, :, 0]-4, 0)) / 4
        days = dates.query("seed==@seed and arm=='original'").day.tolist()
        totals = np.zeros(3)
        for (day, member), event in original.loc[seed].iterrows():
            k = int(event.k)
            row = delta[5*days.index(day)+member]
            totals += [row[k-4:k].sum(), row[k:k+8].sum(), row[k+8:].sum()]
        totals /= len(delta)
        np.testing.assert_allclose(totals.sum(), delta.sum(axis=1).mean(), atol=1e-10)
        phases.append(dict(seed=seed, prior_hour=totals[0], following_2h=totals[1], rest=totals[2]))
    pd.DataFrame(phases).to_csv(dest / "phases.csv", index=False)
    pd.DataFrame(rows).to_csv(dest / "recovery.csv", index=False)
    means.to_csv(dest / "means.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("dest", type=Path)
    args = parser.parse_args()
    summarize(args.out, args.dest)

"""Independently audit eligibility and raw policy outcomes for all new seeds."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def verify(out, tables):
    records = []
    for seed in range(5, 10):
        frame = pd.read_csv(out / f"cases{seed}.csv")
        with np.load(out / f"paired{seed}.npz") as traces:
            original, changed = traces["old"], traces["new"]
        assert original.shape == changed.shape == (2, 200, 95, 3)
        assert np.isfinite(original).all() and np.isfinite(changed).all()
        for m, mode in enumerate(("matched", "frozen")):
            rows = frame[frame["mode"] == mode]
            keys = list(zip(rows.day, rows.member))
            assert keys == [(d, i) for d in sorted(set(rows.day)) for i in range(5)]
            for i, row in enumerate(rows.itertuples()):
                old, new = original[m, i], changed[m, i]
                first = row.first
                if first < 0:
                    assert row.lag == 0
                    fired = -1
                else:
                    assert 1 <= row.lag <= 4
                    fired = next((j for j in range(first + 24, 95)
                                  if np.float32(old[j-1, 1]) <= 3 and old[j, 0] >= 250), -1)
                assert fired == row.issued
                if fired < 0:
                    np.testing.assert_array_equal(old, new)
                else:
                    np.testing.assert_array_equal(old[:fired], new[:fired])
                    np.testing.assert_allclose(old[fired, 0] - new[fired, 0], 250, atol=1e-7, rtol=0)
                diff = np.array([
                    .25 * np.sum(new[:, 0] - old[:, 0]),
                    np.count_nonzero(new[:, 1] > 4) - np.count_nonzero(old[:, 1] > 4),
                    np.count_nonzero(new[:, 2] > 2) - np.count_nonzero(old[:, 2] > 2),
                    .25 * sum(max(n - 4, 0) - max(o - 4, 0) for n, o in zip(new[:, 1], old[:, 1])),
                ])
                np.testing.assert_allclose(diff, [row.air, row.nh, row.do, row.excess], atol=1e-7, rtol=0)
                records.append((mode, row.period, seed, first, fired, *diff))
    raw = pd.DataFrame(records, columns=["mode", "period", "seed", "first", "issued", "air", "nh", "do", "excess"])
    for row in pd.read_csv(tables / "summary.csv").itertuples():
        sub = raw[raw["mode"] == row.mode]
        if row.period != "all":
            sub = sub[sub.period == row.period]
        if row.seed != "all":
            sub = sub[sub.seed == int(row.seed)]
        if row.cohort == "selected":
            sub = sub[sub["first"] >= 0]
        if row.cohort == "fired":
            sub = sub[sub.issued >= 0]
        assert len(sub) == row.n and (sub.issued >= 0).sum() == row.fired
        np.testing.assert_allclose(sub[["air", "nh", "do", "excess"]].mean(),
                                   [row.air, row.nh, row.do, row.excess], atol=1e-7, rtol=0)
        harm = (sub.nh > 0) | (sub["do"] > 0) | (sub.excess > 1e-8)
        assert harm.sum() == row.harmed and (~harm & (sub.air < -.05)).sum() == row.jointly_better
    print("Verified 1,000 paired episodes per mode and all reported subgroup differences.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--tables", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    verify(args.out, args.tables)

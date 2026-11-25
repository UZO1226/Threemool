"""Rebuild both water-quality trajectories with the original sklearn predictor."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from branch import HERE, KEYS, e, p


def verify(out, tables):
    cases = pd.read_csv(tables / "cases.csv")
    events = pd.read_csv(HERE.parent / "e129/cases.csv")
    daily = cases[cases.window == "day"]
    assert daily.groupby("seed_set").size().to_dict() == {"0-4": 82, "5-9": 49}
    assert set(map(tuple, daily[KEYS].values)) == set(map(tuple, events[KEYS].values))
    _, days = e.base.load_selected(e.prepared / "data.pkl")
    env = e.BatchEnv(days, e.Dynamics(e.prepared / "surrogate.pkl"))
    rows, keep, remove = [], [], []
    for seed in range(10):
        rows.append(pd.read_csv(out / f"keys{seed}.csv"))
        with np.load(out / f"trace{seed}.npz") as z:
            keep.append(z["keep"])
            remove.append(z["remove"])
    frame = pd.concat(rows, ignore_index=True)
    full = np.concatenate(keep + remove)
    env.reset(list(zip(frame.day, frame.member)) * 2)
    for t in range(95):
        _, _, _, done, info = env.step(p.air_to_action(full[:, t, 0]))
        reference = full[:, t, 1:]
        actual = np.column_stack((env.state[:, 0], env.state[:, 2:].max(1)))
        np.testing.assert_allclose(actual, reference, rtol=0, atol=2e-6)
        np.testing.assert_array_equal(info["v_nh4"], reference[:, 0] > 4)
        np.testing.assert_array_equal(info["v_do"], reference[:, 1] > 2)
    assert done
    a, b = np.split(full, 2)
    for label, size in [("15min", 1), ("60min", 4), ("day", 95)]:
        idx = np.arange(95)
        valid = np.ones(len(frame), bool) if label == "day" else frame.step.to_numpy() + size <= 95
        mask = (idx >= frame.step.to_numpy()[:, None]) & (idx < frame.step.to_numpy()[:, None] + size)
        checked = frame[valid].merge(cases[cases.window == label], on=KEYS, validate="one_to_one")
        assert len(checked) == sum(valid) == sum(cases.window == label)
        delta = lambda arr: (arr * mask).sum(axis=1)[valid]
        np.testing.assert_allclose(checked.air, delta((b[:, :, 0]-a[:, :, 0]) / 4), rtol=0, atol=1e-8)
        for j, limit, metric in [(1, 4, "nh"), (2, 2, "do")]:
            diff = (b[:, :, j] > limit).astype(int) - (a[:, :, j] > limit)
            np.testing.assert_array_equal(checked[metric], delta(diff))
        np.testing.assert_allclose(checked.excess, delta((np.maximum(b[:, :, 1]-4, 0)
                                                     - np.maximum(a[:, :, 1]-4, 0)) / 4), rtol=0, atol=1e-8)
    print(f"Verified {len(full)} original sklearn trajectories and {len(cases)} paired windows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--tables", type=Path, default=HERE)
    args = parser.parse_args()
    verify(args.out, args.tables)

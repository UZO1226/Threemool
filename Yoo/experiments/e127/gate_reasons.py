"""Post hoc accounting of why the matched gate did not fire in the two seed sets."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def report(prior, current, dest):
    archive = pd.read_csv(prior / "cases.csv")
    records = []
    for seed in range(10):
        if seed < 5:
            frame = archive.query("mode == 'matched' and seed == @seed")
            ids = frame.trace_index
            with np.load(prior / f"traces{seed}.npz") as traces:
                old = traces["old"]
        else:
            all_cases = pd.read_csv(current / f"cases{seed}.csv")
            frame = all_cases[all_cases["mode"] == "matched"].reset_index(drop=True).query("lag > 0")
            ids = frame.index
            with np.load(current / f"paired{seed}.npz") as traces:
                old = traces["old"][0]
        for r, i in zip(frame.itertuples(), ids):
            start = r.first + 24
            nh = old[i, start-1:-1, 1].astype(np.float32)
            flow = old[i, start:, 0]
            assert len(nh) == len(flow)
            if r.issued >= 0:
                reason = "fired"
            elif not len(nh):
                reason = "day_ended"
            elif (nh > 3.).all():
                reason = "nh_never_at_or_below_3"
            else:
                assert not ((nh <= 3.) & (flow >= 250.)).any()
                reason = "insufficient_flow_at_low_nh"
            records.append(dict(seed_set="0-4" if seed < 5 else "5-9", reason=reason, air=r.air))
    frame = pd.DataFrame(records)
    counts = frame.groupby(["seed_set", "reason"]).air.agg(n="size", median_air="median")
    counts.to_csv(dest / "coverage.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prior", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--tables", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    report(args.prior, args.current, args.tables)

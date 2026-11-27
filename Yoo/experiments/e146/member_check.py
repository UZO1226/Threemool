"""Count conflicting simulated rebound labels at exactly identical decision inputs."""
import argparse
from pathlib import Path

import pandas as pd

from paths import FEATURES


def compare(source, dest):
    results = []
    for prefix in (108, 148, 188):
        frame = pd.concat([pd.read_csv(source / f"p{prefix}_s{s}.csv", float_precision="round_trip")
                           for s in range(10, 15)])
        frame = frame.loc[frame.eligible].copy()
        frame["rebound"] = (frame[[f"air_{h}h" for h in range(2, 7)]] >= -1e-8).any(axis=1)
        groups = frame.groupby(["day", "seed", "first"] + FEATURES).rebound.agg(["size", "sum"])
        multi = groups[groups["size"] > 1]
        mixed = multi[(multi["sum"] > 0) & (multi["sum"] < multi["size"])]
        results.append(dict(prefix=prefix, eligible=len(frame), repeated_groups=len(multi),
                            repeated_episodes=multi["size"].sum(), mixed_groups=len(mixed),
                            mixed_episodes=mixed["size"].sum(),
                            inevitable_errors=mixed[["sum"]].assign(negative=mixed["size"]-mixed["sum"]).min(axis=1).sum()))
    pd.DataFrame(results).to_csv(dest, index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("dest", type=Path)
    a = p.parse_args()
    compare(a.source, a.dest)

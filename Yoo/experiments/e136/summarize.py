"""Decompose prior NH4 exposure and verify the intervention endpoints."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e135"))
import report as prev

SRC = Path("/home/jio/fcsrl_src/results/e135/main")
METRICS = prev.METRICS


def exposure(a, b):
    # Combined minus baseline, split into disjoint time intervals.
    ea, eb = np.maximum(a-4, 0), np.maximum(b-4, 0)
    shared = (a > 4) & (b > 4)
    terms = [np.where(shared, ea-eb, 0), np.where(b <= 4, ea, 0), np.where(a <= 4, -eb, 0)]
    np.testing.assert_allclose(sum(terms), ea-eb, atol=1e-12, rtol=0)
    return [float(x.sum(axis=1).mean()/4) for x in terms]


def diagnose():
    result = []
    for s in prev.SEEDS:
        d = SRC / f"seed_{s}"
        with np.load(d / "combined_1000.npz") as a, np.load(d / "base_1000.npz") as b:
            combined, base = a["states"][:, :, 0], b["states"][:, :, 0]
            for t in range(0, 95, 24):
                sl = slice(t, t+24)
                common, only_a, only_b = exposure(combined[:, sl], base[:, sl])
                result.append(dict(seed=s, start_hour=t/4, end_hour=min(t+24,95)/4,
                                   both=common, combined_only=only_a, base_only=only_b,
                                   total=common+only_a+only_b))
    frame = pd.DataFrame(result)
    expected = pd.read_csv(HERE.parent / "e135/paired.csv").query("control=='base'").set_index("seed")
    np.testing.assert_allclose(frame.groupby("seed").total.sum(), expected.nh4_excess, atol=1e-8)
    return frame


def direct(delta):
    return ((delta[...,0] <= .05) & (delta[...,1] <= 1e-8) & (delta[...,2] < -1e-8)
            & (delta[...,3] <= 1e-8))


def summarize(out, dest):
    dest.mkdir(parents=True, exist_ok=True)
    diagnose().to_csv(dest / "exposure.csv", index=False)
    daily, counts, runs = {}, [], []
    for s in prev.SEEDS:
        d = out / f"seed_{s}"
        rec = json.loads((d / "counts.json").read_text())
        for arm in ("base", "original", "severity", "max_air"):
            if arm in ("base","original"):
                name = "base" if arm=="base" else "combined"
                path = SRC / d.name / f"{name}_1000.npz"
            else:
                path = d / f"{arm}.npz"
            v = prev.day_metrics(path)
            daily[s,arm] = v
            if arm in rec:
                counts.append(dict(seed=s,arm=arm,**rec[arm]))
            runs.append(dict(seed=s,arm=arm,within12=v[:,4].mean()<=12,**dict(zip(METRICS,v.mean(axis=0)))))
    pairs, intervals, decision = [], [], {}
    for ref in ("base","original","max_air"):
        fn = prev.improves if ref=="base" else direct
        diffs = np.stack([daily[s,"severity"] - daily[s,ref] for s in prev.SEEDS])
        low, high = prev.block_interval(diffs)
        ds = diffs.mean(axis=1)
        for s, diff in zip(prev.SEEDS, ds):
            pairs.append(dict(seed=s,reference=ref,passed=bool(fn(diff)),**dict(zip(METRICS,diff))))
        for k,m in enumerate(METRICS):
            intervals.append(dict(reference=ref,metric=m,mean=ds[:,k].mean(),low=low[k],high=high[k]))
        wins = int(fn(ds).sum())
        decision[ref] = dict(passed_seeds=wins, mean_pass=bool(fn(ds.mean(axis=0))),
                             candidate=bool(wins>=4 and fn(ds.mean(axis=0))))
    for name, rows in [("runs",runs),("counts",counts),("pairs",pairs),("intervals",intervals)]:
        pd.DataFrame(rows).to_csv(dest / f"{name}.csv", index=False)
    (dest / "decision.json").write_text(json.dumps(decision, indent=2)+"\n")
    print(pd.DataFrame(runs).groupby("arm")[METRICS].mean().round(4))
    print(decision)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("dest", type=Path)
    args = parser.parse_args()
    summarize(args.out, args.dest)

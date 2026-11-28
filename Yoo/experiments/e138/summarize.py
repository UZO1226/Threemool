"""Estimate the contribution of timing and subsequent feedback on matched days."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"e135"))
import report

FIXED = Path("/home/jio/fcsrl_src/results/e137/main")
CONDITIONS = [
    ("original", FIXED, "original"),
    ("early_fixed", FIXED, "early"),
    ("early_feedback", None, "early"),
    ("late_fixed", FIXED, "late"),
    ("late_feedback", None, "late"),
]


def passes(d):
    return bool(d[0]<=.05 and d[1]<=1e-8 and d[2]<-1e-8 and d[3]<=1e-8)


def summarize(out, dest):
    dest.mkdir(parents=True,exist_ok=True)
    daily = {}
    means, pairs, intervals = [], [], []
    for name, source, arm in CONDITIONS:
        d = np.stack([report.day_metrics((source or out)/f"{seed}_{arm}.npz") for seed in report.SEEDS])
        assert d.shape == (5,40,5)
        daily[name] = d
        means.append(dict(arm=name,**dict(zip(report.METRICS,d.mean(axis=(0,1))))))
    for arm in ("early","late"):
        feedback = daily[f"{arm}_feedback"]
        fixed = daily[f"{arm}_fixed"]
        primary = feedback - daily["original"]
        gain = feedback - fixed
        np.testing.assert_allclose(primary, fixed-daily["original"]+gain, rtol=0,atol=1e-10)
        for reference, values in [("original",primary),("fixed",gain)]:
            lo,hi = report.block_interval(values)
            ds = values.mean(axis=1)
            for seed, delta in zip(report.SEEDS,ds):
                pairs.append(dict(arm=arm,reference=reference,seed=seed,passed=passes(delta),
                                  **dict(zip(report.METRICS,delta))))
            for metric, delta, lower, upper in zip(report.METRICS,ds.mean(axis=0),lo,hi):
                intervals.append(dict(arm=arm,reference=reference,metric=metric,mean=delta,low=lower,high=upper))
    events = pd.read_csv(out/"events.csv")
    events["faster"] = events.feedback_min < events.fixed_min
    events["slower"] = events.feedback_min > events.fixed_min
    events["fixed_recovered"] = events.fixed_min < 120
    events["feedback_recovered"] = events.feedback_min < 120
    stats = []
    for (seed,arm),grp in events.groupby(["seed","arm"]):
        daygap = daily[f"{arm}_feedback"] - daily[f"{arm}_fixed"] if arm!="original" else np.zeros_like(daily[arm])
        np.testing.assert_allclose(grp.net.sum()/200, daygap[report.SEEDS.index(seed),:,0].mean(),atol=1e-8)
        stats.append(dict(seed=seed,arm=arm,events=len(grp),
                          **grp[["added","removed","net","fixed_min","feedback_min"]].mean().to_dict(),
                          **grp[["faster","slower","fixed_recovered","feedback_recovered"]].sum().to_dict()))
    for name,records in [("means",means),("pairs",pairs),("intervals",intervals),("recovery",stats)]:
        pd.DataFrame(records).to_csv(dest/f"{name}.csv",index=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    parser.add_argument("dest",type=Path)
    args=parser.parse_args()
    summarize(args.out,args.dest)

"""Resource-matched water outcomes and discrete water-constrained air minima."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e135"))
import report

ARMS = ["base", "actor", "base_risk", "combined"]
SCALES = report.SCALES
NATIVE = SCALES.index(1000)
PAIRS = [("combined","base"), ("combined","base_risk"), ("combined","actor"),
         ("base_risk","base"), ("actor","base")]
WATER = report.METRICS[1:]
WINTER = Path("/home/jio/fcsrl_src/results/e135/main")


def overlap(candidate, control):
    """Same overlap and 21-point mean as the e135 reference, including duplicate flows."""
    curves = []
    for points in (candidate, control):
        points = points[np.argsort(points[:,0])]
        flow = points[:,0]
        if (np.diff(flow)==0).any():
            flow, inverse, counts = np.unique(flow,return_inverse=True,return_counts=True)
            water = np.zeros((len(flow),len(WATER)))
            np.add.at(water,inverse,points[:,1:])
            points = np.column_stack((flow,water/counts[:,None]))
        curves.append(points)
    a,b = curves
    low,high = max(a[0,0],b[0,0]), min(a[-1,0],b[-1,0])
    if high <= low:
        return low,high,np.full(len(WATER),np.nan)
    grid = np.linspace(low,high,21)
    deltas = [np.mean(np.interp(grid,a[:,0],a[:,j])-np.interp(grid,b[:,0],b[:,j]))
              for j in range(1,len(report.METRICS))]
    return low,high,np.array(deltas)


def minimum_air(points, target):
    valid = np.flatnonzero(np.all(points[:,1:4] <= target+1e-8,axis=1))
    if not len(valid):
        return np.nan,np.nan
    index = valid[np.argmin(points[valid,0])]
    return SCALES[index]/1000,points[index,0]


def passes(v):
    return bool(np.isfinite(v).all() and v[0]<-1e-8 and v[1]<-1e-8 and v[2]<=1e-8)


def bootstrap(a,b):
    # Seed is the first axis. The same dates are resampled for every seed, scale and arm.
    rng = np.random.default_rng(135)
    samples = []
    for _ in range(2000):
        seeds = rng.integers(5,size=5)
        days = (rng.integers(40,size=(8,1))+np.arange(5))%40
        weights = np.bincount(days.ravel(),minlength=40)/40
        pts_a,pts_b = [np.einsum("sadk,d->sak",v,weights) for v in (a,b)]
        gaps = np.stack([overlap(pts_a[s],pts_b[s])[2] for s in seeds])
        if np.isfinite(gaps).all():
            samples.append(gaps.mean(axis=0))
    bounds = np.quantile(samples,[.025,.975],axis=0) if samples else np.full((2,len(WATER)),np.nan)
    return len(samples), bounds


def tables(root, dest, prefixes):
    rows, paired, limits, intervals, decisions = [],[],[],[],[]
    for prefix in prefixes:
        directory = WINTER if prefix==188 else root/f"prefix{prefix}/evaluated"
        daily = {arm: np.array([[report.day_metrics(directory/f"seed_{s}/{arm}_{scale}.npz")
                                for scale in SCALES] for s in report.SEEDS]) for arm in ARMS}
        means = {arm: values.mean(axis=2) for arm,values in daily.items()}
        for i,seed in enumerate(report.SEEDS):
            target = means["base"][i,NATIVE,1:4]
            for arm in ARMS:
                points = means[arm][i]
                for scale,point in zip(SCALES,points):
                    rows.append(dict(prefix=prefix,arm=arm,seed=seed,scale=scale/1000,
                                     **dict(zip(report.METRICS,point))))
                scale,air = minimum_air(points,target)
                limits.append(dict(prefix=prefix,seed=seed,arm=arm,min_air=air,scale=scale,
                                   target_nh4_min=target[0],target_excess=target[1],target_do_min=target[2]))
        for arm,ref in PAIRS:
            gaps = []
            for i,seed in enumerate(report.SEEDS):
                a,b = means[arm][i],means[ref][i]
                low,high,gap = overlap(a,b)
                ref_low,ref_high,ref_gap = report.curve_gap(a,b)
                np.testing.assert_allclose([low,high],[ref_low,ref_high])
                np.testing.assert_allclose(gap,ref_gap,rtol=0,atol=1e-10,equal_nan=True)
                paired.append(dict(prefix=prefix,arm=arm,reference=ref,seed=seed,lo_air=low,
                                   hi_air=high,passed=passes(gap),**dict(zip(WATER,gap))))
                gaps.append(gap)
            gap_mean = np.mean(gaps,axis=0)
            wins = sum(passes(g) for g in gaps)
            decisions.append(dict(prefix=prefix,arm=arm,reference=ref,eligible=sum(np.isfinite(g).all() for g in gaps),
                                  passed_seeds=wins,mean_pass=passes(gap_mean),
                                  passed=wins>=4 and passes(gap_mean),**dict(zip(WATER,gap_mean))))
            values = daily[arm][:,NATIVE] - daily[ref][:,NATIVE]
            low,high = report.block_interval(values)
            for metric,mean,l,h in zip(report.METRICS,values.mean(axis=(0,1)),low,high):
                intervals.append(dict(prefix=prefix,arm=arm,reference=ref,endpoint="native",
                                      metric=metric,mean=mean,low=l,high=h,replicates=2000))
            n,(low,high) = bootstrap(daily[arm],daily[ref])
            for metric,mean,l,h in zip(WATER,gap_mean,low,high):
                intervals.append(dict(prefix=prefix,arm=arm,reference=ref,endpoint="matched",
                                      metric=metric,mean=mean,low=l,high=h,replicates=n))
    dest.mkdir(exist_ok=True,parents=True)
    for name,records in [("operating_points",rows),("paired_common_air",paired),("quality_target",limits),
                         ("intervals",intervals),("decision",decisions)]:
        pd.DataFrame(records).to_csv(dest/f"{name}.csv",index=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("dest",type=Path)
    p.add_argument("prefixes",nargs="*",type=int,default=[108,148,188])
    args=p.parse_args()
    tables(args.root,args.dest,args.prefixes)

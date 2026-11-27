"""Count unique decision opportunities and summarize single-switch daily effects."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["prefix","seed","episode","t"]
GROUP = ["prefix","seed","day"]
METRICS = ["air_m3","nh4_min","excess","do_min"]
FLAGS = ["near","different","better","worse","both_versions","means_close_different","means_close_better"]


def load(root, name):
    frames = []
    for prefix in (108,148,188):
        for seed in range(10,15):
            table = pd.read_csv(root/f"p{prefix}_s{seed}/{name}.csv.gz").assign(prefix=prefix,seed=seed)
            frames.append(table)
    return pd.concat(frames,ignore_index=True)


def rates(anchor, pairs, tol):
    sub = pairs[pairs.gap<=tol+1e-12].copy()
    paired = sub.different>0
    close = (sub.nh_gap_min.abs()<=3)&(sub.do_gap_min.abs()<=3)
    sub = sub.assign(near=True,different=paired,better=sub.net>=4,worse=sub.net<=-4,
                     both_versions=(sub.net>=4)&(sub.old_net>=4),means_close_different=paired&close,
                     means_close_better=(sub.net>=4)&close)
    flags = sub.groupby(KEY)[FLAGS].any()
    a = anchor.join(flags,on=KEY,validate="one_to_one")
    a[FLAGS] = a[FLAGS].eq(True)
    assert (a.high | ~a.near).all()
    dates = a.groupby(GROUP)[["high",*FLAGS]].sum().reset_index()
    dates["anchors"] = a.groupby(GROUP).size().to_numpy()
    return dates.assign(tol=tol)


def main(root, dest, actor_root=None):
    anchors, pairs, effects = (load(root,k) for k in ("anchors","pairs","daily"))
    assert len(anchors)==69000 and not anchors.duplicated(KEY).any()
    assert len(effects)==3000 and not effects.duplicated(KEY[:3]).any()
    assert not pairs.duplicated(KEY+["alternative"]).any()
    all_dates = pd.concat([rates(anchors,pairs,t) for t in (0,.025,.05)],ignore_index=True)
    cols = ["anchors","high",*FLAGS]
    summary = all_dates.groupby(["prefix","tol"])[cols].sum()
    bydate = all_dates.groupby(["prefix","tol","day"])[FLAGS].sum()
    for col in FLAGS:
        summary[col+"_days"] = (bydate[col]>0).groupby(["prefix","tol"]).sum()
    for col in ("both_versions","both_versions_days"):
        summary.loc[(108,slice(None)),col] = np.nan
        all_dates.loc[all_dates.prefix==108,"both_versions"] = np.nan
    all_dates.to_csv(dest/"by_seed_day.csv",index=False)
    summary.to_csv(dest/"opportunities.csv")
    selected = effects.start>=0
    strong = pairs[(pairs.gap<=.025+1e-12)&(pairs.net>=4)]
    scheduled = strong.groupby(KEY[:3]).t.min().rename("expected")
    check = effects.join(scheduled,on=KEY[:3])
    np.testing.assert_array_equal(check.expected.fillna(-1), check.start)
    assert (effects.loc[~selected,METRICS]==0).all(axis=None)
    effects["switched"]=selected
    seed = effects.groupby(["prefix","seed"])[METRICS+["first_air_delta_m3","switched"]].mean()
    seed["joint"] = (seed.air_m3 < -.05)&(seed[METRICS[1:]]<=1e-8).all(axis=1)
    seed.to_csv(dest/"daily_by_seed.csv")
    daily = effects.groupby("prefix")[METRICS+["first_air_delta_m3"]].mean()
    daily["switched_episodes"]=effects.groupby("prefix").switched.sum()
    daily["joint_seeds"]=seed.groupby("prefix").joint.sum()
    daily["switched_dates"]=effects.loc[selected].groupby("prefix").day.nunique().reindex(daily.index,fill_value=0)
    daily.to_csv(dest/"daily_effect.csv")
    print(summary.loc[(slice(None),.025),cols+["different_days","better_days","both_versions_days"]].to_string())
    print(daily.to_string())
    if actor_root is not None:
        actor_anchors = load(actor_root,"anchors")
        actor_pairs = load(actor_root,"pairs")
        pd.testing.assert_frame_equal(anchors,actor_anchors)
        baseline = [*KEY,"choice","alternative","alternative_air","gap"]
        pd.testing.assert_frame_equal(pairs[baseline],actor_pairs[baseline])
        counter = rates(actor_anchors,actor_pairs,.025)
        alternative = counter.groupby("prefix")[cols].sum()
        alternative["better_days"] = (counter.groupby(["prefix","day"]).better.sum()>0).groupby("prefix").sum()
        alternative.loc[108,"both_versions"] = np.nan
        alternative.to_csv(dest/"actor_continuation.csv")
        print(alternative.to_string())


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("dest",type=Path)
    parser.add_argument("--actor-root",type=Path)
    args=parser.parse_args()
    main(args.root,args.dest,args.actor_root)

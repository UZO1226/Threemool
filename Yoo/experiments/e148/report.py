"""Aggregate matched decisions and independently recover timing from the four stored paths."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ID = ["prefix", "seed", "decision"]
GROUPS = ["nh4", "no3", "do", "previous_air"]
BLOCKS = ["m3_0_1h", "m3_1_3h", "m3_3_6h", "m3_later"]


def first(mask):
    idx = np.flatnonzero(mask)
    return idx[0]*15 if len(idx) else np.nan


def describe(frame):
    hit = frame.dropna(subset=["onset_min"])
    near = np.isclose(hit.onset_min, hit.nh_min)
    return dict(pairs=len(frame), decisions=len(frame[ID].drop_duplicates()), hits=len(hit),
                median_min=hit.onset_min.median(), q25=hit.onset_min.quantile(.25),
                q75=hit.onset_min.quantile(.75), small_min=frame.small_min.median(),
                nh_before=(hit.nh_min < hit.onset_min).sum(), nh_same=near.sum(),
                nh_later=(hit.nh_min > hit.onset_min).sum(), nh_never=hit.nh_min.isna().sum(),
                positive_gap=(hit.gap>0).sum(),
                selection_larger=(hit.selection.abs()>hit.actor.abs()).sum(),
                **{"top_"+g:(hit.top==g).sum() for g in GROUPS},
                **{"mean_abs_"+g:hit[g].abs().mean() for g in GROUPS+["actor","selection"]},
                **frame[BLOCKS+["current_m3", "previous_m3"]].mean().to_dict())


def report(source, dest):
    events = pd.concat([pd.read_csv(f) for f in sorted(source.glob("events_*.csv"))], ignore_index=True)
    traces = pd.concat([pd.read_csv(f) for f in sorted(source.glob("traces_*.csv.gz"))], ignore_index=True)
    assert len(events) == 525 and not events.duplicated(ID+["current_member","previous_member"]).any()
    assert events.groupby(ID).size().eq(25).all()
    order = ["version", "member", "arm", "minute"]
    timings = []
    for key, case in events.groupby(ID):
        paths = traces[(traces.prefix==key[0]) & (traces.seed==key[1]) & (traces.decision==key[2])].sort_values(order)
        assert not paths.duplicated(order).any()
        cols = ["flow", "nh4", "no3", "do1", "do2", "do3", "prev_air"]
        values = paths[cols].to_numpy().reshape(2,5,2,-1,len(cols))
        # Previous air is the normalized command from the preceding step on each branch.
        np.testing.assert_allclose(values[:,:,:,1:,-1]*9000, values[:,:,:,:-1,0], atol=.001, rtol=0)
        for e in case.itertuples():
            a, b = values[0,e.current_member], values[1,e.previous_member]
            cur, prev = a[1,:,0]-a[0,:,0], b[1,:,0]-b[0,:,0]
            gap = prev-cur
            np.testing.assert_allclose([cur.sum()/4,prev.sum()/4, first(abs(gap)>=250)],
                                       [e.current_m3,e.previous_m3,e.onset_min],atol=1e-7)
            np.testing.assert_allclose(sum(getattr(e, k) for k in BLOCKS),gap.sum()/4,atol=1e-7)
            delta = abs(b-a).max(axis=0)
            # Auxiliary clocks are descriptive first differences between old and current within either arm.
            nh, no3, do, air = (first(delta[:,1]>=.1), first(delta[:,2]>=.1),
                                first(delta[:,3:6].max(1)>=.1), first(delta[:,6]*9000>=250))
            np.testing.assert_allclose(nh, e.nh_min)
            timings.append(dict(zip(ID,key), current_member=e.current_member, previous_member=e.previous_member,
                                no3_start=no3, do_start=do, air_start=air))
    events = events.merge(pd.DataFrame(timings),on=ID+["current_member","previous_member"],validate="one_to_one")
    summaries, cases = [], []
    for prefix, g in events.groupby("prefix"):
        for label, sub in (("all",g), ("old_air_loss",g[g.old_air_loss]), ("same_index",g[g.current_member==g.previous_member])):
            if len(sub):
                summaries.append(dict(prefix=prefix, subset=label, **describe(sub)))
        for key, frame in g.groupby(ID):
            cases.append(dict(zip(ID,key), day=frame.day.iloc[0], failing_previous_members=frame[frame.old_air_loss].previous_member.nunique(),
                              **describe(frame)))
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(dest/"summary.csv",index=False)
    pd.DataFrame(cases).to_csv(dest/"decisions.csv",index=False)
    events.to_csv(dest/"pairs.csv",index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("dest",type=Path)
    a=p.parse_args()
    report(a.source,a.dest)

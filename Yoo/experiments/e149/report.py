"""Summarize paired pulse results at the decision and member-pair levels."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["air_3_6h", "air_m3", "nh4_min", "excess", "do_min"]
KEYS = ["prefix", "seed", "decision", "current_member", "previous_member"]
EPS = 1e-8


def passes(values):
    return (values[:,1] < -.05) & (values[:,2:] <= EPS).all(axis=1)


def summarize(frame):
    before = frame[["gap_before_"+s for s in METRICS]].to_numpy()
    after = frame[["gap_after_"+s for s in METRICS]].to_numpy()
    prev = frame[["prior_before_"+s for s in METRICS]].to_numpy()
    new = frame[["prior_after_"+s for s in METRICS]].to_numpy()
    cut = frame[["prior_cut_change_"+s for s in METRICS]].to_numpy()
    native = frame[["prior_native_change_"+s for s in METRICS]].to_numpy()
    np.testing.assert_allclose(new-prev,cut-native,atol=1e-7,rtol=0)
    no_pulse = ~frame.changed.to_numpy()
    np.testing.assert_array_equal(after[no_pulse], before[no_pulse])
    assert (frame.loc[frame.onset.isna(),"changed"] == False).all()
    result = dict(pairs=len(frame), decisions=len(frame[KEYS[:3]].drop_duplicates()), hits=frame.onset.notna().sum(),
                  projected=frame.projected.sum(), changed=frame.changed.sum(), changed_current=frame.changed_current.sum(),
                  changed_prior=frame.changed_prior.sum(), onset_in_window=frame.onset.between(180,360,inclusive="left").sum(),
                  prior_joint_before=passes(prev).sum(), prior_joint_after=passes(new).sum(),
                  new_joint=(~passes(prev)&passes(new)).sum(), lost_joint=(passes(prev)&~passes(new)).sum(),
                  cut_joint_change=passes(cut).sum(), cut_nh_worse=(cut[:,2]>EPS).sum(),
                  cut_excess_worse=(cut[:,3]>EPS).sum(),cut_do_worse=(cut[:,4]>EPS).sum())
    for i, name in enumerate(METRICS):
        for label, arr in [("gap_before",before),("gap_after",after),("prior_before",prev),("prior_after",new),
                           ("cut_change",cut),("native_change",native)]:
            result[label+"_"+name] = arr[:,i].mean()
        if i < 2:
            d = abs(after[:,i])-abs(before[:,i])
            result["narrower_"+name] = (d < -.05).sum()
            result["wider_"+name] = (d > .05).sum()
            result["mean_abs_before_"+name] = abs(before[:,i]).mean()
            result["mean_abs_after_"+name] = abs(after[:,i]).mean()
    result["gap_and_water"] = bool((abs(after.mean(0)[:2]) < abs(before.mean(0)[:2])).all()
                                   and (cut.mean(0)[2:] <= EPS).all())
    return result


def report(source,dest):
    pairs = pd.concat([pd.read_csv(f) for f in sorted(source.glob("p*_s*.csv"))],ignore_index=True)
    assert len(pairs)==525 and not pairs.duplicated(KEYS).any()
    assert pairs.groupby(KEYS[:3]).size().eq(25).all()
    original = pd.read_csv(Path(__file__).resolve().parents[1]/"e148/pairs.csv")
    matched = pairs.merge(original[KEYS+["onset_min","old_air_loss","m3_3_6h","previous_m3","current_m3"]],
                          on=KEYS,validate="one_to_one")
    assert len(matched)==525 and matched.old_loss.equals(matched.old_air_loss)
    np.testing.assert_allclose(matched.onset,matched.onset_min,equal_nan=True)
    np.testing.assert_allclose(matched.gap_before_air_3_6h,matched.m3_3_6h,atol=1e-7,rtol=0)
    np.testing.assert_allclose(matched.gap_before_air_m3,matched.previous_m3-matched.current_m3,atol=1e-7,rtol=0)
    summaries, decisions = [],[]
    for p,g in pairs.groupby("prefix"):
        assert g.onset.notna().sum() == {148:50,188:350}[p]
        for label,frame in [("all",g),("changed",g[g.changed]),("old_loss",g[g.old_loss])]:
            if len(frame):
                summaries.append(dict(prefix=p,subset=label,**summarize(frame)))
        for (s,d),frame in g.groupby(["seed","decision"]):
            decisions.append(dict(prefix=p,seed=s,decision=d,day=frame.day.iloc[0],**summarize(frame)))
    dest.mkdir(exist_ok=True,parents=True)
    pd.DataFrame(summaries).to_csv(dest/"summary.csv",index=False)
    pd.DataFrame(decisions).to_csv(dest/"decisions.csv",index=False)
    selection = ["day","onset","old_loss","projected","changed","changed_current","changed_prior"]
    outcomes = [f"gap_{when}_{metric}" for when in ("before","after") for metric in METRICS[:2]]
    outcomes += [f"prior_cut_change_{metric}" for metric in METRICS]
    pairs[KEYS+selection+outcomes].to_csv(dest/"pairs.csv",index=False)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("dest",type=Path)
    a=p.parse_args()
    report(a.source,a.dest)

"""Match e151 event-specific warning deadlines using frozen score files."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ARMS = ("base", "no3")


def lead_times(rows, alarm, onsets):
    return (rows.loc[alarm & rows.onset.notna()].groupby("onset").lead.max()
            .reindex(onsets, fill_value=0).astype(int) * 15)


def best_cutoff(rows, score, deadlines):
    required = deadlines[deadlines > 0]
    if required.empty:
        raise ValueError("No reference detections in calibration period")
    deadline = rows.onset.map(required)
    allowed = (15 * rows.lead >= deadline) & deadline.notna()
    peaks = score[allowed].groupby(rows.onset[allowed]).max().reindex(required.index)
    assert peaks.notna().all()
    cutoff = peaks.min()

    # Independent search over all score ranks, eliminating thresholds per event.
    grid = np.unique(score)
    feasible = np.ones(len(grid), dtype=bool)
    for onset, minutes in required.items():
        window = score[(rows.onset == onset) & (rows.index <= onset - pd.Timedelta(minutes=minutes))]
        feasible &= (window.to_numpy()[:, None] >= grid).any(axis=0)
    assert grid[feasible][-1] == cutoff
    return cutoff, peaks


def smoke():
    times = pd.date_range("2021-10-01 10:00", periods=6, freq="15min")
    rows = pd.DataFrame({"onset":[times[2],times[2],pd.NaT,times[5],times[5],pd.NaT],
                         "lead":[2,1,0,2,1,0]},index=times)
    score = pd.Series([.4,.9,.4,.4,.3,.1],index=times)
    required = pd.Series([30,30], index=times[[2,5]])
    cutoff, peaks = best_cutoff(rows,score,required)
    assert cutoff == .4 and peaks.eq(.4).all()
    assert lead_times(rows,score >= cutoff,required.index).tolist() == [30,30]
    assert lead_times(rows,score > cutoff,required.index).tolist() == [15,0]


def run(source, out):
    smoke()
    reference = pd.read_csv(HERE.parent/"e151/summary.csv").set_index(["prefix","arm"])
    summary, detail, previous = [], [], {}
    for n in (108,148,188):
        rows = pd.read_csv(source/f"p{n}.csv",parse_dates=["time","onset"]).set_index("time")
        onsets = pd.DatetimeIndex(rows.onset.dropna().unique()).sort_values()
        assert rows.index.is_unique and rows.lead.gt(0).equals(rows.event.astype(bool))
        delta = (rows.onset - rows.index).dt.total_seconds().dropna() / 60
        np.testing.assert_array_equal(delta, 15 * rows.loc[delta.index,"lead"])
        native = rows.base > reference.loc[(n,"base"),"threshold"]
        deadlines = lead_times(rows,native,onsets)
        required = deadlines > 0
        details = pd.DataFrame({"ref_min":deadlines})
        next_cut = {}
        for arm in ARMS:
            ref = reference.loc[(n,arm)]
            relative = rows[arm]/ref.threshold
            cutoff, peaks = best_cutoff(rows,relative,deadlines)
            next_cut[arm] = cutoff
            details[f"{arm}_eligible_peak"] = peaks
            modes = [("original",1.0,rows[arm] > ref.threshold),
                     ("same_period",cutoff,relative >= cutoff)]
            if previous:
                modes.append(("previous_period",previous[arm],relative >= previous[arm]))
            for mode, multiple, alarm in modes:
                lead = lead_times(rows,alarm,onsets)
                same = lead == deadlines
                missed = required & lead.eq(0)
                late = required & lead.gt(0) & lead.lt(deadlines)
                false = int((alarm & rows.onset.isna()).sum())
                if mode == "original":
                    assert false == ref.fp and (lead > 0).sum() == ref.detected
                    assert lead[lead>0].median() == ref.median_lead_min
                if mode == "same_period":
                    assert not (missed | late).any()
                details[f"{arm}_{mode}_min"] = lead
                summary.append(dict(prefix=n,arm=arm,mode=mode,multiplier=multiple,events=len(lead),
                                    targets=required.sum(),on_time=(required & (lead>=deadlines)).sum(),
                                    early=(required & (lead>deadlines)).sum(),same=(required & same).sum(),
                                    late=late.sum(),missed=missed.sum(),new=((~required) & (lead>0)).sum(),
                                    false_steps=false,negative_steps=rows.event.eq(0).sum()))
        previous = next_cut
        detail.append(details.rename_axis("onset").reset_index().assign(prefix=n))
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(summary).to_csv(out/"summary.csv",index=False,float_format="%.15g")
    pd.concat(detail,ignore_index=True).to_csv(out/"events.csv",index=False,float_format="%.15g")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("out",type=Path)
    args = p.parse_args()
    run(args.source,args.out)

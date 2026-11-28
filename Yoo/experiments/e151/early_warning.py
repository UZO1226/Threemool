"""Evaluate one-hour NH4 onset alarms on recorded DARROW operation."""
import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SENSORS = ["NH4", "DO_1", "DO_2", "DO_3", "NO3"]
RAW = Path("/home/jio/사물/Yoo/data/dynamic-modeling-of-wastewater-treatment-process/train_val_test_online_dataset.csv")


def observations(path):
    records = []
    with path.open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            day = row["DATETIME"][:10]
            if day > "2021-12-19":
                break
            if day >= "2021-05-06":
                records.append([pd.Timestamp(row["DATETIME"]), *(float(row[c]) for c in SENSORS)])
    data = pd.DataFrame(records, columns=["time", *SENSORS]).set_index("time")
    assert len(data) == 228 * 96 and np.isfinite(data).all().all()
    assert data.index.to_series().diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all()
    return data


def features(frame):
    return pd.concat([frame, frame.diff().add_suffix("_d15"), frame.diff(4).add_suffix("_d60")], axis=1)


def decisions(day):
    x = features(day)
    nh = day.NH4.to_numpy()
    times = np.flatnonzero(nh[4:-4] <= 4) + 4
    future = nh[times[:, None] + np.arange(1, 5)]
    above = future > 4
    leads = np.where(above.any(axis=1), above.argmax(axis=1) + 1, 0)
    checks = [next((j for j in range(1, 5) if nh[t+j] > 4), 0) for t in times]
    np.testing.assert_array_equal(leads, checks)
    sample = x.iloc[times].copy()
    sample["lead"] = leads
    sample["onset"] = (sample.index + pd.to_timedelta(leads * 15, unit="min")).where(leads > 0)
    return sample


def smoke():
    idx = pd.date_range("2021-05-06 00:12", periods=16, freq="15min")
    day = pd.DataFrame(np.zeros((16,len(SENSORS))), columns=SENSORS,index=idx)
    day.loc[idx[8], "NH4"] = 4
    baseline = decisions(day)
    assert baseline.lead.eq(0).all()
    day.loc[idx[9], "NH4"] = 4.1
    changed = decisions(day)
    assert changed.loc[idx[8],"lead"] == 1 and changed.loc[idx[5],"lead"] == 4
    assert changed.loc[idx[4],"lead"] == 0 and idx[9] not in changed.index
    day.loc[idx[9]:,SENSORS] += 100
    pd.testing.assert_frame_equal(features(day).loc[idx[4]:idx[8]], baseline.drop(columns=["lead","onset"]).loc[idx[4]:idx[8]])


def evaluate(rows, score, threshold):
    event = rows.lead.to_numpy() > 0
    alarm = score > threshold
    tp = (event & alarm).sum()
    fp = (~event & alarm).sum()
    leads = rows.loc[alarm & event].groupby("onset").lead.max() * 15
    return dict(n=len(rows),pos=event.sum(),ap=average_precision_score(event,score),
                brier=np.mean((score-event)**2),threshold=threshold,tp=tp,fp=fp,
                sensitivity=tp/event.sum(),fpr=fp/(~event).sum(),onsets=rows.onset.nunique(),
                detected=len(leads),median_lead_min=leads.median())


def interval(rows):
    loss = (rows.no3-rows.event)**2 - (rows.base-rows.event)**2
    daily = loss.groupby(rows.index.date).agg(["sum","count"])
    assert len(daily) == 40
    starts = np.random.default_rng(151).integers(0,40,(2000,6))
    indices = ((starts[:,:,None] + np.arange(7)) % 40).reshape(2000,-1)[:,:40]
    draws = daily["sum"].to_numpy()[indices].sum(1) / daily["count"].to_numpy()[indices].sum(1)
    return np.quantile(draws,[.025,.975])


def run(raw, out, paths):
    smoke()
    data = observations(raw)
    frames = [decisions(day) for _, day in data.groupby(data.index.date)]
    out.mkdir(parents=True,exist_ok=True)
    paths.mkdir(parents=True,exist_ok=True)
    summary, contrasts = [], []
    for n in (108,148,188):
        train = pd.concat(frames[:n])
        valid = pd.concat(frames[n:n+40])
        assert train.index.max()+pd.Timedelta(hours=1) < valid.index.min()
        event = train.lead.gt(0)
        scores = valid[["lead","onset"]].assign(event=valid.lead.gt(0).astype(int))
        results = {}
        for arm in ("base","no3"):
            columns = [c for c in features(data.iloc[:5]).columns if arm == "no3" or not c.startswith("NO3")]
            model = make_pipeline(StandardScaler(),LogisticRegression(C=1,solver="lbfgs",max_iter=1000))
            model.fit(train[columns],event)
            cutoff = np.quantile(model.predict_proba(train.loc[~event,columns])[:,1],.95,method="higher")
            prob = model.predict_proba(valid[columns])[:,1]
            scores[arm] = prob
            record = evaluate(valid,prob,cutoff)
            results[arm] = record
            summary.append(dict(prefix=n,arm=arm,start=str(valid.index.min().date()),
                                end=str(valid.index.max().date()),train_n=len(train),
                                prevalence=scores.event.mean(),nh_only_ap=average_precision_score(scores.event,valid.NH4),
                                **record))
        delta = {k: results["no3"][k]-results["base"][k] for k in ("ap","brier","sensitivity","fpr")}
        lo,hi = interval(scores)
        passed = delta["ap"]>0 and delta["brier"]<0 and delta["sensitivity"]>=0 and delta["fpr"]<=0
        contrasts.append(dict(prefix=n,**delta,brier_lo=lo,brier_hi=hi,passed=passed))
        scores.to_csv(paths/f"p{n}.csv",index_label="time")
    pd.DataFrame(summary).to_csv(out/"summary.csv",index=False,float_format="%.9g")
    pd.DataFrame(contrasts).to_csv(out/"comparison.csv",index=False,float_format="%.9g")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    p.add_argument("paths",type=Path)
    p.add_argument("--raw",type=Path,default=RAW)
    args = p.parse_args()
    run(args.raw,args.out,args.paths)

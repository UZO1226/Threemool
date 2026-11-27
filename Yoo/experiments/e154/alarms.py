"""Fit four frozen one-hour alarms and report event-level validation."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"e151"))
sys.path.insert(0, str(HERE.parent/"e152"))
import early_warning as ew
from matched_lead import best_cutoff, lead_times

ARMS = ("base", "no3", "neutral", "refit")


def fit(train, columns, arm):
    selected = [c for c in columns if not (arm == "base" and c.startswith("NO3"))
                and not (arm == "refit" and c == "NO3_d60")]
    scale = StandardScaler().fit(train[selected])
    x = scale.transform(train[selected])
    clf = LogisticRegression(C=1, solver="lbfgs", max_iter=1000).fit(x, train.lead.gt(0))
    beta = clf.coef_[0].copy()
    if arm == "neutral":
        beta[selected.index("NO3_d60")] = 0
    raw = pd.Series(beta/scale.scale_, index=selected)
    bias = float(clf.intercept_[0] - raw @ scale.mean_)
    coef = raw.reindex(columns, fill_value=0).to_numpy()
    score = expit(train[columns].to_numpy() @ coef + bias)
    np.testing.assert_allclose(score, expit(x @ beta + clf.intercept_[0]), rtol=0, atol=1e-12)
    cutoff = float(np.quantile(score[train.lead.eq(0)], .95, method="higher"))
    return {"coef":coef.tolist(), "bias":bias, "cutoff":cutoff}


def probability(x, model):
    return expit(x @ np.array(model["coef"]) + model["bias"])


def run(paths, out):
    out.mkdir(parents=True, exist_ok=True)
    raw = ew.observations(ew.RAW)
    frames = [ew.decisions(d) for _,d in raw.groupby(raw.index.date)]
    columns = list(ew.features(raw.iloc[:5]).columns)
    previous = pd.read_csv(HERE.parent/"e153/summary.csv").query("mode == 'train_budget'").set_index(["prefix","arm"])
    rows, cases, fitted = [], [], {}
    for n in (68,108,148,188):
        train, valid = pd.concat(frames[:n]), pd.concat(frames[n:n+40])
        assert train.index.max() + pd.Timedelta(hours=1) < valid.index.min()
        models = {arm:fit(train,columns,arm) for arm in ARMS}
        fitted[str(n)] = models
        score = {a:pd.Series(probability(valid[columns].to_numpy(),m),index=valid.index) for a,m in models.items()}
        if n >= 108:
            archive = pd.read_csv(paths/f"p{n}.csv",parse_dates=["time"]).set_index("time")
            assert archive.index.equals(valid.index)
            for a in ("base","no3"):
                np.testing.assert_allclose(score[a],archive[a],atol=1e-12,rtol=0)
        onsets = pd.DatetimeIndex(valid.onset.dropna().unique()).sort_values()
        deadlines = lead_times(valid,score["base"]>models["base"]["cutoff"],onsets)
        events = deadlines.to_frame("deadline_min")
        for a,p in score.items():
            cutoff = models[a]["cutoff"]
            alarms = p > cutoff
            leads = lead_times(valid,alarms,onsets)
            events[a] = leads
            oracle,_ = best_cutoff(valid,p,deadlines)
            negatives = valid.lead.eq(0)
            false = int((alarms & negatives).sum())
            ontime = ((deadlines>0)&(leads>=deadlines)).sum()
            if a != "refit":
                ref = previous.loc[(n,a)]
                assert (ontime,false) == (ref.on_time,ref.false_steps)
            rows.append(dict(prefix=n,arm=a,targets=deadlines.gt(0).sum(),on_time=ontime,
                             detected=leads.gt(0).sum(),events=len(leads),false_steps=false,
                             negatives=negatives.sum(),oracle_false=((p>=oracle)&negatives).sum()))
        cases.append(events.rename_axis("onset").reset_index().assign(prefix=n))
    (out/"models.json").write_text(json.dumps({"columns":columns,"models":fitted},indent=2)+"\n")
    pd.DataFrame(rows).to_csv(out/"alarm_summary.csv",index=False)
    pd.concat(cases,ignore_index=True).to_csv(out/"alarm_events.csv",index=False)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.paths,a.out)

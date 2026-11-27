"""Limit the negative nitrate trend contribution in the DARROW one-hour alarm."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e151"))
sys.path.insert(0, str(HERE.parent / "e152"))
import early_warning as ew
from matched_lead import best_cutoff, lead_times


def edit(logits, term):
    floor = -np.log(2)
    return {"half_odds": logits + np.maximum(floor - term, 0),
            "no_suppression": logits + np.maximum(-term, 0),
            "neutral": logits - term}


def smoke():
    term = np.array([-2., -np.log(2), 0., 1.])
    updated = edit(1 + term, term)
    np.testing.assert_allclose(updated["half_odds"], 1 + np.maximum(term, -np.log(2)))
    np.testing.assert_allclose(updated["no_suppression"], 1 + term.clip(min=0))
    np.testing.assert_allclose(updated["neutral"], 1.)


def predict(train, combined, cols, arm):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, solver="lbfgs", max_iter=1000))
    model.fit(train[cols], train.lead.gt(0))
    logits = model.decision_function(combined[cols])
    scores = {arm: logits}
    if arm == "no3":
        i = cols.index("NO3_d60")
        term = model[0].transform(combined[cols])[:, i] * model[1].coef_[0, i]
        scores.update(edit(logits, term))
    return {name: pd.Series(expit(values), index=combined.index) for name, values in scores.items()}


def run(paths, out):
    smoke()
    raw = ew.observations(ew.RAW)
    frames = [ew.decisions(frame) for _, frame in raw.groupby(raw.index.date)]
    archived = pd.read_csv(HERE.parent / "e152/events.csv", parse_dates=["onset"])
    original = pd.read_csv(HERE.parent / "e151/summary.csv").set_index(["prefix","arm"])
    logs, events = [], []
    for n in (68,108,148,188):
        train, valid = pd.concat(frames[:n]), pd.concat(frames[n:n+40])
        assert train.index.max() + pd.Timedelta(hours=1) < valid.index.min()
        combined = pd.concat([train, valid])
        cols = list(ew.features(raw.iloc[:5]).columns)
        scores = predict(train, combined, [c for c in cols if not c.startswith("NO3")], "base")
        scores.update(predict(train, combined, cols, "no3"))
        negative = train.index[train.lead.eq(0)]
        cuts = {arm: np.quantile(score.loc[negative], .95, method="higher") for arm, score in scores.items()}
        val = {arm: score.loc[valid.index] for arm, score in scores.items()}
        alarms = {arm: score > cuts[arm] for arm, score in val.items()}
        onsets = pd.DatetimeIndex(valid.onset.dropna().unique()).sort_values()
        deadlines = lead_times(valid, alarms["base"], onsets)
        required = deadlines.gt(0)
        case_data = deadlines.rename("deadline_min").to_frame()
        if n != 68:
            old = pd.read_csv(paths / f"p{n}.csv", parse_dates=["time"]).set_index("time")
            assert old.index.equals(valid.index)
            for arm in ("base","no3"):
                np.testing.assert_allclose(val[arm], old[arm], atol=1e-12, rtol=0)
                np.testing.assert_allclose(cuts[arm], original.loc[(n,arm),"threshold"], atol=6e-10,rtol=0)
            old_cases = archived.loc[archived.prefix.eq(n)].set_index("onset").sort_index()
            np.testing.assert_array_equal(old_cases.index, onsets)
            np.testing.assert_array_equal(old_cases.ref_min, deadlines)
            np.testing.assert_array_equal(old_cases.no3_original_min, lead_times(valid, alarms["no3"],onsets))
        for arm, score in val.items():
            common_cut = cuts["base" if arm == "base" else "no3"]
            settings = [("train_budget", cuts[arm], alarms[arm]), ("fixed", common_cut, score > common_cut)]
            if required.any():
                cutoff, _ = best_cutoff(valid, score, deadlines)
                settings.append(("oracle", cutoff, score >= cutoff))
            if arm in ("half_odds","no_suppression"):
                assert (score >= val["no3"]).all()
                assert (score[alarms["no3"]] > common_cut).all()
            for mode, cutoff, alarm in settings:
                lead = lead_times(valid, alarm, onsets)
                case_data[f"{arm}_{mode}"] = lead
                missed = required & lead.eq(0)
                late = required & lead.gt(0) & lead.lt(deadlines)
                false = (alarm & valid.lead.eq(0)).sum()
                on_time = (required & lead.ge(deadlines)).sum()
                if mode == "oracle":
                    assert on_time == required.sum()
                logs.append(dict(prefix=n, start=str(valid.index.min().date()), end=str(valid.index.max().date()),
                                 arm=arm, mode=mode, cutoff=cutoff, targets=required.sum(), on_time=on_time,
                                 late=late.sum(), missed=missed.sum(), events=len(lead), detected=lead.gt(0).sum(),
                                 false_steps=false, negatives=valid.lead.eq(0).sum(),
                                 ap=average_precision_score(valid.lead.gt(0),score),
                                 brier=np.mean((score-valid.lead.gt(0))**2)))
        events.append(case_data.rename_axis("onset").reset_index().assign(prefix=n))
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(logs).to_csv(out/"summary.csv",index=False,float_format="%.12g")
    pd.concat(events,ignore_index=True).to_csv(out/"events.csv",index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths",type=Path)
    parser.add_argument("out",type=Path)
    options = parser.parse_args()
    run(options.paths,options.out)

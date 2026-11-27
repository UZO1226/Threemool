"""Test one fixed pre-cut rebound score on two later DARROW periods."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from paths import FEATURES, previous

METRICS = previous.METRICS
EPS = 1e-8


def discrimination(group, threshold):
    actual = group.rebound.to_numpy(bool)
    scores = group.score.to_numpy()
    alarm = scores > threshold
    tp = int((actual & alarm).sum())
    fp = int((~actual & alarm).sum())
    return dict(n=len(group), events=int(actual.sum()), alarms=int(alarm.sum()), tp=tp, fp=fp,
                auc=roc_auc_score(actual, scores) if len(np.unique(actual)) == 2 else np.nan,
                tpr=tp/actual.sum() if actual.any() else np.nan,
                fpr=fp/(~actual).sum() if (~actual).any() else np.nan,
                coverage=(~alarm).mean(), brier=np.square(scores - actual).mean())


def utility(group):
    # Every selected episode uses its single-cut path, all others the native path.
    keep = group.keep.astype(float)
    eligible = group.eligible.astype(float)
    eligible_count = eligible.sum()
    within_day = group.groupby("day")[["keep", "eligible"]].transform("sum")
    day_rate = within_day.keep.div(within_day.eligible.replace(0, np.nan)).fillna(0)
    weights = {"screen": keep, "once": eligible, "oracle": eligible * ~group.rebound,
               "native": 0 * keep,
               "random_seed": eligible * (keep.sum() / max(eligible_count, 1)),
               "random_day": eligible * day_rate}
    assert np.isclose(weights["random_seed"].sum(), keep.sum())
    assert np.isclose(weights["random_day"].sum(), keep.sum())
    return {key: group[METRICS].mul(w, axis=0).mean().to_numpy() for key, w in weights.items()}


def joint(values):
    return (values[0] < -.05) and (values[1:] <= EPS).all()


def analyze(source, dest):
    df = pd.concat([pd.read_csv(source / f"p{n}_s{s}.csv")
                    for n in (108, 148, 188) for s in range(10, 15)], ignore_index=True)
    assert not df.duplicated(["day", "seed", "member"]).any() and len(df) == 3000
    assert df.groupby(["prefix", "seed", "day"]).size().eq(5).all()
    hours = [f"air_{h}h" for h in range(2, 7)]
    df["rebound"] = df[hours].ge(-EPS).any(axis=1)
    eligible = df.eligible
    assert df.loc[eligible, FEATURES + hours].notna().all().all()
    dates = sorted(df.loc[df.prefix == 108, "day"].unique())
    assert len(dates) == 40
    train = df[eligible & df.day.isin(dates[:30])]
    cal = df[eligible & df.day.isin(dates[30:])]
    assert train.day.max() < cal.day.min() and cal.day.max() < df.loc[df.prefix == 148, "day"].min()
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000))
    model.fit(train[FEATURES], train.rebound)
    threshold = float(np.median(model.predict_proba(cal[FEATURES])[:, 1]))
    df["score"] = np.nan
    df.loc[eligible, "score"] = model.predict_proba(df.loc[eligible, FEATURES])[:, 1]
    df["keep"] = eligible & (df.score <= threshold)
    dest.mkdir(parents=True, exist_ok=True)
    scaler, lr = model.steps[0][1], model.steps[1][1]
    spec = dict(features=FEATURES, mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(),
                coefficients=lr.coef_[0].tolist(), intercept=lr.intercept_[0], threshold=threshold,
                fit_days=[dates[0], dates[29]], calibration_days=[dates[30], dates[-1]])
    (dest / "model.json").write_text(json.dumps(spec, indent=2) + "\n")
    s = ((df.loc[eligible, FEATURES].values - scaler.mean_) / scaler.scale_) @ lr.coef_[0] + lr.intercept_[0]
    np.testing.assert_allclose(1 / (1 + np.exp(-s)), df.loc[eligible, "score"], atol=1e-12)
    horizons, rank, control, passed = [], [], [], []
    for n, fold in df.groupby("prefix"):
        permitted = fold[fold.eligible]
        horizons.append(dict(prefix=n, eligible=len(permitted), total=len(fold),
                             any_2_to_6=permitted.rebound.sum(),
                             **{key: int((permitted[key] >= -EPS).sum()) for key in hours + ["air_m3"]},
                             both=int((permitted.rebound & (permitted.air_m3 >= -EPS)).sum())))
        if n == 108:
            continue
        for seed, group in fold.groupby("seed"):
            rank.append(dict(prefix=n, seed=seed, **discrimination(group[group.eligible], threshold)))
            values = utility(group)
            for key, v in values.items():
                control.append(dict(prefix=n, seed=seed, method=key, **dict(zip(METRICS, v))))
            for key in ("native", "random_seed", "random_day"):
                d = values["screen"] - values[key]
                passed.append(dict(prefix=n, seed=seed, comparator=key, joint=joint(d), **dict(zip(METRICS, d))))
        rank.append(dict(prefix=n, seed="pooled", **discrimination(permitted, threshold)))
    for name, rows in (("horizons", horizons), ("ranking", rank), ("control", control), ("paired", passed)):
        pd.DataFrame(rows).to_csv(dest / f"{name}.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("dest", type=Path)
    args = p.parse_args()
    analyze(args.source, args.dest)

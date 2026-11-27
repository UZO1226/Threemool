"""Summarize e150 paired daily simulations and e149's prespecified case labels."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent
METRICS = ["air_m3", "nh4_min", "excess", "do_min"]
KEY = ["prefix", "seed", "episode", "day", "member"]
CASE = ["prefix", "seed", "decision", "day"]
COMPARISONS = [("trend", "once"), ("level", "once"), ("all", "once"),
               ("trend", "all"), ("trend", "level"), ("trend", "native"), ("once", "native")]


def joint(frame):
    return (frame.air_m3 < -.05) & frame[METRICS[1:]].le(1e-8).all(axis=1)


def case_results(reference):
    labels = pd.read_csv(HERE.parent / "e149/decisions.csv")
    labels["success"] = joint(labels.filter(regex="^cut_change_").rename(columns=lambda c: c.removeprefix("cut_change_")))
    representatives = pd.read_csv(HERE.parent / "e148/pairs.csv")[CASE + ["episode"]].drop_duplicates()
    cases = labels[CASE + ["success"]].merge(representatives, on=CASE, validate="one_to_one")
    cases = cases.merge(reference.reset_index(), on=["prefix","seed","day","episode"], validate="one_to_one")
    assert len(cases) == 21 and cases["first"].ge(0).all()
    assert cases.groupby("prefix").success.sum().to_dict() == {148: 0, 188: 9}
    cases = cases[CASE + ["episode","member","first","success","trend","level"]]
    stats = []
    for prefix, group in cases.groupby("prefix"):
        y = group.success
        for arm in ("trend","level"):
            positive = group[arm] > 0
            stats.append(dict(prefix=prefix, arm=arm, cases=len(y), dates=group.day.nunique(),
                              tp=(y & positive).sum(), fp=(~y & positive).sum(),
                              fn=(y & ~positive).sum(), tn=(~y & ~positive).sum(),
                              auc=roc_auc_score(y,group[arm]) if y.nunique() == 2 else np.nan))
    return cases, pd.DataFrame(stats)


def run(paths, out):
    files = [paths / f"p{p}_s{s}.csv" for p in (108,148,188) for s in range(10,15)]
    data = pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
    assert len(data) == 15000 and not data.duplicated(KEY+["mode"]).any()
    arms = {name: rows.set_index(KEY).sort_index() for name,rows in data.groupby("mode")}
    base = arms["once"]
    assert len(base) == 3000 and base.groupby(["prefix","seed"]).size().eq(200).all()
    assert base.groupby(["prefix","seed","day"]).size().eq(5).all()
    for name, arm in arms.items():
        pd.testing.assert_index_equal(arm.index, base.index)
        pd.testing.assert_frame_equal(arm[["first","trend","level"]], base[["first","trend","level"]])
    for name in ("trend","level"):
        arm = arms[name]
        on = base[name].gt(0)
        pd.testing.assert_frame_equal(arm.loc[on, METRICS+["pulse"]],arms["all"].loc[on, METRICS+["pulse"]])
        pd.testing.assert_frame_equal(arm.loc[~on, METRICS+["pulse"]],base.loc[~on, METRICS+["pulse"]])
    cases, confusion = case_results(base)
    byseed, summary = [], []
    for sample in ("all_dates","outside_case_dates"):
        for arm, control in COMPARISONS:
            delta = arms[arm][METRICS].sub(arms[control][METRICS])
            delta["pulse"] = arms[arm].pulse.ge(0)
            delta["control_pulse"] = arms[control].pulse.ge(0)
            delta = delta.reset_index()
            if sample == "outside_case_dates":
                delta = delta[~delta.day.isin(cases.day)]
            for prefix, group in delta.groupby("prefix"):
                means = group.groupby("seed")[METRICS].mean()
                passed = joint(means)
                for seed, row in means.iterrows():
                    byseed.append(dict(sample=sample,prefix=prefix,arm=arm,control=control,
                                       seed=seed,passed=passed.loc[seed],**row.to_dict()))
                pooled = group[METRICS].mean().to_frame().T
                summary.append(dict(sample=sample,prefix=prefix,arm=arm,control=control,
                                    days=group.day.nunique(),paths=len(group),pulses=group.pulse.sum(),
                                    control_pulses=group.control_pulse.sum(),seed_pass=passed.sum(),
                                    mean_pass=joint(pooled).iloc[0],**pooled.iloc[0].to_dict()))
    out.mkdir(exist_ok=True, parents=True)
    for file, table in [("cases",cases), ("classification",confusion), ("by_seed",pd.DataFrame(byseed)),
                        ("summary",pd.DataFrame(summary))]:
        table.to_csv(out / f"{file}.csv", index=False, float_format="%.10g")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths",type=Path)
    p.add_argument("out",type=Path)
    args = p.parse_args()
    run(args.paths,args.out)

"""Count agreement per identical intervention and coverage per original episode."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EP = ["day", "member", "seed"]
KEY = EP + ["lag"]


def summarize(root, out):
    out.mkdir(parents=True, exist_ok=True)
    rows = pd.concat([pd.read_csv(root / f"seed{s}.csv") for s in range(5)], ignore_index=True)
    assert len(rows) == 3054 * 5 and not rows.duplicated(KEY + ["model"]).any()
    assert rows.groupby(KEY).model.nunique().eq(5).all()
    rows["good"] = (rows.air < -.05) & (rows.nh <= 0) & (rows["do"] <= 0)
    native = rows[rows.member == rows.model].sort_values(KEY)
    np.testing.assert_array_equal(native.good, native.source_good)
    assert native.good.sum() == 1210
    grouped = rows.groupby(KEY)
    candidates = grouped.agg(votes=("good", "sum"), nh_min=("nh", "min"), nh_max=("nh", "max"),
                             first_min=("nh4_first", "min"), first_max=("nh4_first", "max"))
    for k in ("score", "source_good"):
        candidates[k] = native.set_index(KEY)[k]
    candidates = candidates.reset_index().sort_values(KEY)
    selected = candidates[candidates.score >= .5].drop_duplicates(EP)
    candidates["selected"] = candidates.set_index(KEY).index.isin(selected.set_index(KEY).index)
    assert candidates.selected.sum() == 7 and candidates[candidates.selected].source_good.sum() == 5
    dates = sorted(candidates.day.unique())
    assert len(dates) == 40
    direction, coverage, support = [], [], []
    for period, days in [("all", dates), ("early", dates[:20]), ("late", dates[20:])]:
        for seed in ["all", *range(5)]:
            part = candidates[candidates.day.isin(days)]
            if seed != "all":
                part = part[part.seed == seed]
            tags = dict(period=period, seed=seed, candidates=len(part))
            counts = part.votes.value_counts()
            support.append(dict(**tags, **{f"votes_{k}": int(counts.get(k, 0)) for k in range(6)}))
            for label, tol in [("first", 1e-8), ("nh", 0)]:
                low, high = part[label + "_min"], part[label + "_max"]
                bins = dict(all_down=(high < -tol), all_up=(low > tol),
                            all_tie=(low >= -tol) & (high <= tol), opposite=(low < -tol) & (high > tol),
                            down_and_tie=(low < -tol) & high.between(-tol, tol),
                            up_and_tie=(high > tol) & low.between(-tol, tol))
                assert np.all(np.sum(list(bins.values()), axis=0) == 1)
                direction.append(dict(**tags, horizon=label, **{k: int(v.sum()) for k, v in bins.items()}))
            for needed in (4, 5):
                yes = part.votes >= needed
                got = yes & part.selected
                ep = len(days)*5*(5 if seed == "all" else 1)
                coverage.append(dict(**tags, votes_required=needed, episodes=ep,
                                     eligible_episodes=len(part.drop_duplicates(EP)),
                                     favorable_candidates=int(yes.sum()),
                                     favorable_episodes=len(part[yes].drop_duplicates(EP)),
                                     selected=int(part.selected.sum()), captured=int(got.sum()),
                                     native_positive=int(part.source_good.sum()),
                                     retained_native_positive=int((yes & part.source_good).sum()),
                                     auc=roc_auc_score(yes, part.score) if yes.nunique() == 2 else np.nan))
    pd.DataFrame(support).to_csv(out / "votes.csv", index=False)
    pd.DataFrame(direction).to_csv(out / "nh_response.csv", index=False)
    pd.DataFrame(coverage).to_csv(out / "coverage.csv", index=False)
    candidates[candidates.selected].to_csv(out / "selected.csv", index=False)
    candidates.to_csv(root / "candidates.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    summarize(args.root, args.out)

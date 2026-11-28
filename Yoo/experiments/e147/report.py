"""Count unique decisions, realized daily savings and agreement across model versions."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

M = ["air_m3", "nh4_min", "excess", "do_min"]
KEY = ["prefix", "seed", "episode"]
DEC = ["prefix", "seed", "decision"]


def success(frame):
    return (frame.air_m3 < -.05) & (frame[M[1:]] <= 1e-8).all(axis=1)


def report(source, dest):
    data = pd.concat([pd.read_csv(source / f"p{p}_s{s}.csv", float_precision="round_trip")
                      for p in (108, 148, 188) for s in range(10, 15)], ignore_index=True)
    assert not data.duplicated(KEY + ["version", "evaluator"]).any()
    assert data.groupby(KEY + ["version"]).size().eq(5).all()
    assert np.isfinite(data[M].values).all()
    data["yes"] = success(data)
    # Keep every observed replica. A model supports a decision only if all its duplicate runs pass.
    groups = data.groupby(DEC + ["version", "evaluator"])
    spread = groups[M].max() - groups[M].min()
    spread["mixed"] = groups.yes.nunique() > 1
    spread["copies"] = groups.size()
    data["supports"] = groups.yes.transform("all")
    current = data[data.version == "current"]
    older = data[data.version == "previous"]
    votes = current.groupby(KEY).supports.sum()
    own = current[current.evaluator == current.origin].set_index(KEY).copy().sort_index()
    own["votes"] = votes
    own["previous_votes"] = older.groupby(KEY).supports.sum()
    for name in M:
        own["ensemble_" + name] = current.groupby(KEY)[name].mean()
        own["previous_" + name] = older.groupby(KEY)[name].mean()
    own = own.reset_index()
    assert len(own) == 3000
    unique = own.drop_duplicates(DEC)
    histogram, effects, byseed, transfer, byday = [], [], [], [], []
    for n, group in own.groupby("prefix"):
        distinct = unique[unique.prefix == n]
        for level in range(6):
            histogram.append(dict(prefix=n, votes=level, visits=int((group.votes == level).sum()),
                                  unique_decisions=int((distinct.votes == level).sum())))
        for method, mask in (("own", group.yes), ("four_or_more", group.votes >= 4),
                             ("unanimous", group.votes == 5),
                             ("other_four", group.votes - group.supports.astype(int) == 4)):
            selected = group[mask]
            result = group[M].mul(mask.astype(float), axis=0)
            summary = dict(prefix=n, method=method, origins=len(group), selected=len(selected),
                           decisions=len(distinct), selected_decisions=len(selected.drop_duplicates(DEC)),
                           dates=selected.day.nunique(), seeds=selected.seed.nunique(),
                           realized_joint=int(selected.yes.sum()), mean_native_air=group.native_daily_air.mean(),
                           conditional_saving=-selected.air_m3.mean(), air_saving_pct=-100*result.air_m3.sum()/group.native_daily_air.sum())
            effects.append(dict(**summary, **result.mean().to_dict()))
            for (seed, day), dg in group.groupby(["seed", "day"]):
                indices = dg.index
                take = mask.loc[indices]
                byday.append(dict(prefix=n, method=method, seed=seed, day=day, n=len(dg),
                                  selected=int(take.sum()), **result.loc[indices].mean().to_dict()))
            for seed, sg in group.groupby("seed"):
                take = mask.loc[sg.index]
                byseed.append(dict(prefix=n, method=method, seed=seed, selected=take.sum(),
                                   unique_selected=len(sg[take].drop_duplicates(DEC)),
                                   **result.loc[sg.index].mean().to_dict()))
        if n != 108:
            primary = group.votes == 5
            chosen = group[primary]
            for k in (4, 5):
                transferable = chosen[chosen.previous_votes >= k]
                transfer.append(dict(prefix=n, previous_votes_at_least=k,
                                     current_visits=len(chosen), retained_visits=len(transferable),
                                     current_unique=len(chosen.drop_duplicates(DEC)),
                                     retained_unique=len(transferable.drop_duplicates(DEC)),
                                     dates=transferable.day.nunique(), seeds=transferable.seed.nunique(),
                                     **{"current_" + m:group["ensemble_" + m].mul(primary).mean() for m in M},
                                     **{m:group["previous_" + m].mul(primary).mean() for m in M}))
    # For each unique unanimous current decision, identify which criteria fail in the earlier version.
    chosen_keys = unique.loc[unique.votes == 5, DEC]
    prior_checks = older.merge(chosen_keys, on=DEC, validate="many_to_one")
    prior_checks["air_ok"] = prior_checks.air_m3 < -.05
    for name in M[1:]:
        prior_checks[name + "_ok"] = prior_checks[name] <= 1e-8
    reqs = ["air_ok"] + [name + "_ok" for name in M[1:]]
    failures = prior_checks.groupby(DEC)[reqs].all().reset_index()
    failures = failures.merge(unique[DEC + ["day", "first", "previous_votes"]], on=DEC, validate="one_to_one")
    assert ((failures.previous_votes == 5) == failures[reqs].all(axis=1)).all()
    dest.mkdir(parents=True, exist_ok=True)
    failures.to_csv(dest / "previous_criteria.csv", index=False)
    duplicate = spread[spread.copies > 1].reset_index().groupby(["prefix", "version"])
    audits = duplicate[M].max().join(duplicate.mixed.sum().rename("mixed_votes")).join(duplicate.size().rename("repeated_evaluations"))
    audits.to_csv(dest / "duplicate_check.csv")
    for name, rows in (("votes", histogram), ("effects", effects), ("by_seed", byseed),
                       ("transfer", transfer), ("by_day", byday)):
        pd.DataFrame(rows).to_csv(dest / f"{name}.csv", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("dest", type=Path)
    args = p.parse_args()
    report(args.source, args.dest)

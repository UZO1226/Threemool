"""Evaluate discrimination at the candidate level and paired outcomes at the episode level."""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

KEY = ["day", "member", "seed"]
METRICS = ["air", "nh", "do", "excess"]


def joint(x):
    return (x.air < -.05) & (x.nh < 0) & (x["do"] <= 0)


def summarize(source, dest):
    dest.mkdir(parents=True, exist_ok=True)
    performance, seasonal, seed_tables, day_tables, absolute = [], [], [], [], []
    for split in ["fit", "autumn", "winter"]:
        opt = pd.read_csv(source / (split + "_options.csv"))
        base = pd.read_csv(source / (split + "_baseline.csv")).set_index(KEY).sort_index()
        opt = opt.sort_values(KEY + ["lag"])
        assert not opt.duplicated(KEY + ["lag"]).any() and base.index.is_unique
        assert ((opt.air < -.05) & (opt.nh <= 0) & (opt["do"] <= 0)).equals(opt.good)
        selected = opt[opt.score >= .5].drop_duplicates(KEY)
        first = opt.drop_duplicates(KEY)
        oracle = opt[opt.good].sort_values(KEY + ["air", "lag"]).drop_duplicates(KEY)
        for seed in ["all", *range(5)]:
            cand, chosen, init, hindsight = [g if seed == "all" else g[g.seed == seed]
                                             for g in (opt, selected, first, oracle)]
            perf = dict(split=split, seed=seed, episodes=len(base) if seed == "all" else
                        len(base.xs(seed, level="seed")), candidates=len(cand), eligible=len(init),
                        positives=int(cand.good.sum()), prevalence=cand.good.mean(),
                        auc=roc_auc_score(cand.good, cand.score) if cand.good.nunique() == 2 else None,
                        selections=len(chosen), correct=int(chosen.good.sum()), precision=chosen.good.mean(),
                        first_correct=int(init.good.sum()), first_precision=init.good.mean(),
                        opportunities=len(hindsight))
            performance.append(perf)
        if split != "winter":
            continue
        dates = sorted(base.index.get_level_values("day").unique())
        actual = {"B": base[METRICS], "A": base[["A_" + k for k in METRICS]].set_axis(METRICS, axis=1)}
        for name, options in [("observed", selected), ("first", first), ("oracle", oracle)]:
            change = options.set_index(KEY)[METRICS].reindex(base.index, fill_value=0.)
            assert change.notna().all().all()
            actual[name] = actual["B"] + change
        for name, values in actual.items():
            absolute.append(dict(rule=name, **values.mean().to_dict()))
        for name in ["observed", "first", "oracle"]:
            for comp in ["B", "A"]:
                delta = actual[name] - actual[comp]
                daily = delta.groupby(["day", "seed"]).mean()
                day_tables.append(daily.assign(rule=name, comparator=comp, joint=joint(daily)).reset_index())
                for period, subset in [("all", dates), ("early", dates[:20]), ("late", dates[20:])]:
                    part = delta.loc[delta.index.get_level_values("day").isin(subset)]
                    per_seed = part.groupby("seed").mean()
                    avg = per_seed.mean()
                    count = int(joint(per_seed).sum())
                    seed_tables.append(per_seed.assign(rule=name, comparator=comp, period=period,
                                                       joint=joint(per_seed)).reset_index())
                    seasonal.append(dict(rule=name, comparator=comp, period=period, **avg.to_dict(),
                                         joint_seeds=count, passed=bool(joint(avg) and count >= 4)))
    for name, values in [("discrimination", performance), ("absolute", absolute), ("paired", seasonal)]:
        pd.DataFrame(values).to_csv(dest / (name + ".csv"), index=False)
    pd.concat(seed_tables).to_csv(dest / "seeds.csv", index=False)
    pd.concat(day_tables).to_csv(dest / "days.csv", index=False)
    tests = [row["passed"] for row in seasonal if row["rule"] == "observed" and row["period"] != "all"]
    assert len(tests) == 4
    (dest / "decision.json").write_text(json.dumps({"candidate": all(tests)}, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("dest", type=Path)
    args = p.parse_args()
    summarize(args.source, args.dest)

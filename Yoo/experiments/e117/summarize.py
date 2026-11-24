"""Paired date and seed summaries of the frozen contrasts."""
import argparse
import json
from pathlib import Path

import pandas as pd

METRICS = ["air", "nh", "do", "excess"]


def joint(frame):
    return (frame.air < -.05) & (frame.nh < 0) & (frame["do"] <= 0)


def summarize(source, out):
    df = pd.read_csv(source / "episodes.csv")
    assert len(df) == 4000 and not df.duplicated(["day", "member", "seed", "arm"]).any()
    dates = sorted(df.day.unique())
    keys = ["day", "member", "seed"]
    by_arm = {name: g.set_index(keys)[METRICS].sort_index() for name, g in df.groupby("arm")}
    seeds, daily, summary = [], [], []
    for a, b in [("C", "B"), ("C", "A"), ("D", "A"), ("B", "A")]:
        delta = (by_arm[a] - by_arm[b]).reset_index()
        assert delta[METRICS].notna().all().all()
        contrast = a + "-" + b
        by_day = delta.groupby(["day", "seed"])[METRICS].mean()
        by_day["joint"] = joint(by_day)
        day = by_day.groupby("day")[METRICS].mean()
        day["joint_seeds"] = by_day.joint.groupby("day").sum()
        daily.append(day.assign(contrast=contrast).reset_index())
        for period, days in [("all", dates), ("early", dates[:20]), ("late", dates[20:])]:
            s = delta[delta.day.isin(days)].groupby("seed")[METRICS].mean()
            s["joint"] = joint(s)
            seeds.append(s.assign(contrast=contrast, period=period).reset_index())
            mean = s[METRICS].mean()
            summary.append(dict(contrast=contrast, period=period, **mean.to_dict(),
                                joint_seeds=int(s.joint.sum()),
                                repeated_days=int((day.loc[days].joint_seeds >= 4).sum()),
                                passed=bool(s.joint.sum() >= 4 and joint(mean))))
    table = pd.DataFrame(summary)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "summary.csv", index=False)
    pd.concat(seeds).to_csv(out / "seeds.csv", index=False)
    pd.concat(daily).to_csv(out / "dates.csv", index=False)
    absolute = df.groupby("arm")[METRICS].mean()
    absolute.to_csv(out / "absolute.csv")
    primary = table[table.contrast.isin(["C-B", "C-A"]) & table.period.isin(["early", "late"])]
    decision = {"candidate": bool(primary.passed.all()), "triggered": int(((df.arm == "A") & (df.trigger >= 0)).sum())}
    (out / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(absolute, table.to_string(index=False), decision, sep="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    summarize(args.source, args.out)

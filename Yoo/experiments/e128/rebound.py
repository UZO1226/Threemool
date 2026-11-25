"""Align the first extra command and the end of initial air savings to observed NH4."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path("/home/jio/fcsrl_src/results")
EVENTS = ("extra", "offset")


def observation(trace, command):
    return float(np.float32(trace[command - 1, 1]))


def measure(old, new, issued):
    assert 2 <= issued < len(old) and old.shape == new.shape
    np.testing.assert_array_equal(old[:issued], new[:issued])
    delta = new[issued:, 0] - old[issued:, 0]
    volume = np.cumsum(delta * .25)
    np.testing.assert_allclose(volume[0], -62.5, rtol=0, atol=1e-7)
    final = volume[-1]
    record = dict(minutes_left=15 * len(delta), daily_air=final, saving=final < -.05)
    extra = np.flatnonzero(delta > 1e-6)
    offset = np.flatnonzero(volume >= -1e-8)
    if len(offset):
        assert len(extra) and extra[0] <= offset[0]
    for event, hits in zip(EVENTS, (extra, offset)):
        if not len(hits):
            continue
        j = int(hits[0])
        t = issued + j
        nh = observation(new, t)
        values = dict(command_min=15*j, event_min=15*(j + int(event == "offset")),
                      flow_gap=delta[j], nh=nh, recent=nh-observation(new, t-1),
                      from_cut=nh-observation(new, issued), paired=nh-observation(old, t))
        record.update({f"{event}_{k}": v for k, v in values.items()})
    return record


def read_cases(root):
    prior = pd.read_csv(root / "e126/main/cases.csv")
    records = []
    for seed in range(10):
        if seed < 5:
            selected = prior.query("mode == 'matched' and seed == @seed and issued >= 0")
            ids = selected.trace_index
            with np.load(root / f"e126/main/traces{seed}.npz") as z:
                old, new = z["old"], z["new"]
        else:
            cases = pd.read_csv(root / f"e127/main/cases{seed}.csv")
            matched = cases[cases["mode"] == "matched"].reset_index(drop=True)
            assert list(zip(matched.day, matched.member)) == [
                (d, m) for d in sorted(matched.day.unique()) for m in range(5)]
            selected = matched[matched.issued >= 0]
            ids = selected.index
            with np.load(root / f"e127/main/paired{seed}.npz") as z:
                old, new = z["old"][0], z["new"][0]
        for row, i in zip(selected.itertuples(), ids):
            values = measure(old[i], new[i], row.issued)
            np.testing.assert_allclose(values["daily_air"], row.air, rtol=0, atol=1e-7)
            records.append(dict(seed_set="0-4" if seed < 5 else "5-9", seed=seed, day=row.day,
                                member=row.member, lag=row.lag, period=row.period, issued=row.issued, **values))
    frame = pd.DataFrame(records)
    assert frame.groupby("seed_set").size().to_dict() == {"0-4": 88, "5-9": 54}
    assert not frame.duplicated(["seed", "member", "day"]).any()
    return frame


def summarize(frame):
    rows = []
    for seed_set, cohort in frame.groupby("seed_set"):
        for period in ("all", "early", "late"):
            season = cohort if period == "all" else cohort[cohort.period == period]
            for seed in ("all", *sorted(season.seed.unique())):
                part = season if seed == "all" else season[season.seed == seed]
                for group, sub in (("all", part), ("saving", part[part.saving]), ("other", part[~part.saving])):
                    if sub.empty:
                        continue
                    follow = sub[sub.minutes_left >= 60]
                    for event in EVENTS:
                        values = sub.filter(regex=f"^{event}_").copy()
                        values.columns = values.columns.str.removeprefix(f"{event}_")
                        hits = values.dropna()
                        row = dict(seed_set=seed_set, period=period, seed=seed, group=group, event=event,
                                   n=len(sub), observed=len(hits), censored=len(sub)-len(hits),
                                   complete_60=len(follow), offset_by_60=(follow.offset_event_min <= 60).sum(),
                                   event_min_median=hits.event_min.median(), command_min_median=hits.command_min.median(),
                                   nh_over_4=(hits.nh > 4).sum(), paired_equal=(hits.paired == 0).sum())
                        for col in ("recent", "from_cut", "paired"):
                            row[col + "_positive"] = (hits[col] > 0).sum()
                            row[col + "_at_least_point1"] = (hits[col] >= .1).sum()
                            row[col + "_median"] = hits[col].median()
                        rows.append(row)
    return pd.DataFrame(rows)


def smoke():
    old = np.full((8, 3), 1000.)
    old[:, 1] = 2.
    new = old.copy()
    new[2:6, 0] += [-250., 100., -100., 255.]
    new[2:6, 1] = [2.5, 2.6, 2.25, 9.]
    x = measure(old, new, 2)
    assert (x["extra_event_min"], x["offset_command_min"], x["offset_event_min"]) == (15, 45, 60)
    assert x["extra_recent"] == .5 and x["offset_nh"] == 2.25
    new[5:, 1] = 0.
    assert measure(old, new, 2) == x
    new[3:, 0] = old[3:, 0]
    y = measure(old, new, 2)
    assert y["saving"] and "offset_event_min" not in y and "extra_event_min" not in y


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--tables", type=Path, default=HERE)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    smoke()
    if not args.smoke:
        cases = read_cases(args.root)
        args.tables.mkdir(parents=True, exist_ok=True)
        cases.to_csv(args.tables / "cases.csv", index=False)
        summarize(cases).to_csv(args.tables / "summary.csv", index=False)

"""Evaluate fixed warning signals at the first air savings offset."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e122"))
import rebound

KEY = rebound.KEY
ROOT = rebound.ROOT
P = rebound.p
METHODS = ("all_offsets", "nh_rise", "air_rise", "both")


def signals_at(cut, boundary):
    # NH4 is measured at the end of a row. Air is decided at its start.
    nh = cut[[boundary - 1, boundary - 3], 1].astype(np.float32).astype(float)
    nh_change = nh[0] - nh[1]
    air_change = cut[boundary, 0] - cut[boundary - 2, 0]
    return dict(nh_change=nh_change, air_change=air_change, nh_rise=nh_change >= .1,
                air_rise=air_change >= 250, both=nh_change >= .1 and air_change >= 250,
                all_offsets=True)


def analyze(base, cut, t, expected):
    delta = np.cumsum(.25 * (cut[t:, 0] - base[t:, 0]))
    assert np.isfinite(delta).all() and t > 0
    first, lasting = rebound.crossings(delta)
    np.testing.assert_allclose([first, lasting, delta[-1]],
                               [expected.first, expected.lasting, expected.final_air],
                               atol=1e-6, rtol=0, equal_nan=True)
    if np.isnan(first):
        return dict(status="no_offset")
    step = int(first // 15)
    recovered = np.flatnonzero(delta[step:] < -1e-8)
    minutes = (recovered[0] + 1) * 15 if len(recovered) else np.nan
    result = dict(recovery_minutes=minutes)
    if t + step == len(cut):
        return dict(**result, status="day_ended")
    if step < 2:
        return dict(**result, status="short_history")
    return dict(**result, status="eligible", **signals_at(cut, t + step))


def replay(out, seed):
    e = P.r.e113
    plan = P.r.read(P.r.ROOT / "plan.json")
    for name, digest in plan["input_sha256"].items():
        P.r.verify(e.prepared / name, digest)
    splits, days = e.base.load_selected(e.prepared / "data.pkl")
    assert splits["val"] == plan["winter"] == sorted(plan["winter"])
    episodes = [(day, member) for day in plan["winter"] for member in range(5)]
    index = {pair: i for i, pair in enumerate(episodes)}
    e.base.DeviceConfig().select_device(-1)
    env = e.BatchEnv(days, e.FastDynamics(e.prepared / "surrogate.pkl"))
    agents = P.r.load(seed, P.r.read(P.r.ROOT / "checksums.json"))[1]
    with np.load(ROOT / "e117/main" / f"seed{seed}.npz") as saved:
        base, trigger = saved["B"], saved["trigger"]
    cases = pd.read_csv(ROOT / "e122/main/cases.csv").query("seed == @seed").drop_duplicates(KEY)
    records = []
    for lag, rows in cases.groupby("lag"):
        cut, _, stamp, chosen = P.simulate(env, episodes, agents, slot=lag)
        np.testing.assert_array_equal(stamp, trigger)
        for row in rows.itertuples():
            i = index[(row.day, row.member)]
            assert chosen[i] == lag and row.t == stamp[i] + lag
            np.testing.assert_allclose(base[i, :row.t], cut[i, :row.t], atol=1e-6, rtol=0)
            records.append({**{k: getattr(row, k) for k in KEY},
                            **analyze(base[i], cut[i], row.t, row)})
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out / f"seed{seed}.csv", index=False)
    print("Verified seed", seed, flush=True)


def summarize(source, dest):
    raw = pd.concat(pd.read_csv(source / f"seed{i}.csv") for i in range(5))
    archived = pd.read_csv(ROOT / "e122/main/cases.csv")
    assert len(raw) == 254 and not raw.duplicated(KEY).any()
    cases = archived.merge(raw, on=KEY, how="left", validate="many_to_one")
    assert len(cases) == 318 and cases.status.notna().all()
    performance, groups = [], []
    for mode, cohort in cases.groupby("mode"):
        for period in ("all", "early", "late"):
            subset = cohort if period == "all" else cohort[cohort.period == period]
            for seed in ("all", *range(5)):
                sample = subset if seed == "all" else subset[subset.seed == seed]
                x = sample[sample.status == "eligible"]
                large = x.group == "large"
                saving = x.group == "saving"
                recovery = x.recovery_minutes.notna()
                tags = dict(mode=mode, period=period, seed=seed, total=len(sample), eligible=len(x),
                            no_offset=(sample.status == "no_offset").sum(),
                            day_ended=(sample.status == "day_ended").sum(),
                            short_history=(sample.status == "short_history").sum())
                for method in METHODS:
                    alarm = x[method].eq(True)
                    tp = (large & alarm).sum()
                    fp = (~large & alarm).sum()
                    performance.append(dict(**tags, method=method, large=large.sum(), tp=tp,
                        nonlarge=(~large).sum(), fp=fp, saving=saving.sum(), saving_alert=(alarm & saving).sum(),
                        persistent=(~recovery).sum(), persistent_alert=(alarm & ~recovery).sum(),
                        recovered=recovery.sum(), recovered_alert=(alarm & recovery).sum(),
                        passed=large.any() and (~large).any() and tp > large.sum()/2 and fp <= (~large).sum()/4))
                for group, g in x.groupby("group"):
                    groups.append(dict(**tags, group=group, n=len(g), recovered=g.recovery_minutes.count(),
                        recovery_min_median=g.recovery_minutes.median(), nh_change_median=g.nh_change.median(),
                        air_change_median=g.air_change.median()))
    dest.mkdir(parents=True, exist_ok=True)
    perf = pd.DataFrame(performance)
    perf.to_csv(dest / "performance.csv", index=False)
    pd.DataFrame(groups).to_csv(dest / "groups.csv", index=False)
    cases.to_csv(source / "cases.csv", index=False)
    halves = perf.query("mode == 'matched' and period != 'all' and seed == 'all' and method == 'both'")
    assert len(halves) == 2
    print("Candidate:", bool(halves.passed.all()))


def smoke():
    base = np.full((8, 3), 1000.)
    cut = base.copy()
    cut[1:5, 0] += [-100, 110, 300, -400]
    cut[2, 1] += .25
    from types import SimpleNamespace
    expected = SimpleNamespace(first=30, lasting=np.nan, final_air=-22.5)
    result = analyze(base, cut, 1, expected)
    assert result["both"] and result["recovery_minutes"] == 30
    changed = cut.copy()
    changed[3:, 1:] = 0
    changed[4:, 0] = 1
    assert signals_at(cut, 3) == signals_at(changed, 3)
    assert analyze(base, base + np.array([-1, 0, 0]), 1,
                   SimpleNamespace(first=np.nan, lasting=np.nan, final_air=-1.75))["status"] == "no_offset"
    assert analyze(base[:3], cut[:3], 1, SimpleNamespace(first=30, lasting=30, final_air=2.5))["status"] == "day_ended"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "replay", "report"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--seed", type=int, choices=range(5))
    parser.add_argument("--tables", type=Path, default=HERE)
    args = parser.parse_args()
    if args.stage == "smoke":
        smoke()
    elif args.stage == "replay":
        if args.seed is None:
            parser.error("replay requires --seed")
        replay(args.out, args.seed)
    else:
        summarize(args.out, args.tables)

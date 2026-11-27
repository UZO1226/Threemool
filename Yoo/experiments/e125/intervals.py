"""Decompose paired daily air and water quality differences after a single cut."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e122"))
import rebound

KEY, ROOT, timing = rebound.KEY, rebound.ROOT, rebound.p
METRICS = ["air", "nh", "do", "excess"]
PHASES = ("through_offset", "next_3h", "hours_3_to_6", "after_6h")


def contributions(trace):
    air, nh, do = trace.T
    return np.column_stack((air * .25, nh > 4, do > 2, np.maximum(nh - 4, 0) * .25))


def split_trace(delta, first):
    n = len(delta)
    edge = n if np.isnan(first) else int(first) // 15
    assert 1 <= edge <= n
    index = np.arange(n)
    phase = np.searchsorted([edge, edge + 12, edge + 24], index, side="right")
    blocks = (n - edge) // 4
    peak = None
    if blocks:
        volumes = delta[edge:edge + 4 * blocks, 0].reshape(blocks, 4).sum(axis=1)
        best = int(np.argmax(volumes))
        if volumes[best] > 0:
            peak = edge + 4 * best
    return phase, peak


def replay(out, seed):
    e = timing.r.e113
    plan = timing.r.read(timing.r.ROOT / "plan.json")
    for name, digest in plan["input_sha256"].items():
        timing.r.verify(e.prepared / name, digest)
    splits, days = e.base.load_selected(e.prepared / "data.pkl")
    assert plan["winter"] == splits["val"] == sorted(plan["winter"])
    episodes = [(d, m) for d in plan["winter"] for m in range(5)]
    index = {pair: i for i, pair in enumerate(episodes)}
    e.base.DeviceConfig().select_device(-1)
    env = e.BatchEnv(days, e.FastDynamics(e.prepared / "surrogate.pkl"))
    agents = timing.r.load(seed, timing.r.read(timing.r.ROOT / "checksums.json"))[1]
    choices = pd.read_csv(ROOT / "e121/main/choices.csv").query("arm == 'risk_air' and seed == @seed")
    offsets = pd.read_csv(ROOT / "e122/main/cases.csv").query("seed == @seed")
    cases = offsets.merge(choices[["mode", *KEY, *METRICS]], on=["mode", *KEY], validate="one_to_one")
    assert len(cases) == len(offsets)
    with np.load(ROOT / "e117/main" / f"seed{seed}.npz") as saved:
        base, trigger = saved["B"], saved["trigger"]
    records = []
    for lag, rows in cases.drop_duplicates(KEY).groupby("lag"):
        cut, _, stamp, chosen = timing.simulate(env, episodes, agents, slot=lag)
        np.testing.assert_array_equal(stamp, trigger)
        for row in rows.itertuples():
            i, t = index[(row.day, row.member)], row.t
            assert chosen[i] == lag and t == stamp[i] + lag
            timing.prior.compare_traces(cut[i, :t], base[i, :t])
            delta = contributions(cut[i, t:]) - contributions(base[i, t:])
            np.testing.assert_allclose(delta.sum(axis=0), [getattr(row, k) for k in METRICS], atol=1e-6, rtol=0)
            first, lasting = rebound.crossings(delta[:, 0].cumsum())
            np.testing.assert_allclose([first, lasting], [row.first, row.lasting], equal_nan=True)
            phase, peak = split_trace(delta, first)
            j = np.arange(len(delta))
            frame = pd.DataFrame(delta, columns=METRICS).assign(j=j, phase=np.array(PHASES)[phase])
            frame["peak"] = False if peak is None else (peak <= j) & (j < peak + 4)
            frame["follow"] = False if peak is None or peak + 8 > len(j) else (peak + 4 <= j) & (j < peak + 8)
            records.append(frame.assign(**{k: getattr(row, k) for k in KEY}))
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(records).to_csv(out / f"seed{seed}.csv", index=False)
    print("verified", seed, flush=True)


def windows(frame, first):
    peak = frame[frame.peak]
    follow = frame[frame.follow]
    assert len(peak) in (0, 4) and len(follow) in (0, 4)
    groups = {p: frame[frame.phase == p] for p in PHASES}
    groups.update(day=frame, peak=peak, follow=follow, two_hours=frame[frame.peak | frame.follow] if len(follow) else follow)
    for name, part in groups.items():
        if part.empty:
            continue
        yield dict(window=name, steps=len(part), **part[METRICS].sum().to_dict(),
                   peak_min_after_offset=int(peak.j.min()) * 15 - first if name == "peak" else np.nan)


def report(source, dest):
    traces = pd.concat(pd.read_csv(source / f"seed{s}.csv") for s in range(5))
    assert not traces.duplicated([*KEY, "j"]).any()
    archive = pd.read_csv(ROOT / "e122/main/cases.csv")
    unique = archive.drop_duplicates(KEY)
    grouped = traces.groupby(KEY)
    assert len(grouped) == len(unique) == 254
    records = []
    for row in unique.itertuples():
        key = {k: getattr(row, k) for k in KEY}
        frame = grouped.get_group(tuple(key.values())).sort_values("j")
        np.testing.assert_array_equal(frame.j, np.arange(row.remaining_min // 15))
        for record in windows(frame, row.first):
            records.append({**key, **record})
    cases = archive[["mode", "period", "group", *KEY]].merge(pd.DataFrame(records), on=KEY)
    cases.to_csv(source / "windows.csv", index=False)
    large = cases[cases.group == "large"]
    summaries = []
    for mode, cohort in large.groupby("mode"):
        for period in ("all", "early", "late"):
            part = cohort if period == "all" else cohort[cohort.period == period]
            for seed in ("all", *range(5)):
                x = part if seed == "all" else part[part.seed == seed]
                day = x[x.window == "day"]
                if day.empty:
                    continue
                for window, w in x.groupby("window"):
                    totals = w[METRICS].sum()
                    summaries.append(dict(mode=mode, period=period, seed=seed, window=window, n=len(w),
                        cases=len(day), hours=w.steps.sum() * .25, **totals.to_dict(),
                        air_per_hour=totals.air / (w.steps.sum() * .25),
                        share=totals.air / day.air.sum() if window in PHASES else np.nan,
                        nh_lower=(w.nh < 0).sum(), nh_equal=(w.nh == 0).sum(), nh_higher=(w.nh > 0).sum(),
                        excess_lower=(w.excess < -1e-8).sum(), excess_higher=(w.excess > 1e-8).sum(),
                        peak_min_median=w.peak_min_after_offset.median() if window == "peak" else np.nan))
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(dest / "summary.csv", index=False)
    large[large.window.isin(["day", "peak", "follow", "two_hours"])].to_csv(dest / "large_cases.csv", index=False)


def smoke():
    b = np.array([[100, 4., 2.], [200, 5., 1.], [300, 3., 3.]])
    np.testing.assert_array_equal(contributions(b).sum(0), [150, 1, 1, .25])
    d = np.zeros((28, 4))
    d[:, 0] = [-2, 2] + [1] * 8 + [0] * 16 + [50, 50]
    phase, peak = split_trace(d, 30)
    np.testing.assert_array_equal(np.bincount(phase), [2, 12, 12, 2])
    assert peak == 2 and split_trace(d, np.nan)[1] is None
    assert split_trace(-np.ones((10, 4)), 30)[1] is None


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
        report(args.out, args.tables)

"""Paired native outcomes, circular date-block intervals and common-air curves."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["air_m3", "nh4_min", "nh4_excess", "do_min", "cost"]
CONTROLS = ["base", "actor", "extra", "base_risk", "actor_penalty"]
ARMS = CONTROLS + ["combined"]
SEEDS = list(range(10, 15))
SCALES = [800, 900, 1000, 1100, 1250]


def day_metrics(raw):
    with np.load(raw) as z:
        air, state = z["air"], z["states"]
    assert air.shape == (200, 95) and state.shape == (200, 95, 5)
    assert np.isfinite(state).all() and ((air >= 0) & (air <= 9000)).all()
    nh = state[:, :, 0]
    n = (nh > 4).sum(axis=1)
    d = (state[:, :, 2:].max(axis=2) > 2).sum(axis=1)
    metrics = np.column_stack((air.sum(axis=1) / 4, 15*n, np.maximum(nh-4, 0).sum(axis=1)/4,
                               15*d, n+d))
    return metrics.reshape(40, 5, len(METRICS)).mean(axis=1)


def improves(delta):
    return (delta[..., 0] < -.05) & (delta[..., 1:4] <= 1e-8).all(axis=-1)


def block_interval(daily):
    rng = np.random.default_rng(135)
    estimates = []
    for _ in range(2000):
        s = rng.integers(5, size=5)
        days = (rng.integers(40, size=(8, 1)) + np.arange(5)) % 40
        estimates.append(daily[s][:, days.ravel()].mean(axis=(0, 1)))
    return np.quantile(estimates, [.025, .975], axis=0)


def curve_gap(a, b):
    # Collapse duplicate air values before interpolation of every water outcome.
    a, b = [pd.DataFrame(x, columns=METRICS).groupby("air_m3").mean() for x in (a, b)]
    lo = max(a.index.min(), b.index.min())
    hi = min(a.index.max(), b.index.max())
    if hi <= lo:
        return lo, hi, [np.nan] * (len(METRICS)-1)
    grid = np.linspace(lo, hi, 21)
    return lo, hi, [(np.interp(grid, a.index, a[k]) - np.interp(grid, b.index, b[k])).mean()
                    for k in METRICS[1:]]


def report(out, dest):
    daily = {(arm, seed, scale):day_metrics(out / f"seed_{seed}/{arm}_{scale}.npz")
             for arm in ARMS for seed in SEEDS for scale in SCALES}
    means = {k:v.mean(axis=0) for k,v in daily.items()}
    orig = pd.read_csv(Path(__file__).resolve().parents[1] / "e133/runs.csv").set_index(["method", "seed"])
    for arm, method in [("base","fcsrl"), ("actor","fcsrl_actor"), ("extra","actor_extra_ppo")]:
        for seed in SEEDS:
            expected = orig.loc[(method,seed),["air_m3","nh4_minutes","nh4_excess_mg_h_L","do_minutes","cost"]]
            np.testing.assert_allclose(means[arm,seed,1000], expected.to_numpy(float), atol=1e-8, rtol=0)
    rows = [dict(arm=k[0], seed=k[1], scale=k[2]/1000, **dict(zip(METRICS,v))) for k,v in means.items()]
    frame = pd.DataFrame(rows)
    primary, paired, intervals, curves = {}, [], [], []
    for control in CONTROLS:
        differences = np.stack([daily["combined",s,1000]-daily[control,s,1000] for s in SEEDS])
        seed_diffs = differences.mean(axis=1)
        mean = seed_diffs.mean(axis=0)
        low, high = block_interval(differences)
        passes = improves(seed_diffs)
        primary[control] = dict(joint_seeds=int(passes.sum()), mean_joint=bool(improves(mean)),
                                all_upper_nonpositive=bool((high[:4] <= 0).all()))
        for i, metric in enumerate(METRICS):
            intervals.append(dict(control=control, metric=metric, delta=mean[i], low=low[i], high=high[i]))
        for i, s in enumerate(SEEDS):
            paired.append(dict(control=control, seed=s, joint=passes[i], **dict(zip(METRICS, seed_diffs[i]))))
    for method, controls in [("combined", CONTROLS), ("actor", ["base", "extra"])]:
        for control in controls:
            for s in SEEDS:
                lo, hi, values = curve_gap(*[np.stack([means[a,s,k] for k in SCALES])
                                             for a in (method,control)])
                curves.append(dict(method=method, control=control, seed=s, lo_air=lo, hi_air=hi,
                                   overlap=hi>lo, **dict(zip(METRICS[1:],values))))
    scaled = []
    for scale in SCALES:
        diffs = np.stack([means["combined",s,scale]-means["base",s,1000] for s in SEEDS])
        scaled.append(dict(scale=scale/1000, joint_seeds=int(improves(diffs).sum()),
                           **dict(zip(METRICS, diffs.mean(axis=0)))))
    native = frame[frame.scale == 1].groupby("arm")[METRICS].mean()
    dest.mkdir(parents=True, exist_ok=True)
    native.to_csv(dest / "native.csv")
    for name, table in [("operating_points",frame),("paired",paired),("intervals",intervals),("curves",curves),("fixed_scales",scaled)]:
        pd.DataFrame(table).to_csv(dest / f"{name}.csv", index=False)
    b = primary["base"]
    result = dict(comparisons=primary, primary_candidate=b["joint_seeds"]>=4 and b["mean_joint"],
                  supported=b["joint_seeds"]>=4 and b["mean_joint"] and b["all_upper_nonpositive"])
    (dest / "decision.json").write_text(json.dumps(result, indent=2)+"\n")
    print(native.round(3).to_string())
    print(result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("dest", type=Path)
    args = p.parse_args()
    report(args.out, args.dest)

"""Check selected multipliers and compare paired winter days and training seeds."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["air_m3", "nh4_minutes", "nh4_excess_mg_h_L", "do_minutes"]
METHODS = ["fcsrl", "fcsrl_actor", "actor_extra_ppo"]
CONTROLS = ["fcsrl", "actor_extra_ppo"]
SEEDS = list(range(10, 15))


def gap(a, b):
    return 200 * abs(a - b) / (a + b)


def checked_days(out, row, dates):
    stem = f"{row['method']}_{row['seed']}"
    episodes = pd.read_json(out / f"{stem}.json", convert_dates=False)
    assert list(zip(episodes.day, episodes.member)) == [(d, m) for d in dates for m in range(5)]
    with np.load(out / f"{stem}.npz") as arrays:
        air, state = arrays["air"], arrays["states"]
    assert state.shape == (200, 95, 5) and air.shape == (200, 95)
    assert np.isfinite(state).all() and np.isfinite(air).all()
    np.testing.assert_allclose(air.sum(axis=1) / 4, episodes.air_m3, rtol=0, atol=1e-8)
    nh, do = state[:, :, 0], state[:, :, 2:].max(axis=2)
    np.testing.assert_array_equal((nh > 4).sum(axis=1), episodes.nh4_cost)
    np.testing.assert_array_equal((do > 2).sum(axis=1), episodes.do_cost)
    np.testing.assert_allclose(np.maximum(nh - 4, 0).sum(axis=1) / 4, episodes.nh4_excess_mg_h_L,
                               rtol=0, atol=1e-8)
    episodes["nh4_minutes"] = 15 * episodes.nh4_cost
    episodes["do_minutes"] = 15 * episodes.do_cost
    return episodes.groupby("day")[METRICS].mean()


def joint(delta):
    return (delta.nh4_minutes < -1e-8 and delta.nh4_excess_mg_h_L < -1e-8
            and delta.do_minutes <= 1e-8)


def report(out, dest):
    dates = json.loads(Path("/home/jio/fcsrl_src/results/e50/prepared/data_manifest.json").read_text())["splits"]["val"]
    selections = json.loads((out / "selection.json").read_text())
    assert sorted((r["method"], r["seed"]) for r in selections) == sorted((m,s) for m in METHODS for s in SEEDS)
    rows, days = [], {}
    for row in selections:
        values = {int(k): v for k, v in row["candidates"].items()}
        target = row["target"]
        key = lambda a: (abs(values[a] - target), abs(a - 1000), a)
        coarse = range(500, 1501, 100)
        best = min(coarse, key=key)
        expected = set(coarse) | {best + d for d in (-50, -25, 25, 50) if 500 <= best + d <= 1500}
        assert set(values) == expected and row["alpha_milli"] == min(values, key=key)
        assert values[row["alpha_milli"]] == row["train_air"]
        daily = checked_days(out, row, dates)
        days[row["method"], row["seed"]] = daily
        rows.append(dict(method=row["method"], seed=row["seed"], alpha=row["alpha_milli"] / 1000,
                         target=target, train_air=row["train_air"],
                         target_error_pct=100 * abs(row["train_air"] / target - 1), **daily.mean()))
    runs = pd.DataFrame(rows).set_index(["method", "seed"])
    assert runs.target.nunique() == 1
    pairs, per_day = [], []
    for seed in SEEDS:
        actor = runs.loc[("fcsrl_actor", seed)]
        for control in CONTROLS:
            other = runs.loc[(control, seed)]
            train_gap = gap(actor.train_air, other.train_air)
            winter_gap = gap(actor.air_m3, other.air_m3)
            target_ok = max(actor.target_error_pct, other.target_error_pct) <= 1
            comparable = bool(target_ok and train_gap <= 1 and winter_gap <= 3)
            a, b = days["fcsrl_actor", seed], days[control, seed]
            daily = a - b
            daily["gap_pct"] = gap(a.air_m3, b.air_m3)
            daily["seed"], daily["control"] = seed, control
            per_day.append(daily.reset_index())
            delta = actor[METRICS] - other[METRICS]
            np.testing.assert_allclose(delta, daily[METRICS].mean(), rtol=0, atol=1e-8)
            pairs.append(dict(seed=seed, control=control, train_targets_ok=target_ok,
                              train_gap_pct=train_gap, winter_gap_pct=winter_gap,
                              matched_days=int((daily.gap_pct <= 3).sum()), comparable=comparable,
                              improves=bool(comparable and joint(delta)), **delta))
    pairs = pd.DataFrame(pairs)
    by_seed = pairs.groupby("seed")[["comparable", "improves"]].all()
    enough = int(by_seed.comparable.sum()) >= 4
    result = {"both_comparable": by_seed.index[by_seed.comparable].tolist(),
              "both_improved": by_seed.index[by_seed.improves].tolist(),
              "decision": ("insufficient_matches" if not enough else
                           "timing_analysis_supported" if by_seed.improves.sum() >= 4 else "criterion_not_met")}
    dest.mkdir(parents=True, exist_ok=True)
    runs.to_csv(dest / "runs.csv")
    pairs.to_csv(dest / "paired.csv", index=False)
    pd.concat(per_day).to_csv(dest / "days.csv", index=False)
    (dest / "decision.json").write_text(json.dumps(result, indent=2) + "\n")
    print(pairs.round(3).to_string(index=False))
    print(result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("dest", type=Path)
    args = p.parse_args()
    report(args.out, args.dest)

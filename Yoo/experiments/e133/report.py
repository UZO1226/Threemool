"""Verify the 15 final rollouts and apply the frozen e133 paired decision."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = ("fcsrl", "fcsrl_actor", "actor_extra_ppo")
CONTROLS = ("fcsrl", "actor_extra_ppo")
SEEDS = range(10, 15)
OUTCOMES = ["air_m3", "nh4_minutes", "nh4_excess_mg_h_L", "do_minutes"]
PREP = Path("/home/jio/fcsrl_src/results/e50/prepared")


def read(path):
    return json.loads(path.read_text())


def check_rollout(dest, metrics, dates):
    episodes = read(dest / "winter_episodes.json")
    assert [(r["day"], r["member"]) for r in episodes] == [
        (day, member) for day in dates for member in range(5)
    ]
    with np.load(dest / "winter_rollout.npz") as raw:
        states, air = raw["states"], raw["air"]
    assert states.shape == (200, 95, 5) and air.shape == (200, 95)
    assert np.isfinite(states).all() and np.isfinite(air).all()
    nh4, do = states[:, :, 0], states[:, :, 2:].max(axis=2)
    daily = {
        "air_m3": air.sum(axis=1) * 0.25,
        "nh4_cost": (nh4 > 4).sum(axis=1),
        "do_cost": (do > 2).sum(axis=1),
        "nh4_excess_mg_h_L": np.maximum(nh4 - 4, 0).sum(axis=1) * 0.25,
        "do_excess_mg_h_L": np.maximum(do - 2, 0).sum(axis=1) * 0.25,
    }
    for key, values in daily.items():
        np.testing.assert_allclose(values, [r[key] for r in episodes], rtol=0, atol=1e-8)
        np.testing.assert_allclose(values.mean(), metrics[key], rtol=0, atol=1e-8)
    total = daily["nh4_cost"] + daily["do_cost"]
    np.testing.assert_allclose(total.mean(), metrics["cost"], rtol=0, atol=1e-8)
    assert bool(total.mean() <= 12) == metrics["mean_cost_within_12"]


def run_row(out, method, seed, config, dates):
    dest = out / method / f"seed_{seed}"
    assert read(dest / "effective_config.json") == dict(config, method=method, seed=seed)
    training = read(dest / "metrics.json")
    assert training["steps"] == 401280 and training["normalizer_and_prior_frozen"]
    assert training["method"] == method and training["seed"] == seed
    metrics = read(dest / "winter_metrics.json")
    assert metrics["n_episodes"] == 200
    for key, value in training["val_final"].items():
        np.testing.assert_allclose(value, metrics[key], rtol=0, atol=1e-8)
    check_rollout(dest, metrics, dates)
    logs = read(dest / "train_log.json")
    assert [r["update"] for r in logs] == list(range(1, 67))
    rng = np.random.default_rng(seed)
    for row in logs:
        assert row["member"] == rng.integers(5)
        rng.integers(188, size=64)
        if method in ("fcsrl_actor", "actor_extra_ppo"):
            assert row["preserve_fit_samples"] == 2048 and row["preserve_check_samples"] == 1024
            assert row["extra_attempted_steps"] in (0, 50)
            for critic in ("reward", "cost"):
                assert row[f"before_{critic}_error"] == row[f"after_{critic}_error"]
    return dict(method=method, seed=seed, air_m3=metrics["air_m3"],
                nh4_minutes=15 * metrics["nh4_cost"], nh4_excess_mg_h_L=metrics["nh4_excess_mg_h_L"],
                do_minutes=15 * metrics["do_cost"], cost=metrics["cost"],
                within_12=metrics["mean_cost_within_12"], elapsed_seconds=training["elapsed_seconds"],
                extra_steps=int(sum(r.get("extra_attempted_steps", 0) for r in logs)),
                accepted_updates=int(sum(r.get("preserve_accepted", 0) for r in logs)),
                extra_cpu_seconds=sum(r.get("extra_cpu_seconds", 0) for r in logs),
                extra_wall_seconds=sum(r.get("extra_wall_seconds", 0) for r in logs))


def joint(delta):
    return (delta["air_m3"] < -0.05) & (delta[OUTCOMES[1:]].to_numpy() <= 1e-8).all(axis=-1)


def report(out, tables):
    config = read(PREP.parent / "main/runs/fcsrl/seed_0/effective_config.json")
    dates = read(PREP / "data_manifest.json")["splits"]["val"]
    runs = pd.DataFrame([run_row(out, m, s, config, dates) for m in METHODS for s in SEEDS])
    groups = {m: runs[runs.method == m].set_index("seed") for m in METHODS}
    actor = groups["fcsrl_actor"]
    paired = []
    means_pass = {}
    for control in CONTROLS:
        columns = OUTCOMES + ["cost"]
        delta = actor[columns] - groups[control][columns]
        means_pass[control] = bool(joint(delta.mean()))
        delta["joint"] = joint(delta)
        delta["control"] = control
        paired.append(delta.reset_index())
    pairs = pd.concat(paired, ignore_index=True)
    both = pairs.groupby("seed")["joint"].all()
    base = groups["fcsrl"]
    cost_wins = int((actor.cost < base.cost).sum())
    summary = runs.groupby("method", sort=False)[OUTCOMES + ["cost"]].mean()
    summary["within_12_seeds"] = runs.groupby("method").within_12.sum()
    decision = dict(mean_joint=means_pass, paired_joint_counts={
        k: int(v) for k, v in pairs.groupby("control").joint.sum().items()},
        joint_seeds_both=[int(s) for s in both[both].index],
        replication_pass=bool(all(means_pass.values()) and both.sum() >= 4),
        original_cost_wins=cost_wins, original_e50_pass=bool(cost_wins >= 4 and
            actor.cost.mean() < base.cost.mean() and actor.air_m3.mean() <= base.air_m3.mean() * 1.02),
        extra_steps_matched=bool(actor.extra_steps.equals(groups["actor_extra_ppo"].extra_steps)))
    tables.mkdir(parents=True, exist_ok=True)
    runs.to_csv(tables / "runs.csv", index=False)
    pairs.to_csv(tables / "paired.csv", index=False)
    summary.to_csv(tables / "summary.csv")
    (tables / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    print(summary.to_string())
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--tables", type=Path, required=True)
    args = parser.parse_args()
    report(args.out, args.tables)

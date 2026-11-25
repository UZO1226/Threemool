"""Paired single-reduction trajectories and prospective time selection."""
import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e117"))
import air_feedback as prior
from darrow_env import AIR_MAX, action_to_air, air_to_action

r = prior.replay
METRICS = prior.METRICS
INPUTS = ["nh4", "no3", "do1", "do2", "do3", "flow", "temperature", "sin", "cos",
          "influent", "previous_air", "nh_change", "proposal", "reduction", "lag"]
KEYS = ["day", "member", "seed"]


def scores(model, x):
    if model is None or 1 not in model.classes_:
        return np.zeros(len(x))
    return model.predict_proba(x)[:, list(model.classes_).index(1)]


def simulate(env, episodes, agents, slot=0, model=None):
    obs = env.reset(episodes)
    previous = obs[:, 0].copy()
    n = len(obs)
    trigger = np.full(n, -1)
    chosen = np.zeros(n, dtype=int)
    anchor = np.zeros(n)
    features = np.full((n, 4, len(INPUTS)), np.nan)
    trace = np.empty((n, 95, 3))
    for t in range(95):
        native = r.action(agents["weighted"], obs)
        flow = action_to_air(native)
        other = r.action(agents["uniform"], obs)
        change = obs[:, 0] - previous
        fire = ((trigger < 0) & (t > 0) & (obs[:, 0] >= 3) & (change > 0)
                & (action_to_air(other) - flow >= 250))
        trigger[fire] = t
        anchor[fire] = flow[fire]
        act = native.astype(float)
        act[fire] = other[fire]
        lag = t - trigger
        active = (trigger >= 0) & (lag >= 1) & (lag <= 4) & (chosen == 0)
        if active.any():
            alternate = obs.copy()
            alternate[active, 10] = anchor[active] / AIR_MAX
            reduction = (flow - action_to_air(r.action(agents["weighted"], alternate))) / 2
            ids = np.flatnonzero(active & (reduction >= 1))
            if len(ids):
                x = np.column_stack((obs[ids], change[ids], flow[ids], reduction[ids], lag[ids]))
                features[ids, lag[ids] - 1] = x
                take = (scores(model, x) >= .5) if slot == "online" else (lag[ids] == slot)
                chosen_ids = ids[take]
                act[chosen_ids] = air_to_action(flow[chosen_ids] - reduction[chosen_ids])
                chosen[chosen_ids] = lag[chosen_ids]
        previous = obs[:, 0].copy()
        obs, _, _, done, info = env.step(act)
        trace[:, t] = np.column_stack((info["air"], env.state[:, 0], env.state[:, 2:].max(1)))
    assert done and np.isfinite(trace).all()
    return trace, features, trigger, chosen


def phase(dates, agents_by_seed, env, model=None, winter=False):
    episodes = [(d, m) for d in dates for m in range(5)]
    options, records = [], []
    for seed, agents in agents_by_seed.items():
        keys = pd.DataFrame(episodes, columns=KEYS[:2]).assign(seed=seed)
        base, x, trigger, _ = simulate(env, episodes, agents)
        metrics = prior.daily(*base.transpose(2, 0, 1))
        base_row = keys.assign(trigger=trigger)
        base_row[METRICS] = metrics
        eligible = np.isfinite(x[:, :, 0])
        expected = base.copy()
        chosen = np.zeros(len(base), dtype=int)
        if winter:
            path = r.ROOT.parents[1] / "e117/main" / f"seed{seed}.npz"
            with np.load(path) as z:
                np.testing.assert_array_equal(trigger, z["trigger"])
                prior.compare_traces(base, z["B"])
                base_row[["A_" + k for k in METRICS]] = prior.daily(*z["A"].transpose(2, 0, 1))
        records.append(base_row)
        for j in range(4):
            branch, branch_x, stamp, taken = simulate(env, episodes, agents, slot=j+1)
            ids = np.flatnonzero(eligible[:, j])
            np.testing.assert_array_equal(stamp, trigger)
            np.testing.assert_array_equal(taken, (j+1)*eligible[:, j])
            np.testing.assert_array_equal(branch_x[ids, j], x[ids, j])
            prefix = (~eligible[:, j, None]) | (np.arange(95) < trigger[:, None] + j + 1)
            np.testing.assert_array_equal(branch[prefix], base[prefix])
            t = trigger[ids] + j + 1
            np.testing.assert_allclose(base[ids, t, 0] - branch[ids, t, 0], x[ids, j, -2], atol=1e-7)
            data = keys.iloc[ids].copy()
            data[INPUTS] = x[ids, j]
            deltas = prior.daily(*branch.transpose(2, 0, 1))[ids] - metrics[ids]
            data[METRICS] = deltas
            data["good"] = (data.air < -.05) & (data.nh <= 0) & (data["do"] <= 0)
            options.append(data)
            if winter and len(ids):
                take = ids[(chosen[ids] == 0) & (scores(model, x[ids, j]) >= .5)]
                expected[take] = branch[take]
                chosen[take] = j+1
        if winter:
            actual, _, stamp, selected = simulate(env, episodes, agents, slot="online", model=model)
            np.testing.assert_array_equal(stamp, trigger)
            np.testing.assert_array_equal(selected, chosen)
            np.testing.assert_array_equal(actual, expected)
        print(dates[0], "seed", seed, "eligible", int(eligible.sum()), flush=True)
    return pd.concat(options, ignore_index=True), pd.concat(records, ignore_index=True)


def main(out, smoke):
    e = r.e113
    plan = r.read(r.ROOT / "plan.json")
    for name, digest in plan["input_sha256"].items():
        r.verify(e.prepared / name, digest)
    hashes = r.read(r.ROOT / "checksums.json")
    splits, days = e.base.load_selected(e.prepared / "data.pkl")
    fall, winter = plan["held"], plan["winter"]
    assert fall == sorted(fall) and len(set(fall)) == 48 and len(set(winter)) == 40
    assert max(fall) < min(winter) and winter == splits["val"]
    assert set(fall).isdisjoint(plan["train"])
    e.base.DeviceConfig().select_device(-1)
    env = e.BatchEnv(days, e.FastDynamics(e.prepared / "surrogate.pkl"))
    agents = {seed: r.load(seed, hashes)[1] for seed in (range(1) if smoke else range(5))}
    if smoke:
        samples, b = phase(fall[:4], agents, env)
        assert len(samples) and len(b) == 20
        return
    out.mkdir(parents=True, exist_ok=False)
    for name, dates in [("fit", fall[:32]), ("autumn", fall[32:]), ("winter", winter)]:
        if name == "winter":
            model = joblib.load(out / "selector.joblib")
        samples, baseline = phase(dates, agents, env, model if name == "winter" else None, name == "winter")
        if name == "fit":
            model = None
            if len(samples):
                model = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=20,
                                               random_state=0, n_jobs=1)
                model.fit(samples[INPUTS].to_numpy(), samples.good)
            joblib.dump(model, out / "selector.joblib")
        samples["score"] = scores(model, samples[INPUTS].to_numpy()) if len(samples) else []
        samples.to_csv(out / (name + "_options.csv"), index=False)
        baseline.to_csv(out / (name + "_baseline.csv"), index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    main(args.out, args.smoke)

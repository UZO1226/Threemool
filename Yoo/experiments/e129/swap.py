"""Query a frozen policy on paired event observations and exact feature swaps."""
import argparse
from math import comb
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e116"))
import divergence as prior
from darrow_env import AIR_MAX, air_to_action

r = prior.replay
e = r.e113
ROOT = r.ROOT.parents[1]
TOL = .05


def gather(env, episodes, trace, decision):
    obs = env.reset(episodes)
    inputs = np.empty_like(obs)
    for t in range(int(decision.max()) + 1):
        at = decision == t
        if at.any():
            inputs[at] = obs[at]
            np.testing.assert_array_equal(obs[at, 0], trace[at, t-1, 1].astype(np.float32))
            np.testing.assert_array_equal(obs[at, 10], (trace[at, t-1, 0] / AIR_MAX).astype(np.float32))
        obs, _, _, _, info = env.step(air_to_action(trace[:, t, 0]))
        np.testing.assert_allclose(info["air"], trace[:, t, 0], rtol=0, atol=1e-8)
        np.testing.assert_allclose(env.state[:, 0], trace[:, t, 1], rtol=0, atol=2e-6)
        np.testing.assert_allclose(env.state[:, 2:].max(1), trace[:, t, 2], rtol=0, atol=2e-6)
    return inputs


def queries(agent, control, changed):
    np.testing.assert_array_equal(control[:, 5:10], changed[:, 5:10])
    output = []
    for mask in range(16):
        x = control.copy()
        for bit, cols in enumerate(prior.GROUPS.values()):
            if mask & (1 << bit):
                x[:, cols] = changed[:, cols]
        output.append(r.action_to_air(r.action(agent, x)))
    return np.stack(output)


def check_shapley(values, phi):
    for bit in range(4):
        without = [m for m in range(16) if not m & (1 << bit)]
        expected = sum((values[m | (1 << bit)] - values[m]) / (4 * comb(3, m.bit_count())) for m in without)
        np.testing.assert_allclose(phi[:, bit], expected, rtol=0, atol=1e-8)
    np.testing.assert_allclose(phi.sum(axis=1), values[-1] - values[0], atol=1e-8, rtol=0)


def analyze_seed(events, seed, env):
    if seed < 5:
        frame = pd.read_csv(ROOT / "e126/main/cases.csv").query("mode == 'matched' and seed == @seed")
        checkpoint = r.ROOT
        path = ROOT / f"e126/main/traces{seed}.npz"
    else:
        frame = pd.read_csv(ROOT / f"e127/main/cases{seed}.csv").query("mode == 'matched'").reset_index(drop=True)
        frame["trace_index"] = frame.index
        checkpoint = ROOT / "e127/main"
        path = checkpoint / f"paired{seed}.npz"
    keys = ["seed", "day", "member", "lag", "issued"]
    rows = events.merge(frame[[*keys, "trace_index"]], on=keys, validate="one_to_one")
    assert len(rows) == len(events)
    ids = rows.trace_index.to_numpy(dtype=int)
    with np.load(path) as archive:
        control, changed = [(archive[k] if seed < 5 else archive[k][0])[ids] for k in ("old", "new")]
    episodes = list(zip(rows.day, rows.member))
    step = rows.issued.to_numpy() + rows.extra_command_min.to_numpy(dtype=int) // 15
    i = np.arange(len(rows))
    a, b = (gather(env, episodes, trace, step) for trace in (control, changed))
    np.testing.assert_allclose(b[:, 0].astype(float) - a[:, 0], rows.extra_paired, atol=1e-8)
    np.testing.assert_allclose(b[:, 0], rows.extra_nh, atol=1e-8)
    agent = e.base.common.make_agent("fcsrl_actor", seed, e.prepared, e.base.common.config(66))
    agent.load(checkpoint / "weighted" / f"seed_{seed}/model_final")
    v = queries(agent, a, b)
    old_flow, new_flow = control[i, step, 0], changed[i, step, 0]
    np.testing.assert_allclose(v[0], old_flow, atol=TOL, rtol=0)
    np.testing.assert_allclose(v[15], new_flow, atol=TOL, rtol=0)
    gap = new_flow - old_flow
    np.testing.assert_allclose(gap, rows.extra_flow_gap, atol=1e-8, rtol=0)
    phi = prior.decompose(v)
    check_shapley(v, phi)
    result = rows[[*keys, "seed_set", "period", "saving", "extra_command_min", "extra_paired", "daily_air"]].copy()
    result["gap"] = gap
    result["previous_flow_gap"] = changed[i, step-1, 0] - control[i, step-1, 0]
    result["endpoint_error"] = np.maximum(abs(v[0] - old_flow), abs(v[15] - new_flow))
    for bit, (name, cols) in enumerate(prior.GROUPS.items()):
        effect = v[15] - v[15 ^ (1 << bit)]
        equal = (a[:, cols] == b[:, cols]).all(axis=1)
        np.testing.assert_allclose(effect[equal], 0, atol=1e-8, rtol=0)
        result["swap_" + name] = effect
        result["phi_" + name] = phi[:, bit]
    result["swap_both"] = v[15] - v[6]  # Keep NO3 and DO from the late-cut path.
    for mask, values in enumerate(v):
        result[f"flow_mask_{mask}"] = values
    return result


def summarize(cases):
    output = []
    methods = ["gap", *["swap_" + k for k in prior.GROUPS], "swap_both", *["phi_" + k for k in prior.GROUPS]]
    for seed_set, cohort in cases.groupby("seed_set"):
        for period in ("all", "early", "late"):
            season = cohort if period == "all" else cohort[cohort.period == period]
            for seed in ("all", *sorted(season.seed.unique())):
                part = season if seed == "all" else season[season.seed == seed]
                for group, x in (("all", part), ("saving", part[part.saving]), ("loss", part[~part.saving])):
                    if x.empty:
                        continue
                    for method in methods:
                        values = x[method]
                        output.append(dict(seed_set=seed_set, period=period, seed=seed, group=group, method=method,
                                           n=len(x), mean=values.mean(), median=values.median(), total=values.sum(),
                                           positive=(values > TOL).sum(), negative=(values < -TOL).sum(),
                                           near_zero=(values.abs() <= TOL).sum()))
    return pd.DataFrame(output)


def main(dest):
    events = pd.read_csv(HERE.parent / "e128/cases.csv").dropna(subset=["extra_command_min"])
    assert events.groupby("seed_set").size().to_dict() == {"0-4": 82, "5-9": 49}
    plan = r.read(r.ROOT / "plan.json")
    for name, digest in plan["input_sha256"].items():
        r.verify(e.prepared / name, digest)
    splits, days = e.base.load_selected(e.prepared / "data.pkl")
    assert splits["val"] == plan["winter"] and set(events.day) <= set(splits["val"])
    e.base.DeviceConfig().select_device(-1)
    env = e.BatchEnv(days, e.FastDynamics(e.prepared / "surrogate.pkl"))
    cases = pd.concat([analyze_seed(g, s, env) for s, g in events.groupby("seed")], ignore_index=True)
    dest.mkdir(parents=True, exist_ok=True)
    cases.to_csv(dest / "cases.csv", index=False)
    summarize(cases).to_csv(dest / "summary.csv", index=False)
    print(f"Reproduced both original decisions at {len(cases)} events. Max error {cases.endpoint_error.max():.8f} m3/h.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, default=HERE)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    prior.smoke()
    if not args.smoke:
        main(args.tables)

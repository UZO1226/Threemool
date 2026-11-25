"""Localize e130 air losses and audit two observable signals."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e116"))
import divergence as attribution
from darrow_env import air_to_action

r, e = attribution.replay, attribution.replay.e113
ROOT = r.ROOT.parents[1]
BANDS = [(0, 4), (4, 12), (12, 24), (24, 95)]
KEY = ["seed", "day", "member"]


def first(mask):
    pos = np.flatnonzero(mask)
    return int(pos[0]) if len(pos) else None


def signals(trace, t):
    assert 3 <= t < 95
    nh, air = trace[t-1, 1], trace[t-1, 0]
    dn, da = nh - trace[t-3, 1], air - trace[t-3, 0]
    return dict(nh4=nh, nh_change=dn, prev_air=air, air_change=da, rise=dn > 0, both=dn > 0 and da > 0)


def timeline(a, b, t):
    delta = b[t:, 0] - a[t:, 0]
    crossing = first(delta.cumsum() / 4 > .05)
    rebound = first((np.arange(len(delta)) >= 4) & (delta >= 250))
    rec = dict(cross_min=(crossing+1)*15 if crossing is not None else np.nan,
               rebound_min=rebound*15 if rebound is not None else np.nan)
    for i, (lo, hi) in enumerate(BANDS):
        interval = delta[lo:hi]
        rec[f"air{i}"] = interval.sum() / 4
        rec[f"steps{i}"] = len(interval)
    for name, k in [("start", t), ("hour", t+4)]:
        if k < len(a):
            rec.update({f"{name}_{col}": value for col, value in signals(b, k).items()})
    if rebound is not None:
        rec["rebound_nh4"] = b[t+rebound-1, 1]
    return rec


def match(row, candidates):
    eligible = candidates[(candidates.seed == row.seed) & (candidates.day != row.day)
                          & (candidates.air < -.05)].copy()
    dt = abs(eligible.step - row.step) / 8
    dq = abs(eligible.gap - row.gap) / 125
    eligible["distance"] = dt**2 + dq**2
    return eligible[(dt <= 1) & (dq <= 1)].sort_values(["distance", "day", "member"])


def smoke():
    a = np.zeros((95, 3))
    b = a.copy()
    b[88:, 0] = [-200, 100, 100, 0, 400, 1, 2]
    b[87, 1] = 1
    row = timeline(a, b, 88)
    assert row["cross_min"] == 75 and row["rebound_min"] == 60
    assert not row["hour_rise"] and row["start_rise"] and row["air0"] == 0 and row["steps1"] == 3
    assert "hour_rise" not in timeline(a, b, 91)
    q = pd.DataFrame(dict(seed=[8]*4, day=list("abdc"), member=[0]*4, air=[100, -1, -2, -3],
                          step=[50, 58, 59, 58], gap=[100, 225, 100, 226]))
    assert match(q.iloc[0], q).day.tolist() == ["b"]
    attribution.smoke()


def read_events():
    archived = pd.read_csv(HERE.parent / "e130/cases.csv")
    daily = archived.query("window == 'day'").set_index(KEY, verify_integrity=True)
    hourly = archived.query("window == '60min'").set_index(KEY, verify_integrity=True)
    records, traces = [], {}
    for seed in range(10):
        keys = pd.read_csv(ROOT / f"e130/main/keys{seed}.csv")
        with np.load(ROOT / f"e130/main/trace{seed}.npz") as z:
            keep, remove = z["keep"], z["remove"]
        for ix, row in keys.iterrows():
            key = (seed, row.day, row.member)
            a, b = keep[ix], remove[ix]
            t = int(row.step)
            np.testing.assert_array_equal(a[:t], b[:t])
            record = timeline(a, b, t)
            ref = daily.loc[key]
            np.testing.assert_array_equal([row.step, row.gap], [ref.step, ref.gap])
            np.testing.assert_allclose(sum(record[f"air{i}"] for i in range(4)), ref.air, rtol=0, atol=1e-7)
            if t + 4 <= 95:
                np.testing.assert_allclose(record["air0"], hourly.loc[key, "air"], rtol=0, atol=1e-7)
            assert sum(record[f"steps{i}"] for i in range(4)) == 95-t
            records.append(dict(**row, **record, air=ref.air, joint=ref.joint, nh_minutes=15*ref.nh))
            traces[key] = (a, b)
    cases = pd.DataFrame(records)
    assert len(cases) == len(traces) == len(daily) == 131
    cases["large"] = cases.groupby("seed_set").air.rank(ascending=False, method="first") <= 4
    return cases, traces


def alerts(cases):
    rows = []
    for batch, cohort in cases.groupby("seed_set"):
        groups = [("all", "all", cohort)]
        for col in ["seed", "member"]:
            groups.extend((col, level, x) for level, x in cohort.groupby(col))
        for grouping, level, part in groups:
            for time in ("start", "hour"):
                for rule in ("rise", "both"):
                    name = f"{time}_{rule}"
                    x = part.dropna(subset=[name])
                    hit = x[name].astype(bool)
                    before = x.cross_min.isna() | (x.cross_min > (60 if time == "hour" else 0))
                    rows.append(dict(seed_set=batch, grouping=grouping, level=level, time=time, rule=rule,
                        n=len(x), large=x.large.sum(), caught=(hit & x.large).sum(),
                        caught_before_cross=(hit & x.large & before).sum(), other=(~x.large).sum(),
                        false_alarm=(hit & ~x.large).sum(), joint=x.joint.sum(), flagged_joint=(hit & x.joint).sum()))
    return pd.DataFrame(rows)


def replay_obs(env, episodes, traces):
    obs = env.reset(episodes)
    history = []
    for t in range(95):
        history.append(obs)
        obs, _, _, _, info = env.step(air_to_action(traces[:, t, 0]))
        np.testing.assert_allclose(info["air"], traces[:, t, 0], rtol=0, atol=1e-8)
        np.testing.assert_allclose(env.state[:, 0], traces[:, t, 1], rtol=0, atol=2e-6)
        np.testing.assert_allclose(env.state[:, 2:].max(axis=1), traces[:, t, 2], rtol=0, atol=2e-6)
    return np.stack(history)


def explain(selected, traces):
    plan = r.read(r.ROOT / "plan.json")
    for name, digest in plan["input_sha256"].items():
        r.verify(e.prepared / name, digest)
    splits, days = e.base.load_selected(e.prepared / "data.pkl")
    assert splits["val"] == plan["winter"]
    e.base.DeviceConfig().select_device(-1)
    env = e.BatchEnv(days, e.FastDynamics(e.prepared / "surrogate.pkl"))
    assert set(selected.seed) == {8}
    agent = e.base.common.make_agent("fcsrl_actor", 8, e.prepared, e.base.common.config(66))
    agent.load(ROOT / "e127/main/weighted/seed_8/model_final")
    episodes = list(zip(selected.day, selected.member))
    a, b = (np.stack(v) for v in zip(*(traces[(8, *key)] for key in episodes)))
    control, changed = [replay_obs(env, episodes, v) for v in (a, b)]
    phi = attribution.attribution(agent, control, changed)
    native = []
    for obs in [control, changed]:
        native.append(np.stack([r.action_to_air(r.action(agent, group)) for group in obs]))
    output = []
    for i, row in enumerate(selected.itertuples()):
        t = int(row.step)
        for pred, actual in zip(native, (a, b)):
            np.testing.assert_allclose(pred[t+1:, i], actual[i, t+1:, 0], rtol=0, atol=.05)
        np.testing.assert_array_equal(control[t, i], changed[t, i])
        delta = b[i, t:, 0] - a[i, t:, 0]
        direct = np.zeros(len(delta))
        direct[0] = -row.gap
        contribution = phi[t:, i]
        np.testing.assert_allclose(direct + contribution.sum(1), delta, rtol=0, atol=.05)
        for j, v in enumerate(contribution):
            output.append(dict(day=row.day, member=row.member, seed=row.seed, elapsed_min=j*15,
                flow_gap=delta[j], direct=direct[j], **dict(zip(attribution.GROUPS, v))))
    return pd.DataFrame(output)


def run(dest):
    cases, traces = read_events()
    top = cases[(cases.seed_set == "5-9") & cases.large].sort_values("air", ascending=False)
    selected = list(top.index)
    pairs = []
    for idx, row in top.iterrows():
        pool = match(row, cases)
        rec = dict(loss_day=row.day, loss_member=row.member, eligible=len(pool))
        if len(pool):
            chosen = pool.iloc[0]
            selected.append(chosen.name)
            rec.update(save_day=chosen.day, save_member=chosen.member, distance=chosen.distance,
                       time_diff_min=15*abs(row.step-chosen.step), gap_diff=abs(row.gap-chosen.gap))
        pairs.append(rec)
    steps = explain(cases.loc[list(dict.fromkeys(selected))], traces)
    dest.mkdir(parents=True, exist_ok=True)
    for name, frame in [("cases", cases), ("alerts", alerts(cases)), ("matches", pd.DataFrame(pairs)), ("policy", steps)]:
        frame.to_csv(dest / f"{name}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, default=HERE)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    smoke()
    if not args.smoke:
        run(args.tables)

"""Run a one-decision previous-air intervention after a recorded reduction."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e130"))
import branch

p, e, root = branch.p, branch.e, branch.root
METRICS = ["air", "nh", "do", "excess"]
KEY = ["seed", "day", "member"]


def eligible(t, origin, stamp, last, older):
    return (t >= origin + 4) & (stamp < 0) & (last - older >= 250)


def correction(native, alternate):
    half = (native - alternate) / 2
    return np.where(half >= 1, half, 0.)


def smoke():
    np.testing.assert_array_equal(eligible(7, np.array([4, 3, 3, 3]), np.array([-1, -1, 6, -1]),
                                           np.array([600, 600, 600, 599]), 350), [False, True, False, False])
    np.testing.assert_array_equal(correction(np.array([10, 12, 13, 11]), 11), [0, 0, 1, 0])


def load_cases(seed, cohort):
    if cohort == "diagnostic":
        source = root / "e130/main"
        keys = pd.read_csv(source / f"keys{seed}.csv").rename(columns={"step": "origin"})
        with np.load(source / f"trace{seed}.npz") as z:
            expected, keep = z["remove"], z["keep"]
        return keys, expected, keep
    if seed < 5:
        source = root / "e126/main"
        keys = pd.read_csv(source / "cases.csv").query("seed == @seed and mode == 'frozen' and issued >= 0")
        with np.load(source / f"traces{seed}.npz") as z:
            expected = z["new"][keys.trace_index]
    else:
        source = root / "e127/main"
        frozen = pd.read_csv(source / f"cases{seed}.csv").query("mode == 'frozen'").reset_index(drop=True)
        keys = frozen[frozen.issued >= 0]
        with np.load(source / f"paired{seed}.npz") as z:
            expected = z["new"][1, keys.index]
    np.testing.assert_allclose(p.prior.daily(*expected.transpose(2, 0, 1)),
                               keys[["new_" + k for k in METRICS]], rtol=0, atol=1e-7)
    return keys.assign(origin=keys.issued), expected, None


def simulate(env, agent, keys, expected, dates, apply):
    episodes = list(zip(keys.day, keys.member))
    slots = [dates.index(day) * 5 + m for day, m in episodes]
    assert len(set(slots)) == len(keys)
    obs = env.reset(episodes)
    batch = np.zeros((200, obs.shape[1]), dtype=obs.dtype)
    trace = np.empty_like(expected)
    stamp = np.full(len(keys), -1)
    cut = np.zeros(len(keys))
    origin = keys.origin.to_numpy(int)
    for t in range(95):
        batch[slots] = obs
        action = p.r.action(agent, batch)[slots].astype(float)
        native = p.action_to_air(action)
        prefix = t <= origin
        action[prefix] = p.air_to_action(expected[prefix, t, 0])
        if t >= 2:
            last, older = trace[:, t-1, 0], trace[:, t-2, 0]
            use = eligible(t, origin, stamp, last, older)
            if use.any():
                np.testing.assert_allclose(obs[:, 10], last / p.AIR_MAX, rtol=0, atol=1e-7)
                batch[np.asarray(slots)[use], 10] = older[use] / p.AIR_MAX
                alternate = p.action_to_air(p.r.action(agent, batch)[slots])
                reduce = correction(native, alternate)
                now = use & (reduce > 0)
                stamp[now], cut[now] = t, reduce[now]
                if apply:
                    action[now] = p.air_to_action(native[now] - reduce[now])
        obs, _, _, done, info = env.step(action)
        trace[:, t] = np.column_stack((info["air"], env.state[:, 0], env.state[:, 2:].max(1)))
    assert done
    return trace, stamp, cut


def run(seed, out):
    plan = p.r.read(p.r.ROOT / "plan.json")
    for name, value in plan["input_sha256"].items():
        p.r.verify(e.prepared / name, value)
    splits, days = e.base.load_selected(e.prepared / "data.pkl")
    dates = splits["val"]
    assert dates == sorted(plan["winter"]) and len(dates) == 40
    e.base.DeviceConfig().select_device(-1)
    agent = e.base.common.make_agent("fcsrl_actor", seed, e.prepared, e.base.common.config(66))
    source = p.r.ROOT if seed < 5 else root / "e127/main"
    agent.load(source / "weighted" / f"seed_{seed}/model_final")
    env = e.BatchEnv(days, e.FastDynamics(e.prepared / "surrogate.pkl"))
    out.mkdir(parents=True, exist_ok=True)
    for cohort in ["diagnostic", "frozen"]:
        keys, expected, keep = load_cases(seed, cohort)
        base, stamp, cut = simulate(env, agent, keys, expected, dates, False)
        p.prior.compare_traces(base, expected)
        updated, new_stamp, new_cut = simulate(env, agent, keys, expected, dates, True)
        np.testing.assert_array_equal(stamp, new_stamp)
        np.testing.assert_array_equal(cut, new_cut)
        fired = stamp >= 0
        prefix = np.arange(95) < np.where(fired, stamp, 95)[:, None]
        np.testing.assert_array_equal(updated[prefix], base[prefix])
        idx = np.flatnonzero(fired)
        np.testing.assert_allclose(updated[idx, stamp[idx], 0] - base[idx, stamp[idx], 0], -cut[idx],
                                   rtol=0, atol=1e-8)
        keys = keys[[*KEY, "period", "origin"]].assign(cohort=cohort, stamp=stamp, direct_air=-cut/4)
        end = p.prior.daily(*updated.transpose(2, 0, 1))
        keys[METRICS] = end - p.prior.daily(*base.transpose(2, 0, 1))
        if keep is not None:
            keys[["vs_keep_" + k for k in METRICS]] = end - p.prior.daily(*keep.transpose(2, 0, 1))
        np.savez_compressed(out / f"{cohort}{seed}.npz", baseline=base, updated=updated)
        keys.to_csv(out / f"{cohort}{seed}.csv", index=False)
        print(cohort, seed, len(keys), int(fired.sum()), flush=True)


def joint(df):
    return (df.air < -.05) & (df.nh <= 0) & (df["do"] <= 0) & (df.excess <= 1e-8)


def report(out, dest):
    dates = p.r.read(p.r.ROOT / "plan.json")["winter"]
    large = pd.read_csv(HERE.parent / "e131/cases.csv").set_index(KEY).large
    parts = []
    for cohort in ["diagnostic", "frozen"]:
        x = pd.concat([pd.read_csv(out / f"{cohort}{s}.csv") for s in range(10)]).set_index(KEY, verify_integrity=True)
        if cohort == "frozen":
            x = x.reindex(pd.MultiIndex.from_product([range(10), dates, range(5)], names=KEY))
            x["cohort"] = cohort
            x[METRICS + ["direct_air"]] = x[METRICS + ["direct_air"]].fillna(0.)
            x[["origin", "stamp"]] = x[["origin", "stamp"]].fillna(-1).astype(int)
        else:
            x["large"] = large.reindex(x.index)
            assert len(x) == len(large) and x.large.notna().all()
        x = x.reset_index()
        x["period"] = np.where(x.day.isin(dates[:20]), "early", "late")
        x["seed_set"] = np.where(x.seed < 5, "0-4", "5-9")
        x["later_air"] = x.air - x.direct_air
        x["joint"] = joint(x)
        parts.append(x)
    cases = pd.concat(parts, ignore_index=True)
    rows = []
    for (cohort, batch), block in cases.groupby(["cohort", "seed_set"]):
        for period, sub in [("all", block), *block.groupby("period")]:
            for seed, group in [("all", sub), *sub.groupby("seed")]:
                row = dict(cohort=cohort, seed_set=batch, period=period, seed=seed, n=len(group),
                           origins=(group.origin >= 0).sum(), fired=(group.stamp >= 0).sum(), joint=group.joint.sum())
                means = group[METRICS + ["direct_air", "later_air"]].mean()
                row.update(means.to_dict(), passed=bool(joint(means)))
                if cohort == "diagnostic":
                    row.update(group[["vs_keep_" + k for k in METRICS]].mean().to_dict())
                rows.append(row)
    dest.mkdir(parents=True, exist_ok=True)
    cases.to_csv(dest / "cases.csv", index=False)
    pd.DataFrame(rows).to_csv(dest / "summary.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--seed", type=int, choices=range(10))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tables", type=Path, default=HERE)
    args = parser.parse_args()
    smoke()
    if args.seed is not None:
        run(args.seed, args.out)
    elif not args.smoke:
        report(args.out, args.tables)

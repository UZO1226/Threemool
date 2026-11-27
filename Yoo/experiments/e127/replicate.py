"""Rebuild e121 selections on fresh policies, then replay the frozen e126 rule."""
import argparse
from runpy import run_path
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "e126"), str(HERE.parent / "e121")]
import late_cut as late
import risk_air

p, e = late.timing, late.E
ROOT = late.ROOT
METRICS = late.METRICS
MODES = ["matched", "frozen"]


def load_pair(out, seed):
    agents = {}
    schedule_check = run_path(str(HERE.parent / "e113/report.py"))["verify_schedule"]
    plan = p.r.read(out / "plan.json")
    for arm in ("uniform", "weighted"):
        path = out / arm / f"seed_{seed}"
        cfg = p.r.read(path / "effective_config.json")
        original = p.r.read(p.r.ROOT / arm / "seed_0/effective_config.json")
        assert cfg == {**original, "seed": seed}
        schedule_check(plan, arm, seed, path)
        assert p.r.read(path / "metrics.json")["steps"] == 401280
        assert (path / "winter_metrics.json").is_file()
        agent = e.base.common.make_agent("fcsrl_actor", seed, e.prepared, e.base.common.config(66))
        agent.load(path / "model_final")
        agents[arm] = agent
    return agents


def selections(features, frame, dates):
    ids, slots = np.nonzero(np.isfinite(features[:, :, 0]))
    options = pd.DataFrame(features[ids, slots], columns=p.INPUTS)
    scores = frame.iloc[ids].reset_index(drop=True).assign(lag=slots + 1)
    scores["period"] = np.where(scores.day.isin(dates[:20]), "early", "late")
    source = ROOT / "e121/main"
    models = joblib.load(source / "estimators.joblib")
    info = p.r.read(source / "fit.json")
    risk, saving = risk_air.predictors(models, options)
    scores["risk_air"] = np.where(risk <= info["risk_cutoff"], saving, np.nan)
    scores["air_only"] = saving
    consensus = joblib.load(ROOT / "e120/main/selectors.joblib")["consensus"]
    scores["consensus"] = p.scores(consensus, options.to_numpy())
    matched = risk_air.matched(scores)["risk_air"]
    frozen = risk_air.old.choose(scores, "risk_air", "frozen", info["thresholds"]["risk_air"])
    picks = np.stack([choice.set_index(p.KEYS).lag.reindex(pd.MultiIndex.from_frame(frame), fill_value=0)
                      for choice in (matched, frozen)]).astype(int)
    return picks, risk_air.Online("risk_air", models, info)


def replay(out, seed, regression):
    plan = p.r.read(out / "plan.json")
    assert plan == p.r.read(p.r.ROOT / "plan.json")
    for name, digest in plan["input_sha256"].items():
        p.r.verify(e.prepared / name, digest)
    e.base.DeviceConfig().select_device(-1)
    agents = load_pair(p.r.ROOT if regression else out, seed)
    splits, days = e.base.load_selected(e.prepared / "data.pkl")
    dates = plan["winter"]
    assert splits["val"] == dates == sorted(dates)
    episodes = [(day, member) for day in dates for member in range(5)]
    frame = pd.DataFrame(episodes, columns=["day", "member"]).assign(seed=seed)
    dyn = e.FastDynamics(e.prepared / "surrogate.pkl")
    env, probe = e.BatchEnv(days, dyn), late.LateCutEnv(days, dyn)
    boost, features, trigger, _ = p.simulate(env, episodes, agents)
    picks, gate = selections(features, frame, dates)
    online, _, online_trigger, online_choice = p.simulate(env, episodes, agents, slot="online", model=gate)
    np.testing.assert_array_equal(online_trigger, trigger)
    np.testing.assert_array_equal(online_choice, picks[1])
    old = np.stack([boost, boost])
    new = old.copy()
    issued = np.full(picks.shape, -1)
    for lag in range(1, 5):
        active = np.any(picks == lag, axis=0)
        if not active.any():
            continue
        native, _, stamp, chosen = p.simulate(env, episodes, agents, slot=lag)
        np.testing.assert_array_equal(stamp, trigger)
        assert (chosen[active] == lag).all()
        probe.first = np.where(active, stamp + lag, -1)
        cut, _, _, _ = p.simulate(probe, episodes, agents, slot=lag)
        for i in np.flatnonzero(active):
            t = probe.issued[i]
            np.testing.assert_array_equal(native[i, :t if t >= 0 else 95], cut[i, :t if t >= 0 else 95])
            if t >= 0:
                np.testing.assert_allclose(native[i, t, 0] - cut[i, t, 0], 250., atol=1e-7, rtol=0)
        for m in range(2):
            use = picks[m] == lag
            old[m, use], new[m, use], issued[m, use] = native[use], cut[use], probe.issued[use]
    np.testing.assert_array_equal(online, old[1])
    rows = []
    for m, mode in enumerate(MODES):
        records = frame.assign(mode=mode, lag=picks[m], issued=issued[m],
                               first=np.where(picks[m] > 0, trigger + picks[m], -1),
                               period=np.where(frame.day.isin(dates[:20]), "early", "late"))
        native = p.prior.daily(*old[m].transpose(2, 0, 1))
        cut = p.prior.daily(*new[m].transpose(2, 0, 1))
        for j, metric in enumerate(METRICS):
            records["old_" + metric] = native[:, j]
            records["new_" + metric] = cut[:, j]
            records[metric] = cut[:, j] - native[:, j]
        rows.append(records)
    cases = pd.concat(rows, ignore_index=True)
    if regression:
        archived = pd.read_csv(ROOT / "e126/main/cases.csv").query("seed == @seed")
        keys = ["mode", *late.KEY]
        selected = cases[cases.lag > 0].set_index(keys).sort_index()
        ref = archived.set_index(keys).sort_index()
        assert selected.index.equals(ref.index)
        np.testing.assert_allclose(selected[["first", "issued", *METRICS]], ref[["first", "issued", *METRICS]],
                                   rtol=0, atol=1e-7)
        print("seed 0 reproduces all", len(ref), "archived selected rows", flush=True)
        return
    np.savez_compressed(out / f"paired{seed}.npz", old=old, new=new)
    cases.to_csv(out / f"cases{seed}.csv", index=False)
    print("verified", seed, flush=True)


def summarize(out, dest):
    cases = pd.concat(pd.read_csv(out / f"cases{s}.csv") for s in range(5, 10))
    assert len(cases) == 2000 and not cases.duplicated(["mode", *p.KEYS]).any()
    cases["fired"] = cases.issued >= 0
    cases["harm"] = (cases.nh > 0) | (cases["do"] > 0) | (cases.excess > 1e-8)
    cases["joint"] = (cases.air < -.05) & ~cases.harm
    rows = []
    for mode, cohort in cases.groupby("mode"):
        for period in ("all", "early", "late"):
            part = cohort if period == "all" else cohort[cohort.period == period]
            for seed in ("all", *range(5, 10)):
                x = part if seed == "all" else part[part.seed == seed]
                for group, sub in [("whole", x), ("selected", x[x.lag > 0]), ("fired", x[x.fired])]:
                    if sub.empty:
                        continue
                    avg = sub[METRICS].mean()
                    rows.append(dict(mode=mode, period=period, seed=seed, cohort=group, n=len(sub),
                        selected=(sub.lag > 0).sum(), fired=sub.fired.sum(), jointly_better=sub.joint.sum(),
                        harmed=sub.harm.sum(), air_saved=(sub.air < -.05).sum(),
                        passed=avg.air < -.05 and avg.nh <= 0 and avg["do"] <= 0 and avg.excess <= 1e-8,
                        **avg.to_dict(), **sub[["old_" + k for k in METRICS] + ["new_" + k for k in METRICS]].mean().to_dict()))
    dest.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(dest / "summary.csv", index=False)
    halves = table.query("mode == 'matched' and period != 'all' and seed == 'all' and cohort == 'selected'")
    print("Replicated:", len(halves) == 2 and bool(halves.passed.all()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("regression", "replay", "report"))
    parser.add_argument("out", type=Path)
    parser.add_argument("--seed", type=int, choices=range(5, 10))
    parser.add_argument("--tables", type=Path, default=HERE)
    args = parser.parse_args()
    if args.stage == "report":
        summarize(args.out, args.tables)
    elif args.stage == "regression":
        replay(args.out, 0, True)
    elif args.seed is None:
        parser.error("replay requires --seed")
    else:
        replay(args.out, args.seed, False)

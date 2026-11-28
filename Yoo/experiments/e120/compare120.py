"""Fit the two selectors and compare episode-matched coverage and prospective control."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from labels import cross, p, r

EP = p.KEYS
KEY = cross.KEY
M = p.METRICS
ARMS = ("native", "consensus")


def data(name):
    return pd.read_csv(cross.SOURCE / (name + ".csv"), float_precision="round_trip")


def ensemble_votes(directory, options):
    raw = pd.concat([pd.read_csv(directory / f"seed{s}.csv") for s in range(5)])
    assert len(raw) == 5 * len(options) and not raw.duplicated(KEY + ["model"]).any()
    assert raw.groupby(KEY).model.nunique().eq(5).all()
    raw["positive"] = (raw.air < -.05) & (raw.nh <= 0) & (raw["do"] <= 0)
    keys = pd.MultiIndex.from_frame(options[KEY])
    local = raw[raw.member == raw.model].set_index(KEY).loc[keys]
    np.testing.assert_array_equal(local.positive, options.good)
    return raw.groupby(KEY).positive.sum().loc[keys].to_numpy()


def fit(out):
    train = data("fit_options")
    autumn = data("autumn_options")
    assert max(train.day) < min(autumn.day)
    votes = ensemble_votes(out / "labels", train)
    info = {}
    models = {}
    for arm in ARMS:
        y = train.good if arm == "native" else votes == 5
        model = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=20,
                                       n_jobs=1, random_state=0)
        model.fit(train[p.INPUTS].to_numpy(), y)
        score = p.scores(model, autumn[p.INPUTS].to_numpy())
        if arm == "native":
            np.testing.assert_allclose(p.scores(model, train[p.INPUTS].to_numpy()), train.score, atol=1e-12, rtol=0)
            np.testing.assert_allclose(score, autumn.score, atol=1e-12, rtol=0)
        peaked = autumn.assign(priority=score).groupby(EP).priority.max()
        threshold = float(peaked.nlargest(40).min())
        models[arm] = model
        info[arm] = dict(train_candidates=len(train), train_positive=int(np.sum(y)), threshold=threshold,
                         autumn_selections=int((peaked >= threshold).sum()), autumn_episodes=400)
    joblib.dump(models, out / "selectors.joblib")
    (out / "fit.json").write_text(json.dumps(info, indent=2) + "\n")
    print(info, flush=True)


def first(frame):
    return frame.sort_values(KEY).drop_duplicates(EP).copy()


def choose(frame, arm, mode, threshold):
    if mode == "frozen":
        return first(frame[frame[arm] >= threshold])
    picks = []
    for _, g in frame.groupby(["seed", "period"]):
        peaks = g.groupby(EP, as_index=False)[arm].max()
        top = peaks.sort_values([arm, "day", "member"], ascending=[False, True, True]).head(10)
        assert len(top) == 10
        eligible = g[g[arm] >= top[arm].min()].merge(top[EP], on=EP, validate="many_to_one")
        selected = first(eligible)
        assert len(selected) == 10
        picks.append(selected)
    return pd.concat(picks)


class AtThreshold:
    classes_ = np.array([False, True])

    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold

    def predict_proba(self, x):
        take = p.scores(self.model, x) >= self.threshold
        return np.column_stack((~take, take))


def verify_online(base, dates, models, fits, actions, values):
    e = r.e113
    e.base.DeviceConfig().select_device(-1)
    split, days = e.base.load_selected(e.prepared / "data.pkl")
    assert split["val"] == dates
    env = e.BatchEnv(days, e.FastDynamics(e.prepared / "surrogate.pkl"))
    episodes = [(d, m) for d in dates for m in range(5)]
    hashes = r.read(r.ROOT / "checksums.json")
    for seed in range(5):
        agents = r.load(seed, hashes)[1]
        keys = [(d,m,seed) for d,m in episodes]
        for arm in ARMS:
            gate = AtThreshold(models[arm], fits[arm]["threshold"])
            trace, _, trigger, lag = p.simulate(env, episodes, agents, slot="online", model=gate)
            np.testing.assert_array_equal(lag, actions[arm].reindex(keys, fill_value=0))
            np.testing.assert_array_equal(trigger, base.loc[keys, "trigger"])
            np.testing.assert_allclose(p.prior.daily(*trace.transpose(2,0,1)), values[arm].loc[keys], rtol=0, atol=1e-6)
        print("Winter online replay verified", seed, flush=True)


def joint(x):
    return (x.air < -.05) & (x.nh < 0) & (x["do"] <= 0)


def evaluate(out, tables):
    models = joblib.load(out / "selectors.joblib")
    fits = json.loads((out / "fit.json").read_text())
    options = data("winter_options")
    base = data("winter_baseline").set_index(EP).sort_index()
    assert not options.duplicated(KEY).any() and len(base) == 1000 and base.index.is_unique
    dates = sorted(set(base.index.get_level_values("day")))
    scores = options[KEY].copy()
    scores["period"] = np.where(scores.day.isin(dates[:20]), "early", "late")
    for arm in ARMS:
        scores[arm] = p.scores(models[arm], options[p.INPUTS].to_numpy())
    np.testing.assert_allclose(scores.native, options.score, rtol=0, atol=1e-12)
    picks = {(mode, arm): choose(scores, arm, mode, fits[arm]["threshold"])[KEY]
             for mode in ("matched", "frozen") for arm in ARMS}
    outcomes = options.copy()
    outcomes["votes"] = ensemble_votes(r.ROOT.parents[1] / "e119/main", options)
    records, per_seed, capture, chosen_rows, ranks = [], [], [], [], []
    for arm in ARMS:
        for name, target in [("native", options.good), ("consensus", outcomes.votes == 5)]:
            ranks.append(dict(arm=arm, target=name, auc=roc_auc_score(target, scores[arm])))
    for mode in ("matched", "frozen"):
        vals = {"B": base[M], "A": base[["A_" + m for m in M]].set_axis(M, axis=1)}
        actions = {}
        for arm in ARMS:
            choices = picks[mode, arm].merge(outcomes, on=KEY, validate="one_to_one").set_index(EP)
            vals[arm] = vals["B"] + choices[M].reindex(base.index, fill_value=0)
            actions[arm] = choices.lag
            chosen_rows.append(choices.reset_index()[KEY + ["reduction","votes","good"] + M].assign(mode=mode,arm=arm))
            for name, ds in [("all", dates), ("early", dates[:20]), ("late", dates[20:])]:
                g = choices[choices.index.get_level_values("day").isin(ds)]
                capture.append(dict(mode=mode,arm=arm,period=name,chosen=len(g),native=int(g.good.sum()),
                                    four=int((g.votes >= 4).sum()),five=int((g.votes == 5).sum()),
                                    mean_initial_cut=g.reduction.mean(), **vals[arm].loc[
                                        vals[arm].index.get_level_values("day").isin(ds)].mean().to_dict()))
        if mode == "frozen":
            verify_online(base, dates, models, fits, actions, vals)
        for comp in ("native","B","A"):
            delta = vals["consensus"] - vals[comp]
            for name, ds in [("all",dates), ("early",dates[:20]), ("late",dates[20:])]:
                g = delta.loc[delta.index.get_level_values("day").isin(ds)].groupby("seed").mean()
                per_seed.append(g.assign(mode=mode, comparator=comp, period=name, joint=joint(g)).reset_index())
                mean = g.mean()
                count = int(joint(g).sum())
                records.append(dict(mode=mode, comparator=comp, period=name, **mean.to_dict(),
                                    joint_seeds=count, passed=bool(joint(mean) and count >= 4)))
    tables.mkdir(parents=True, exist_ok=True)
    control = pd.DataFrame(records)
    success = pd.DataFrame(capture)
    halves = control[control.period != "all"]
    matched = halves[(halves["mode"]=="matched") & (halves.comparator=="native")]
    counts = success[(success["mode"]=="matched") & (success.period!="all")].pivot(index="period",columns="arm",values="five")
    operational = halves[(halves["mode"]=="frozen") & halves.comparator.isin(["B","A"])]
    assert len(matched)==2 and len(operational)==4
    decision = dict(matched_pass=bool(matched.passed.all() and (counts.consensus > counts.native).all()),
                    operational_pass=bool(operational.passed.all()))
    for name, table in [("control",control),("success",success),("seeds",pd.concat(per_seed)),("rank",pd.DataFrame(ranks))]:
        table.to_csv(tables / (name+".csv"),index=False)
    pd.concat(chosen_rows).to_csv(out / "choices.csv",index=False)
    (tables / "decision.json").write_text(json.dumps(decision,indent=2)+"\n")
    print(decision)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fit","evaluate"])
    parser.add_argument("out", type=Path)
    parser.add_argument("--tables", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    if args.stage == "fit":
        fit(args.out)
    else:
        evaluate(args.out, args.tables)

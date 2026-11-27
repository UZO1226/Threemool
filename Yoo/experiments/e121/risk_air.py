"""Compare water-risk-filtered expected air savings with frozen e120 priorities."""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e120"))
import compare120 as old

EP, KEY, M = old.EP, old.KEY, old.M
ROOT = old.r.ROOT.parents[1]
PREV = ROOT / "e120/main"
ARMS = ["consensus", "air_only", "risk_air"]


def targets(directory, options):
    raw = pd.concat([pd.read_csv(directory / f"seed{s}.csv") for s in range(5)])
    assert len(raw) == len(options)*5 and not raw.duplicated(KEY + ["model"]).any()
    assert raw.groupby(KEY).model.nunique().eq(5).all()
    raw["harm"] = (raw.nh > 0) | (raw["do"] > 0)
    raw["good"] = (~raw.harm) & (raw.air < -.05)
    ids = pd.MultiIndex.from_frame(options[KEY])
    source = raw[raw.model == raw.member].set_index(KEY).loc[ids]
    np.testing.assert_allclose(source[M], options[M], rtol=0, atol=1e-6)
    table = raw.groupby(KEY).agg(risk=("harm","mean"), air=("air","mean"), votes=("good","sum")).loc[ids]
    return table.assign(saving=-table.air).reset_index(drop=True)


def predictors(estimators, data):
    x = data[old.p.INPUTS].to_numpy()
    return estimators["risk"].predict(x), estimators["saving"].predict(x)


def fit(out):
    out.mkdir(parents=True, exist_ok=False)
    train, autumn = [old.data(s + "_options") for s in ("fit","autumn")]
    assert max(train.day) < min(autumn.day) and len(train) == 2246
    truth = targets(PREV / "labels", train)
    models = {}
    for y in ["risk","saving"]:
        model = RandomForestRegressor(n_estimators=100, max_depth=3, min_samples_leaf=20,
                                      max_features=1.0, random_state=0, n_jobs=1)
        models[y] = model.fit(train[old.p.INPUTS].to_numpy(), truth[y])
    risk, saving = predictors(models, autumn)
    ceiling = float(np.median(risk))
    eligible = risk <= ceiling
    summary = dict(risk_cutoff=ceiling, loss_cutoff=float(train.air.quantile(.9)), thresholds={},
                   calibration_episodes={}, calibration_eligible={})
    for arm, score in [("risk_air",np.where(eligible, saving, np.nan)), ("air_only",saving)]:
        peaks = autumn.assign(score=score).groupby(EP).score.max().dropna()
        threshold = float(peaks.nlargest(40).min()) if len(peaks) else float("inf")
        summary["thresholds"][arm] = threshold
        summary["calibration_episodes"][arm] = int((peaks >= threshold).sum())
        summary["calibration_eligible"][arm] = len(peaks)
    summary["thresholds"]["consensus"] = json.loads((PREV / "fit.json").read_text())["consensus"]["threshold"]
    joblib.dump(models, out / "estimators.joblib")
    (out / "fit.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(summary, flush=True)


def matched(scores):
    blocks = {arm: [] for arm in ARMS}
    for _, part in scores.groupby(["period","seed"]):
        k = min(10, len(part[part.risk_air.notna()].drop_duplicates(EP)))
        for arm in ARMS:
            if not k:
                blocks[arm].append(part.iloc[:0])
                continue
            valid = part.dropna(subset=[arm])
            peaks = valid.groupby(EP,as_index=False)[arm].max().sort_values(
                [arm,"day","member"], ascending=[False,True,True]).head(k)
            selected = old.first(valid[valid[arm] >= peaks[arm].min()].merge(peaks[EP], on=EP))
            assert len(selected) == k
            blocks[arm].append(selected)
    return {arm:pd.concat(group)[KEY] for arm, group in blocks.items()}


class Online:
    classes_ = [0,1]

    def __init__(self, arm, models, fit_info):
        self.arm, self.models, self.info = arm, models, fit_info

    def predict_proba(self, x):
        take = self.models["saving"].predict(x) >= self.info["thresholds"][self.arm]
        if self.arm == "risk_air":
            take &= self.models["risk"].predict(x) <= self.info["risk_cutoff"]
        return np.column_stack([~take,take])


def replay(dates, base, actions, values, models, info):
    e = old.r.e113
    e.base.DeviceConfig().select_device(-1)
    split, days = e.base.load_selected(e.prepared / "data.pkl")
    assert dates == split["val"]
    env = e.BatchEnv(days, e.FastDynamics(e.prepared / "surrogate.pkl"))
    episodes = [(d,m) for d in dates for m in range(5)]
    hashes = old.r.read(old.r.ROOT / "checksums.json")
    for seed in range(5):
        agents = old.r.load(seed, hashes)[1]
        keys = [(d,m,seed) for d,m in episodes]
        for arm in ("risk_air","air_only"):
            gate = Online(arm,models,info)
            trace, _, trigger, chosen = old.p.simulate(env,episodes,agents,slot="online",model=gate)
            np.testing.assert_array_equal(trigger,base.loc[keys,"trigger"])
            np.testing.assert_array_equal(chosen,actions[arm].reindex(keys,fill_value=0))
            np.testing.assert_allclose(old.p.prior.daily(*trace.transpose(2,0,1)),values[arm].loc[keys],atol=1e-6,rtol=0)
        print("Winter replay verified",seed,flush=True)


def evaluate(out, tables):
    opts = old.data("winter_options")
    base = old.data("winter_baseline").set_index(EP).sort_index()
    dates = sorted(set(base.index.get_level_values("day")))
    assert len(base)==1000 and len(opts)==3054
    info = json.loads((out/"fit.json").read_text())
    models = joblib.load(out/"estimators.joblib")
    risk, saving = predictors(models,opts)
    consensus = joblib.load(PREV/"selectors.joblib")["consensus"]
    scores = opts[KEY].assign(risk=risk, air_only=saving,
                              consensus=old.p.scores(consensus, opts[old.p.INPUTS].to_numpy()),
                              risk_air=np.where(risk <= info["risk_cutoff"], saving, np.nan),
                              period=np.where(opts.day.isin(dates[:20]),"early","late"))
    picks = {"matched":matched(scores), "frozen":{
        arm:old.choose(scores,arm,"frozen",info["thresholds"][arm])[KEY] for arm in ARMS}}
    prior_choices = pd.read_csv(PREV/"choices.csv")
    for mode, choices in picks.items():
        if mode=="frozen" or len(choices["consensus"])==100:
            archived = prior_choices[(prior_choices["mode"]==mode)&(prior_choices.arm=="consensus")]
            assert set(archived[KEY].itertuples(index=False,name=None)) == set(choices["consensus"].itertuples(index=False,name=None))
    truth = targets(ROOT/"e119/main",opts)
    opts = opts.assign(votes=truth.votes, harm=(opts.nh>0)|(opts["do"]>0),
                       large=opts.air>info["loss_cutoff"], predicted_risk=risk, predicted_saving=saving,
                       ensemble_saving=truth.saving, ensemble_risk=truth.risk)
    records, by_seed, selected, paired, accuracy = [], [], [], [], []
    periods = [("all",dates),("early",dates[:20]),("late",dates[20:])]
    for period, ds in periods:
        g = opts[opts.day.isin(ds)]
        accuracy.append(dict(period=period,candidates=len(g), risk_auc=roc_auc_score(g.harm,g.predicted_risk),
                             risk_mae=(g.predicted_risk-g.ensemble_risk).abs().mean(),
                             native_spearman=(-g.air).corr(g.predicted_saving,method="spearman"),
                             ensemble_spearman=g.ensemble_saving.corr(g.predicted_saving,method="spearman"),
                             loss_auc=roc_auc_score(g.large,-g.predicted_saving)))
    for mode, choices in picks.items():
        values = {"B":base[M], "A":base[["A_"+k for k in M]].set_axis(M,axis=1)}
        actions = {}
        for arm, chosen in choices.items():
            x = chosen.merge(opts,on=KEY,validate="one_to_one").set_index(EP)
            actions[arm] = x.lag
            values[arm] = base[M]+x[M].reindex(base.index,fill_value=0)
            selected.append(x.reset_index().assign(arm=arm,mode=mode))
            for period, ds in periods:
                g = x[x.index.get_level_values("day").isin(ds)]
                v = values[arm].loc[values[arm].index.get_level_values("day").isin(ds)]
                records.append(dict(mode=mode,arm=arm,period=period,chosen=len(g), harmed=int(g.harm.sum()),
                                    big_losses=int(g.large.sum()), gross_extra_m3=g.air.clip(lower=0).sum(),
                                    five=int((g.votes==5).sum()), **v.mean().to_dict()))
        for comp in ["consensus","air_only","A","B"]:
            diff = values["risk_air"]-values[comp]
            for period, ds in periods:
                d = diff.loc[diff.index.get_level_values("day").isin(ds)].groupby("seed").mean()
                avg = d.mean()
                count = int(old.joint(d).sum())
                by_seed.append(d.assign(mode=mode, comparator=comp, period=period,joint=old.joint(d)).reset_index())
                paired.append(dict(mode=mode,comparator=comp,period=period,**avg.to_dict(),
                                   joint_seeds=count, passed=bool(old.joint(avg) and count>=4)))
        if mode=="frozen":
            replay(dates,base,actions,values,models,info)
    pairs = pd.DataFrame(paired)
    half = pairs[pairs.period!="all"]
    primary = half[(half["mode"]=="matched") & (half.comparator=="consensus")]
    oper = half[(half["mode"]=="frozen") & half.comparator.isin(["A","B"])]
    assert len(primary)==2 and len(oper)==4
    decision = dict(matched_pass=bool(len(picks["matched"]["risk_air"])==100 and primary.passed.all()),
                    operational_pass=bool(oper.passed.all()))
    tables.mkdir(parents=True, exist_ok=True)
    for name, table in [("metrics",pd.DataFrame(records)),("paired",pairs),("seeds",pd.concat(by_seed)),
                        ("prediction",pd.DataFrame(accuracy))]:
        table.to_csv(tables/(name+".csv"),index=False)
    pd.concat(selected).to_csv(out/"choices.csv",index=False)
    (tables/"decision.json").write_text(json.dumps(decision,indent=2)+"\n")
    print(decision)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["fit","evaluate"])
    parser.add_argument("out",type=Path)
    parser.add_argument("--tables",type=Path,default=HERE)
    args=parser.parse_args()
    if args.stage=="fit":
        fit(args.out)
    else:
        evaluate(args.out,args.tables)

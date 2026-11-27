"""Check recurrence of apparent NH4 recovery under the frozen e142 specification."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e142"))
import hour
from surrogate_dw import train_gbm

PREFIXES = (28, 68, 108, 148, 188)
DISCOVERY = {"2021-10-30", "2021-11-04"}
MODELS = ("direct", "gbm", "no_new_air", "held_air")
COHORTS = ("near", "wider", "legacy_near")


def pick(nh, air, upper):
    starts = []
    for t in range(4, len(nh)-4):
        eligible = 4 < nh[t] <= upper and air[t]-air[t-1] >= 500
        if eligible and (not starts or t-starts[-1]>=4):
            starts.append(t)
    return starts


def cohorts(days, dates):
    sets = {name:set() for name in COHORTS}
    for d in dates:
        row = days[d]
        for name, upper in (("near",4.5), ("wider",5.)):
            sets[name].update((d,t) for t in pick(row["state"][:,0],row["air"],upper))
    events = hour.replay.W.events(hour.replay.W.windows(days,dates),500.)
    legacy = [(e["day"],e["t"]) for e in events]
    sets["legacy_near"] = {(d,t) for d,t in legacy if 4<days[d]["state"][t,0]<=4.5
                           and days[d]["air"][t]-days[d]["air"][t-1]>=500}
    return sets, legacy


def coefficients(days, history):
    return {(model,h):hour.train(days, history, h, model=="direct")
            for model in ("direct","no_new_air") for h in range(1,5)}


def evaluate(days, keys, coef, gbm_path):
    dyn = hour.replay.E.Dynamics(gbm_path)
    ds, ts = zip(*keys)
    def values(k, offset):
        return np.stack([days[d][k][t+offset] for d,t in keys])

    truth = np.stack([values("state",j)[:,0] for j in range(5)],axis=1)
    path = {model:[] for model in MODELS}
    member = np.tile(np.arange(5),len(keys))
    state = np.repeat(values("state",0),5,axis=0)
    for j in range(4):
        args = [np.repeat(values(k,delta),5,axis=0) for k,delta in (("exo",j),("zone",j),("zone",j-1))]
        state = dyn.step(state,*args,member)
        path["gbm"].append(state.reshape(len(keys),5,5)[:,:,0].mean(axis=1))
        for m in ("direct","no_new_air"):
            beta, bias = coef[m,j+1]
            x = np.concatenate([hour.inputs(days[d],np.array([t]),j+1,m=="direct") for d,t in keys])
            path[m].append(np.maximum(0,truth[:,0]+bias+x@beta))
            if m=="direct":
                # Zero change from the pre-command zone flows in this very same predictor.
                path["held_air"].append(np.maximum(0,truth[:,0]+bias+x[:,:-5]@beta[:-5]))
    path = {m:np.column_stack(p) for m,p in path.items()}
    result = pd.DataFrame(dict(day=ds,t=ts,initial=truth[:,0],dq=values("air",0)-values("air",-1),
                               actual=hour.first_cross(truth[:,1:],truth[:,0])))
    for m in MODELS:
        result[m] = hour.first_cross(path[m],truth[:,0])
    return result, truth, path


def verify_e142(n,legacy,keys,truth,path,coef):
    src = hour.ROOT / f"e142/main/prefix{n}.npz"
    hashes = {148:"f222b27ea054eed8bde3dee5867ef10af4ccb25e93c604d0c475825d61396805",
              188:"4b16ff235f313744823e1d1714f8ea80f3cd28c946d4f2a04d40fabd8d39dd90"}
    assert hour.digest(src)==hashes[n]
    lookup = {key:i for i,key in enumerate(keys)}
    indices = [lookup[k] for k in legacy]
    with np.load(src) as archived:
        assert list(zip(archived["days"],archived["times"]))==legacy
        np.testing.assert_array_equal(truth[indices],archived["truth"])
        for m in ("gbm","no_new_air","direct"):
            np.testing.assert_allclose(path[m][indices],archived[m],rtol=0,atol=1e-9)
        for (m,h),(beta,bias) in coef.items():
            np.testing.assert_allclose(np.r_[bias,beta],archived[f"{m}_{h}"],atol=1e-10,rtol=0)


def summarize(frame, schedule, period):
    rows = []
    for cohort in COHORTS:
        data = frame.loc[frame[cohort] & ~frame.day.isin(DISCOVERY)]
        recovered = data.actual>0
        for m in MODELS:
            called = data[m]>0
            fp = called & ~recovered
            rows.append(dict(schedule=schedule,period=period,cohort=cohort,model=m,windows=len(data),
                    days=data.day.nunique(),recovered=recovered.sum(),not_recovered=(~recovered).sum(),
                    negative_days=data.loc[~recovered,"day"].nunique(),false_recovery=fp.sum(),
                    false_days=data.loc[fp,"day"].nunique(),detected=(recovered & called).sum(),
                    exact=((data.actual==data[m])&recovered).sum(),
                    direct_only=(~recovered & (data.direct>0) & ~called).sum(),
                    other_only=(fp & (data.direct==0)).sum()))
    return rows


def main(out):
    nh = np.full(96,4.25)
    air = np.zeros(96)
    air[[4,6,8,91,92]]=500
    assert pick(nh,air,4.5)==[4,8,91]
    nh[4],nh[8]=4,4.51
    assert pick(nh,air,4.5)==[6,91]
    hour.smoke()
    out.mkdir(parents=True,exist_ok=False)
    prep=hour.replay.E.DATA
    expected=json.loads((HERE.parent/"e77/summary.json").read_text())["sha256"]
    for p in (prep/"data.pkl",prep/"surrogate.pkl"):
        assert hour.digest(p)==expected[str(p)]
    split,days = hour.replay.E.S.base.load_selected(prep/"data.pkl")
    dates=split["train"]+split["val"]
    assert len(set(dates))==228 and dates==sorted(dates)
    table = pd.read_pickle(prep/"data.pkl")["data"]
    fits, frames, rows = {}, [], []
    for n in PREFIXES:
        history, target = dates[:n], dates[n:n+40]
        assert len(target)==40 and history[-1]<target[0]
        if n<108:
            gbm = out / f"gbm{n}.pkl"
            train_gbm(table.loc[history[0]:history[-1]], dict(train=history,val=history[int(.8*n):]),gbm)
        elif n<188:
            folder=hour.ROOT/f"e139/main/prefix{n}/prepared"
            gbm=folder/"surrogate.pkl"
            meta=json.loads((folder/"manifest.json").read_text())
            assert meta["train"]==history and hour.digest(gbm)==meta["hashes"][gbm.name]
        else:
            gbm=prep/"surrogate.pkl"
        fits[n] = coefficients(days,history),gbm
        sets, legacy = cohorts(days,target)
        keys=sorted(set(legacy).union(*sets.values()))
        for fit in ([n,148] if n==188 else [n]):
            coef, model = fits[fit]
            result, truth, paths = evaluate(days,keys,coef,model)
            if fit==n and n in (148,188):
                verify_e142(n,legacy,keys,truth,paths,coef)
            for name, members in sets.items():
                result[name]=[k in members for k in keys]
            result=result[result[list(COHORTS)].any(axis=1)].assign(fit=fit,period=n)
            frames.append(result)
            schedule="rolling" if fit==n else "frozen148"
            rows.extend(summarize(result,schedule,str(n)))
        print(n,target[0],target[-1],flush=True)
    all_events=pd.concat(frames,ignore_index=True)
    rolling=all_events.loc[all_events.fit==all_events.period]
    assert not rolling.duplicated(["day","t"]).any()
    for label,subset in (("new120",rolling[rolling.period<148]),("previous80",rolling[rolling.period>=148]),
                         ("all200",rolling)):
        rows.extend(summarize(subset,"rolling",label))
    # The autumn fold itself is also the first period of the frozen-148 comparison.
    rows.extend(summarize(rolling[rolling.period==148],"frozen148","148"))
    idx=rolling.legacy_near & rolling.day.isin(DISCOVERY) & (rolling.actual==0) & (rolling.direct>0)
    assert set(zip(rolling.loc[idx,"day"],rolling.loc[idx,"t"]))=={("2021-10-30",51),("2021-11-04",64)}
    all_events.to_csv(out/"windows.csv",index=False)
    pd.DataFrame(rows).to_csv(out/"summary.csv",index=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    main(parser.parse_args().out)

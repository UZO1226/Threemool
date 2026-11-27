"""One-hour conditional DARROW forecasts on two frozen historical splits."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e104"))
import replay

ROOT = replay.E.DATA.parents[1]
MODELS = ("gbm", "persistence", "direct", "no_new_air")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(day, times, h, with_air):
    state, exo, zone = (day[k] for k in ("state", "exo", "zone"))
    path = times[:, None] + np.arange(h)
    previous = zone[times - 1]
    cols = [state[times], state[times]-state[times-1], state[times]-state[times-4],
            exo[times], previous, exo[path].mean(axis=1)]
    if with_air:
        cols.append(zone[path].mean(axis=1)-previous)
    return np.column_stack(cols)


def train(days, train_dates, h, with_air):
    times = np.arange(4, 92)
    x = np.vstack([inputs(days[d], times, h, with_air) for d in train_dates])
    y = np.concatenate([days[d]["state"][times+h, 0] - days[d]["state"][times, 0] for d in train_dates])
    assert len(y) == len(train_dates)*88 and np.isfinite(x).all() and np.isfinite(y).all()
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-10] = 1
    # Centering the response leaves the intercept unpenalized.
    matrix = np.vstack(((x-center)/scale, np.eye(x.shape[1])))
    rhs = np.r_[y-y.mean(), np.zeros(x.shape[1])]
    upper = np.full(x.shape[1], np.inf)
    if with_air:
        upper[-5:] = 0
    solution = lsq_linear(matrix, rhs, bounds=(-np.inf, upper), method="bvls")
    assert solution.success
    beta = solution.x / scale
    return beta, y.mean()-center@beta


def smoke():
    rng = np.random.default_rng(142)
    d = {k:rng.normal(size=(96,width)) for k,width in (("state",5),("zone",5),("exo",9))}
    t = np.array([17])
    for h in range(1,5):
        ref = inputs(d,t,h,True)
        changed = {k:v.copy() for k,v in d.items()}
        changed["state"][18:] += 500
        changed["zone"][17+h:] += 900
        changed["exo"][17+h:] -= 40
        np.testing.assert_array_equal(ref,inputs(changed,t,h,True))
        np.testing.assert_allclose(ref[0,-5:], d["zone"][17:17+h].mean(0)-d["zone"][16])
        changed["zone"][17:17+h] += 100
        np.testing.assert_array_equal(inputs(changed,t,h,False),inputs(d,t,h,False))
    initial = np.array([5,3,4])
    np.testing.assert_array_equal(first_cross(np.array([[5,4,3,5],[3,4,5,4],[4,4,4,4]]),initial), [2,3,0])


def first_cross(path, initial):
    flips = (path>4) != (initial[:, None]>4)
    return np.where(flips.any(axis=1), flips.argmax(axis=1)+1, 0)


def report(dates, events, truth, preds, majority):
    ids = np.array([dates.index(e["day"]) for e in events])
    rises = np.array([e["dq"]>0 for e in events])
    initial = truth[:,0]
    real = first_cross(truth[:,1:],initial)
    reference_error = np.abs(preds["gbm"][:,-1]-truth[:,-1])
    result = []
    groups = {"all":np.ones(len(real),dtype=bool)}
    if majority is not None:
        groups["majority"] = majority
    for name, pred in preds.items():
        cross = first_cross(pred, initial)
        err = abs(pred[:,-1]-truth[:,-1])
        for group, mask in groups.items():
            for direction, flag in (("up",rises),("down",~rises)):
                take = mask & flag
                date_ids = np.unique(ids[take])
                daily = np.array([[err[(ids==d)&take].mean(), reference_error[(ids==d)&take].mean()] for d in date_ids])
                diff = daily[:,0]-daily[:,1]
                ci = replay.W.block_ci(diff, date_ids, len(dates))
                recovery = take & (initial>4)
                result.append(dict(model=name, group=group, direction=direction, windows=take.sum(), days=len(daily),
                    mae=daily[:,0].mean(), gap=diff.mean(), low=ci[0], high=ci[1],
                    recoveries=(recovery&(real>0)).sum(), detected_recovery=(recovery&(real>0)&(cross>0)).sum(),
                    exact_recovery=(recovery&(real>0)&(cross==real)).sum(), nonrecoveries=(recovery&(real==0)).sum(),
                    false_recovery=(recovery&(real==0)&(cross>0)).sum(),
                    crossings=(take&(real>0)).sum(), hits=(take&(real>0)&(cross>0)).sum(),
                    exact=(take&(real>0)&(cross==real)).sum(), noncrossings=(take&(real==0)).sum(),
                    false_alarms=(take&(real==0)&(cross>0)).sum()))
    return pd.DataFrame(result)


def run_split(n, days, train_dates, dates, model_path, out):
    assert len(train_dates)==n and len(dates)==40 and train_dates[-1]<dates[0]
    events = replay.W.events(replay.W.windows(days,dates),500.)
    truth, exo, zone = replay.pack(events,days)
    dyn = replay.E.Dynamics(model_path)
    nwin = len(events)
    members = np.tile(np.arange(5),nwin)
    state = np.repeat(truth[:,0],5,axis=0)
    gbm = []
    for j in range(4):
        args = [np.repeat(v,5,axis=0) for v in (exo[:,j],zone[:,j+1],zone[:,j])]
        state = dyn.step(state, *args, members)
        gbm.append(state.reshape(nwin,5,5).copy())
    gbm = np.stack(gbm,axis=1)
    truth = truth[:,:5,0]
    preds = dict(gbm=gbm[:,:,:,0].mean(axis=2), persistence=np.repeat(truth[:,0:1],4,axis=1))
    majority = None
    if n==188:
        old = ROOT / "e141/main/windows.csv"
        assert digest(old)=="edc586c037c46862de09c14cdc57ba3066b4f4b72fa8682914811e5f2e30a716"
        archive = pd.read_csv(old)
        assert list(zip(archive.day,archive.t)) == [(e["day"],e["t"]) for e in events]
        majority = archive.votes.to_numpy()>=3
        np.testing.assert_array_equal(first_cross(preds["gbm"],truth[:,0]),archive.model_cross)
        np.testing.assert_array_equal(first_cross(truth[:,1:],truth[:,0]),archive.observed_cross)
        with np.load(ROOT / "e104/main/replay.npz") as ref:
            np.testing.assert_allclose(gbm[:,[0,3]],ref["rec"][:,:2],atol=1e-10,rtol=0)
    weights = {}
    for name, with_air in (("direct",True),("no_new_air",False)):
        path = []
        for h in range(1,5):
            beta, bias = train(days,train_dates,h,with_air)
            x = np.vstack([inputs(days[e["day"]], np.array([e["t"]]),h,with_air) for e in events])
            pred = np.maximum(0,truth[:,0]+x@beta+bias)
            assert np.isfinite(pred).all()
            if with_air:
                assert max(beta[-5:]) <= 1e-12
            weights[f"{name}_{h}"] = np.r_[bias,beta]
            path.append(pred)
        preds[name] = np.column_stack(path)
    np.savez_compressed(out/f"prefix{n}.npz",truth=truth,days=[e["day"] for e in events],
                        times=[e["t"] for e in events], dq=[e["dq"] for e in events],**preds,**weights)
    return report(dates,events,truth,preds,majority).assign(prefix=n)


def main(out):
    smoke()
    out.mkdir(parents=True,exist_ok=False)
    prep = replay.E.DATA
    expected = json.loads((HERE.parent/"e77/summary.json").read_text())["sha256"]
    for path in (prep/"data.pkl",prep/"surrogate.pkl"):
        assert digest(path)==expected[str(path)]
    splits, days = replay.E.S.base.load_selected(prep/"data.pkl")
    dates = splits["train"]+splits["val"]
    early = ROOT/"e139/main/prefix148/prepared"
    manifest = json.loads((early/"manifest.json").read_text())
    assert manifest["train"]==dates[:148] and manifest["val"]==dates[148:188]
    assert digest(early/"surrogate.pkl")==manifest["hashes"]["surrogate.pkl"]
    tables = [run_split(n,days,dates[:n],dates[n:n+40],folder/"surrogate.pkl",out)
              for n,folder in ((148,early),(188,prep))]
    table = pd.concat(tables,ignore_index=True)
    table.to_csv(out/"summary.csv",index=False)
    decisions = []
    for n in (148,188):
        scores = table[(table.prefix==n)&(table.group=="all")].set_index(["direction","model"])
        base, cand, control = (scores.loc[("up",m)] for m in ("gbm","direct","no_new_air"))
        bd, cd = (scores.loc[("down",m)] for m in ("gbm","direct"))
        gates = dict(exact_recovery=bool(base.recoveries and cand.exact_recovery>base.exact_recovery),
                     up_mae=cand.mae<base.mae, up_false_recovery=cand.false_recovery<=base.false_recovery,
                     down_mae=cd.mae<=bd.mae, down_false_alarms=cd.false_alarms<=bd.false_alarms,
                     action_value=cand.mae<control.mae)
        decisions.append(dict(prefix=n, **gates, passed=all(gates.values())))
    pd.DataFrame(decisions).to_csv(out/"decision.csv",index=False)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    main(parser.parse_args().out)

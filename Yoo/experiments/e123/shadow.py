"""Issue prospective one-interval offset alarms from leave-one-model-out baselines."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e119"))
import agreement as cross

r = cross.r
ROOT = r.ROOT.parents[1]
KEY = cross.KEY
METHODS = ["shadow", "hold", "oracle"]


def measure(cut, native, other, first_proposal):
    # Row j is flow already decided at elapsed j*15, covering [j*15,(j+1)*15).
    n = min(len(cut), 13)
    cut, native, other = cut[:n], native[:n], other[:, :n]
    assert other.shape == (4,n)
    np.testing.assert_allclose(other[:,0], first_proposal, atol=.01, rtol=0)
    assert cut[0] < first_proposal
    expected = {"shadow":other.mean(0), "hold":np.full(n,first_proposal), "oracle":native}
    history = pd.DataFrame(dict(minute=np.arange(n)*15, cut_flow=cut, native_flow=native))
    alarms = {}
    for name, flow in expected.items():
        projected = np.cumsum((cut-flow)*.25)
        # The first cut is at time zero. Warnings start at the *next* decision.
        hits = np.flatnonzero(projected[1:] >= -1e-8)
        alarms[name] = int(15*(hits[0]+1)) if len(hits) else np.nan
        history[name+"_end_delta"] = projected
        history[name+"_baseline_error"] = np.cumsum((flow-native)*.25)
    for i, flow in enumerate(other):
        history[f"other_flow_{i+1}"] = flow
    return alarms, history


def run(out, seed, smoke):
    e = r.e113
    plan = r.read(r.ROOT/"plan.json")
    for name, digest in plan["input_sha256"].items():
        r.verify(e.prepared/name, digest)
    dates = plan["winter"]
    splits, days = e.base.load_selected(e.prepared/"data.pkl")
    assert dates == splits["val"] == sorted(dates)
    episodes = [(day,m) for day in dates for m in range(5)]
    index = {ep:i for i,ep in enumerate(episodes)}
    cases = pd.read_csv(ROOT/"e122/main/cases.csv")
    ref = pd.read_csv(ROOT/"e121/main/choices.csv",float_precision="round_trip").query("arm=='risk_air'")
    assert len(cases)==len(ref)==318 and not cases.duplicated(["mode",*KEY]).any()
    selected = ref.merge(cases[["mode",*KEY,"first","t"]],on=["mode",*KEY],validate="one_to_one")
    assert len(selected)==318
    selected = selected[selected.seed==seed].drop_duplicates(KEY)
    hashes = r.read(r.ROOT/"checksums.json")
    e.base.DeviceConfig().select_device(-1)
    agent = r.load(seed,hashes)[1]["weighted"]
    env = e.BatchEnv(days,e.FastDynamics(e.prepared/"surrogate.pkl"))
    with np.load(ROOT/"e117/main"/f"seed{seed}.npz") as z:
        saved, trigger = z["B"], z["trigger"]
    flows = saved[:,:,0]
    states,zones = cross.prefix(env,episodes,flows,saved)
    summaries, curves = [], []
    for lag in sorted(selected.lag.unique())[:1 if smoke else None]:
        rows = selected[selected.lag==lag].copy()
        rows.index = [index[(row.day,row.member)] for row in rows.itertuples()]
        ids = rows.index.to_numpy()
        when = np.full(len(episodes),-1)
        when[ids] = rows.t
        np.testing.assert_array_equal(when[ids],trigger[ids]+lag)
        args = (env,episodes,agent,rows,flows,states,zones,when)
        cut = cross.branch(*args,shift=0,cut=True)[:,:,0]
        other = np.stack([cross.branch(*args,shift=k,cut=False)[:,:,0] for k in range(1,5)])
        for row in rows.itertuples():
            i,t = row.Index,row.t
            raw = cut[i,t:]
            native = flows[i,t:]
            truth = np.cumsum((raw-native)*.25)
            np.testing.assert_allclose(truth[-1],row.air,rtol=0,atol=1e-6)
            np.testing.assert_allclose(truth[0],-row.reduction*.25,rtol=0,atol=1e-7)
            hit = np.flatnonzero(truth>=-1e-8)
            first = int((hit[0]+1)*15) if len(hit) else np.nan
            np.testing.assert_allclose(first,row.first,equal_nan=True)
            alarms, h = measure(raw,native,other[:,i,t:],row.proposal)
            reference = first-15 if 30<=first<=195 else np.nan
            np.testing.assert_allclose(alarms["oracle"],reference,equal_nan=True)
            key = {k:getattr(row,k) for k in KEY}
            summaries.append(dict(**key,**alarms))
            curves.append(h.assign(**key))
    if smoke:
        assert summaries
        return
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(summaries).to_csv(out/f"alarms{seed}.csv",index=False,mode="x")
    pd.concat(curves).to_csv(out/f"curves{seed}.csv",index=False,mode="x")
    print("Verified seed",seed,flush=True)


def report(source, dest):
    cases = pd.read_csv(ROOT/"e122/main/cases.csv")
    a = pd.concat([pd.read_csv(source/f"alarms{s}.csv") for s in range(5)])
    h = pd.concat([pd.read_csv(source/f"curves{s}.csv") for s in range(5)])
    assert len(a)==254 and not a.duplicated(KEY).any()
    full = cases.merge(a,on=KEY,validate="many_to_one")
    assert len(full)==318
    curves = cases[["mode","period",*KEY]].merge(h,on=KEY)
    performance, errors = [], []
    for mode,g in full.groupby("mode"):
        for period in ["all","early","late"]:
            season = g if period=="all" else g[g.period==period]
            for seed in ["all",*range(5)]:
                x = season if seed=="all" else season[season.seed==seed]
                large = x.group=="large"
                save = x.group=="saving"
                for name in METHODS+["nh_alarm","air_alarm"]:
                    alarm = x[name].notna()
                    early = large & (x[name] < x["first"])
                    fp = ~large & alarm
                    lead = (x.loc[early,"first"]-x.loc[early,name])
                    performance.append(dict(mode=mode,period=period,seed=seed,method=name,n=len(x),
                        large=int(large.sum()),early=int(early.sum()),lead_median=lead.median() if len(lead) else np.nan,
                        late=int((alarm&large&~early).sum()),missed=int((large&~alarm).sum()),
                        nonlarge=int((~large).sum()), false_alert=int(fp.sum()), savers=int(save.sum()),
                        saver_alert=int((save&alarm).sum()), only_method=int((alarm&x.oracle.isna()).sum()),
                        only_oracle=int((~alarm&x.oracle.notna()).sum()),passed=bool(large.any() and (~large).any()
                        and early.sum()>large.sum()/2 and fp.sum()<=(~large).sum()/4)))
            z=curves[curves["mode"]==mode]
            if period!="all":
                z=z[z.period==period]
            for end in (60,180):
                at_end=z[z.minute+15==end]
                for name in METHODS:
                    err=at_end[name+"_baseline_error"]
                    errors.append(dict(mode=mode,period=period,end_min=end,method=name,n=len(err),
                                       mean_error=err.mean(),mae=err.abs().mean()))
    dest.mkdir(parents=True,exist_ok=True)
    perf=pd.DataFrame(performance)
    perf.to_csv(dest/"performance.csv",index=False)
    pd.DataFrame(errors).to_csv(dest/"baseline_error.csv",index=False)
    full.to_csv(source/"cases.csv",index=False)
    halves=perf[(perf["mode"]=="matched")&(perf.seed=="all")&(perf.period!="all")&(perf.method=="shadow")]
    assert len(halves)==2
    (dest/"decision.json").write_text(json.dumps({"candidate":bool(halves.passed.all())},indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["replay","smoke","report"])
    parser.add_argument("out",type=Path)
    parser.add_argument("--seed",type=int,choices=range(5),default=0)
    parser.add_argument("--tables",type=Path,default=HERE)
    args=parser.parse_args()
    if args.stage=="report":
        report(args.out,args.tables)
    else:
        run(args.out,args.seed,args.stage=="smoke")

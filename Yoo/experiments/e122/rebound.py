"""Replay e121 cuts and align physical air use and available warnings to each decision."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e118"))
import timing as p

ROOT = p.r.ROOT.parents[1]
KEY = ["day", "member", "seed", "lag"]
SIGNALS = ["nh_alarm", "air_alarm"]


def crossings(delta):
    over = delta >= -1e-8
    assert not over[0]
    first = (np.flatnonzero(over)[0] + 1)*15 if over.any() else np.nan
    lasting = (np.flatnonzero(~over)[-1] + 2)*15 if over[-1] else np.nan
    return first, lasting


def first_alarm(hit):
    return (np.flatnonzero(hit)[0]+1)*15 if hit.any() else np.nan


def measure(base, cut, t, nh_start, proposal):
    # Boundary k is just after interval k and just before the next control decision.
    d = np.cumsum((cut[t:,0]-base[t:,0])*.25)
    first, lasting = crossings(d)
    n = min(12, len(d)-1)
    nh = cut[t:t+n,1].astype(np.float32).astype(float)
    flow = cut[t+1:t+1+n,0]
    metrics = dict(first=first, lasting=lasting, remaining_min=len(d)*15,
                   nh_alarm=first_alarm(nh-nh_start>=.1), air_alarm=first_alarm(flow-proposal>=250),
                   final_air=d[-1], initial_air=d[0], warning_window_min=n*15)
    for h in [15,60,180,360]:
        metrics[f"d{h}"] = d[h//15-1] if len(d)>=h//15 else np.nan
    return metrics, d


def replay(out, smoke):
    e = p.r.e113
    plan = p.r.read(p.r.ROOT/"plan.json")
    for name, digest in plan["input_sha256"].items():
        p.r.verify(e.prepared/name, digest)
    choices = pd.read_csv(ROOT/"e121/main/choices.csv", float_precision="round_trip")
    cutoff = json.loads((ROOT/"e121/main/fit.json").read_text())["loss_cutoff"]
    choices = choices[choices.arm=="risk_air"]
    assert choices.groupby("mode").size().to_dict()=={"frozen":218,"matched":100}
    assert not choices.duplicated(["mode", *KEY[:3]]).any()
    unique = choices.drop_duplicates(KEY)
    splits, days = e.base.load_selected(e.prepared/"data.pkl")
    dates = plan["winter"]
    assert splits["val"] == dates == sorted(dates)
    episodes = [(d,m) for d in dates for m in range(5)]
    index = {ep:i for i,ep in enumerate(episodes)}
    e.base.DeviceConfig().select_device(-1)
    env = e.BatchEnv(days, e.FastDynamics(e.prepared/"surrogate.pkl"))
    hashes = p.r.read(p.r.ROOT/"checksums.json")
    records, curves = [], []
    for seed in ([0] if smoke else range(5)):
        agents = p.r.load(seed, hashes)[1]
        base, _, trigger, _ = p.simulate(env, episodes, agents)
        with np.load(ROOT/"e117/main"/f"seed{seed}.npz") as z:
            p.prior.compare_traces(base,z["B"])
            np.testing.assert_array_equal(trigger,z["trigger"])
        for lag in range(1,5):
            ref = unique[(unique.seed==seed)&(unique.lag==lag)]
            if ref.empty:
                continue
            cut, features, stamp, chosen = p.simulate(env,episodes,agents,slot=lag)
            np.testing.assert_array_equal(trigger,stamp)
            for r in ref.itertuples():
                i = index[(r.day,r.member)]
                t = int(stamp[i]+lag)
                assert chosen[i]==lag and t>0
                np.testing.assert_array_equal(cut[i,:t],base[i,:t])
                np.testing.assert_allclose(features[i,lag-1], [getattr(r,k) for k in p.INPUTS], rtol=0, atol=1e-8)
                np.testing.assert_allclose(float(np.float32(cut[i,t-1,1])), r.nh4, rtol=0, atol=1e-7)
                np.testing.assert_allclose(base[i,t,0],r.proposal,atol=.01,rtol=0)
                result, curve = measure(base[i],cut[i],t,r.nh4,r.proposal)
                np.testing.assert_allclose(result["initial_air"],-.25*r.reduction,atol=1e-7,rtol=0)
                np.testing.assert_allclose(result["final_air"],r.air,atol=1e-6,rtol=0)
                assert (r.air>cutoff)==r.large
                key=dict(day=r.day,member=r.member,seed=seed,lag=lag)
                cls="large" if r.large else ("saving" if r.air<-.05 else "other")
                records.append(dict(**key, t=t, period="early" if r.day in dates[:20] else "late",
                                    group=cls, **result))
                curves.extend(dict(**key,minute=(j+1)*15,delta_air=v) for j,v in enumerate(curve))
        print("seed",seed,"verified",flush=True)
    if smoke:
        assert records
        return
    modes = choices[["mode",*KEY]]
    cases = modes.merge(pd.DataFrame(records),on=KEY,validate="many_to_one")
    assert len(cases)==len(choices)
    out.mkdir(parents=True,exist_ok=False)
    cases.to_csv(out/"cases.csv",index=False)
    pd.DataFrame(curves).to_csv(out/"curves.csv",index=False)


def summarize(source, dest):
    cases = pd.read_csv(source/"cases.csv")
    groups, alarms = [], []
    for mode, selections in cases.groupby("mode"):
        for period in ["all","early","late"]:
            season = selections if period=="all" else selections[selections.period==period]
            for seed in ["all",*range(5)]:
                sub = season if seed=="all" else season[season.seed==seed]
                tags = dict(mode=mode,period=period,seed=seed)
                for group in ["large","saving","other"]:
                    g = sub[sub.group==group]
                    row = dict(**tags,group=group,n=len(g),crossed=g["first"].count(),
                               lasting_n=g.lasting.count(),recrossed=((g["first"].notna()) &
                               ((g.lasting>g["first"]) | g.lasting.isna())).sum(),
                               full_warning_window=int((g.warning_window_min==180).sum()))
                    for field in ["first","lasting","d15","d60","d180","d360","final_air"]:
                        vals = g[field].dropna()
                        row.update({field+"_n":len(vals), **{field+"_"+k:v for k,v in
                            zip(["q25","median","q75"], vals.quantile([.25,.5,.75]).to_numpy())}})
                    groups.append(row)
                large = sub.group=="large"
                for signal in SIGNALS:
                    notified = sub[signal].notna()
                    timely = large & (sub[signal]<sub["first"])
                    fp = notified & ~large
                    saves = sub.group=="saving"
                    lead = (sub.loc[timely,"first"]-sub.loc[timely,signal])
                    alarms.append(dict(**tags,signal=signal,large=int(large.sum()),early=int(timely.sum()),
                        late_or_equal=int((large&notified&~timely).sum()),missed=int((large&~notified).sum()),
                        lead_median=lead.median() if len(lead) else np.nan,
                        nonlarge=int((~large).sum()),false_alert=int(fp.sum()),savers=int(saves.sum()),
                        saver_alert=int((saves&notified).sum()), passed=bool(large.any() and (~large).any()
                        and timely.sum()>large.sum()/2 and fp.sum()<=(~large).sum()/4)))
    dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(groups).to_csv(dest/"timing.csv",index=False)
    a=pd.DataFrame(alarms)
    a.to_csv(dest/"alarms.csv",index=False)
    decision=a[(a["mode"]=="matched")&(a.seed=="all")&(a.period!="all")].groupby("signal").passed.all()
    (dest/"decision.json").write_text(json.dumps(decision.to_dict(),indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["smoke","replay","report"])
    parser.add_argument("out",type=Path)
    parser.add_argument("--tables",type=Path,default=HERE)
    args=parser.parse_args()
    if args.stage=="report":
        summarize(args.out,args.tables)
    else:
        replay(args.out,args.stage=="smoke")

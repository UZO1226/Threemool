"""Compare quality under recorded aeration with observed subsequent water levels."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import folds
from darrow_env import air_to_action


def water(states):
    nh = states[:,:,0]
    do = states[:,:,2:].max(axis=2)
    return np.column_stack(((nh>4).sum(axis=1)*15, np.maximum(nh-4,0).sum(axis=1)/4,
                            (do>2).sum(axis=1)*15))


def replay(root,dest):
    rows = []
    for n in (108,148,188):
        if n==188:
            prep = folds.trial.PREP
            splits,days = folds.base.load_selected(prep/"data.pkl")
        else:
            _,prep = folds.paths(root,n)
            splits,days = folds.load_days(prep/"data.pkl")
        dates = splits["val"]
        env = folds.trial.fast.BatchEnv(days,folds.trial.fast.FastDynamics(prep/"surrogate.pkl"))
        episodes = folds.old.all_episodes(dates,5)
        env.reset(episodes)
        predicted = []
        for t in range(95):
            env.step(air_to_action(env.x["air"][:,t]))
            predicted.append(env.state.copy())
        pred = np.stack(predicted,axis=1)
        obs = env.x["state"][:,1:,:]
        sim,real = [water(x).reshape(40,5,3).mean(axis=1) for x in (pred,obs)]
        mae = np.abs(pred[:,:,0]-obs[:,:,0]).reshape(40,5,95).mean(axis=(1,2))
        for i,day in enumerate(dates):
            data = days[day]
            row = dict(prefix=n,day=day,air_m3=np.clip(data["air"][:-1],0,9000).sum()/4,
                       clipped_steps=int(((data["air"][:-1]<0)|(data["air"][:-1]>9000)).sum()),
                       nh4_mae=mae[i],temp=data["exo"][:-1,1].mean(),inflow=data["exo"][:-1,0].mean())
            for j,m in enumerate(("nh4_min","nh4_excess","do_min")):
                row["observed_"+m] = real[i,j]
                row["replay_"+m] = sim[i,j]
            rows.append(row)
    dest.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(dest,index=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    replay(args.root,args.output)

"""Check support in recorded training inputs at the measured NH4 rise events."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from response import S,onsets


def observations(g):
    e=g['exo'][:-1]
    air=g['air'][np.maximum(np.arange(95)-1,0)]
    return np.column_stack((g['state'][:-1],e[:,0],e[:,1],e[:,2],e[:,3],e[:,4],air))


def run(source,out):
    splits,days=S.base.load_selected(S.PREPARED/'data.pkl')
    train=splits['train']
    x=np.concatenate([observations(days[d]) for d in train])
    scale=np.diff(np.quantile(x,[.25,.75],axis=0),axis=0)[0]
    assert np.all(scale>0)
    nn=NearestNeighbors(n_neighbors=96).fit(x/scale)
    scores=[]
    for split,dates in [('train',train),('winter',splits['val'])]:
        for day in dates:
            ts=onsets(days[day]['state'][:,0])
            if not ts:
                continue
            dist,ids=nn.kneighbors(observations(days[day])[ts]/scale)
            for t,ds,ix in zip(ts,dist,ids):
                if split=='train':
                    ds=ds[ix//95!=train.index(day)]
                scores.append((day,t,split,ds[0]/np.sqrt(x.shape[1])))
    scored=pd.DataFrame(scores,columns=['day','t','split','distance'])
    threshold=scored.loc[scored.split=='train','distance'].quantile(.90)
    scored['rare']=scored.distance>threshold
    events=pd.read_csv(source/'events.csv.gz').query("kind=='same_input' and period!='train'").copy()
    events['command']=events.air_before+events.first_delta
    groups=['day','period','t','method']
    assert events.groupby(groups).size().eq(5).all()
    spread=events.groupby(groups).command.std(ddof=0).rename('air_sd').reset_index()
    winter=scored[scored.split=='winter'].drop(columns='split')
    merged=spread.merge(winter,on=['day','t'],validate='many_to_one',how='left')
    assert len(merged)==len(winter)*2 and merged.notna().all().all()
    by=['period','method','rare']
    daily=merged.groupby(by+['day']).air_sd.mean()
    report=daily.groupby(level=by).mean().to_frame()
    report['days']=daily.groupby(level=by).size()
    report['events']=merged.groupby(by).size()
    report['train_threshold']=threshold
    out.mkdir(parents=True,exist_ok=False)
    report.to_csv(out/'coverage.csv')
    scored.to_csv(out/'distances.csv',index=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    run(args.source,args.out)

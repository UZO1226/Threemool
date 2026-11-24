"""Pair held-date labels with size-matched included-date labels on the same events."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'e109'))
import evaluate as prior

ROOT = prior.ROOT
MODES = ('included', 'excluded')


def replicas(root, mode, originals, dates):
    rows = []
    for k in range(5):
        folder = root/f'fold{k}'
        lists = json.loads((folder/'dates.json').read_text())
        assert lists['fit']==[d for i,d in enumerate(dates) if i//7%5!=k]
        assert lists['held']==[d for i,d in enumerate(dates) if i//7%5==k]
        with np.load(folder/f'{mode}.npz') as z:
            base, _ = prior.oracle.losses(z['preds'][0],z['truth'])
            corrected, _ = prior.oracle.losses(z['preds'][1],z['truth'])
            labels = prior.oracle.decisions(base-corrected)[1]
            for i,(day,t) in enumerate(zip(z['dates'],z['t'])):
                assert (day in lists['fit'])==(mode=='included')
                fold = dates.index(day)//7%5
                assert k==(fold if mode=='excluded' else (fold+1)%5)
                np.testing.assert_array_equal(originals[day,t],z['truth'][i])
                rows.append((day,t,labels[i],base[i,0]))
    cols = ['date','t',mode,mode+'_base_nh4_mae']
    return pd.DataFrame(rows,columns=cols)


def label_summary(g):
    w = prior.weights(g)
    fields = ('original',*MODES)
    labels = {m:dict(positives=int(g[m].sum()),rate=np.average(g[m],weights=w),
                     original_agreement=np.average(g[m]==g.original,weights=w),
                     base_mae=np.average(g[m+'_base_nh4_mae'],weights=w)) for m in fields}
    return dict(n=len(g), days=g.date.nunique(), labels=labels,
                paired=prior.oracle.cross(g.included,g.excluded))


def analyze(root, out):
    raw = pd.read_pickle(ROOT/'e50/prepared/data.pkl')['data']
    hashes = dict(line.split()[::-1] for line in (HERE.parent/'e106/sha256.txt').read_text().splitlines())
    a,b,_,n = prior.load('fit',raw,hashes)
    a['original'] = a.useful
    a['original_base_nh4_mae'] = b[:,0]
    with np.load(ROOT/'e106/main/fit.npz') as old:
        dates = old['dates'].tolist()
        originals = {(dates[d],t):x for d,t,x,dq in zip(old['ids'],old['t'],old['truth'],old['dq']) if dq>0}
    for m in MODES:
        a = a.merge(replicas(root,m,originals,dates),on=['date','t'],how='left',validate='one_to_one')
    assert len(a)==len(originals)==194 and a.original.sum()==17 and not a.isna().any().any()
    a['fold']=a.day//7%5
    stats = label_summary(a)
    diff = (a.excluded.astype(float)-a.included).groupby(a.day).mean().reindex(range(n)).to_numpy()
    starts=np.random.default_rng(110).integers(n,size=(2000,(n+6)//7))
    draws=((starts[:,:,None]+np.arange(7))%n).reshape(2000,-1)[:,:n]
    stats['difference_ci']=np.quantile(np.nanmean(diff[draws],axis=1),[.025,.975]).tolist()
    stats['folds']={str(k):label_summary(g) for k,g in a.groupby('fold')}
    out.mkdir(exist_ok=False,parents=True)
    a.to_csv(out/'labels.csv',index=False)
    baseline=json.loads((HERE.parent/'e109/model.json').read_text())
    x,w=a[prior.FEATURES].to_numpy(),prior.weights(a)
    locks={}
    for m in MODES:
        y=a[m].to_numpy()
        model=prior.fit(x,y,w)
        prevalence=np.average(y,weights=w)
        threshold=prior.cutoff(prior.predict(model,x),w,prevalence)
        locks[m]=dict(models=dict(fall=baseline['models']['fall'],useful=model),
                      cutoff=dict(fall=baseline['cutoff']['fall'],useful=threshold),prevalence=prevalence)
    (out/'locked.json').write_text(json.dumps(locks,indent=2)+'\n')
    result=dict(training=stats,evaluation={})
    reference=json.loads((HERE.parent/'e109/summary.json').read_text())
    for split in ('confirm','winter'):
        df,b,c,n=prior.load(split,raw,hashes)
        result['evaluation'][split]={}
        for m,lock in locks.items():
            scored=df.copy()
            scores=prior.evaluate(scored,b,c,n,lock)
            assert scores['positives']==reference[split]['positives']
            np.testing.assert_allclose(scores['methods']['base']['mae'],reference[split]['methods']['base']['mae'])
            result['evaluation'][split][m]=scores
            scored.to_csv(out/f'{split}_{m}.csv',index=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    parser.add_argument('out',type=Path)
    args=parser.parse_args()
    analyze(args.root,args.out)

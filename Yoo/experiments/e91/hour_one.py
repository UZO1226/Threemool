"""Landmark diagnostic after one hour of identical post-cut operation."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'e90'))
import rebound as R

S = R.S
HOUR = 4


class PrefixRecorder:
    def __init__(self, dynamics):
        self.dynamics = dynamics
        self.states = []

    def step(self, *args):
        state = self.dynamics.step(*args)
        if len(self.states) < HOUR:
            self.states.append(state.copy())
        return state


def replay(rows, archive, keys, q, agent, dyn):
    totals, features = [], []
    for start in range(0,len(keys),250):
        ix = slice(start,start+250)
        args = (rows,archive,keys[ix],q[ix],agent)
        a, base = R.simulate(*args,dyn,False)
        prefix = PrefixRecorder(dyn)
        b, cut = R.simulate(*args,prefix,True)
        tape = cut.copy()
        tape[:,HOUR:] = np.minimum(base[:,HOUR:],cut[:,HOUR:])
        replay_prefix = PrefixRecorder(dyn)
        d, _ = R.simulate(*args,replay_prefix,True,tape)
        np.testing.assert_array_equal(prefix.states,replay_prefix.states)
        np.testing.assert_array_equal(b[:,:HOUR],d[:,:HOUR])
        np.testing.assert_allclose(b[:,HOUR:,0]-d[:,HOUR:,0],
                                   np.maximum(0,b[:,HOUR:,0]-a[:,HOUR:,0]),rtol=0,atol=R.TOL)
        day,t,m = keys[ix].T
        first = archive['states'][day*5+m,t-1,0]
        nh = prefix.states[-1][:,0]
        np.testing.assert_array_equal(b[:,HOUR-1,1],nh>4)
        np.testing.assert_allclose(b[:,HOUR-1,3],np.maximum(0,nh-4)*.25)
        features.append(np.column_stack((nh,nh-first)))
        totals.append(np.stack([s.sum(axis=1) for s in (a,b,d)],axis=1))
    return np.concatenate(features),np.concatenate(totals)


def benefit(totals):
    diff = (totals[:,1]-totals[:,2])[:,[1,3]]
    return (diff<=R.TOL).all(axis=1) & (diff < -R.TOL).any(axis=1)


def cutoff(scores,y,w):
    if not y.any():
        return None
    selected = np.flatnonzero(y)
    order = selected[np.argsort(-scores[selected],kind='stable')]
    i = np.searchsorted(np.cumsum(w[order]),.90*w[order].sum())
    return float(scores[order[i]])


def fit(x,y,days,xt):
    fit_mask = days < 100
    cal = ~fit_mask
    assert fit_mask.any() and cal.any()
    w = S.weights(days[fit_mask])
    scale = StandardScaler().fit(x[fit_mask],sample_weight=w)
    if len(np.unique(y[fit_mask])) == 2:
        model = LogisticRegression(C=1,max_iter=2000).fit(scale.transform(x[fit_mask]),y[fit_mask],sample_weight=w)
        assert model.n_iter_[0] < 2000
        cs = model.decision_function(scale.transform(x[cal]))
        score = model.decision_function(scale.transform(xt))
    else:
        cs = np.full(cal.sum(),float(y[fit_mask][0]))
        score = np.full(len(xt),cs[0])
    weight = S.weights(days[cal])
    threshold = cutoff(cs,y[cal],weight)
    keep = np.ones(len(score),bool) if threshold is None else score>=threshold
    recall = None if threshold is None else float(np.average(cs[y[cal]]>=threshold,weights=weight[y[cal]]))
    assert recall is None or recall >= .9-1e-10
    return score,keep,dict(cutoff=threshold,cal_recall=recall,fit_pos=int(y[fit_mask].sum()),
                           cal_pos=int(y[cal].sum()),fit_n=int(fit_mask.sum()),cal_n=int(cal.sum()))


def effects(paths,keep,w):
    a,b,d=paths.transpose(1,0,2)
    choice = d+np.asarray(keep)[:,None]*(b-d)
    return dict(vs_base=np.average(choice-a,weights=w,axis=0).tolist(),
                vs_keep=np.average(choice-b,weights=w,axis=0).tolist(),retained=float(np.sum(keep)))


def classify(day,paths,score,keep):
    y=benefit(paths)
    w=S.weights(day)
    def rate(v,mask):
        return float(np.average(v[mask],weights=w[mask])) if mask.any() else None
    coin=np.empty(len(day))
    for date in np.unique(day):
        group=day==date
        coin[group]=keep[group].mean()
    return dict(auc=float(roc_auc_score(y,score,sample_weight=w)) if len(np.unique(y))==2 else None,
                recall=rate(keep,y),precision=rate(y,keep),fpr=rate(keep,~y),positive=int(y.sum()),
                missed=int((y&~keep).sum()),outcome=effects(paths,keep,w),matched_coin=effects(paths,coin,w))


def run(root,out,smoke):
    torch.set_num_threads(1)
    dates,rows=S.E.load_train()
    assert dates[99]<dates[100]<=dates[139]<dates[140]
    hashes=json.loads((HERE.parent/'e77/summary.json').read_text())['sha256']
    assert all(S.digest(Path(p))==v for p,v in hashes.items())
    dyn=S.E.Dynamics(S.E.DATA/'surrogate.pkl')
    refs=json.loads((HERE.parent/'e89/summary.json').read_text())
    prev=json.loads((HERE.parent/'e90/summary.json').read_text())
    probe_refs=json.loads((HERE.parent/'e86/summary.json').read_text())
    records=[]
    if not smoke:
        out.mkdir(parents=True,exist_ok=False)
    for ref in refs:
        seed=ref['seed']
        agent=S.E.restore(seed)
        initial=S.E.S.fingerprint(agent)
        src=[root/'e88/main'/f'{seed}_early_down.npz',root/'e78/main'/f'{seed}_train48_down.npz',
             root/'e88/main'/f'probe{seed}.npz']
        assert [S.digest(p) for p in src]==ref['input_sha256']
        query_file=root/'e85/main'/f'probe{seed}.npz'
        assert S.digest(query_file)==probe_refs[seed]['score_sha256']
        with np.load(query_file) as z:
            full,q=z['keys'],z['q']
            expected=z['totals'][:,:,-1]
        late,keys=S.read_candidates(src[1],dates[140:])
        subset=S.headroom.nonoverlap(full)
        np.testing.assert_array_equal(full[subset],keys)
        if smoke:
            _,v=replay(rows[140:],late,full[:5],q[:5],agent,dyn)
            np.testing.assert_allclose(v[:,:2],expected[:5],rtol=0,atol=1e-7)
            assert cutoff(np.arange(10),np.ones(10,bool),np.ones(10))==1
            print('identical prefix, archived totals and recall threshold verified')
            return
        xt,qt=replay(rows[140:],late,full,q,agent,dyn)
        np.testing.assert_allclose(qt[:,:2],expected,rtol=0,atol=1e-7)
        xt,qt=xt[subset],qt[subset]
        match=root/'e89/main'/f'matches{seed}.npz'
        assert S.digest(match)==prev[seed]['match_sha256']
        with np.load(match) as z:
            np.testing.assert_array_equal(z['keys'],keys)
            support=z['supported']
        with np.load(root/'e90/main'/f'paths{seed}.npz') as z:
            np.testing.assert_array_equal(z['supported'],support)
            np.testing.assert_array_equal(z['keys'],keys)
            np.testing.assert_allclose(qt[:,:2],z['cumulative'][:,:2,-1],rtol=0,atol=1e-7)
        early,k=S.read_candidates(src[0],dates[:140])
        day,t,m=k.T
        x,tr=replay(rows[:140],early,k,early['raw'][day*5+m,t],agent,dyn)
        with np.load(src[2]) as z:
            np.testing.assert_array_equal(z['train_keys'],k)
            np.testing.assert_allclose(tr[:,:2],z['train_totals'],rtol=0,atol=1e-7)
        labels=benefit(tr)
        score_list,keep_list,settings=[],[],{}
        for width,name in [(1,'level'),(2,'trend')]:
            scores,keep,params=fit(x[:,:width],labels,day,xt[:,:width])
            score_list.append(scores)
            keep_list.append(keep)
            settings[name]=params
        result={}
        for name,ix in [('supported',support),('all',np.ones(len(keys),bool))]:
            group,paths=keys[ix,0],qt[ix]
            w=S.weights(group)
            diff=paths[:,2]-paths[:,1]
            oracle=~((diff[:,0] < -R.TOL) & (diff[:,1:]<=R.TOL).all(1))
            stats=dict(n=len(paths),days=len(np.unique(group)),always=effects(paths,np.ones(len(paths)),w),
                       remove=effects(paths,np.zeros(len(paths)),w),oracle=effects(paths,oracle,w))
            for mode,sc,keep in zip(('level','trend'),score_list,keep_list):
                stats[mode]=classify(group,paths,sc[ix],keep[ix])
            result[name]=stats
        assert S.E.S.fingerprint(agent)==initial
        np.savez_compressed(out/f'hour{seed}.npz',keys=keys,supported=support,features=xt,totals=qt,
                            train_keys=k,train_x=x,train_totals=tr,scores=np.stack(score_list),keep=np.stack(keep_list))
        records.append(dict(seed=seed,settings=settings,groups=result))
        primary=result['supported']['trend']
        print(seed,primary['auc'],primary['recall'],primary['outcome']['vs_base'],flush=True)
    (out/'summary.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=S.ROOT)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    run(args.root,args.out,args.smoke)

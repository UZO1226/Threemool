"""Attribute archived policy commands and evaluate selection before the first recovery."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import paired as P

GROUPS = ([0], [1], [2,3,4], [10])
FEATURES = ['NH4','NH4_change','DO1','DO2','DO3','previous_air']
WINTER = Path('/home/jio/fcsrl_src/results/e81/main')


def attribute(left, right, predict):
    assert left.shape == right.shape
    np.testing.assert_array_equal(left[:,5:10],right[:,5:10])
    f = []
    for subset in range(16):
        x = left.copy()
        for i,cols in enumerate(GROUPS):
            if subset & (1 << i):
                x[:,cols] = right[:,cols]
        f.append(predict(x))
    phi = np.zeros((len(left),4))
    for i in range(4):
        for s in range(16):
            if not s & (1 << i):
                phi[:,i] += (f[s | (1 << i)]-f[s])/(4*math.comb(3,s.bit_count()))
    np.testing.assert_allclose(phi.sum(1),f[-1]-f[0],atol=1e-8,rtol=0)
    return phi,f[0],f[-1]


def smoke():
    left = np.zeros((2,11))
    right = left.copy()
    right[:,[0,1,2,10]] = [[1,2,3,4],[0,0,0,0]]
    def predict(x):
        return x[:,0]*x[:,10] + 3*x[:,1] - 2*x[:,2]
    phi,_,_ = attribute(left,right,predict)
    np.testing.assert_allclose(phi,[[2,6,-6,2],[0,0,0,0]])
    print('mask and interaction check passed')


def classification(y, score):
    take = score>=.5
    tp = int((take & y).sum())
    return dict(n=len(y),positive=int(y.sum()),selected=int(take.sum()),tp=tp,
                precision=tp/int(take.sum()) if take.any() else None,
                recall=tp/int(y.sum()) if y.any() else None,
                auc=float(roc_auc_score(y,score)) if len(set(y)) == 2 else None)


def read_traces(path):
    with np.load(path) as data:
        return {k:data[k] for k in data.files}


def read_training(root, dates):
    xs,ys,counts = [],[],[]
    for seed in range(5):
        with np.load(root/f'train{seed}.npz') as rows:
            np.testing.assert_array_equal(rows['dates'],np.asarray(dates,str))
            xs.append(rows['x'])
            ys.append(rows['y'])
            counts.append([len(ys[-1]),int(ys[-1].sum()),len(np.unique(rows['ep']//5))])
    x,y = np.concatenate(xs),np.concatenate(ys)
    assert len(y)>0
    if len(np.unique(y)) == 1:
        return lambda new: np.full(len(new),float(y[0])),counts,{}
    model = make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=2000))
    model.fit(x,y)
    scaler,logit = model.steps[0][1],model.steps[1][1]
    params = dict(features=FEATURES, mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
                  coef=logit.coef_[0].tolist(),intercept=float(logit.intercept_[0]))
    return lambda new:model.predict_proba(new)[:,1],counts,params


def evaluate(root):
    splits, days = P.setup()
    score,counts,params = read_training(root,splits['train'])
    dates = splits['val']
    saved = json.loads((P.HERE.parent/'e81/records.json').read_text())
    results,labels,scores = [],[],[]
    for seed in range(5):
        agent = P.E.restore(seed)
        group = {m:read_traces(WINTER/f'{seed}_{m}.npz') for m in P.R.MODES}
        amounts = {m:P.metrics(z) for m,z in group.items()}
        for m,z in group.items():
            np.testing.assert_array_equal(z['dates'], np.array(dates,str))
            ref = next(r for r in saved if r['mode']==m and r['seed']==seed)
            np.testing.assert_allclose(amounts[m].mean(0),[ref[k] for k in P.R.METRICS],atol=1e-8,rtol=0)
        pulse,rec = group['pulse'],group['recover']
        left,right = [P.observations(z,dates,days) for z in (pulse,rec)]
        ep,t,x,y = P.events(pulse,rec,left)
        np.testing.assert_array_equal(left[ep,t],right[ep,t])
        prob = score(x)
        take = np.zeros(200,dtype=bool)
        take[ep] = prob>=.5
        amounts['selected'] = np.where(take[:,None],amounts['recover'],amounts['pulse'])
        labels.append(y)
        scores.append(prob)

        def predict(obs):
            return P.R.prior.action_to_air(agent.act_det(obs)[:,0])
        # Same batch shape for all masks. Inputs at t include the action executed at t-1.
        contributions = []
        for step in range(95):
            if np.array_equal(left[:,step],right[:,step]):
                q = predict(left[:,step])
                other = q
                phi = np.zeros((len(q),4))
            else:
                phi,q,other = attribute(left[:,step],right[:,step],predict)
            np.testing.assert_allclose(q,pulse['raw'][:,step],atol=.02,rtol=0)
            np.testing.assert_allclose(other,rec['raw'][:,step],atol=.02,rtol=0)
            contributions.append(phi)
        phi = np.stack(contributions, axis=1)
        observed = rec['raw'] - pulse['raw']
        error = np.max(np.abs(phi.sum(2)-observed))
        daily = phi.sum(1)*.25
        np.testing.assert_allclose(daily.sum(1),observed.sum(1)*.25,atol=.05,rtol=0)
        has_next = t+1<95
        immediate = phi[ep[has_next],t[has_next]+1]
        row = dict(seed=seed,n_next=len(immediate),days=len(set(ep//5)),
                   immediate=immediate.mean(0).tolist(),immediate_abs=np.abs(immediate).mean(0).tolist(),
                   daily=daily.mean(0).tolist(),max_endpoint_delta_error=float(error),
                   metrics={k:v.mean(0).tolist() for k,v in amounts.items()},
                   class_stats=classification(y,prob))
        np.savez_compressed(root/f'winter{seed}.npz',ep=ep,t=t,dates=np.asarray(dates,str),x=x,y=y,
                            score=prob,selected=take,phi=phi,**amounts_to_arrays(amounts))
        results.append(row)
        print('winter audited',seed,flush=True)
    output = dict(groups=['NH4','NO3','DO1-3','previous_air'],train_counts=counts,model=params,
                  rows=results,pooled=classification(np.concatenate(labels),np.concatenate(scores)))
    (root/'summary.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')


def amounts_to_arrays(amounts):
    return {f'day_{name}':v for name,v in amounts.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    if args.smoke:
        smoke()
    else:
        evaluate(args.root)

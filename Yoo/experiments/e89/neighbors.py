"""Check local agreement of matched six-hour effects across DARROW dates."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'e88'))
import selection as S

ROOT = S.ROOT
TOL = S.headroom.TOL


def prehistory(rows, archive, keys):
    day, t, member = keys.T
    assert np.all(t >= 4)
    ep = day*5 + member
    nh = archive['states'][:, :, 0]
    past = nh[ep, np.maximum(t-5, 0)].copy()
    for i in np.flatnonzero(t == 4):
        past[i] = rows[day[i]]['state'][0, 0]
    air = archive['air'][ep[:, None], t[:, None] + np.arange(-4, 0)]
    return np.column_stack((nh[ep, t-1]-past, air[:, -1]-air[:, 0], air.mean(axis=1)-air[:, -1]))


def inputs(rows, archive, keys):
    obs, _ = S.before_action(rows, archive, keys)
    return obs.astype(float), np.column_stack((obs, prehistory(rows, archive, keys)))


def weighted_quantile(values, w, p):
    order = np.argsort(values, kind='stable')
    pos = np.searchsorted(np.cumsum(w[order]), p*w.sum())
    return float(values[order[pos]])


def find_neighbors(train, query, day, weights):
    scaler = StandardScaler().fit(train, sample_weight=weights)
    ref = scaler.transform(train)
    distance = cdist(ref, ref)/np.sqrt(ref.shape[1])
    distance[day[:, None] == day] = np.inf
    nearest = distance.min(axis=1)
    assert np.isfinite(nearest).all()
    limits = np.array([weighted_quantile(nearest, weights, p) for p in (.5, .95)])
    d = cdist(scaler.transform(query), ref)/np.sqrt(ref.shape[1])
    # Take the nearest row from each date, then the five closest dates.
    best = []
    for date in np.unique(day):
        ix = np.flatnonzero(day == date)
        best.append(ix[d[:, ix].argmin(axis=1)])
    best = np.sort(np.stack(best, axis=1), axis=1)
    by_date = np.take_along_axis(d, best, axis=1)
    order = by_date.argsort(axis=1, kind='stable')[:, :5]
    ids = np.take_along_axis(best, order, axis=1)
    return ids, np.take_along_axis(d, ids, axis=1), limits


def event(change):
    return np.column_stack((change[:, 0] < -TOL, change[:, 1:] > TOL))


def positive(change):
    return (change[:, 0] <= -25.) & (change[:, 1:] <= TOL).all(axis=1)


def summarize(keys, train_change, query_change, neighbors, subset, prior):
    if not subset.any():
        return None
    ids = neighbors[subset]
    w = S.weights(keys[subset, 0])
    y, yt = positive(train_change), positive(query_change[subset])
    vote = y[ids].mean(axis=1)
    nearest = train_change[ids[:, 0]]
    target = query_change[subset]
    def avg(x):
        return np.average(x, weights=w, axis=0)
    return dict(n=int(subset.sum()), days=len(np.unique(keys[subset, 0])),
                mismatch=float(avg(y[ids[:, 0]] != yt)), brier=float(avg((vote-yt)**2)),
                chance_mismatch=float(avg(np.where(yt, 1-prior, prior))),
                chance_brier=float(avg((prior-yt)**2)), prevalence=float(avg(yt)),
                component_disagreement=avg(event(nearest) != event(target)).tolist(),
                component_mae=avg(np.abs(nearest-target)).tolist())


def run(root, out):
    dates, rows = S.E.load_train()
    refs = json.loads((HERE.parent/'e88/summary.json').read_text())
    e85 = json.loads((HERE.parent/'e85/summary.json').read_text())['rows']
    data_path = S.E.DATA/'data.pkl'
    model_inputs = json.loads((HERE.parent/'e77/summary.json').read_text())['sha256']
    assert S.digest(data_path) == model_inputs[str(data_path)]
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for ref in refs:
        seed = ref['seed']
        probe = root/'e88/main'/f'probe{seed}.npz'
        early = root/'e88/main'/f'{seed}_early_down.npz'
        late = root/'e78/main'/f'{seed}_train48_down.npz'
        assert S.digest(late) == e85[seed]['source_sha256']
        with np.load(probe) as z:
            train, val = z['train_dates'], z['dates']
            np.testing.assert_array_equal(np.concatenate((train, val)), np.asarray(dates, str))
            assert max(train) < min(val)
            tr_keys, keys = z['train_keys'], z['keys']
            tr_change = z['train_totals'][:, 1] - z['train_totals'][:, 0]
            change = z['change']
            np.testing.assert_array_equal(positive(change), z['positive'])
        assert positive(tr_change).sum() == ref['train_pos'] and positive(change).sum() == ref['eval_pos']
        a, k = S.read_candidates(early, train)
        b, kt = S.read_candidates(late, val)
        np.testing.assert_array_equal(k, tr_keys)
        np.testing.assert_array_equal(kt, keys)
        x, xt = inputs(rows[:140], a, k), inputs(rows[140:], b, kt)
        w = S.weights(k[:, 0])
        prior = np.average(positive(tr_change), weights=w)
        peers, distances, limits = map(np.stack, zip(*(find_neighbors(v, q, k[:, 0], w) for v, q in zip(x,xt))))
        assert np.all(np.diff(distances, axis=2)>=0)
        groups = dict(all=np.ones(len(keys), bool),
                      supported=(distances[:,:,0] <= limits[:,1,None]).all(axis=0),
                      close=(distances[:,:,0] <= limits[:,0,None]).all(axis=0))
        result = {g: {method:summarize(keys,tr_change,change,ids,mask,prior)
                      for method,ids in zip(('current','history'), peers)} for g,mask in groups.items()}
        records.append(dict(seed=seed, train_n=len(k), eval_n=len(kt), input_sha256=[S.digest(p) for p in (early,late,probe)],
                            limits=limits.tolist(), metrics=result))
        np.savez_compressed(out/f'matches{seed}.npz', train_keys=k, keys=kt, ids=peers, distances=distances,
                            limits=limits, train_x=x[1], x=xt[1], **groups)
        print(seed, len(kt), groups['supported'].sum(), groups['close'].sum(), flush=True)
    (out/'summary.json').write_text(json.dumps(records, indent=2, allow_nan=False)+'\n')


def smoke():
    keys=np.array([[0,4,0],[0,8,0]])
    nh=np.arange(12,dtype=float)[None,:,None]
    archive=dict(states=nh,air=100*np.arange(12)[None,:])
    rows=[dict(state=np.array([[-1.]]))]
    expected=np.array([[4.,300.,-150.],[4.,300.,-150.]])
    np.testing.assert_array_equal(prehistory(rows,archive,keys),expected)
    # No decision-time or later value can alter its own history.
    for key, truth in zip(keys,expected):
        t=key[1]
        altered={name:v.copy() for name,v in archive.items()}
        for v in altered.values():
            v[:,t:]=999
        np.testing.assert_array_equal(prehistory(rows,altered,key[None,:])[0],truth)
    day=np.array([0,0,1,2,3,4,5])
    x=np.array([0.,0.,1.,2.,3.,4.,5.])[:,None]
    ids,_,_=find_neighbors(x,x[:1],day,S.weights(day))
    np.testing.assert_array_equal(ids[0], [0,2,3,4,5])
    assert weighted_quantile(np.array([2,0,1]),np.array([1,9,1]),.5)==0


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.smoke:
        smoke()
    else:
        run(args.root,args.out)

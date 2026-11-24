"""Replay the observed DARROW zone flows from measured or recursively predicted states."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'e77'))
import diagnose as E

spec = importlib.util.spec_from_file_location('e37_windows', HERE.parent/'e37/run.py')
W = importlib.util.module_from_spec(spec)
spec.loader.exec_module(W)

STEPS = np.array([1,4,12])


def forecasts(dyn, truth, exo, zone):
    n, m = len(truth), dyn.n_members
    member = np.tile(np.arange(m), n)
    repeat = lambda v: np.repeat(v, m, axis=0)
    state = repeat(truth[:,0])
    recursive, measured = [], []
    for h in range(1,exo.shape[1]+1):
        args = (repeat(exo[:,h-1]), repeat(zone[:,h]), repeat(zone[:,h-1]), member)
        state = dyn.step(state,*args)
        if h in STEPS:
            recursive.append(state.reshape(n,m,-1).copy())
            one = dyn.step(repeat(truth[:,h-1]),*args)
            measured.append(one.reshape(n,m,-1))
    return np.stack(recursive,1), np.stack(measured,1)


def pack(events,days):
    truth, exo, zone = [], [], []
    for w in events:
        g,t = days[w['day']],w['t']
        truth.append(g['state'][t:t+13])
        exo.append(g['exo'][t:t+12])
        zone.append(g['zone'][t-1:t+12])
    return tuple(map(np.stack,(truth,exo,zone)))


def smoke():
    class Toy:
        n_members = 2
        def step(self,s,e,z,p,m):
            return s + e + z - p + m[:,None]
    u = np.arange(12.)[None,:,None]
    z = 2*np.arange(13.)[None,:,None]
    x = (10+3*np.arange(13.))[None,:,None]
    rec, local = forecasts(Toy(),x,u,z)
    for i,h in enumerate(STEPS):
        for m in range(2):
            assert rec[0,i,m,0] == 10+h*(h-1)/2+h*(2+m)
            assert local[0,i,m,0] == x[0,h-1,0]+h-1+2+m
    np.testing.assert_array_equal(rec[:,0],local[:,0])


def run(out):
    data, model = [E.DATA/s for s in ('data.pkl','surrogate.pkl')]
    ref = json.loads((HERE.parent/'e77/summary.json').read_text())
    for p in (data,model):
        assert hashlib.sha256(p.read_bytes()).hexdigest()==ref['sha256'][str(p)]
    splits,days = E.S.base.load_selected(data)
    train,val = splits['train'],splits['val']
    dyn = E.Dynamics(model)
    assert dyn.n_members==5 and len(val)==40
    threshold = max(500.,np.quantile([abs(w['dq']) for w in W.windows(days,train)],.9))
    events = W.events(W.windows(days,val),threshold)
    names = np.array([str(w['day']) for w in events])
    times = np.array([w['t'] for w in events])
    dq = np.array([w['dq'] for w in events])
    truth,exo,zone = pack(events,days)
    np.testing.assert_allclose(zone[:,1].sum(1)-zone[:,0].sum(1),dq,atol=1e-8,rtol=0)
    with np.load(E.DATA.parents[1]/'e37/main/trajectories.npz') as old:
        n = len(events)
        np.testing.assert_array_equal(names, old['days'][:n].astype(str))
        np.testing.assert_array_equal(times,old['t'][:n])
        np.testing.assert_array_equal(truth, old['actual'][:n])
    assert (len(events),(dq>0).sum(),(dq<0).sum())==(132,67,65)
    rec,local = forecasts(dyn,truth,exo,zone)
    np.testing.assert_array_equal(rec[:,0],local[:,0])
    _,check = W.simulate(events,days,dyn)
    np.testing.assert_allclose(rec,check[:,:,STEPS].transpose(0,2,1,3),atol=1e-10,rtol=0)
    assert np.isfinite(rec).all() and np.isfinite(local).all()
    # e77 includes late-day anchors with only four future steps.
    anchors = [(d,t) for d in train[140:] for t in range(4,89,4)
               if abs(days[d]['air'][t]-days[d]['air'][t-1])>=500]
    x = np.stack([days[d]['state'][t:t+5] for d,t in anchors])
    e = np.stack([days[d]['exo'][t:t+4] for d,t in anchors])
    z = np.stack([days[d]['zone'][t-1:t+4] for d,t in anchors])
    short,_ = forecasts(dyn,x,e,z)
    up = (z[:,1].sum(1)-z[:,0].sum(1))>0
    for name, on in [('increase',up),('decrease',~up)]:
        expected = ref['observed'][name]
        assert on.sum() == expected['n']
        np.testing.assert_allclose(abs(short[on,-1,:,0].mean(1)-x[on,-1,0]).mean(),
                                   expected['model_mae'],atol=1e-8,rtol=0)
        np.testing.assert_allclose(abs(x[on,0,0]-x[on,-1,0]).mean(),expected['persistence_mae'])
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'replay.npz', date_index=np.array([val.index(w['day']) for w in events]),
                        dates=names, t=times, dq=dq, truth=truth, rec=rec, local=local)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    smoke()
    if args.out:
        run(args.out)

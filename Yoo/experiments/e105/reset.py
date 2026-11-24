"""Reset selected measured components at hour one and replay the next eight inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'e104'))
import replay as prior

MODES = {'none': (), 'nh4': (0,), 'do': (2,3,4), 'nh4_do': (0,2,3,4), 'all': (0,1,2,3,4)}


def advance(dyn, initial, measured, exo, zone):
    n, members, state_dim = initial.shape
    idx = np.tile(np.arange(members),n)
    starts, ends = [], []
    for channels in MODES.values():
        state = initial.copy()
        if channels:
            state[:,:,channels] = measured[:,None,channels]
        starts.append(state.copy())
        state = state.reshape(-1,state_dim)
        for k in range(len(exo[0])):
            inputs = [np.repeat(v,members,axis=0) for v in (exo[:,k],zone[:,k+1],zone[:,k])]
            state = dyn.step(state,*inputs,idx)
        ends.append(state.reshape(n,members,state_dim))
    return np.stack(starts,axis=1),np.stack(ends,axis=1)


def smoke():
    class Toy:
        def step(self,s,e,a,p,m):
            return s + e + a - p + m[:,None]
    initial = np.arange(20.).reshape(2,2,5)
    measured = np.array([[90.,91.,92.,93.,94.],[60.,61.,62.,63.,64.]])
    exo = np.ones((2,8,1))
    zone = np.tile(np.arange(9.)[None,:,None],(2,1,1))
    starts,ends = advance(Toy(),initial,measured,exo,zone)
    for j, channels in enumerate(MODES.values()):
        for column in range(5):
            wanted = np.repeat(measured[:,column,None],2,1) if column in channels else initial[:,:,column]
            np.testing.assert_array_equal(starts[:,j,:,column],wanted)
            np.testing.assert_array_equal(ends[:,j,:,column],wanted+[16,24])


def run(out):
    data,model = [prior.E.DATA/s for s in ('data.pkl','surrogate.pkl')]
    ref = json.loads((HERE.parent/'e77/summary.json').read_text())['sha256']
    for p in (data,model):
        assert hashlib.sha256(p.read_bytes()).hexdigest()==ref[str(p)]
    splits,days = prior.E.S.base.load_selected(data)
    dyn = prior.E.Dynamics(model)
    with np.load(data.parents[2]/'e104/main/replay.npz') as z:
        dates,t,day,dq = [z[k] for k in ('dates','t','date_index','dq')]
        val = splits['val']
        events = [dict(day=val[d], t=step) for d,step in zip(day,t)]
        np.testing.assert_array_equal(dates, np.array([str(w['day']) for w in events]))
        truth,exo,zone = prior.pack(events,days)
        np.testing.assert_array_equal(truth,z['truth'])
        np.testing.assert_allclose(zone[:,1].sum(1)-zone[:,0].sum(1),dq,rtol=0,atol=1e-8)
        initial = z['rec'][:,1]
        early,_ = prior.forecasts(dyn,truth[:,:5],exo[:,:4],zone[:,:5])
        np.testing.assert_array_equal(initial,early[:,-1])
        starts,ends = advance(dyn,initial,truth[:,4],exo[:,4:],zone[:,4:])
        np.testing.assert_array_equal(ends[:,0],z['rec'][:,-1])
    # A full restart from measurements with the recorded lagged flow should match independently.
    shifted = [dict(day=w['day'],t=w['t']+4) for w in events]
    check = np.repeat(truth[:,4,None,:],dyn.n_members,axis=1)
    for m in range(dyn.n_members):
        state = check[:,m].copy()
        for k in range(8):
            args = [np.stack([days[w['day']][key][w['t']+k+off] for w in shifted])
                    for key,off in [('exo',0),('zone',0),('zone',-1)]]
            state = dyn.step(state,*args,np.full(len(state),m))
        np.testing.assert_array_equal(ends[:,-1,m],state)
    assert np.isfinite(ends).all()
    out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'reset.npz',dates=dates,t=t,day=day,dq=dq,truth=truth[:,[0,4,12]],
                        starts=starts,ends=ends,modes=list(MODES))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    smoke()
    if args.out:
        run(args.out)

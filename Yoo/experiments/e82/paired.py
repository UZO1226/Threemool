"""Produce paired episode features and training outcomes for the e81 recovery decision."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'e81'))
import recover as R

E = R.E
SOURCE = Path('/home/jio/fcsrl_src/results/e78/main')
TOL = 1e-8


def observations(z, dates, days):
    rows = [days[d] for d in dates for _ in range(5)]
    exo = np.stack([r['exo'][:95,:5] for r in rows]) / [1000,10,1,1,50]
    initial = np.stack([r['state'][0] for r in rows])
    old_air = np.array([r['air'][0] for r in rows])[:,None]
    state = np.concatenate([initial[:,None,:], z['states'][:,:-1]], axis=1)
    air = np.concatenate([old_air,z['air'][:,:-1]], axis=1)
    return np.concatenate([state,exo,air[:,:,None]/9000], axis=2).astype(np.float32)


def metrics(z):
    states = z['states']
    return np.column_stack([z['air'].sum(1)*.25, (states[:,:,0]>4).sum(1),
                            (states[:,:,2:].max(2)>2).sum(1), np.maximum(states[:,:,0]-4,0).sum(1)*.25])


def success(delta):
    return (delta[:,0] < -TOL) & (delta[:,1:] <= TOL).all(1)


def events(pulse, recovered, obs):
    cut = recovered['delta'] < 0
    ep = np.flatnonzero(cut.any(1))
    t = cut[ep].argmax(1)
    assert (t >= 1).all()
    shared = cut.cumsum(1) == 0
    for key in ('air', 'states'):
        np.testing.assert_array_equal(pulse[key][shared], recovered[key][shared])
    nh = obs[ep,t,0]
    features = np.column_stack([nh, nh-obs[ep,t-1,0], obs[ep,t,2:5], obs[ep,t,10]*9000])
    assert (features[:,0]<=4).all() and (features[:,1]<=0).all()
    outcomes = metrics(recovered) - metrics(pulse)
    np.testing.assert_allclose(outcomes[~cut.any(1)],0,atol=TOL,rtol=0)
    return ep,t,features,success(outcomes[ep])


def setup():
    torch.set_num_threads(1)
    hashes = json.loads((HERE.parent/'e77/summary.json').read_text())['sha256']
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    splits, days = E.S.base.load_selected(E.DATA/'data.pkl')
    assert not set(splits['train']) & set(splits['val'])
    return splits,days


def training(out, smoke):
    splits,days = setup()
    dates,rows = E.load_train()
    assert dates == splits['train']
    env = E.BatchEnv(days,E.Dynamics(E.DATA/'surrogate.pkl'))
    support = R.prior.fit_support(rows[:140])
    hashes = json.loads((HERE.parent/'e79/summary.json').read_text())['e78_input_sha256']
    out.mkdir(parents=True,exist_ok=False)
    for seed in ([0] if smoke else range(5)):
        agent = E.restore(seed)
        path = SOURCE/f'tree{seed}.joblib'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[path.name]
        tree = joblib.load(path)
        batch = dates[:2] if smoke else dates
        pulse,recovered = [R.rollout(m,agent,tree,support,env,batch) for m in ('pulse','recover')]
        obs = observations(pulse,batch,days)
        for t in [0,1,94]:
            q = R.prior.action_to_air(agent.act_det(obs[:,t])[:,0])
            np.testing.assert_allclose(q,pulse['raw'][:,t],atol=.02,rtol=0)
        ep,t,x,y = events(pulse,recovered,obs)
        np.savez_compressed(out/f'train{seed}.npz', dates=np.asarray(batch,str), ep=ep,t=t,x=x,y=y)
        print('training pairs',seed,len(y),int(y.sum()),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    training(args.out,args.smoke)

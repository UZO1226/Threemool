"""Cross all five frozen transition models with each e102 starting state and action."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'e96'))
import effects as E

S = E.S
TOL = E.TOL


class ShiftModel:
    def __init__(self, dynamics, shift):
        self.dynamics = dynamics
        self.shift = shift
        self.initial = None

    def step(self, state, exo, zone, prev_zone, member):
        if self.initial is None:
            self.initial = tuple(x.copy() for x in (state, exo, zone, prev_zone, member))
        return self.dynamics.step(state, exo, zone, prev_zone, (member+self.shift) % 5)


def weighted(days, values):
    return np.average(values, weights=S.weights(days), axis=0)


def valid(delta):
    return (delta[...,0] < -TOL) & (delta[...,1:] <= TOL).all(axis=-1)


def select(delta):
    good = valid(delta)
    air = np.where(good, delta[...,0], np.inf)
    air[...,0] = 0.
    return air.argmin(-1)


def metrics(days, delta, source):
    n = len(days)
    chosen = select(delta)
    joint = valid(delta)
    index = np.arange(n)[:,None]
    all_zero = (chosen == 0).all(1)
    unanimous_cut = (chosen == chosen[:,:1]).all(1) & ~all_zero
    common = joint[:,:,1:].all(1).any(1)
    info = dict(n=n, dates=len(np.unique(days)), all_zero=int(all_zero.sum()),
                unanimous_cut=int(unanimous_cut.sum()), common=int(common.sum()))
    a, b = np.where(~np.eye(5, dtype=bool))
    diffs = np.stack([delta[:,:,j,0]-delta[:,:,i,0] for i,j in [(0,1),(0,2),(1,2)]], axis=-1)
    signs = np.where(abs(diffs)<=TOL, 0, np.sign(diffs))
    info['sign_match'] = weighted(days,(signs[:,a]==signs[:,b]).mean(1)).tolist()
    for name, only_source in [('cross',False),('origin',True)]:
        weight = (a[None,:]==source[:,None]) if only_source else np.ones((n,len(a)),bool)
        picks = chosen[:,a]
        evaluated = delta[index,b,picks]
        def avg(x):
            return weighted(days, (x*weight).sum(1)/weight.sum(1))
        acted = picks > 0
        den = avg(acted)
        good_air = evaluated[:,:,0]<-TOL
        good_quality = (evaluated[:,:,1:]<=TOL).all(2)
        same = picks==chosen[:,b]
        outcomes = {}
        for k,v in [('joint',good_air&good_quality),('air',good_air),('quality',good_quality),('best',same)]:
            outcomes[k] = float(avg(v&acted)/den) if den else None
        effect = [float(avg(evaluated[:,:,j])) for j in range(4)]
        info[name] = dict(acted=float(den), cases=int((acted & weight).sum()), same=float(avg(same)),
                          delta=effect, **outcomes)
    return info


def run(out):
    S.torch.set_num_threads(1)
    assert all(S.digest(Path(k)) == v for k,v in E.ref('e77')['sha256'].items())
    dates, rows = S.E.load_train()
    dyn = S.E.Dynamics(S.E.DATA/'surrogate.pkl')
    out.mkdir(parents=True,exist_ok=False)
    records = []
    for seed in range(5):
        path = S.ROOT/'e88/main'/f'{seed}_early_down.npz'
        assert S.digest(path) == E.ref('e89')[seed]['input_sha256'][0]
        archive, _ = S.read_candidates(path,dates[:140])
        agent = S.E.restore(seed)
        with np.load(S.ROOT/'e102/main'/f'size{seed}.npz') as z:
            keys, q, ref = z['keys'], z['q'], z['totals']
            x, raw = S.before_action(rows[:140], archive, keys)
            np.testing.assert_array_equal(q,raw)
            np.testing.assert_array_equal(x[:,0],z['nh4'])
        totals = np.empty((len(keys),5,3,4))
        for start in range(0,len(keys),250):
            ids = np.arange(start,min(start+250,len(keys)))
            for arm,cut in enumerate((0.,250.,500.)):
                original = None
                for shift in range(5):
                    model = ShiftModel(dyn,shift)
                    result = S.P.branch(rows[:140],model,agent,keys[ids],q[ids],-cut,archive)[:,-1]
                    target = (keys[ids,2]+shift)%5
                    totals[ids,target,arm] = result
                    if shift == 0:
                        original = model.initial
                        np.testing.assert_allclose(result,ref[ids,arm],atol=1e-7,rtol=0)
                    else:
                        for current, baseline in zip(model.initial,original):
                            np.testing.assert_array_equal(current,baseline)
        assert np.isfinite(totals).all()
        np.savez_compressed(out/f'transfer{seed}.npz',keys=keys,totals=totals)
        delta = totals-totals[:,:,:1]
        for block,start in enumerate((70,105), 1):
            on = (keys[:,0]>=start)&(keys[:,0]<start+35)
            records.append(dict(seed=seed,block=block,**metrics(keys[on,0],delta[on],keys[on,2])))
        print(f'seed {seed} complete',flush=True)
    (out/'summary.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')


def smoke():
    delta = np.zeros((1,5,3,4))
    def check():
        return metrics(np.array([70]),delta,np.array([0]))
    assert check()['cross']['joint'] is None and check()['all_zero']==1
    delta[:,:,1:,0] = -1.
    assert check()['unanimous_cut']==1 and (select(delta)==1).all()
    delta[:,0,2,0] = -2.
    g = check()
    assert g['cross']['best'] < 1 and g['cross']['joint']==1 and g['common']==1
    delta[:,0,1:,1] = 1.
    assert check()['common']==0 and check()['origin']['acted']==0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path)
    args = parser.parse_args()
    smoke()
    if args.out:
        run(args.out)

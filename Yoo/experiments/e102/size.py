"""Pair fixed and ammonia-dependent first-cut amplitudes on historical windows."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'e96'))
import effects as E

S = E.S
MODES = ('base', 'half', 'full', 'adaptive', 'random', 'oracle')


def level(nh4):
    return np.where(nh4 <= 2, 2, np.where(nh4 < 4, 1, 0))


def measure(days, delta, choices):
    n = len(days)
    rows = np.arange(n)
    np.testing.assert_array_equal(delta[:,0], 0)
    safe = (delta[:,:,1:] <= E.TOL).all(2)
    best = np.where(safe, delta[:,:,0], np.inf).argmin(1)
    modes = dict(base=delta[:,0], half=delta[:,1], full=delta[:,2], adaptive=delta[rows,choices],
                 random=np.zeros((n,4)), oracle=delta[rows,best])
    for d in np.unique(days):
        on = days == d
        probs = np.bincount(choices[on], minlength=3) / on.sum()
        modes['random'][on] = (delta[on]*probs[None,:,None]).sum(1)
        np.testing.assert_allclose(np.dot(probs,[0,250,500]), np.mean(choices[on]*250))
        np.testing.assert_allclose(modes['random'][on].mean(0), probs @ delta[on].mean(0), atol=1e-8)
    results = {}
    for name, values in modes.items():
        avg = np.average(values, weights=S.weights(days), axis=0)
        np.testing.assert_allclose(avg, np.mean([values[days==d].mean(0) for d in np.unique(days)],0), atol=1e-8)
        results[name] = avg.tolist()
    return dict(n=n, dates=len(np.unique(days)), levels=np.bincount(choices,minlength=3).tolist(),
                outcomes=results)


def run(out):
    S.torch.set_num_threads(1)
    assert all(S.digest(Path(k)) == v for k,v in E.ref('e77')['sha256'].items())
    dates, rows = S.E.load_train()
    dyn = S.E.Dynamics(S.E.DATA/'surrogate.pkl')
    out.mkdir(parents=True, exist_ok=False)
    summary = []
    for seed in range(5):
        root = S.ROOT
        archive_file = root/'e88/main'/f'{seed}_early_down.npz'
        label_file = root/'e88/main'/f'probe{seed}.npz'
        hashes = E.ref('e89')[seed]['input_sha256']
        assert [S.digest(archive_file), S.digest(label_file)] == [hashes[0], hashes[2]]
        archive, k = S.read_candidates(archive_file, dates[:140])
        x, q = S.before_action(rows[:140], archive, k)
        with np.load(label_file) as z:
            np.testing.assert_array_equal(k, z['train_keys'])
            endpoints = z['train_totals']
        agent = S.E.restore(seed)
        batch = slice(0,250)
        for arm, offset in enumerate((0.,-500.)):
            replay = S.P.branch(rows[:140], dyn, agent, k[batch], q[batch], offset, archive)
            np.testing.assert_allclose(replay[:,-1], endpoints[batch,arm], rtol=0, atol=1e-7)
        parts = np.concatenate([np.flatnonzero((k[:,0]>=start)&(k[:,0]<start+35)) for start in (70,105)])
        with np.load(root/'e101/main'/f'rank{seed}.npz') as z:
            np.testing.assert_array_equal(k[parts], z['keys'])
            np.testing.assert_allclose(endpoints[parts,1]-endpoints[parts,0],z['truth'],atol=1e-7,rtol=0)
            keep = parts[(z['pred'][:,1:] <= E.TOL).all(1)]
        half = []
        for a in range(0,len(keep),250):
            ix = keep[a:a+250]
            half.append(S.P.branch(rows[:140], dyn, agent, k[ix], q[ix], -250., archive)[:,-1])
        totals = np.stack((endpoints[keep,0], np.concatenate(half), endpoints[keep,1]), axis=1)
        delta = totals-totals[:,:1]
        choices = level(x[keep,0])
        np.savez_compressed(out/f'size{seed}.npz', keys=k[keep], nh4=x[keep,0], q=q[keep], totals=totals)
        for block,start in enumerate((70,105)):
            on = (k[keep,0]>=start)&(k[keep,0]<start+35)
            g = measure(k[keep,0][on], delta[on], choices[on])
            assert g['n'] == E.ref('e101')[seed*2+block]['groups']['quality']['n']
            summary.append(dict(seed=seed, block=block+1, **g))
        print(f'seed {seed} complete', flush=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out',type=Path)
    args = p.parse_args()
    np.testing.assert_array_equal(level(np.array([1.,2.,3.,4.,5.])), [2,2,1,0,0])
    np.testing.assert_array_equal(np.array([0,250,500])*.25, [0,62.5,125])
    if args.out:
        run(args.out)

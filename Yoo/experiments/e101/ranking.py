"""Evaluate out-of-fold six-hour air savings ranks inside two historical blocks."""
import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e96'))
import effects as E

S = E.S


def top_weights(score):
    k = (len(score)+1)//2
    boundary = np.sort(score)[-k]
    above, tied = score > boundary, score == boundary
    return above.astype(float) + tied*((k-above.sum())/tied.sum())


def compare(day, pred, truth, start):
    score = -pred[:,0]
    value = truth.copy()
    value[:,0] *= -1
    rho = None
    if len(np.unique(score)) > 1 and len(np.unique(value[:,0])) > 1:
        rho = float(spearmanr(score, value[:,0]).statistic)
    tops, controls = [], []
    empty = singles = ties = slots = 0
    for d in range(start, start+35):
        mask = day == d
        n = int(mask.sum())
        if n < 2:
            empty += n == 0
            singles += n == 1
            continue
        scores, values = score[mask], value[mask]
        take = top_weights(scores)
        k = (n+1)//2
        np.testing.assert_allclose(take.sum(), k)
        selected = np.average(values, axis=0, weights=take)
        control = values.mean(0)
        if n <= 8:
            samples = list(itertools.combinations(range(n), k))
            ranks = [tuple(sorted(scores[list(ix)], reverse=True)) for ix in samples]
            best = max(ranks)
            means = np.array([values[list(ix)].mean(0) for ix in samples])
            np.testing.assert_allclose(selected, means[[r == best for r in ranks]].mean(0), atol=1e-8)
            np.testing.assert_allclose(control, means.mean(0), atol=1e-8)
        tops.append(selected)
        controls.append(control)
        ties += int(np.all(scores == scores[0]))
        slots += k
    result = dict(n=len(day), empty=empty, singleton=singles, dates=len(tops),
                  ties=ties, slots=slots, rho=rho, top=None, random=None, lift=None)
    if tops:
        result.update(top=np.mean(tops, axis=0).tolist(), random=np.mean(controls, axis=0).tolist(),
                      lift=float(np.mean(np.array(tops)[:,0]-np.array(controls)[:,0])))
    return result


def run(out):
    dates, rows = S.E.load_train()
    assert all(a < b for a,b in zip(dates[:139],dates[1:140]))
    data = S.E.DATA/'data.pkl'
    assert S.digest(data) == E.ref('e77')['sha256'][str(data)]
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for seed in range(5):
        archive_path = S.ROOT/'e88/main'/f'{seed}_early_down.npz'
        label_path = S.ROOT/'e88/main'/f'probe{seed}.npz'
        hashes = E.ref('e89')[seed]['input_sha256']
        assert S.digest(archive_path) == hashes[0] and S.digest(label_path) == hashes[2]
        arc, keys = S.read_candidates(archive_path, dates[:140])
        x = S.before_action(rows[:140], arc, keys)[0].astype(float)
        with np.load(label_path) as z:
            np.testing.assert_array_equal(keys, z['train_keys'])
            target = z['train_totals'][:,1] - z['train_totals'][:,0]
        ordered_keys, forecasts, outcomes = [], [], []
        for start in (70,105):
            fit = keys[:,0] < start
            valid = (keys[:,0] >= start) & (keys[:,0] < start+35)
            assert keys[fit,0].max() < keys[valid,0].min()
            pred, _ = E.predict(x[fit], target[fit], keys[fit,0], x[valid])
            assert np.isfinite(pred).all() and np.isfinite(target[valid]).all()
            ok = (pred[:,1:] <= E.TOL).all(1)
            groups = {}
            for name, mask in [('quality',ok),('all', np.ones(len(pred),bool))]:
                groups[name] = compare(keys[valid,0][mask], pred[mask], target[valid][mask], start)
            records.append(dict(seed=seed, start=start, groups=groups))
            ordered_keys.append(keys[valid])
            forecasts.append(pred)
            outcomes.append(target[valid])
        k, p, y = map(np.concatenate, (ordered_keys, forecasts, outcomes))
        with np.load(S.ROOT/'e100/main'/f'cut{seed}.npz') as z:
            np.testing.assert_array_equal(k[:,0], z['cal_day'])
            np.testing.assert_allclose(y[:,0]-p[:,0], z['cal_residual'], atol=1e-7, rtol=0)
        np.savez_compressed(out/f'rank{seed}.npz', keys=k, pred=p, truth=y)
    (out/'summary.json').write_text(json.dumps(records, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    np.testing.assert_array_equal(top_weights(np.array([3.,2.,2.,1.])), [1.,.5,.5,0.])
    x = np.zeros((4,4))
    x[:,0] = [-4,-3,-2,-1]
    for sign in (1,-1):
        stats = compare(np.array([70]*4), x, sign*x, 70)
        np.testing.assert_allclose([stats['rho'], stats['lift']], [sign,sign])
    assert compare(np.array([70]), x[:1], x[:1], 70)['lift'] is None
    if args.out:
        run(args.out)

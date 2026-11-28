"""Apply chronological air-error margins to the frozen e96 first-cut rule."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e96'))
import effects as E

S = E.S


def margin(residual, days):
    order = np.argsort(residual)
    weights = S.weights(days)[order]
    index = np.searchsorted(np.cumsum(weights), .8 * weights.sum())
    return max(0., float(residual[order[index]]))


def joint(delta):
    return (delta[..., 0] < -E.TOL) & (delta[..., 1:] <= E.TOL).all(axis=-1)


def allowed(pred, old, allowance):
    return old & (pred[:, 0] + allowance < -E.TOL)


def measure(day, first, pred, truth, allowance):
    old = first & joint(pred)
    cut = allowed(pred, old, allowance)
    options = dict(base=np.zeros(len(day)), original=old, calibrated=cut,
                   random=E.matched(day, first, cut), retained=E.matched(day, old, cut))
    values = {}
    for name, take in options.items():
        delta = np.average(truth * take[:, None], axis=0, weights=S.weights(day))
        values[name] = dict(n=int(round(take.sum())), hits=float(np.dot(take.astype(float), joint(truth))),
                            delta=delta.tolist())
        by_day = []
        for d in np.unique(day):
            m = day == d
            if name in ('random', 'retained'):
                c = (first if name == 'random' else old) & m
                expectation = truth[c].mean(axis=0)*cut[m].sum()/m.sum() if c.any() else np.zeros(4)
            else:
                expectation = (truth[m]*take[m,None]).mean(0)
            by_day.append(expectation)
        np.testing.assert_allclose(np.mean(by_day, axis=0), delta, atol=1e-8, rtol=0)
    coverage = np.average(truth[first,0]-pred[first,0] <= allowance, weights=S.weights(day[first]))
    return dict(n=len(day), candidates=int(first.sum()), coverage=float(coverage), outcomes=values)


def run(out):
    dates, rows = S.E.load_train()
    assert all(a < b for a, b in zip(dates, dates[1:]))
    data = S.E.DATA/'data.pkl'
    assert S.digest(data) == E.ref('e77')['sha256'][str(data)]
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for seed in range(5):
        paths = [S.ROOT/'e88/main'/f'{seed}_early_down.npz', S.ROOT/'e78/main'/f'{seed}_train48_down.npz',
                 S.ROOT/'e88/main'/f'probe{seed}.npz']
        assert [S.digest(p) for p in paths] == E.ref('e89')[seed]['input_sha256']
        arc, k = S.read_candidates(paths[0], dates[:140])
        x = S.before_action(rows[:140], arc, k)[0].astype(float)
        arc, keys = S.read_candidates(paths[1], dates[140:])
        xt = S.before_action(rows[140:], arc, keys)[0].astype(float)
        with np.load(paths[2]) as z:
            np.testing.assert_array_equal(k, z['train_keys'])
            np.testing.assert_array_equal(keys, z['keys'])
            target = z['train_totals'][:,1] - z['train_totals'][:,0]
            truth = z['change']
        day = k[:,0]
        residuals, days, blocks = [], [], []
        for start, stop in [(70,105), (105,140)]:
            fit = day < start
            cal = (day >= start) & (day < stop)
            assert day[fit].max() < day[cal].min()
            pred, _ = E.predict(x[fit], target[fit], day[fit], x[cal])
            residuals.extend(target[cal,0]-pred[:,0])
            days.extend(day[cal])
            blocks.append(dict(fit=int(fit.sum()), cal=int(cal.sum()), dates=len(np.unique(day[cal]))))
        days, residuals = np.array(days), np.array(residuals)
        allowance = margin(residuals, days)
        pred, _ = E.predict(x, target, day, xt)
        with np.load(S.ROOT/'e96/main'/f'prediction{seed}.npz') as z:
            np.testing.assert_array_equal(keys, z['keys'])
            np.testing.assert_allclose(pred, z['pred'], atol=1e-7, rtol=0)
            np.testing.assert_array_equal(truth, z['truth'])
            first, supported = z['first'], z['supported']
        groups = {}
        for name, mask in [('supported',supported), ('all',np.ones(len(keys), bool))]:
            g = measure(keys[mask,0], first[mask], pred[mask], truth[mask], allowance)
            past = E.ref('e96')[seed]['groups'][name]
            assert g['candidates'] == past['candidates'] and g['n'] == past['n']
            old = g['outcomes']['original']
            assert old['n'] == past['outcomes']['forecast']['n']
            np.testing.assert_allclose(old['delta'], past['outcomes']['forecast']['delta'], atol=1e-8, rtol=0)
            groups[name] = g
        old = first & joint(pred)
        np.savez_compressed(out/f'cut{seed}.npz', keys=keys, pred=pred, truth=truth, first=first,
                            supported=supported, original=old, calibrated=allowed(pred, old, allowance),
                            cal_day=days, cal_residual=residuals, margin=allowance)
        records.append(dict(seed=seed, blocks=blocks, margin=allowance, groups=groups))
    (out/'summary.json').write_text(json.dumps(records, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    assert margin(np.array([1.,1.,1.,1.,9.]), np.array([0,0,0,0,1])) == 9
    assert margin(np.array([-4.]), np.array([0])) == 0
    old = np.array([True,True,False])
    p = np.array([[-5.,0,0,0],[-6.,0,0,0],[-8.,0,0,0]])
    np.testing.assert_array_equal(allowed(p,old,5), [False,True,False])
    E.smoke()
    if args.out:
        run(args.out)

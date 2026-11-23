"""Measure hindsight headroom for one supported aeration cut followed by native FCSRL."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/jio/fcsrl_src/results')
TOL = 1e-8


def nonoverlap(keys):
    day, step, member = keys.T
    take = np.zeros(len(keys), dtype=bool)
    next_start = {}
    for i in np.lexsort((step, member, day)):
        group = (day[i], member[i])
        if step[i] >= next_start.get(group, 0):
            take[i] = True
            next_start[group] = step[i] + 24
    return take


def measures(totals):
    change = totals[:, 1] - totals[:, 0]
    saved = -change[:, 0]
    clean = (change[:, 1:] <= TOL).all(axis=1)
    good = clean & (saved > TOL)
    useful = clean & (saved >= 25.)
    optimal = np.where(good, saved, 0.)
    # Independently minimize air across admissible branches for each case.
    admissible = (totals[:, :, 1:] <= totals[:, :1, 1:] + TOL).all(axis=2)
    best = np.where(admissible, totals[:, :, 0], np.inf).min(axis=1)
    np.testing.assert_allclose(totals[:, 0, 0] - best, optimal, atol=TOL, rtol=0)
    return change, good, useful, optimal


def summarize(keys, totals):
    change, good, useful, optimal = measures(totals)
    day = keys[:, 0]
    days, idx, counts = np.unique(day, return_inverse=True, return_counts=True)
    weight = 1 / counts[idx]
    upper = float(np.average(optimal, weights=weight))
    base = float(np.average(totals[:, 0, 0], weights=weight))
    useful_days = int(np.unique(day[useful]).size)
    return dict(n=len(day), eligible_days=len(days), good=int(good.sum()), useful=int(useful.sum()),
                good_days=int(np.unique(day[good]).size), useful_days=useful_days,
                passes=bool(useful_days >= 10 and useful.mean() >= .05),
                oracle_m3=upper, base_m3=base, oracle_pct=100 * upper/base,
                median_good_m3=float(np.median(optimal[good])) if good.any() else None,
                always_cut_delta=np.average(change, axis=0, weights=weight).tolist(),
                good_date_weighted_pct=100 * float(np.average(good, weights=weight)))


def agreement(keys, useful):
    groups, inv, n = np.unique(keys[:, :2], axis=0, return_inverse=True, return_counts=True)
    complete = n == 5
    agree = complete & (np.bincount(inv, weights=useful) >= 4)
    return dict(complete=int(complete.sum()), agree=int(agree.sum()),
                eligible_days=int(np.unique(groups[complete, 0]).size),
                agree_days=int(np.unique(groups[agree, 0]).size))


def smoke():
    keys = np.array([[0, 28, 0], [1, 8, 0], [0, 4, 0], [0, 8, 0], [0, 8, 1]])
    np.testing.assert_array_equal(nonoverlap(keys), [1, 1, 1, 0, 1])
    totals = np.zeros((5, 2, 4))
    totals[:, 0, 0] = 200
    totals[:, 1, 0] = [175, 195, 200, 250, 140]
    totals[-1, 1, 3] = .01
    _, good, useful, optimal = measures(totals)
    np.testing.assert_array_equal(good, [1, 1, 0, 0, 0])
    np.testing.assert_array_equal(useful, [1, 0, 0, 0, 0])
    assert np.isclose(summarize(keys, totals)['oracle_m3'], (25/4+5)/2)
    np.testing.assert_array_equal(optimal, [25, 5, 0, 0, 0])


def run(root, out):
    previous = json.loads((HERE.parent / 'e86/summary.json').read_text())
    results = []
    out.mkdir(parents=True, exist_ok=False)
    all_dates = None
    for ref in previous:
        seed = ref['seed']
        paths = [root/'e85/main'/f'probe{seed}.npz', root/'e86/main'/f'seed{seed}.npz']
        hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        assert hashes[0] == ref['score_sha256']
        with np.load(paths[0]) as probe, np.load(paths[1]) as replay:
            keys, totals, dates = probe['keys'], probe['totals'][:, :, -1], probe['dates']
            np.testing.assert_array_equal(keys, replay['keys'])
            np.testing.assert_array_equal(replay['costs'].sum(axis=2), totals[:, :, 1:3])
        if all_dates is None:
            all_dates = dates
        np.testing.assert_array_equal(dates, all_dates)
        assert len(set(dates)) == 48 and np.all((keys[:, 0]>=0) & (keys[:, 0]<48))
        assert np.all(np.isin(keys[:, 2], np.arange(5))) and np.all(np.isin(keys[:, 1], np.arange(4,69,4)))
        assert np.unique(keys, axis=0).shape == keys.shape and np.isfinite(totals).all()
        assert len(keys) == ref['metrics']['e85_harm']['n']
        keep = nonoverlap(keys)
        change, good, useful, optimal = measures(totals)
        np.savez_compressed(out/f'cases{seed}.npz', dates=dates, keys=keys, change=change,
                            good=good, useful=useful, optimal=optimal, nonoverlap=keep)
        results.append(dict(seed=seed, sha256=hashes, all=summarize(keys, totals),
                            nonoverlap=summarize(keys[keep], totals[keep]),
                            agreement=agreement(keys, useful)))
    (out/'summary.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--out', type=Path)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.smoke:
        smoke()
    elif args.out:
        run(args.root, args.out)
    else:
        p.error('supply --out or --smoke')

"""Forecast first-cut harm from observations available before the cut."""
import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e91'))
import hour_one as hour

S = hour.S
TOL = 1e-8


def harm(paths):
    quality_delta = paths[:, 1:, 1:] - paths[:, :1, 1:]
    return (quality_delta > TOL).any(axis=2).all(axis=1)


def matched_fraction(days, cut):
    return np.bincount(days, weights=cut)[days] / np.bincount(days)[days]


def metrics(days, paths, scores, skip):
    risk = harm(paths)
    w = S.weights(days)
    cut = ~skip
    coin = matched_fraction(days, cut)
    delta = paths[:, 1] - paths[:, 0]

    def rate(values, group):
        return float(np.average(values[group], weights=w[group])) if group.any() else None

    effects = {}
    for name, take in [('always_cut', np.ones(len(cut))), ('gate', cut),
                       ('random', coin), ('perfect_skip', ~risk)]:
        weighted = np.average(take[:, None] * delta, weights=w, axis=0)
        effects[name] = weighted.tolist()
        independent = []
        for day in np.unique(days):
            mask = days == day
            if name == 'random':
                independent.append(delta[mask].mean(axis=0) * cut[mask].mean())
            else:
                independent.append((delta[mask] * take[mask, None]).mean(axis=0))
        np.testing.assert_allclose(weighted, np.mean(independent, axis=0), rtol=0, atol=1e-8)
    np.testing.assert_allclose(np.bincount(days, weights=coin), np.bincount(days, weights=cut))
    return dict(n=len(days), days=len(set(days)), positive=int(risk.sum()),
                missed=int((risk & cut).sum()), cut=int(cut.sum()),
                auc=float(roc_auc_score(risk, scores, sample_weight=w)) if risk.any() and not risk.all() else None,
                recall=rate(skip, risk), random_recall=rate(1-coin, risk),
                fpr=rate(skip, ~risk), precision=rate(risk, skip), delta=effects)


def smoke():
    paths = np.zeros((4, 3, 4))
    paths[0, 1, 1] = paths[0, 2, 2] = paths[1, 1, 3] = 1
    paths[2, 1:, 0] = 10
    paths[3, 1:, 3] = TOL
    np.testing.assert_array_equal(harm(paths), [1, 0, 0, 0])
    days = np.array([0, 0, 0, 1])
    stats = metrics(days, paths, np.arange(4), np.array([False, True, False, True]))
    possible = []
    for chosen in itertools.combinations(range(3), 2):
        possible.append(paths[list(chosen), 1].sum(axis=0) / 6)
    np.testing.assert_allclose(stats['delta']['random'], np.mean(possible, axis=0))


def run(out):
    def ref(experiment):
        return json.loads((HERE.parent / experiment / 'summary.json').read_text())

    r89, r92 = ref('e89'), ref('e92')
    raw = S.E.DATA / 'data.pkl'
    assert S.digest(raw) == ref('e77')['sha256'][str(raw)]
    dates, rows = S.E.load_train()
    assert dates[99] < dates[100] < dates[139] < dates[140]
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for seed in range(5):
        archives = [S.ROOT / 'e88/main' / f'{seed}_early_down.npz',
                    S.ROOT / 'e78/main' / f'{seed}_train48_down.npz']
        assert [S.digest(p) for p in archives] == r89[seed]['input_sha256'][:2]
        path = S.ROOT / 'e91/main' / f'hour{seed}.npz'
        assert S.digest(path) == r92[seed]['sha256']
        with np.load(path) as z:
            inputs = []
            for archive_path, date_part, row_part, prefix, landmark in zip(
                    archives, (dates[:140], dates[140:]), (rows[:140], rows[140:]),
                    ('train_', ''), ('train_x', 'features')):
                archive, keys = S.read_candidates(archive_path, date_part)
                np.testing.assert_array_equal(keys, z[prefix + 'keys'])
                assert keys[:, 1].min() >= 1
                x, _ = S.before_action(row_part, archive, keys)
                # The later level minus its change reconstructs the initial NH4.
                np.testing.assert_allclose(x[:, 0], z[landmark][:, 0] - z[landmark][:, 1],
                                           rtol=1e-6, atol=1e-6)
                assert x.shape == (len(keys), 11) and np.isfinite(x).all()
                inputs.append(x)
            train_day = z['train_keys'][:, 0]
            scores, skip, settings = hour.fit(inputs[0], harm(z['train_totals']), train_day, inputs[1])
            groups = {}
            for name, mask in [('supported', z['supported']), ('all', np.ones(len(skip), bool))]:
                group = metrics(z['keys'][mask, 0], z['totals'][mask], scores[mask], skip[mask])
                expected = r92[seed]['groups'][name]
                assert group['n'] == expected['n']
                assert group['positive'] == expected['fixed']['unavoidable_quality_loss_windows']
                groups[name] = group
            np.savez_compressed(out / f'gate{seed}.npz', keys=z['keys'], supported=z['supported'],
                                score=scores, skip=skip, risk=harm(z['totals']))
        records.append(dict(seed=seed, settings=settings, groups=groups))
    (out / 'summary.json').write_text(json.dumps(records, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    smoke()
    if args.out is not None:
        run(args.out)

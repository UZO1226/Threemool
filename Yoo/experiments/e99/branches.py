"""Decompose the frozen e98 late-air predictions into baseline and cut volumes."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e98'))
import landmark as lm

S = lm.S


def training_data(rows, archive, keys, q, dyn, agent, expected):
    panels, pairs = [], []
    for start in range(0, len(keys), 250):
        sl = slice(start, start + 250)
        trace = lm.ObservedStates(dyn)
        b, _ = lm.rebound.simulate(rows, archive, keys[sl], q[sl], agent, trace, True)
        a = S.P.branch(rows, dyn, agent, keys[sl], q[sl], 0., archive)
        np.testing.assert_allclose(a[:, -1], expected[sl, 0], atol=1e-7, rtol=0)
        np.testing.assert_allclose(b.sum(1), expected[sl, 1], atol=1e-7, rtol=0)
        states = np.stack(trace.prefix, axis=1)
        volume = b[:, :, 0]
        panels.append(np.column_stack([lm.measured(states, volume, 0, 4),
                                       lm.measured(states, volume, 4, 12)]))
        pairs.append(np.column_stack([a[:, 2, 0] - a[:, 1, 0], volume[:, 12:].sum(1)]))
    return np.concatenate(panels), np.concatenate(pairs)


def metrics(day, first, selected, residual):
    def average(x):
        return np.average(x, weights=S.weights(day[first]), axis=0)
    a, b = residual[first].transpose(2, 0, 1)
    ma, mb = average(a*a), average(b*b)
    cross = average(a*b)
    delta = average((b-a)**2)
    cov = cross - average(a)*average(b)
    np.testing.assert_allclose(delta, ma+mb-2*cross, atol=1e-7, rtol=1e-12)
    contribution = np.average(residual * selected[:, None, None], weights=S.weights(day), axis=0)
    return dict(n=len(day), candidates=int(first.sum()), selected=int(selected.sum()),
                mse_a=ma.tolist(), mse_b=mb.tolist(), mse_delta=delta.tolist(),
                cross=cross.tolist(), covariance=cov.tolist(),
                from_a=(-contribution[:, 0]).tolist(), from_b=contribution[:, 1].tolist())


def run(out):
    S.torch.set_num_threads(1)
    assert all(S.digest(Path(k)) == v for k, v in lm.prior.ref('e77')['sha256'].items())
    dates, rows = S.E.load_train()
    assert dates[139] < dates[140]
    dyn = S.E.Dynamics(S.E.DATA/'surrogate.pkl')
    records = []
    out.mkdir(parents=True, exist_ok=False)
    for seed in range(5):
        root = S.ROOT
        paths = [root/'e88/main'/f'{seed}_early_down.npz', root/'e78/main'/f'{seed}_train48_down.npz',
                 root/'e88/main'/f'probe{seed}.npz']
        assert [S.digest(p) for p in paths] == lm.prior.ref('e89')[seed]['input_sha256']
        early, k = S.read_candidates(paths[0], dates[:140])
        x, q = S.before_action(rows[:140], early, k)
        with np.load(paths[2]) as z:
            np.testing.assert_array_equal(k, z['train_keys'])
            panel, targets = training_data(rows[:140], early, k, q, dyn, S.E.restore(seed), z['train_totals'])
        late, keys = S.read_candidates(paths[1], dates[140:])
        xt, qt = S.before_action(rows[140:], late, keys)
        probe = root/'e85/main'/f'probe{seed}.npz'
        assert S.digest(probe) == lm.prior.ref('e86')[seed]['score_sha256']
        with np.load(probe) as z:
            ix = S.headroom.nonoverlap(z['keys'])
            np.testing.assert_array_equal(keys, z['keys'][ix])
            np.testing.assert_array_equal(qt, z['q'][ix])
            horizons = z['totals'][ix]
            truth = horizons[:, :, 2, 0] - horizons[:, :, 1, 0]
        frozen = root/'e98/main'/f'landmark{seed}.npz'
        with np.load(frozen) as z:
            np.testing.assert_array_equal(keys, z['keys'])
            np.testing.assert_allclose(truth[:, 1]-truth[:, 0], z['truth'], atol=1e-7, rtol=0)
            forecasts = []
            for width in (0, 4, 8):
                train = np.column_stack((x, panel[:, :width]))
                evaluation = np.column_stack((xt, z['panel'][:, :width]))
                p, _ = lm.prior.predict(train, targets, k[:, 0], evaluation)
                forecasts.append(p)
            pred = np.stack(forecasts, axis=1)
            np.testing.assert_allclose(pred[:, :, 1]-pred[:, :, 0], z['pred'], atol=1e-7, rtol=0)
            first, selected, supported = z['first'], z['selected'], z['supported']
        residual = truth[:, None, :] - pred
        groups = {}
        for name, mask in [('supported', supported), ('all', np.ones(len(keys), bool))]:
            g = metrics(keys[mask, 0], first[mask], selected[mask], residual[mask])
            old = lm.prior.ref('e98')[seed]['groups'][name]
            for field in ('n', 'candidates', 'selected'):
                assert g[field] == old[field]
            np.testing.assert_allclose(np.sqrt(g['mse_delta']), old['rmse'], atol=1e-7, rtol=0)
            np.testing.assert_allclose(np.add(g['from_a'], g['from_b']),
                                       old['actual_effect']-np.array(old['pred_effect']), atol=1e-7, rtol=0)
            groups[name] = g
        np.savez_compressed(out/f'branches{seed}.npz', keys=keys, truth=truth, pred=pred,
                            first=first, selected=selected, supported=supported)
        records.append(dict(seed=seed, train_n=len(k), frozen_sha256=S.digest(frozen), groups=groups))
        print(f'seed {seed} complete', flush=True)
    (out/'summary.json').write_text(json.dumps(records, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    toy = np.array([[[3., 3.], [0., 3.], [-3., 0.]]])
    m = metrics(np.array([0]), np.array([True]), np.array([True]), toy)
    np.testing.assert_allclose(m['mse_delta'], [0, 9, 9])
    np.testing.assert_allclose(m['from_a'], [-3, 0, 3])
    np.testing.assert_allclose(m['from_b'], [3, 3, 0])
    if args.out:
        run(args.out)

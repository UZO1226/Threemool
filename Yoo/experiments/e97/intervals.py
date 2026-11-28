"""Attribute the frozen six-hour air forecast to disjoint feedback intervals."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e96'))
import effects as prior

S = prior.S
LABELS = ('0~1시간', '1~3시간', '3~6시간')


def split_air(totals):
    cumulative = totals[:, 1, :, 0] - totals[:, 0, :, 0]
    return np.diff(cumulative, axis=1, prepend=0.)


def training_targets(rows, archive, keys, q, agent, dyn, expected):
    horizon = []
    for start in range(0, len(keys), 250):
        part = slice(start, start + 250)
        paired = np.stack([S.P.branch(rows, dyn, agent, keys[part], q[part], correction, archive)
                           for correction in (0., -S.R.STEP)], axis=1)
        np.testing.assert_allclose(paired[:, :, -1], expected[part], rtol=0, atol=1e-7)
        horizon.append(split_air(paired))
    return np.concatenate(horizon)


def analyze(day, pred, true, center, candidates, take):
    w = S.weights(day[candidates])
    p, y = pred[candidates], true[candidates]
    errors = np.average((p-y)**2, weights=w, axis=0)
    base = np.average((center-y)**2, weights=w, axis=0)
    assert (base > 0).all()
    saving = y < -prior.TOL
    def direction(v):
        return np.average((v < -prior.TOL) == saving, weights=w, axis=0).tolist()
    all_weights = S.weights(day)
    def contribution(x):
        return np.average(x*take[:,None], weights=all_weights, axis=0).tolist()
    return dict(n=len(day), candidates=int(candidates.sum()), selected=int(take.sum()),
                rmse=np.sqrt(errors).tolist(), constant_rmse=np.sqrt(base).tolist(),
                skill=(1-errors/base).tolist(), accuracy=direction(p), constant_accuracy=direction(center),
                pred_effect=contribution(pred), actual_effect=contribution(true))


def run(out):
    S.torch.set_num_threads(1)
    assert all(S.digest(Path(path)) == val for path,val in prior.ref('e77')['sha256'].items())
    dates, rows = S.E.load_train()
    assert dates[139] < dates[140] and tuple(S.P.HORIZONS) == (4,12,24)
    dyn = S.E.Dynamics(S.E.DATA/'surrogate.pkl')
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for seed in range(5):
        agent = S.E.restore(seed)
        root = S.ROOT
        sources = [root/'e88/main'/f'{seed}_early_down.npz', root/'e78/main'/f'{seed}_train48_down.npz',
                   root/'e88/main'/f'probe{seed}.npz']
        assert [S.digest(p) for p in sources] == prior.ref('e89')[seed]['input_sha256']
        archive, train_keys = S.read_candidates(sources[0], dates[:140])
        x, q = S.before_action(rows[:140], archive, train_keys)
        late, keys = S.read_candidates(sources[1], dates[140:])
        xt, qt = S.before_action(rows[140:], late, keys)
        with np.load(sources[2]) as z:
            np.testing.assert_array_equal(train_keys, z['train_keys'])
            np.testing.assert_array_equal(keys, z['keys'])
            tr = training_targets(rows[:140], archive, train_keys, q, agent, dyn, z['train_totals'])
        probe = root/'e85/main'/f'probe{seed}.npz'
        assert S.digest(probe) == prior.ref('e86')[seed]['score_sha256']
        with np.load(probe) as z:
            ix = S.headroom.nonoverlap(z['keys'])
            np.testing.assert_array_equal(z['keys'][ix], keys)
            np.testing.assert_array_equal(qt, z['q'][ix])
            true = split_air(z['totals'][ix])
        pred, center = prior.predict(x.astype(float), tr, train_keys[:,0], xt.astype(float))
        with np.load(root/'e96/main'/f'prediction{seed}.npz') as z:
            np.testing.assert_array_equal(keys, z['keys'])
            np.testing.assert_allclose(true.sum(1), z['truth'][:,0], atol=1e-7, rtol=0)
            np.testing.assert_allclose(pred.sum(1), z['pred'][:,0], atol=1e-7, rtol=0)
            np.testing.assert_allclose(center.sum(), z['train_mean'][0], atol=1e-7, rtol=0)
            first, supported = z['first'], z['supported']
            quality = (z['pred'][:,1:] <= prior.TOL).all(1)
            take = first & quality & (pred.sum(1) < -prior.TOL)
            np.testing.assert_array_equal(take, first & quality & (z['pred'][:,0] < -prior.TOL))
        groups = {}
        for name, mask in [('supported', supported), ('all', np.ones(len(keys),bool))]:
            g = analyze(keys[mask,0], pred[mask], true[mask], center, first[mask], take[mask])
            ref = prior.ref('e96')[seed]['groups'][name]
            assert (g['n'],g['candidates'],g['selected']) == (ref['n'],ref['candidates'],ref['outcomes']['forecast']['n'])
            np.testing.assert_allclose(sum(g['actual_effect']), ref['outcomes']['forecast']['delta'][0], atol=1e-7)
            groups[name] = g
        records.append(dict(seed=seed, train_n=len(tr), groups=groups))
        np.savez_compressed(out/f'interval{seed}.npz', keys=keys, pred=pred, truth=true, center=center,
                            first=first, supported=supported, selected=take)
        print(seed, len(tr), groups['supported']['skill'], flush=True)
    (out/'summary.json').write_text(json.dumps(records, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    sample = np.zeros((1,2,3,4))
    sample[0,0,:,0] = [30,90,180]
    sample[0,1,:,0] = [20,120,175]
    np.testing.assert_array_equal(split_air(sample), [[-10,40,-35]])
    if args.out:
        run(args.out)

"""Predict paired action effects before the first cut on frozen DARROW windows."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e88'))
import selection as S

TOL = 1e-8


def ref(exp):
    return json.loads((HERE.parent / exp / 'summary.json').read_text())


def predict(x, target, days, xt):
    w = S.weights(days)
    scaler = StandardScaler().fit(x, sample_weight=w)
    train = scaler.transform(x)
    later = scaler.transform(xt)
    reg = Ridge(alpha=1.).fit(train, target, sample_weight=w)
    pred = reg.predict(later)
    separate = [Ridge(alpha=1.).fit(train, y, sample_weight=w).predict(later) for y in target.T]
    np.testing.assert_allclose(pred, np.column_stack(separate), atol=1e-8, rtol=1e-10)
    return pred, np.average(target, axis=0, weights=w)


def select(pred, truth, candidate):
    saved, safe = pred[:, 0] < -TOL, (pred[:, 1:] <= TOL).all(1)
    real_saved, real_safe = truth[:, 0] < -TOL, (truth[:, 1:] <= TOL).all(1)
    return dict(prior=candidate, forecast=candidate & saved & safe,
                known_air=candidate & real_saved & safe, known_quality=candidate & saved & real_safe,
                oracle=candidate & real_saved & real_safe)


def matched(days, candidate, take):
    prob = np.zeros(len(days))
    if candidate.any():
        d = days[candidate]
        prob[candidate] = (np.bincount(d, weights=take[candidate]) / np.maximum(np.bincount(d), 1))[d]
    np.testing.assert_allclose(np.bincount(days, weights=prob), np.bincount(days, weights=take))
    return prob


def measure(days, candidate, pred, truth, center):
    options = select(pred, truth, candidate)
    options['random'] = matched(days, candidate, options['forecast'])
    w = S.weights(days)
    outcomes = {}
    for name, take in options.items():
        effect = np.average(truth * take[:, None], axis=0, weights=w)
        outcomes[name] = dict(n=float(take.sum()), delta=effect.tolist(),
                              hits=float(np.dot(take.astype(float), options['oracle'])))
    cw = S.weights(days[candidate])
    target = truth[candidate]
    mse = np.average((pred[candidate]-target)**2, axis=0, weights=cw)
    reference = np.average((center-target)**2, axis=0, weights=cw)
    skill = [float(1-a/b) if b > 0 else None for a,b in zip(mse, reference)]
    return dict(n=len(days), candidates=int(candidate.sum()), days=len(set(days)), outcomes=outcomes,
                rmse=np.sqrt(mse).tolist(), reference_rmse=np.sqrt(reference).tolist(), skill=skill)


def smoke():
    day = np.array([0, 0, 0, 1, 1])
    candidate = np.array([0, 1, 1, 0, 0], bool)
    truth = np.array([[-10,0,0,0], [-5,0,0,-1], [-2,0,1,0], [-3,0,0,0], [-1,0,0,0.]])
    take = select(truth, truth, candidate)['forecast']
    np.testing.assert_array_equal(take, [0, 1, 0, 0, 0])
    prob = matched(day, candidate, take)
    np.testing.assert_allclose(prob, [0,.5,.5,0,0])
    stats = measure(day, candidate, truth, truth, truth.mean(0))
    np.testing.assert_allclose(stats['outcomes']['random']['delta'], (truth[1]+truth[2])/12)
    assert stats['skill'] == [1., None, 1., 1.]


def run(out):
    S.torch.set_num_threads(1)
    data = S.E.DATA / 'data.pkl'
    assert S.digest(data) == ref('e77')['sha256'][str(data)]
    dates, rows = S.E.load_train()
    assert dates[139] < dates[140]
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for seed in range(5):
        root = S.ROOT
        early = root / 'e88/main' / f'{seed}_early_down.npz'
        late = root / 'e78/main' / f'{seed}_train48_down.npz'
        labels = root / 'e88/main' / f'probe{seed}.npz'
        choices = root / 'e94/main' / f'pulse{seed}.npz'
        assert [S.digest(p) for p in (early, late, labels)] == ref('e89')[seed]['input_sha256']
        assert S.digest(choices) == ref('e95')[seed]['sha256']
        agent = S.E.restore(seed)
        features = []
        with np.load(labels) as z:
            for path, date_part, row_part, prefix in [(early, dates[:140], rows[:140], 'train_'),
                                                      (late, dates[140:], rows[140:], '')]:
                archive, keys = S.read_candidates(path, date_part)
                np.testing.assert_array_equal(keys, z[prefix+'keys'])
                x, q = S.before_action(row_part, archive, keys)
                np.testing.assert_allclose(S.R.action_to_air(agent.act_det(x)[:,0]), q, rtol=0, atol=.02)
                features.append(x.astype(float))
            train = z['train_totals'][:, 1] - z['train_totals'][:, 0]
            train_days = z['train_keys'][:, 0]
            truth = z['change']
        with np.load(choices) as z:
            np.testing.assert_array_equal(keys, z['keys'])
            np.testing.assert_allclose(truth, z['b']-z['a'], rtol=0, atol=1e-7)
            first, supported = z['first'], z['supported']
        pred, center = predict(features[0], train, train_days, features[1])
        groups = {}
        for name, mask in [('supported', supported), ('all', np.ones(len(keys), bool))]:
            g = measure(keys[mask,0], first[mask], pred[mask], truth[mask], center)
            prior = ref('e93')[seed]['groups'][name]
            upper = ref('e95')[seed]['groups'][name]
            assert g['candidates'] == prior['cut'] and g['n'] == upper['n']
            np.testing.assert_allclose(g['outcomes']['prior']['delta'], prior['delta']['gate'], rtol=0, atol=1e-7)
            np.testing.assert_allclose(-g['outcomes']['oracle']['delta'][0], upper['first_choice_alone_m3'], rtol=0, atol=1e-7)
            groups[name] = g
        np.savez_compressed(out / f'prediction{seed}.npz', keys=keys, first=first, supported=supported,
                            pred=pred, truth=truth, train_mean=center)
        records.append(dict(seed=seed, train_n=len(train), groups=groups))
    (out/'summary.json').write_text(json.dumps(records, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    smoke()
    if args.out:
        run(args.out)

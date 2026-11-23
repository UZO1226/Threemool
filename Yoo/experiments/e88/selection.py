"""Estimate the incremental value of frozen FCSRL features for selective aeration cuts."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'e87'))
sys.path.insert(0, str(HERE.parent/'e85'))
import headroom
import probe

E, R, P = probe.E, probe.R, probe.P
ROOT = headroom.ROOT
METHODS = ('raw', 'learned', 'random_features')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_decisions(keys, score, threshold):
    """Choose the first on its own score, otherwise choose the next arrival."""
    keep = np.zeros(len(keys), bool)
    take = np.zeros(len(keys), bool)
    for m in np.unique(keys[:, 2]):
        group = np.flatnonzero(keys[:, 2] == m)
        ordered = group[np.lexsort((keys[group, 1], keys[group, 0]))]
        pairs = ordered[:len(ordered)//2*2].reshape(-1, 2)
        first, second = pairs.T
        keep[pairs] = True
        take[first] = score[first] >= threshold
        take[second] = ~take[first]
        np.testing.assert_array_equal(take[pairs].sum(axis=1), 1)
    return keep, take


def weights(days):
    _, ix, counts = np.unique(days, return_inverse=True, return_counts=True)
    return len(days) / (len(counts) * counts[ix])


def summary(keys, change, positive, take):
    take = np.asarray(take, dtype=float)
    delta = np.average(change*take[:, None], axis=0, weights=weights(keys[:, 0]))
    return dict(candidates=len(keys), dates=len(set(keys[:, 0])), cuts=float(take.sum()),
                precision=float(np.dot(positive, take)/take.sum()) if take.any() else None,
                delta=delta.tolist())


def before_action(rows, archive, keys):
    day, t, member = keys.T
    ep = day*5 + member
    # Recreate only information available before the current transition.
    obs = R.observations(rows, keys[:, :2])
    obs[:, :5] = archive['states'][ep, t-1]
    obs[:, 10] = archive['air'][ep, t-1]/9000
    return obs, archive['raw'][ep, t]


def read_candidates(path, dates):
    with np.load(path) as z:
        np.testing.assert_array_equal(z['dates'], np.asarray(dates, str))
        archive = {k:z[k] for k in ('states', 'air', 'direction', 'raw')}
    ep, ix = np.where(archive['direction'][:, P.TIMES] == -1)
    keys = np.column_stack((ep//5, P.TIMES[ix], ep%5))
    return archive, keys[headroom.nonoverlap(keys)]


def train_labels(rows, archive, keys, q, dyn, agent):
    totals = np.empty((len(keys), 2, 4))
    for start in range(0, len(keys), 250):
        b = slice(start, start+250)
        for arm, offset in enumerate((0., -R.STEP)):
            totals[b, arm] = P.branch(rows, dyn, agent, keys[b], q[b], offset, archive)[:, -1]
    return totals


@torch.no_grad()
def features(agent, seed, x):
    obs = agent._t(agent.obs_normalizer(x))
    policy_input = agent.policy_input(obs)
    np.testing.assert_array_equal(policy_input[:, :11].numpy(), obs.numpy())
    assert policy_input.shape == (len(obs), 75)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8800 + seed)
        encoder = deepcopy(agent.fixed_encoder)
        for layer in encoder.zs_net.modules():
            if isinstance(layer, torch.nn.Linear):
                layer.reset_parameters()
        noise = encoder.zs(obs)
    return (obs.numpy(), policy_input.numpy(), torch.cat((obs, noise), dim=1).numpy())


def scores(x, y, w, xt):
    scaler = StandardScaler().fit(x, sample_weight=w)
    x = scaler.transform(x)
    model = LogisticRegression(C=1., max_iter=2000).fit(x, y, sample_weight=w)
    assert max(model.n_iter_) < 2000
    return model.decision_function(scaler.transform(xt)), float(np.median(model.decision_function(x)))


def run(out):
    torch.set_num_threads(1)
    hashes = json.loads((HERE.parent/'e77/summary.json').read_text())['sha256']
    assert all(digest(Path(p)) == h for p,h in hashes.items())
    labels_ref = json.loads((HERE.parent/'e87/summary.json').read_text())
    tree_hash = json.loads((HERE.parent/'e79/summary.json').read_text())['e78_input_sha256']
    source_ref = json.loads((HERE.parent/'e85/summary.json').read_text())['rows']
    dates, rows = E.load_train()
    train, hold = dates[:140], dates[140:]
    assert max(train) < min(hold)
    support = R.fit_support(rows[:140])
    dyn = E.Dynamics(E.DATA/'surrogate.pkl')
    out.mkdir(parents=True, exist_ok=False)
    records = []
    for seed in range(5):
        agent = E.restore(seed)
        fingerprint = E.S.fingerprint(agent)
        tree_path = probe.SOURCE/f'tree{seed}.joblib'
        assert digest(tree_path) == tree_hash[tree_path.name]
        tree = joblib.load(tree_path)
        R.evaluate(seed, agent, tree, support, 'down', 'early', train, dict(zip(train, rows[:140])), dyn, out)
        archive, train_keys = read_candidates(out/f'{seed}_early_down.npz', train)
        x, q = before_action(rows[:140], archive, train_keys)
        np.testing.assert_allclose(R.action_to_air(agent.act_det(x)[:,0]), q, atol=.02, rtol=0)
        totals = train_labels(rows[:140], archive, train_keys, q, dyn, agent)
        _, _, y, _ = headroom.measures(totals)
        val_source = probe.SOURCE/f'{seed}_train48_down.npz'
        assert digest(val_source) == source_ref[seed]['source_sha256']
        validation, keys = read_candidates(val_source, hold)
        xt, qt = before_action(rows[140:], validation, keys)
        np.testing.assert_allclose(R.action_to_air(agent.act_det(xt)[:,0]), qt, atol=.02, rtol=0)
        source = ROOT/'e85/main'/f'probe{seed}.npz'
        assert digest(source) == labels_ref[seed]['sha256'][0]
        with np.load(source) as z:
            keep = headroom.nonoverlap(z['keys'])
            np.testing.assert_array_equal(z['keys'][keep], keys)
            np.testing.assert_array_equal(z['q'][keep], qt)
            change, _, yt, _ = headroom.measures(z['totals'][keep, :, -1])
        assert yt.sum() == labels_ref[seed]['nonoverlap']['useful']
        for obs, flow in [(x,q), (xt,qt)]:
            np.testing.assert_array_equal(R.command(obs, flow, tree, support, 'down'), -np.ones(len(obs)))
        all_features = features(agent, seed, np.concatenate((x,xt)))
        modes = {}
        predictions = []
        for method, feat in zip(METHODS, all_features):
            score, threshold = scores(feat[:len(x)], y, weights(train_keys[:,0]), feat[len(x):])
            keep, take = paired_decisions(keys, score, threshold)
            modes[method] = dict(auc=float(roc_auc_score(yt, score, sample_weight=weights(keys[:,0]))),
                                 threshold=threshold, matched=summary(keys[keep], change[keep], yt[keep], take[keep]),
                                 free=summary(keys, change, yt, score >= threshold))
            predictions.append(score)
        modes['coin'] = summary(keys[keep], change[keep], yt[keep], np.full(keep.sum(), .5))
        assert all(modes[m]['matched']['cuts'] == modes['coin']['cuts'] for m in METHODS)
        assert fingerprint == E.S.fingerprint(agent)
        records.append(dict(seed=seed, train_n=len(x), train_pos=int(y.sum()), train_days=len(set(train_keys[:,0])),
                            eval_n=len(xt), eval_pos=int(yt.sum()), matched_dropped=int((~keep).sum()), modes=modes))
        np.savez_compressed(out/f'probe{seed}.npz', train_dates=np.asarray(train,str), dates=np.asarray(hold,str),
                            train_keys=train_keys, train_totals=totals, keys=keys, change=change,
                            positive=yt, scores=np.stack(predictions), matched=keep)
        print(seed, len(x), len(xt), modes['learned']['matched'], flush=True)
    (out/'summary.json').write_text(json.dumps(records, indent=2, allow_nan=False)+'\n')


def smoke():
    keys = np.array([[2,4,0], [0,4,0], [1,4,0], [0,4,1], [1,4,1]])
    score = np.array([9,1,10,-1,-10])
    keep, take = paired_decisions(keys, score, 0)
    np.testing.assert_array_equal(keep, [0,1,1,1,1])
    np.testing.assert_array_equal(take, [0,1,0,0,1])
    score[[0,2,4]] *= -1
    np.testing.assert_array_equal(paired_decisions(keys, score, 0)[1], take)
    np.testing.assert_allclose(weights(np.array([0,0,1])), [.75,.75,1.5])
    agent = E.restore(0)
    x = np.zeros((4,11), dtype=np.float32)
    a, b, c = features(agent, 0, x)
    np.testing.assert_array_equal(a,b[:,:11])
    np.testing.assert_array_equal(c,features(agent,0,x)[2])
    assert not np.array_equal(b[:,11:],c[:,11:])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.smoke:
        smoke()
    else:
        run(args.out)

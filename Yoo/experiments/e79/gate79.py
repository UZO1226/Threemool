"""Evaluate a training-day recurrence requirement for e78's positive leaves."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e78'))
from report import audit, wins
import run as prior

E = prior.E
TOL = 1e-8


def distinct_days(keys, mask):
    return int(len(np.unique(keys[mask, 0])))


class GatedTree:
    def __init__(self, tree, permitted):
        self.tree = tree
        self.permitted = permitted

    def predict(self, x):
        labels = self.tree.predict(x)
        allowed = np.isin(self.tree.apply(x), self.permitted)
        return np.where((labels == 1) & ~allowed, 0, labels)


def calibrate(tree, fit, rows, agent, dynamics, support):
    keys = np.array([(d, t) for d in range(140, 188) for t in range(4, 89, 4)])
    x = prior.observations(rows, keys)
    q = prior.action_to_air(agent.act_det(x)[:, 0])
    proposed = (tree.predict(x) == 1) & prior.allowed(x, q, support)[1]
    keys, x, q = keys[proposed], x[proposed], q[proposed]
    leaves = tree.apply(x) if len(x) else np.array([], dtype=int)
    corroborated = np.zeros(len(keys), dtype=bool)
    for start in range(0, len(keys), 160):
        batch = slice(start, start + 160)
        branches = []
        for delta in [0, prior.STEP]:
            obs, totals, _ = E.simulate(rows, dynamics, keys[batch], q[batch] + delta, agent)
            np.testing.assert_array_equal(obs, x[batch])
            branches.append(totals)
        corroborated[batch] = E.classify(*branches)[1] == 2

    train_leaf = tree.apply(fit['x'])
    train_pred = tree.predict(fit['x'])
    assert np.all(fit['keys'][:, 0] < 140)
    details = []
    for leaf in np.unique(train_leaf[train_pred == 1]):
        positive = (train_leaf == leaf) & (fit['labels'] == 1)
        selected = leaves == leaf
        fit_days = distinct_days(fit['keys'], positive)
        cal_days = distinct_days(keys, selected & corroborated)
        details.append(dict(leaf=int(leaf), fit_positive=int(sum(positive)), fit_days=fit_days,
                            cal_proposed=int(sum(selected)), cal_positive=int(sum(selected & corroborated)),
                            cal_days=cal_days, permit=min(fit_days, cal_days) >= 3))
    return details


def mechanism(gated, both, down):
    if any(gated[k] >= both[k] - TOL for k in ['air', 'do']):
        return False
    for metric in ['nh', 'excess']:
        gain = down[metric] - both[metric]
        if gain <= TOL or down[metric] - gated[metric] < .5 * gain - TOL:
            return False
    return True


def assess(group):
    base, both, down, gated = [group[k] for k in ('base', 'both', 'down', 'gated')]
    delta = {k: gated[k] - base[k] for k in ['air', 'nh', 'do', 'excess']}
    return dict(joint=bool(wins(delta, False)), secondary=bool(mechanism(gated, both, down)))


def main(source, out):
    torch.set_num_threads(1)
    old = json.loads((source / 'records.json').read_text())
    assert old == json.loads((HERE.parent / 'e78/records.json').read_text())
    hashes = json.loads((HERE.parent / 'e77/summary.json').read_text())['sha256']
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    splits, days = E.S.base.load_selected(E.DATA / 'data.pkl')
    dates, rows = E.load_train()
    assert dates == splits['train'] and len(set(dates)) == 188
    dynamics = E.Dynamics(E.DATA / 'surrogate.pkl')
    support = prior.fit_support(rows[:140])
    out.mkdir(parents=True, exist_ok=False)
    records, evidence, checksums = [], [], {}
    for seed in range(5):
        agent = E.restore(seed)
        fingerprint = E.S.fingerprint(agent)
        tree_file = source / f'tree{seed}.joblib'
        fit_file = source / f'fit{seed}.npz'
        for path in (fit_file, tree_file):
            checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        tree = joblib.load(tree_file)
        with np.load(fit_file) as fit:
            details = calibrate(tree, fit, rows, agent, dynamics, support)
            positive = [r['leaf'] for r in details]
            allowed = [r['leaf'] for r in details if r['permit']]
            labels = tree.predict(fit['x'])
            np.testing.assert_array_equal(GatedTree(tree, positive).predict(fit['x']), labels)
            np.testing.assert_array_equal(GatedTree(tree, []).predict(fit['x']), np.minimum(labels, 0))
        evidence.append(dict(seed=seed, leaves=details))
        control = [r for r in old if r['split'] == 'winter' and r['seed'] == seed]
        assert sorted(r['mode'] for r in control) == ['base', 'both', 'down']
        for row in control:
            audit(source, row)
        records.extend(control)
        stats = prior.evaluate(seed, agent, GatedTree(tree, allowed), support, 'both', 'winter',
                               splits['val'], days, dynamics, out)
        audit(out, dict(seed=seed, split='winter', mode='both', **stats))
        with np.load(out / f'{seed}_winter_both.npz') as actual:
            np.testing.assert_array_equal(actual['dates'], np.asarray(splits['val'], dtype=str))
            if not allowed or set(allowed) == set(positive):
                mode = 'both' if allowed else 'down'
                with np.load(source / f'{seed}_winter_{mode}.npz') as ref:
                    for key in ['dates', 'states', 'direction', 'air', 'raw']:
                        np.testing.assert_array_equal(actual[key], ref[key])
        records.append(dict(seed=seed, mode='gated', split='winter', **stats))
        assert E.S.fingerprint(agent) == fingerprint
        print(f'seed {seed}, retained {len(allowed)}/{len(positive)} leaves', flush=True)
    metrics = ['air', 'nh', 'do', 'excess', 'up_steps', 'down_steps', 'limit_pct']
    modes = ['base', 'down', 'both', 'gated']
    means = {m: {k: float(np.mean([r[k] for r in records if r['mode'] == m]))
                 for k in metrics} for m in modes}
    paired = [assess({r['mode']: r for r in records if r['seed'] == seed}) for seed in range(5)]
    decision = {k: bool(ok and sum(r[k] for r in paired) >= 4) for k, ok in assess(means).items()}
    result = dict(means=means, paired=paired, decision=decision, evidence=evidence, records=records,
                  e78_input_sha256=checksums)
    (out / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.out)

"""Compare frozen feasibility scores with six-hour targets on the same e85 cuts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e85'))
import probe
from ppo_fcsrl import feasibility_targets
from darrow_env import NH4_MAX, STEP_H

E, R = probe.E, probe.R
SCORES = Path('/home/jio/fcsrl_src/results/e85/main')
H = 24
TOL = 1e-8


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def peak(cost, tail, gamma):
    """Input axes are step, candidate. Compare the training recursion to closed form."""
    next_value = np.zeros_like(cost)
    next_value[-1] = tail
    mask = np.zeros_like(cost)
    target = feasibility_targets(cost, next_value, mask, mask, gamma)[0]
    observed = np.max(gamma ** np.arange(len(cost))[:, None] * cost, axis=0)
    np.testing.assert_allclose(target, np.maximum(observed, gamma ** len(cost) * tail),
                               atol=1e-12, rtol=1e-12)
    return target, observed


def replay(rows, dyn, agent, keys, q, archive, cut, expected):
    windows = {i: {k: arr[t:t+H+1] for k, arr in rows[d].items()}
               for i, (d, t, _) in enumerate(keys)}
    env = E.BatchEnv(windows, dyn)
    day, time, member = keys.T
    env.reset(list(enumerate(member)))
    ep = day * 5 + member
    env.state = archive['states'][ep, time-1].copy()
    env.prev_air = archive['air'][ep, time-1].copy()
    env.prev_zone = env.prev_air[:, None] * np.array([rows[d]['frac'][t-1] for d, t in zip(day, time)])
    obs = env.obs()
    np.testing.assert_allclose(R.action_to_air(agent.act_det(obs)[:, 0]), q, atol=.02, rtol=0)
    metrics = []
    for i in range(H):
        action = R.air_to_action(q - R.STEP * cut) if i == 0 else agent.act_det(obs)[:, 0]
        obs, _, cost, done, info = env.step(action)
        np.testing.assert_array_equal(cost, info['v_nh4'] + info['v_do'])
        if cut and i == 0:
            np.testing.assert_allclose(env.state, archive['states'][ep, time], atol=1e-10, rtol=0)
        metrics.append(np.column_stack((info['air'] * STEP_H, info['v_nh4'], info['v_do'],
                                        np.maximum(0, env.state[:, 0] - NH4_MAX) * STEP_H)))
    assert done
    metrics = np.stack(metrics)
    totals = metrics.cumsum(axis=0)[np.array(probe.P.HORIZONS)-1].transpose(1, 0, 2)
    np.testing.assert_allclose(totals, expected, atol=1e-7, rtol=0)
    tail = agent.next_feasibility(agent.obs_normalizer(obs)).astype(float)
    return metrics[:, :, 1:3].transpose(1, 0, 2), tail


def ranked(day, scores, labels):
    _, inverse, counts = np.unique(day, return_inverse=True, return_counts=True)
    days = [int(np.unique(day[mask]).size) for mask in (~labels, labels)]
    auc = float(roc_auc_score(labels, scores, sample_weight=1/counts[inverse])) if min(days) else None
    return dict(n=int(len(day)), positive=int(labels.sum()), class_days=days, auc=auc,
                passed=bool(min(days) >= 5 and auc is not None and auc >= .60))


def evaluate(keys, normal, cut, costs, tails, totals, gamma):
    gap = cut-normal
    nh = costs[:, :, :, 0]
    combined = costs.sum(axis=-1)
    native, observed = peak(combined.transpose(2, 0, 1).reshape(H, -1), tails.ravel(), gamma)
    nh_observed = np.max(nh * gamma ** np.arange(H), axis=-1)
    targets = dict(bootstrap_peak=native.reshape(-1, 2), observed_peak=observed.reshape(-1, 2),
                   nh_peak=nh_observed, combined_count=combined.sum(axis=-1), nh_count=nh.sum(axis=-1))
    results = {}
    for name, values in targets.items():
        diff = values[:, 1] - values[:, 0]
        results[name] = dict(**ranked(keys[:, 0], gap, diff > TOL), ties=int((abs(diff) <= TOL).sum()),
                             decreases=int((diff < -TOL).sum()))
    change = totals[:, 1, -1] - totals[:, 0, -1]
    results['e85_harm'] = ranked(keys[:, 0], gap, (change[:, 1] >= 1) | (change[:, 3] >= .1))
    results['absolute'] = ranked(keys[:, 0], normal, combined[:, 0].max(axis=1) > 0)
    return results


def run(out, smoke):
    torch.set_num_threads(1)
    inputs = json.loads((HERE.parent/'e77/summary.json').read_text())['sha256']
    assert all(digest(Path(p)) == value for p, value in inputs.items())
    reference = json.loads((HERE.parent/'e85/summary.json').read_text())['rows']
    dates, rows = E.load_train()
    dyn = E.Dynamics(E.DATA/'surrogate.pkl')
    results = []
    if smoke:
        actual, observed = peak(np.array([[0., 0.], [1., 0.]]), np.array([0., 1.]), .9)
        np.testing.assert_allclose(actual, [.9, .81])
        np.testing.assert_allclose(observed, [.9, 0.])
    else:
        out.mkdir(parents=True, exist_ok=False)
    for seed in range(1 if smoke else 5):
        agent = E.restore(seed)
        fingerprint = E.S.fingerprint(agent)
        gamma = agent.repr_config.feasibility_discount
        assert gamma == .9
        source = probe.SOURCE / f'{seed}_train48_down.npz'
        score_path = SCORES / f'probe{seed}.npz'
        assert digest(source) == reference[seed]['source_sha256']
        with np.load(source) as raw, np.load(score_path) as z:
            np.testing.assert_array_equal(z['dates'], np.asarray(dates[140:], str))
            np.testing.assert_array_equal(raw['dates'], z['dates'])
            archive = {k: raw[k] for k in ('states', 'air')}
            keys, q, totals = z['keys'], z['q'], z['totals']
            normal, cut = z['normal'], z['cut']
        if smoke:
            keys, q, totals = keys[:5], q[:5], totals[:5]
        costs = np.empty((len(keys), 2, H, 2))
        tails = np.empty((len(keys), 2))
        for start in range(0, len(keys), 250):
            b = slice(start, start+250)
            for action in (0, 1):
                costs[b, action], tails[b, action] = replay(rows[140:], dyn, agent, keys[b], q[b], archive,
                                                            action, totals[b, action])
        assert np.isfinite(tails).all() and E.S.fingerprint(agent) == fingerprint
        if smoke:
            print('target recurrence and archived rollout smoke passed')
            return
        stats = evaluate(keys, normal, cut, costs, tails, totals, gamma)
        assert abs(stats['e85_harm']['auc'] - reference[seed]['auc']) < 1e-12
        assert stats['e85_harm']['positive'] == reference[seed]['positive']
        np.savez_compressed(out/f'seed{seed}.npz', keys=keys, costs=costs.astype(np.uint8), tails=tails)
        results.append(dict(seed=seed, score_sha256=digest(score_path), metrics=stats))
        print(seed, len(keys), stats['bootstrap_peak'], flush=True)
    (out/'summary.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.out, args.smoke)

"""Training-date paired first actions, four-step closed-loop costs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e58'))
import severity as S
from darrow_env import BatchEnv, Dynamics, EXO, STATE, ZONES, air_to_action
from darrow_env import all_episodes

DATA = Path('/home/jio/fcsrl_src/results/e50/prepared')
POLICIES = Path('/home/jio/fcsrl_src/results/e58/main/runs/baseline')
TOL = 1e-8


def load_train():
    raw = pd.read_pickle(DATA / 'data.pkl')
    dates = list(raw['days']['train'])
    assert dates == sorted(dates) and len(dates) == 188
    typical = np.array([raw['zone_fractions'][z] for z in ZONES])
    rows = []
    for date in dates:
        g = raw['data'].loc[date]
        zone = g[ZONES].to_numpy(float)
        total = zone.sum(1, keepdims=True)
        rows.append(dict(state=g[STATE].to_numpy(float), exo=g[EXO].to_numpy(float),
                         air=g.air.to_numpy(float), zone=zone,
                         frac=np.where(total > 1, zone / np.maximum(total, 1), typical)))
    return dates, rows


def support(rows, anchors):
    def features(d, t):
        x = rows[d]
        return [x['state'][t, 0], x['air'][t-1], *x['exo'][t, :2]]
    pool = [(d, t) for d in range(140) for t in range(1, 95)]
    x = np.array([features(*key) for key in pool])
    scale = x.std(0)
    nn = NearestNeighbors(n_neighbors=194).fit(x / scale)
    ids = nn.kneighbors(np.array([features(*key) for key in anchors]) / scale, return_distance=False)
    pool = np.array(pool)
    bounds = []
    for (d, t), matches in zip(anchors, ids):
        neighbors = pool[matches]
        neighbors = neighbors[neighbors[:, 0] != d][:100]
        assert len(neighbors) == 100
        bounds.append(np.percentile([rows[i]['air'][j] for i, j in neighbors], [25, 75]))
    return np.array(bounds)


def simulate(rows, dyn, keys, first=None, agent=None):
    # Run each anchor as a short episode, restoring the measured prior actuator state.
    windows = {i: {k: v[t:t+5] for k, v in rows[d].items()} for i, (d, t) in enumerate(keys)}
    env = BatchEnv(windows, dyn)
    env.reset(all_episodes(windows, 5))
    env.prev_zone = np.repeat([rows[d]['zone'][t-1] for d, t in keys], 5, axis=0)
    env.prev_air = np.repeat([rows[d]['air'][t-1] for d, t in keys], 5)
    obs = env.obs()
    initial = obs[::5].copy()
    sums = np.zeros((len(obs), 4))
    for k in range(4):
        if first is None:
            a = env.historical_action()
        elif k == 0:
            a = air_to_action(np.repeat(first, 5))
        else:
            a = agent.act_det(obs)[:, 0]
        obs, _, cost, done, info = env.step(a)
        assert np.allclose(env.prev_zone.sum(1), info['air'])
        assert np.array_equal(cost, info['v_nh4'] + info['v_do'])
        sums += np.column_stack([info['air'] * .25, info['v_nh4'], info['v_do'],
                                 np.maximum(env.state[:, 0] - 4, 0) * .25])
    assert done
    return initial, sums.reshape(-1, 5, 4), env.state[:, 0].reshape(-1, 5)


def restore(seed):
    agent = S.make_agent('baseline', seed, DATA, S.base.common.config(66))
    agent.load(POLICIES / f'seed_{seed}/model_final')
    return agent


def classify(low, high):
    delta = high - low
    air, nh, do, exc = delta.transpose(2, 0, 1)
    cheap = (air >= 25) & (nh >= 0) & (do >= 0) & (exc >= -TOL)
    useful = ((nh <= -1) | (exc <= -.1)) & (nh <= 0) & (do <= 0) & (exc <= TOL)
    assert not np.any(cheap & useful)
    votes = np.where(cheap, 1, np.where(useful, 2, 0))
    labels = np.where(cheap.sum(1) >= 4, 1, np.where(useful.sum(1) >= 4, 2, 0))
    return votes, labels


def replay(rows, dyn, keys):
    keys = [(d, t) for d, t in keys if d >= 140 and abs(rows[d]['air'][t] - rows[d]['air'][t-1]) >= 500]
    _, _, pred = simulate(rows, dyn, keys)
    result = {}
    for sign, name in [(1, 'increase'), (-1, 'decrease')]:
        ids = [i for i, (d,t) in enumerate(keys) if sign * (rows[d]['air'][t] - rows[d]['air'][t-1]) > 0]
        actual = np.array([rows[d]['state'][t+4, 0] for d,t in (keys[i] for i in ids)])
        initial = np.array([rows[d]['state'][t, 0] for d,t in (keys[i] for i in ids)])
        days = len({keys[i][0] for i in ids})
        model_mae = float(np.abs(pred[ids].mean(1) - actual).mean())
        persistence = float(np.abs(initial - actual).mean())
        result[name] = dict(n=len(ids), dates=days, model_mae=model_mae, persistence_mae=persistence,
                            passed=days >= 10 and model_mae < persistence)
    return result


def probe(x, keys, labels):
    train = keys[:, 0] < 140
    classifier = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, class_weight='balanced', random_state=77)
    classifier.fit(x[train], labels[train])
    pred = classifier.predict(x[~train])
    truth = labels[~train]
    matrix = confusion_matrix(truth, pred, labels=[0, 1, 2])
    metrics = {}
    for label, name in [(1, 'low'), (2, 'high')]:
        ids = truth == label
        days = len(np.unique(keys[~train][ids, 0]))
        precision = float(matrix[label,label] / max(1, matrix[:,label].sum()))
        recall = float(matrix[label,label] / max(1, ids.sum()))
        metrics[name] = dict(n=int(ids.sum()), days=days, fraction=float(ids.mean()), precision=precision, recall=recall,
                             passed=bool(days >= 10 and ids.mean() >= .05 and min(precision, recall) >= .6))
    return metrics, matrix.tolist(), pred


def main(out, smoke):
    torch.set_num_threads(1)
    dates, rows = load_train()
    dyn = Dynamics(DATA / 'surrogate.pkl')
    assert dyn.n_members == 5
    keys = [(d,t) for d in range(len(rows)) for t in range(4, 89, 4)]
    bounds = support(rows, keys)
    keys = np.asarray(keys)
    valid = (bounds[:,1] - bounds[:,0] >= 500) & (bounds[:,0] >= 0) & (bounds[:,1] <= 9000)
    selected = keys[valid]
    ranges = bounds[valid]
    if smoke:
        agent = restore(0)
        a = simulate(rows, dyn, selected[:12], ranges[:12,0], agent)
        b = simulate(rows, dyn, selected[:12], ranges[:12,0], agent)
        assert all(np.array_equal(u,v) for u,v in zip(a,b))
        assert not classify(a[1], b[1])[1].any()
        env = BatchEnv({0:rows[0]}, dyn)
        obs = env.reset(all_episodes([0], 5))
        with np.load(POLICIES/'seed_0/train_rollout.npz') as z:
            wanted = z['states'][:5,:4]
        for t in range(4):
            obs, *_ = env.step(agent.act_det(obs)[:,0])
            np.testing.assert_allclose(env.state, wanted[:,t], atol=2e-5, rtol=0)
        print('smoke pass')
        return
    out.mkdir(parents=True, exist_ok=False)
    result = dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),
                  fit_dates=140, probe_dates=48, eligible_fit=int((selected[:,0]<140).sum()),
                  eligible_probe=int((selected[:,0]>=140).sum()), observed=replay(rows,dyn,keys), policies=[])
    for seed in range(5):
        agent = restore(seed)
        blocks = []
        for start in range(0,len(selected),256):
            batch = slice(start,start+256)
            x, lo, _ = simulate(rows,dyn,selected[batch],ranges[batch,0],agent)
            _, hi, _ = simulate(rows,dyn,selected[batch],ranges[batch,1],agent)
            blocks.append((x,lo,hi))
        x,lo,hi = [np.concatenate(v) for v in zip(*blocks)]
        votes, labels = classify(lo,hi)
        m, matrix, pred = probe(x,selected,labels)
        result['policies'].append(dict(seed=seed, metrics=m, confusion=matrix))
        np.savez_compressed(out/f'seed{seed}.npz', keys=selected, dates=np.array(dates,dtype=str),
                            bounds=ranges, obs=x, low=lo, high=hi, votes=votes, labels=labels, predictions=pred)
        print('completed seed',seed,flush=True)
    result['eligible_policies'] = sum(all(v['passed'] for v in p['metrics'].values()) for p in result['policies'])
    result['advance'] = result['eligible_policies'] >= 4 and all(v['passed'] for v in result['observed'].values())
    inputs = [DATA/'data.pkl', DATA/'surrogate.pkl',DATA/'prior.pt',DATA/'normalizer.json']
    for seed in range(5):
        inputs.extend(sorted((POLICIES/f'seed_{seed}/model_final').glob('*')))
    result['sha256'] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    main(args.out, args.smoke)

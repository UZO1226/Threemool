"""Predict observed NH4 declines and test the fixed e106 calibration choice."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e106'))
import calibrate as C

OLD = C.E.DATA.parents[1] / 'e106/main'
FEATURES = ['nh4', 'trend', 'flow']
MODELS = {'level': ['nh4'], 'trend': ['nh4', 'trend'], 'flow': ['nh4', 'flow'], 'full': FEATURES}
ERRORS = ['segment_mae', 'nh4_3h', 'do_3h']


def features(state, exo, t):
    return [state[t, 0], state[t, 0]-state[t-4, 0], exo[t, 0]]


def smoke():
    x = np.arange(80).reshape(16, 5)
    u = np.arange(32).reshape(16, 2)
    assert features(x, u, 4) == [20, 20, 8]
    assert features(x, u, 4) == features(x[:5], u[:5], 4)
    np.testing.assert_allclose(weights(pd.DataFrame({'day': [0, 0, 1]})), [.75, .75, 1.5])


def weights(df):
    w = 1 / df.groupby('day')['day'].transform('size').to_numpy()
    return w / w.mean()


def load(name, dates, days):
    with np.load(OLD / f'{name}.npz') as z:
        np.testing.assert_array_equal(z['dates'], np.array(dates, dtype=str))
        ids, times = z['ids'], z['t']
        events = C.base.W.events(C.base.W.windows(days, dates), 500.0)
        assert [(e['day'], e['t'], e['dq']) for e in events] == [
            (dates[d], t, q) for d, t, q in zip(ids, times, z['dq'])]
        rows = []
        for d, t, truth in zip(ids, times, z['truth']):
            g = days[dates[d]]
            np.testing.assert_array_equal(truth, g['state'][t:t+13])
            x = features(g['state'][:t+1], g['exo'][:t+1], t)
            rows.append([str(dates[d]), d, t, *x, truth[12, 0]-truth[4, 0]])
        df = pd.DataFrame(rows, columns=['date', 'day', 't', *FEATURES, 'change'])
        df['fall'] = (df.change < -.1).astype(int)
        df['direction'] = np.where(z['dq']>0, 'increase', 'decrease')
        assert z['gains'][0] == 0 and z['gains'][1] == .05
        for prefix, p in zip(['base', 'corrected'], z['preds'][:2]):
            errors = C.metrics(z['truth'], p)
            for col in ERRORS:
                df[f'{prefix}_{col}'] = errors[:, C.METRICS.index(col)]
    return df


def fit(df):
    w = weights(df)
    fitted = {'constant': float(np.average(df.fall, weights=w))}
    for key, cols in MODELS.items():
        scale = StandardScaler().fit(df[cols], sample_weight=w)
        lr = LogisticRegression(C=1., tol=1e-10, max_iter=1000).fit(
            scale.transform(df[cols]), df.fall, sample_weight=w)
        fitted[key] = dict(mean=scale.mean_.tolist(), scale=scale.scale_.tolist(),
                           coef=lr.coef_[0].tolist(), intercept=lr.intercept_[0])
    return fitted


def evaluate(df, model, n_days):
    df = df.copy()
    y, w = df.fall.to_numpy(), weights(df)
    avg = lambda v: float(np.average(v, weights=w))
    df['constant'] = model['constant']
    for key, cols in MODELS.items():
        m = model[key]
        logit = ((df[cols].to_numpy()-m['mean'])/m['scale']) @ m['coef'] + m['intercept']
        df[key] = 1 / (1+np.exp(-logit))
    scores = {}
    for key in ['constant', *MODELS]:
        scores[key] = dict(brier=avg((df[key]-y)**2), auc=float(roc_auc_score(y, df[key], sample_weight=w)))
    chosen = df.full >= .5
    mean_errors = {}
    for col in ERRORS:
        a, b = df['base_'+col], df['corrected_'+col]
        df['selected_'+col] = np.where(chosen, b, a)
        mean_errors[col] = [avg(a), avg(b), avg(df['selected_'+col])]
    daily = ((df.full-y)**2 - (df.level-y)**2).groupby(df.day).mean().reindex(range(n_days)).to_numpy()
    rng = np.random.default_rng(107)
    starts = rng.integers(n_days, size=(2000, (n_days+6)//7))
    draws = ((starts[:, :, None] + np.arange(7)) % n_days).reshape(2000, -1)[:, :n_days]
    sampled = daily[draws]
    ci = np.quantile(np.nanmean(sampled[np.isfinite(sampled).any(axis=1)], axis=1), [.025, .975])
    full = scores['full']
    stats = dict(events=len(df), days=df.day.nunique(), positives=int(sum(y)), prevalence=avg(y),
                 selected=int(chosen.sum()), tpr=avg(chosen*y)/avg(y), fpr=avg(chosen*(1-y))/avg(1-y),
                 scores=scores, brier_difference_ci=ci.tolist(), errors=mean_errors,
                 classification_pass=(full['auc']>.5 and full['brier']<min(scores[k]['brier'] for k in ('constant','level'))))
    return df, stats


def run(out):
    for line in (HERE.parent/'e106/sha256.txt').read_text().splitlines():
        sha, name = line.split()
        assert hashlib.sha256((OLD/name).read_bytes()).hexdigest() == sha
    data = C.E.DATA/'data.pkl'
    ref = json.loads((HERE.parent/'e77/summary.json').read_text())
    assert hashlib.sha256(data.read_bytes()).hexdigest() == ref['sha256'][str(data)]
    train, rows = C.E.load_train()
    days = dict(zip(train, rows))
    a = load('fit', train[:140], days)
    out.mkdir(parents=True, exist_ok=False)
    fitted = {d: fit(s) for d, s in a.groupby('direction')}
    (out/'model.json').write_text(json.dumps(fitted, indent=2)+'\n')
    a.to_csv(out/'fit.csv', index=False)
    splits, all_days = C.E.S.base.load_selected(data)
    assert splits['train'] == train and max(train) < min(splits['val'])
    result = {}
    for name, dates in [('confirm', train[140:]), ('winter', splits['val'])]:
        result[name] = {}
        scored = []
        for direction, sample in load(name, dates, all_days).groupby('direction'):
            frame, stats = evaluate(sample, fitted[direction], len(dates))
            result[name][direction] = stats
            scored.append(frame)
        pd.concat(scored).sort_index().to_csv(out/f'{name}.csv', index=False)
        up, down = [result[name][d]['errors'] for d in ('increase', 'decrease')]
        result[name]['calibration_pass'] = bool(
            all(up[k][2] < up[k][0] for k in ERRORS[:2]) and
            all(down[k][2] <= down[k][0] for k in ERRORS) and up['do_3h'][2] <= up['do_3h'][0])
    (out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    smoke()
    if args.out:
        run(args.out)

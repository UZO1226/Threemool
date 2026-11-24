"""Compare whole-trajectory choices on observed DARROW inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/jio/fcsrl_src/results')
METRICS = ['nh4_3h', 'segment_mae', 'do_3h']
EPS = 1e-12


def decisions(gain):
    endpoint = gain[:, 0] > EPS
    joint = endpoint & (gain[:, 1:] >= -EPS).all(axis=1)
    return endpoint, joint


def losses(pred, truth):
    nh = pred[..., 0].mean(axis=2)
    segment = nh[:, 2] - nh[:, 1]
    obs_segment = truth[:, 12, 0] - truth[:, 4, 0]
    do = pred[:, 2, :, 2:].max(axis=2).mean(axis=1)
    return np.column_stack([abs(nh[:, 2]-truth[:, 12, 0]), abs(segment-obs_segment),
                            abs(do-truth[:, 12, 2:].max(axis=1))]), segment


def cross(flag, joint):
    return dict(both=int(sum(flag & joint)), flag_only=int(sum(flag & ~joint)),
                useful_only=int(sum(~flag & joint)), neither=int(sum(~flag & ~joint)))


def score(frame, base, corrected, pred_change, n_days):
    day = frame.day.to_numpy()
    observed = frame.change.to_numpy()
    endpoint, joint = decisions(base-corrected)
    fall, short = observed < -.1, pred_change > observed + EPS
    selected = frame.prob_full.to_numpy() >= .5
    choices = dict(base=np.zeros(len(day), bool), always=np.ones(len(day), bool),
                   e107=selected, observed_fall=fall, endpoint=endpoint, joint=joint)
    dates = np.unique(day)
    rng = np.random.default_rng(108)
    starts = rng.integers(n_days, size=(2000, (n_days+6)//7))
    draws = ((starts[:, :, None]+np.arange(7))%n_days).reshape(2000, -1)[:, :n_days]
    methods = {}
    for name, choose in choices.items():
        err = np.where(choose[:, None], corrected, base)
        per = np.stack([err[day==d].mean(axis=0) for d in dates])
        methods[name] = dict(chosen=int(sum(choose)), mae=dict(zip(METRICS, per.mean(axis=0))))
        if name in ('e107', 'endpoint', 'joint'):
            gain = np.full(n_days, np.nan)
            gain[dates] = [base[day==d, 0].mean() for d in dates] - per[:, 0]
            samples = gain[draws]
            samples = samples[np.isfinite(samples).any(axis=1)]
            methods[name]['gain_ci'] = np.quantile(np.nanmean(samples, axis=1), [.025, .975]).tolist()
    flags = dict(observed_fall=fall, underpredicted_drop=short, fall_and_short=fall & short,
                 e107=selected, endpoint=endpoint)
    out = frame[['date', 'day', 't', 'direction']].copy()
    for key, flag in dict(**flags, joint=joint).items():
        out[key] = flag
    for i, key in enumerate(METRICS):
        out['gain_'+key] = base[:, i]-corrected[:, i]
    stats = dict(n=len(day), days=len(dates), useful=int(joint.sum()), methods=methods,
                 counts={key: cross(flag, joint) for key, flag in flags.items()})
    return out, stats


def verify_scores(frame, params, raw):
    cols = {'level':['nh4'], 'trend':['nh4','trend'], 'flow':['nh4','flow'], 'full':['nh4','trend','flow']}
    for row in frame.itertuples():
        g = raw.loc[row.date]
        nh = g.nh4.iloc[row.t]
        values = dict(nh4=nh, trend=nh-g.nh4.iloc[row.t-4], flow=g.q_in.iloc[row.t])
        for k, val in values.items():
            np.testing.assert_allclose(getattr(row, k), val, rtol=1e-13, atol=1e-13)
        for k, names in cols.items():
            m = params[row.direction][k]
            logit = m['intercept'] + sum((values[f]-u)*c/s for f,u,c,s in
                                         zip(names,m['mean'],m['coef'],m['scale']))
            np.testing.assert_allclose(getattr(row,'prob_'+k), 1/(1+np.exp(-logit)), atol=1e-12, rtol=0)


def smoke():
    eps = np.array([[1.,0.,1.], [1.,-.1,2.], [0.,1.,1.], [-1.,2.,1.]])
    end, joint = decisions(eps)
    np.testing.assert_array_equal(end, [1,1,0,0])
    np.testing.assert_array_equal(joint, [1,0,0,0])
    assert cross(end,joint) == dict(both=1, flag_only=1, useful_only=0, neither=2)


def run(source, out):
    old = HERE.parent/'e107'
    params = json.loads((old/'model.json').read_text())
    assert params == json.loads((source/'model.json').read_text())
    ref = json.loads((source/'summary.json').read_text())
    raw = pd.read_pickle(ROOT/'e50/prepared/data.pkl')['data']
    expected = dict(line.split()[::-1] for line in (HERE.parent/'e106/sha256.txt').read_text().splitlines())
    result, frames = {}, []
    for split in ('confirm','winter'):
        frame = pd.read_csv(source/f'{split}.csv', float_precision='round_trip')
        path = ROOT/f'e106/main/{split}.npz'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected[path.name]
        with np.load(path) as z:
            assert list(z['gains']) == [0,.05]
            np.testing.assert_array_equal(frame.date, z['dates'][z['ids']])
            np.testing.assert_array_equal(frame.day, z['ids'])
            np.testing.assert_array_equal(frame.t, z['t'])
            np.testing.assert_array_equal(frame.direction=='increase', z['dq']>0)
            np.testing.assert_allclose(frame.change,z['truth'][:,12,0]-z['truth'][:,4,0], atol=1e-12)
            base, pred_change = losses(z['preds'][0],z['truth'])
            corrected, _ = losses(z['preds'][1],z['truth'])
            n_days = len(z['dates'])
        verify_scores(frame, params, raw)
        result[split] = {}
        for direction, sub in frame.groupby('direction'):
            i = sub.index
            flags, s = score(sub, base[i], corrected[i], pred_change[i], n_days)
            for k in METRICS:
                actual = [s['methods'][m]['mae'][k] for m in ('base','always','e107')]
                np.testing.assert_allclose(actual, ref[split][direction]['errors'][k], rtol=0, atol=1e-12)
            flags['split'] = split
            frames.append(flags)
            result[split][direction] = s
    out.mkdir(parents=True, exist_ok=False)
    pd.concat(frames).to_csv(out/'choices.csv', index=False)
    (out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, default=ROOT/'e107/corrected')
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    smoke()
    if args.out:
        run(args.source, args.out)

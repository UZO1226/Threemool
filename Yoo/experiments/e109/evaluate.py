"""Fit two identical selectors to different targets and evaluate frozen choices."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'e108'))
import run as oracle

ROOT = oracle.ROOT
FEATURES = ['nh4', 'trend', 'flow']


def weights(frame):
    w = 1/frame.groupby('day')['t'].transform('size').to_numpy()
    return w/w.mean()


def cutoff(scores, w, prevalence):
    order = np.argsort(scores, kind='stable')
    cdf = np.cumsum(w[order])/sum(w)
    i = min(np.searchsorted(cdf, 1-prevalence), len(order)-1)
    return float(scores[order[i]])


def fit(x, y, w):
    scaler = StandardScaler().fit(x, sample_weight=w)
    model = dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist())
    if len(np.unique(y)) == 1:
        return dict(**model, constant=float(y[0]))
    lr = LogisticRegression(C=1., tol=1e-10, max_iter=1000).fit(scaler.transform(x), y, sample_weight=w)
    return dict(**model, coef=lr.coef_[0].tolist(), intercept=lr.intercept_[0])


def predict(model, x):
    if 'constant' in model:
        return np.full(len(x), model['constant'])
    return expit(((x-model['mean'])/model['scale']) @ model['coef'] + model['intercept'])


def load(split, raw, hashes):
    path = ROOT/f'e106/main/{split}.npz'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[path.name]
    df = pd.read_csv(ROOT/f'e107/corrected/{split}.csv', float_precision='round_trip')
    with np.load(path) as z:
        up = z['dq']>0
        np.testing.assert_array_equal(df.date,z['dates'][z['ids']])
        np.testing.assert_array_equal(df.t,z['t'])
        np.testing.assert_array_equal(df.day,z['ids'])
        np.testing.assert_array_equal(df.direction=='increase',up)
        np.testing.assert_array_equal(z['gains'][:2], [0, .05])
        base, _ = oracle.losses(z['preds'][0,up], z['truth'][up])
        corrected, _ = oracle.losses(z['preds'][1,up], z['truth'][up])
        df = df.loc[up].copy()
        np.testing.assert_array_equal(df.fall, z['truth'][up,12,0]-z['truth'][up,4,0]<-.1)
        n_days = len(z['dates'])
    df['useful'] = oracle.decisions(base-corrected)[1]
    for row in df.itertuples():
        g = raw.loc[row.date]
        values = [g.nh4.iloc[row.t],g.nh4.iloc[row.t]-g.nh4.iloc[row.t-4],g.q_in.iloc[row.t]]
        np.testing.assert_allclose([row.nh4,row.trend,row.flow],values,rtol=0,atol=1e-10)
    return df, base, corrected, n_days


def interval(gain, ids, n):
    per = pd.Series(gain).groupby(ids).mean().reindex(range(n)).to_numpy()
    starts = np.random.default_rng(109).integers(n,size=(2000,(n+6)//7))
    draws = ((starts[:,:,None]+np.arange(7))%n).reshape(2000,-1)[:,:n]
    values = per[draws]
    return np.quantile(np.nanmean(values[np.isfinite(values).any(1)],axis=1),[.025,.975]).tolist()


def evaluate(df, base, corrected, n, lock):
    y = df.useful.to_numpy()
    w = weights(df)
    x = df[FEATURES].to_numpy()
    choose = dict(base=np.zeros(len(y),bool), always=np.ones(len(y),bool), oracle=y)
    diagnostics = {}
    for target, model in lock['models'].items():
        score = predict(model,x)
        df['score_'+target] = score
        choose[target] = score>=.5
        choose[target+'_quota'] = score>=lock['cutoff'][target]
        diagnostics[target] = dict(auc=roc_auc_score(y,score,sample_weight=w), low=score.min(), high=score.max())
    np.testing.assert_allclose(df.score_fall, df.prob_full, atol=1e-10,rtol=0)
    results = {}
    for key, mask in choose.items():
        err = np.where(mask[:,None],corrected,base)
        results[key] = dict(selected=int(mask.sum()), tp=int(sum(mask & y)), fp=int(sum(mask & ~y)),
                            mae=np.average(err,axis=0,weights=w).tolist())
        df['choose_'+key] = mask
    direct_err = np.where(choose['useful'][:,None],corrected,base)
    gain_ci = {}
    for key in ['base','fall']:
        other = np.where(choose[key][:,None],corrected,base)
        gain_ci[key] = interval(other[:,0]-direct_err[:,0],df.day.to_numpy(),n)
    dif = np.array(results['useful']['mae'])-results['base']['mae']
    return dict(n=len(y), days=df.date.nunique(), positives=int(y.sum()), prevalence=np.average(y,weights=w),
                diagnostics=diagnostics, brier_direct=np.average((df.score_useful-y)**2,weights=w),
                brier_constant=np.average((lock['prevalence']-y)**2,weights=w), methods=results,
                gain_ci=gain_ci, passed=bool((dif[:2]<0).all() and dif[2]<=0))


def smoke():
    np.testing.assert_allclose(cutoff(np.array([.1,.7,.4]),np.array([1.,2.,1.]),.25),.7)
    x=np.arange(6.).reshape(2,3)
    np.testing.assert_array_equal(predict(fit(x,np.zeros(2),np.ones(2)),x),[0,0])
    oracle.smoke()


def run(out):
    data = ROOT/'e50/prepared/data.pkl'
    sha = json.loads((HERE.parent/'e77/summary.json').read_text())['sha256'][str(data)]
    assert hashlib.sha256(data.read_bytes()).hexdigest()==sha
    raw = pd.read_pickle(data)['data']
    hashes = dict(line.split()[::-1] for line in (HERE.parent/'e106/sha256.txt').read_text().splitlines())
    a, _, _, _ = load('fit',raw,hashes)
    w = weights(a)
    x = a[FEATURES].to_numpy()
    models = {target:fit(x,a[target].to_numpy(),w) for target in ('fall','useful')}
    prev = np.average(a.useful,weights=w)
    thresholds = {k:cutoff(predict(m,x),w,prev) for k,m in models.items()}
    original = json.loads((HERE.parent/'e107/model.json').read_text())['increase']['full']
    for key, val in original.items():
        np.testing.assert_allclose(models['fall'][key],val,rtol=0,atol=1e-10)
    lock = dict(models=models, cutoff=thresholds, prevalence=prev, n=len(a), positives=int(a.useful.sum()),
                dates=[a.date.min(),a.date.max()])
    out.mkdir(parents=True,exist_ok=False)
    (out/'model.json').write_text(json.dumps(lock,indent=2)+'\n')
    ref = json.loads((HERE.parent/'e108/summary.json').read_text())
    truth = pd.read_csv(ROOT/'e108/main/choices.csv')
    result = {}
    for split in ('confirm','winter'):
        df,b,c,n = load(split,raw,hashes)
        assert df.date.min()>lock['dates'][1]
        saved = truth[(truth.split==split)&(truth.direction=='increase')]
        np.testing.assert_array_equal(df[['date','t']],saved[['date','t']])
        np.testing.assert_array_equal(df.useful,saved.joint)
        result[split] = evaluate(df,b,c,n,lock)
        for k,v in [('fall','e107'),('oracle','joint'),('always','always'),('base','base')]:
            prior = ref[split]['increase']['methods'][v]
            curr = result[split]['methods'][k]
            assert curr['selected']==prior['chosen']
            np.testing.assert_allclose(curr['mae'],[prior['mae'][s] for s in oracle.METRICS])
        df.to_csv(out/f'{split}.csv',index=False)
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    smoke()
    if args.out:
        run(args.out)

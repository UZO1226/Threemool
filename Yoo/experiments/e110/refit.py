"""Fit one diagnostic replica and replay both date cohorts it serves."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'e106'))
import calibrate as C
from surrogate_dw import train_gbm


def fold_id(day):
    return (day//7)%5


def smoke():
    blocks = fold_id(np.arange(140))
    for k in range(5):
        assert sum(blocks == k) == 28
        assert sum(blocks != k) == 112
        assert np.all(blocks[(blocks+1)%5 == k] != k)
    C.smoke()


def run(k, root):
    data = C.E.DATA/'data.pkl'
    ref = json.loads((HERE.parent/'e77/summary.json').read_text())
    assert hashlib.sha256(data.read_bytes()).hexdigest() == ref['sha256'][str(data)]
    table = pd.read_pickle(data)['data']
    dates, rows = C.E.load_train()
    days = dict(zip(dates[:140], rows[:140]))
    dates = dates[:140]
    included = [d for j,d in enumerate(dates) if fold_id(j)!=k]
    held = [d for j,d in enumerate(dates) if fold_id(j)==k]
    assert set(included).isdisjoint(held) and len(included)==112 and len(held)==28
    out = root/f'fold{k}'
    out.mkdir(parents=True,exist_ok=False)
    print('Training replica', k, flush=True)
    train_gbm(table, dict(train=included,val=held), out/'model.pkl')
    dyn = C.E.Dynamics(out/'model.pkl')
    for name, selected in [('excluded',held),('included',[d for j,d in enumerate(dates) if (fold_id(j)+1)%5==k])]:
        assert (set(selected).issubset(included)) == (name=='included')
        ids,t,dq,drive,x,e,z = C.cohort(selected,days)
        u = dq>0
        pred = np.stack([C.forecasts(dyn,x[u,0],e[u],z[u],drive[u],g) for g in [0.,.05]])
        assert np.isfinite(pred).all() and dyn.n_members==5
        np.savez_compressed(out/f'{name}.npz', dates=np.array(selected,dtype=str)[ids[u]],
                            t=t[u], truth=x[u], preds=pred)
    (out/'dates.json').write_text(json.dumps(dict(fit=included,held=held),default=str,indent=2)+'\n')
    print('Completed replica',k, flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--fold',type=int,choices=range(5))
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    smoke()
    if args.fold is not None:
        run(args.fold,args.out)

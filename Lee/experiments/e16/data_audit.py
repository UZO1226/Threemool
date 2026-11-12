"""Verify provenance and characterize training/development shift without test evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

p=argparse.ArgumentParser()
p.add_argument('--data',type=Path,required=True)
p.add_argument('--raw',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
expected={'train_val_test_online_dataset.csv':'0a8d06be567d012f5d3791763c9d910c83aac4787d5183a7a505c9b4fc7c9a82',
 'train_val_test_dynamic_influent.csv':'8c22eba0527371792511af966d52641ddf35209f6b25394da387fccd0380c81e'}
hashes={name:hashlib.sha256((a.raw/name).read_bytes()).hexdigest() for name in expected}
assert hashes==expected,'Raw input differs from protocol v1.1'
d=pd.read_pickle(a.data)
assert [len(d['days'][s]) for s in ['train','val','test']]==[188,40,41]
assert max(d['days']['train'])<min(d['days']['val'])<min(d['days']['test'])
raw=pd.read_csv(a.raw/'train_val_test_online_dataset.csv',sep='\t',parse_dates=['DATETIME']).set_index('DATETIME')
cols={'nh4':'NH4','no3':'NO3','do1':'DO_1','do2':'DO_2','do3':'DO_3','q_in':'Q_inf','temp':'Temp'}
stats={}
for split in ['train','val']:
    table=pd.concat([d['data'].loc[day] for day in d['days'][split]])
    assert np.isfinite(table.to_numpy(float)).all()
    for col,source in cols.items():
        np.testing.assert_allclose(table[col],raw.loc[table.index,source],rtol=1e-10,atol=1e-10)
    np.testing.assert_allclose(table.air,raw.loc[table.index,[f'Q_air_{i}' for i in range(1,6)]].sum(1))
    stats[split]={col:{'min':float(table[col].min()),'q05':float(table[col].quantile(.05)),
        'median':float(table[col].median()),'q95':float(table[col].quantile(.95)),
        'max':float(table[col].max())} for col in [*cols,'air','inf_s_nh']}
    stats[split]['violation_pct']={'nh4':float((table.nh4>4).mean()*100),
        'do':float((table[['do1','do2','do3']].max(1)>2).mean()*100)}
train=pd.concat([d['data'].loc[day] for day in d['days']['train']])
val=pd.concat([d['data'].loc[day] for day in d['days']['val']])
outside={col:float(((val[col]<train[col].min())|(val[col]>train[col].max())).mean()*100)
         for col in ['temp','q_in','inf_s_nh','nh4','do1','do2','do3','air']}
result={'raw_hashes_match_protocol':True,'raw_hashes':hashes,'prepared_sha256':hashlib.sha256(a.data.read_bytes()).hexdigest(),
    'prepared_matches_raw_train_val':True,'test_evaluated':False,'statistics':stats,
    'validation_outside_training_range_pct':outside}
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps({'raw_hashes_match_protocol':True,'validation_outside_training_range_pct':outside}),flush=True)

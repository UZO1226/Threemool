"""Choose inner dates from timestamps alone and freeze reference scoring units."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'Lee/code/darrow'))
from surrogate_dw import STATE

p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
p.add_argument('--reference',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();raw=pd.read_pickle(a.data);old=json.loads(a.reference.read_text())
train=raw['days']['train'];indices=set(np.rint(np.linspace(0,len(train)-1,30)).astype(int))
cal=[d for k,d in enumerate(train) if k in indices];fit=[d for k,d in enumerate(train) if k not in indices]
assert len(fit)==158 and len(cal)==30 and not set(fit)&set(cal)
assert set(fit+cal)==set(train) and max(fit+cal)<min(raw['days']['val'])
assert not set(fit+cal)&set(raw['days']['test'])
sd=np.concatenate([raw['data'].loc[d][STATE].to_numpy(float) for d in old['fit_days']]).std(0)
result={'fit_days':fit,'calibration_days':cal,'validation':{'old':old['validation']['old']},
    'scoring_sd':sd.tolist(),'parent_reference_sha256':hashlib.sha256(a.reference.read_bytes()).hexdigest(),
    'split_checks_passed':True,'official_test_evaluated':False,
    'fit_min_temperature':min(float(raw['data'].loc[d].temp.min()) for d in fit),
    'old_fit_min_temperature':min(float(raw['data'].loc[d].temp.min()) for d in old['fit_days'])}
a.out.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({k:result[k] for k in ['split_checks_passed','fit_min_temperature','old_fit_min_temperature']}),flush=True)

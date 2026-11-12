"""Confirm full validation factual replay without the terminal accounting alias."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from quantile_env import FrozenDynamics,evaluate,ROOT,WORKSPACE
from surrogate_dw import GBMDynamics

p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True)
p.add_argument('--freeze',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
a=p.parse_args();dyn=FrozenDynamics(a.freeze);raw=pd.read_pickle(a.data)
s=json.loads((dyn.path/'summary.json').read_text());reference=s['validation'][s['selected']]
new,rows=evaluate(a.data,dyn,replay=True)
np.testing.assert_allclose(new['nh4_viol_pct'],reference['nh4_violation']['pred'],rtol=1e-12)
np.testing.assert_allclose(new['do_viol_pct'],reference['do_violation']['pred'],rtol=1e-12)
old_path=WORKSPACE/dyn.manifest['original_environment_path'];old,_=evaluate(a.data,GBMDynamics(old_path),replay=True)
original=json.loads((ROOT/'Lee/experiments/e16/summary.json').read_text())['validation']['old']
np.testing.assert_allclose(old['nh4_viol_pct'],original['nh4_violation']['pred'],rtol=1e-12)
np.testing.assert_allclose(old['do_viol_pct'],original['do_violation']['pred'],rtol=1e-12)
air=float(np.mean([raw['data'].loc[d].air.iloc[:95].sum()*.25 for d in raw['days']['val']]))
np.testing.assert_allclose(new['air_m3'],air,rtol=1e-12)
np.testing.assert_allclose(old['air_m3'],air,rtol=1e-12)
observed_cost=float(np.mean([((raw['data'].loc[d].nh4.iloc[1:]>4).astype(int)
    +(raw['data'].loc[d][['do1','do2','do3']].iloc[1:].max(axis=1)>2).astype(int)).sum() for d in raw['days']['val']]))
result={'old_corrected_accounting':old,'new_frozen':new,'observed_daily_cost':observed_cost,
    'recorded_air_m3':air,'matches_forecast_audit':True,'test_evaluated':False,
    'scope':'Factual prediction comparison, not an agent improvement'}
a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

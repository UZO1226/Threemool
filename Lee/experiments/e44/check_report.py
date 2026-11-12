"""Reject missing seeds, false air-match flags and smoke performance claims."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from report import summarize

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
def score(nh4,air=100000):
    return {'air_m3':air,'cost':nh4*.95,'nh4_viol_pct':nh4,'do_viol_pct':0.,
        'reward':-air/2250,'days_within_limit_pct':80.}
pairs=[{'policy_seed':seed,'kind':'agent','evaluation_environment':'new','baseline_method':'ppolag_fcsrl',
    'candidate_method':'ppolag_fcsrl_split','freeze_canonical_sha256':'synthetic fixture',
    'baseline_training_environment':'new','candidate_training_environment':'new',
    'test_evaluated':False,'smoke_only':False,'training_steps':401280,'checkpoint_replay_verified':True,
    'baseline_native':score(10),'candidate_native':score(9,95000),
    'candidate_matched':score(8+seed*.1),'matching':{'matched_within_one_percent':True}} for seed in range(5)]
assert summarize(pairs)['meets_prespecified_simulation_criterion']
bad=deepcopy(pairs);bad[0]['candidate_matched']['air_m3']=105000
assert not summarize(bad)['checks']['all_five_seeds_air_matched']
bad=deepcopy(pairs)
for pair in bad:pair['candidate_matched']=score(11)
assert not summarize(bad)['meets_prespecified_simulation_criterion']
for bad in (pairs[:4],[*pairs[:4],pairs[0]]):
    try:summarize(bad)
    except AssertionError:pass
    else:raise AssertionError('Missing or duplicate seed accepted')
bad=deepcopy(pairs);bad[0]['smoke_only']=True
try:summarize(bad)
except AssertionError:pass
else:raise AssertionError('Smoke run accepted as full performance evidence')
result={'synthetic_fixture_only':True,'real_policy_performance_tested':False,
    'improvement_and_regression_distinguished':True,'air_mismatch_independently_rechecked':True,
    'all_five_unique_seeds_required':True,'smoke_runs_rejected':True}
a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

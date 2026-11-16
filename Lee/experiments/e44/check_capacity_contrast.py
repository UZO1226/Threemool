"""Verify reference identity and actual mutual air matching in the contrast."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from capacity_contrast import contrast

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
def rows(method,nh4,air):
    return [{'policy_seed':seed,'kind':'agent','baseline_method':'ppolag_fcsrl','candidate_method':method,
        'evaluation_environment':'new','baseline_training_environment':'new','candidate_training_environment':'new',
        'training_steps':401280,'smoke_only':False,'test_evaluated':False,'checkpoint_replay_verified':True,
        'freeze_canonical_sha256':'synthetic fixture','baseline_checkpoint_hashes':{'model.pt':f'fixture{seed}'},
        'baseline_native':{'air_m3':100000},'matching':{'matched_within_one_percent':True},
        'candidate_matched':{'air_m3':air,'cost':nh4*.95,'nh4_viol_pct':nh4,'do_viol_pct':0,
            'reward':-air/2250,'days_within_limit_pct':80}} for seed in range(5)]
split=rows('ppolag_fcsrl_split',8,99980);capacity=rows('ppolag_fcsrl_capacity',9,100030)
assert contrast(split,capacity)['constraint_identity_benefit_supported_in_simulation']
bad=deepcopy(capacity);bad[0]['baseline_checkpoint_hashes']={'model.pt':'different reference'}
try:contrast(split,bad)
except AssertionError:pass
else:raise AssertionError('Different baseline checkpoint accepted')
left=rows('ppolag_fcsrl_split',8,101000);right=rows('ppolag_fcsrl_capacity',9,99000)
result=contrast(left,right);assert result['checks']['all_variants_match_reference_air']
assert not result['checks']['all_variants_match_each_other_air']
assert not result['constraint_identity_benefit_supported_in_simulation']
a.out.write_text(json.dumps({'synthetic_fixture_only':True,'common_reference_checkpoint_required':True,
    'mutual_air_difference_independently_checked':True,'real_policy_performance_tested':False},indent=2));print('Capacity contrast checks passed',flush=True)

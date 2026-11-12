"""Summarize five policy-seed pairs without treating model draws as seeds."""
import argparse
import json
from pathlib import Path

import numpy as np

KEYS=('air_m3','cost','nh4_viol_pct','do_viol_pct','reward','days_within_limit_pct')


def summarize(pairs):
    assert len(pairs)==5 and sorted(p['policy_seed'] for p in pairs)==list(range(5))
    pairs=sorted(pairs,key=lambda p:p['policy_seed'])
    for key in ('kind','evaluation_environment','baseline_method','candidate_method','freeze_canonical_sha256',
            'baseline_training_environment','candidate_training_environment'):
        assert len({p[key] for p in pairs})==1
    assert all(not p['test_evaluated'] and not p['smoke_only'] and p['training_steps']==401280 for p in pairs)
    assert all(p['checkpoint_replay_verified'] for p in pairs)
    rng=np.random.default_rng(2026);indices=rng.integers(0,5,size=(10000,5))
    results={}
    for mode in ('candidate_native','candidate_matched'):
        if any(p[mode] is None for p in pairs):continue
        rows={}
        for key in KEYS:
            baseline=np.array([p['baseline_native'][key] for p in pairs]);candidate=np.array([p[mode][key] for p in pairs])
            assert np.isfinite(baseline).all() and np.isfinite(candidate).all()
            difference=candidate-baseline;interval=np.quantile(difference[indices].mean(1),[.025,.975])
            rows[key]={'baseline_mean':float(baseline.mean()),'candidate_mean':float(candidate.mean()),
                'baseline_sd':float(baseline.std(ddof=1)),'candidate_sd':float(candidate.std(ddof=1)),
                'baseline_values':baseline.tolist(),'candidate_values':candidate.tolist(),
                'paired_difference':difference.tolist(),'paired_mean_difference':float(difference.mean()),
                'paired_seed_bootstrap95_ci':interval.tolist()}
        results[mode]=rows
    matched=results.get('candidate_matched')
    def independently_matched(pair):
        if not pair['matching'] or not pair['matching']['matched_within_one_percent'] or pair['candidate_matched'] is None:return False
        target=pair['baseline_native']['air_m3'];actual=pair['candidate_matched']['air_m3']
        return (abs(actual-target)<=target*.01 if target>0 else abs(actual)<1e-9)
    all_matched=all(independently_matched(p) for p in pairs)
    checks={'all_five_seeds_air_matched':all_matched,
        'cost_difference_interval_below_zero':matched is not None and matched['cost']['paired_seed_bootstrap95_ci'][1]<0,
        'nh4_difference_interval_below_zero':matched is not None and matched['nh4_viol_pct']['paired_seed_bootstrap95_ci'][1]<0,
        'do_difference_interval_not_above_zero':matched is not None and matched['do_viol_pct']['paired_seed_bootstrap95_ci'][1]<=0}
    return {'kind':pairs[0]['kind'],'evaluation_environment':pairs[0]['evaluation_environment'],
        'baseline_method':pairs[0]['baseline_method'],'candidate_method':pairs[0]['candidate_method'],
        'baseline_training_environment':pairs[0]['baseline_training_environment'],
        'candidate_training_environment':pairs[0]['candidate_training_environment'],
        'freeze_canonical_sha256':pairs[0]['freeze_canonical_sha256'],
        'policy_seeds':list(range(5)),'metrics':results,'checks':checks,
        'meets_prespecified_simulation_criterion':all(checks.values()),'test_evaluated':False,
        'uncertainty_unit':'Five policy training seeds, sample SD ddof1 and10000 paired seed-bootstrap resamples. Model members, noise draws and dates are not independent policy seeds.',
        'limitations':'Exploratory reused validation and frozen surrogate; matched actions differ from native policy; no real-plant causal claim.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pairs',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=summarize([json.loads(path.read_text()) for path in a.pairs])
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result['checks']),flush=True)

"""Contrast split/capacity variants at the same original-FCSRL reference air."""
import argparse
import json
from pathlib import Path

import numpy as np


def contrast(split_pairs,capacity_pairs):
    def indexed(pairs,method):
        assert len(pairs)==5 and sorted(p['policy_seed'] for p in pairs)==list(range(5))
        for p in pairs:
            assert p['kind']=='agent' and p['baseline_method']=='ppolag_fcsrl' and p['candidate_method']==method
            assert p['evaluation_environment']==p['baseline_training_environment']==p['candidate_training_environment']=='new'
            assert p['training_steps']==401280 and not p['smoke_only'] and not p['test_evaluated']
            assert p['checkpoint_replay_verified'] and p['candidate_matched'] is not None
        return {p['policy_seed']:p for p in pairs}
    split=indexed(split_pairs,'ppolag_fcsrl_split');capacity=indexed(capacity_pairs,'ppolag_fcsrl_capacity')
    assert len({p['freeze_canonical_sha256'] for p in split_pairs+capacity_pairs})==1
    reference=[];matched=[];mutual=[]
    for seed in range(5):
        s,c=split[seed],capacity[seed]
        assert s['baseline_checkpoint_hashes']==c['baseline_checkpoint_hashes']
        np.testing.assert_allclose(s['baseline_native']['air_m3'],c['baseline_native']['air_m3'],rtol=1e-12)
        target=s['baseline_native']['air_m3'];reference.append(target)
        actual=[p['candidate_matched']['air_m3'] for p in (s,c)]
        matched.append(all(p['matching']['matched_within_one_percent'] and
            abs(p['candidate_matched']['air_m3']-target)<=max(target*.01,1e-9) for p in (s,c)))
        mutual.append(abs(actual[0]-actual[1])<=max(target*.01,1e-9))
    indices=np.random.default_rng(2026).integers(0,5,size=(10000,5));metrics={}
    for key in ('air_m3','cost','nh4_viol_pct','do_viol_pct','reward','days_within_limit_pct'):
        s=np.array([split[i]['candidate_matched'][key] for i in range(5)])
        c=np.array([capacity[i]['candidate_matched'][key] for i in range(5)])
        assert np.isfinite(s).all() and np.isfinite(c).all()
        difference=s-c
        metrics[key]={'split_values':s.tolist(),'capacity_values':c.tolist(),'split_mean':float(s.mean()),
            'capacity_mean':float(c.mean()),'split_sd':float(s.std(ddof=1)),'capacity_sd':float(c.std(ddof=1)),
            'paired_difference':difference.tolist(),'paired_seed_bootstrap95_ci':np.quantile(difference[indices].mean(1),[.025,.975]).tolist()}
    checks={'all_variants_match_reference_air':all(matched),'all_variants_match_each_other_air':all(mutual),
        'cost_interval_below_zero':metrics['cost']['paired_seed_bootstrap95_ci'][1]<0,
        'nh4_interval_below_zero':metrics['nh4_viol_pct']['paired_seed_bootstrap95_ci'][1]<0,
        'do_interval_not_above_zero':metrics['do_viol_pct']['paired_seed_bootstrap95_ci'][1]<=0}
    return {'policy_seeds':list(range(5)),'reference_fcsrl_air_m3':reference,'metrics':metrics,'checks':checks,
        'constraint_identity_benefit_supported_in_simulation':all(checks.values()),'test_evaluated':False,
        'freeze_canonical_sha256':split[0]['freeze_canonical_sha256'],
        'scope':'Both variants adjusted independently to the same original FCSRL reference air. Five paired policy seeds, exploratory reused validation, no real-plant claim.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split-pairs',type=Path,nargs='+',required=True)
    p.add_argument('--capacity-pairs',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();result=contrast([json.loads(x.read_text()) for x in a.split_pairs],[json.loads(x.read_text()) for x in a.capacity_pairs])
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result['checks']),flush=True)

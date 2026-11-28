"""Independently recompute final metrics and baseline guards from saved arrays."""
import argparse
import json
from pathlib import Path
import numpy as np


def main(root):
    reports=sorted((root/'jobs').glob('*/result.json'));assert len(reports)==16
    verified=[]
    for path in reports:
        r=json.loads(path.read_text());z=np.load(path.parent/'final_rollout.npz')
        st,air,cost=z['states'],z['airs'],z['costs'];nh=st[:,:,0];cs=np.cumsum(nh,axis=1)
        actual=dict(air_m3=(air*.25).sum(1),step_cost=((nh>4)|(cs>380)).sum(1),
                    nh4_excess_mg_h_L=(np.maximum(nh-4,0)*.25).sum(1),nh4_steps=(nh>4).sum(1),
                    over_budget_steps=(cs>380).sum(1),do_steps=(st[:,:,2:].max(2)>2).sum(1))
        np.testing.assert_array_equal(cost,((nh>4)|(cs>380)).astype(float))
        np.testing.assert_allclose(air,z['zones'].sum(2),rtol=0,atol=1e-8)
        plan=np.load(path.parent/'plan_8.npz');control=np.load(root/f"{r['split']}_transfer_control.npz")
        for key,value in actual.items():
            np.testing.assert_allclose(value,plan[key],rtol=0,atol=1e-8)
            np.testing.assert_allclose(value.mean(),r['final'][key],rtol=0,atol=1e-8)
            # Feedback control is saved in target-member order (1,2,3,4,0).
            base=np.roll(control[f'feedback_{key}'].reshape(12,5),1,axis=1).reshape(-1)
            np.testing.assert_allclose(base.mean(),r['baseline'][key],rtol=0,atol=1e-8)
            assert np.all(value<=base+1e-8),(path.parent.name,key)
        short=np.load(path.parent/'plan_2.npz');key='air_m3' if r['mode']=='air' else 'step_cost'
        assert np.all(plan[key]<=short[key]+1e-8)
        verified.append(path.parent.name)
    out=dict(verified_jobs=verified,all_16_passed=True,independent_saved_array_accounting=True,
             per_episode_baseline_guards=True,budget_primary_monotonicity=True)
    (root/'final_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)

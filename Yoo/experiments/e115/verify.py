"""Independently replay the cross-member case and verify decision times from native observations."""
import argparse
import json
from pathlib import Path

import numpy as np
from replay import ROOT, action_to_air, e113, load, read


def audit(source):
    dates=read(ROOT/'plan.json')['winter']
    _,days=e113.base.load_selected(e113.prepared/'data.pkl')
    hashes=read(ROOT/'checksums.json')
    episodes=[(day,m) for day in dates for m in range(5)]
    env=e113.BatchEnv(days,e113.Dynamics(e113.prepared/'surrogate.pkl'))
    e113.base.DeviceConfig().select_device(-1)
    checks=[]
    for seed in range(5):
        archive,agents=load(seed,hashes)
        weighted=archive['weighted']
        due=np.full(200,-1)
        obs=env.reset(episodes)
        for t in range(95):
            if t:
                env.t=t
                env.state=weighted['states'][:,t-1]
                env.prev_air=weighted['air'][:,t-1]
                obs=env.obs()
                prev_nh=(weighted['states'][:,t-2,0] if t>1 else env.x['state'][:,0,0]).astype(np.float32)
                alternative=action_to_air(agents['uniform'].act_det(obs)[:,0])
                due[(due==-1)&(obs[:,0]>=3)&(obs[:,0]>prev_nh)&(alternative-weighted['air'][:,t]>=250)]=t
        with np.load(source/f'online{seed}.npz') as z:
            np.testing.assert_array_equal(z['trigger'],due)
        if seed!=1:
            continue
        idx=dates.index('2021-12-06')
        for step in (0,7):
            obs=env.reset(episodes)
            trace=[]
            for t in range(95):
                policy=agents['uniform' if t==step else 'weighted']
                obs,_,_,_,v=env.step(policy.act_det(obs)[:,0])
                trace.append(np.column_stack([v['air'],env.state[:,0],env.state[:,2:].max(1)]))
            original=np.stack(trace,axis=1)[idx*5:idx*5+5]
            with np.load(source/f'branches{seed}.npz') as z:
                k=z['keys']
                mask=(k[:,0]==idx)&(k[:,1]==2)&(k[:,3]==step)
                np.testing.assert_array_equal(k[mask,4],np.arange(5))
                np.testing.assert_allclose(original,z['trace'][mask,:,:3],atol=2e-6,rtol=0)
            checks.append(dict(day=dates[idx],seed=seed,step=step,members=5))
    print(json.dumps(dict(native_trigger_episodes=1000,sklearn_replay=checks),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    audit(p.parse_args().source)

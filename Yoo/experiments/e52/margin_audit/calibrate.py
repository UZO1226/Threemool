"""기록 공기량 재생에서 e52/e53 여유 선정의 수치를 재검산한다. 정책 학습 없음."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import sklearn

HERE=Path(__file__).resolve().parent
EXP=HERE.parents[1]
sys.path.insert(0,str(EXP/'e45'))
from evaluate import load_selected
from darrow_env import BatchEnv,Dynamics,all_episodes,action_to_air
from common import sha,save

RESULTS=Path('/home/jio/fcsrl_src/results')


def run(a):
    assert sklearn.__version__=='1.7.2'
    a.out.mkdir(parents=True,exist_ok=False)
    prepared=RESULTS/'e30/prepared'
    model=RESULTS/'e35/zone_fix/corrected_surrogate.pkl'
    splits,days=load_selected(prepared/'data.pkl')
    dyn=Dynamics(model)
    hashes={str(p):sha(p) for p in [prepared/'data.pkl',model,Path(__file__)]}
    output=dict(created_at=datetime.now(timezone.utc).isoformat(),inputs=hashes,
                source_commit='58ab2812d545731e1b4dd413d414f971eb39c2ac',
                test_evaluated=False,weight_updates=0,selection='delta remains 0.45; no retuning',splits={})
    start=time.monotonic()
    for split in ['train','val']:
        env=BatchEnv(days,dyn)
        episodes=all_episodes(splits[split],5)
        env.reset(episodes)
        states=[];sds=[]
        for t in range(95):
            action=env.historical_action()
            air=action_to_air(action)
            zone=air[:,None]*env.x['frac'][:,t]
            x=np.column_stack([env.state,env.x['exo'][:,t],zone/1000.,env.prev_zone/1000.])
            sd=np.stack([m[0].predict(x) for m in dyn.models]).std(0)
            _,_,_,done,_=env.step(action)
            states.append(env.state[:,0].copy());sds.append(sd)
        assert done
        states=np.stack(states,1);sds=np.stack(sds,1)
        actual=np.stack([days[d]['state'][1:,0] for d in splits[split]])
        actual_cost=float((actual>4).sum(1).mean())
        cost=lambda margin:float((states+margin>4).sum(1).mean())
        grid={f'{delta:.2f}':cost(delta) for delta in np.arange(101)/100}
        fixed=(states+.45>4);adaptive=(states+10*sds>4)
        output['splits'][split]=dict(n_days=len(splits[split]),actual_cost=actual_cost,plain_cost=cost(0),
            fixed_cost=cost(.45),adaptive_cost=cost(10*sds),mean_sd=float(sds.mean()),
            fixed_vs_adaptive_event_disagreement_pct=float(100*(fixed!=adaptive).mean()),
            both_count=int((fixed&adaptive).sum()),only_fixed_count=int((fixed&~adaptive).sum()),
            only_adaptive_count=int((~fixed&adaptive).sum()),delta_grid=grid,
            nearest_train_delta=min(grid,key=lambda d:abs(grid[d]-actual_cost)) if split=='train' else None)
        np.savez_compressed(a.out/f'{split}_recorded_replay.npz',nh4=states,sd=sds,
                            day=np.array([d for d,_ in episodes]),member=np.array([m for _,m in episodes]))
        print(split,{k:v for k,v in output['splits'][split].items() if k!='delta_grid'},flush=True)
    assert all(sha(p)==v for p,v in hashes.items())
    output['elapsed_seconds']=time.monotonic()-start
    output['hashes_unchanged']=True
    save(a.out/'calibration.json',output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())

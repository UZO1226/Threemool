"""훈련 날짜에서 초기 정책과 기록 운전 재생의 두 한도를 점검한다."""
import argparse
from pathlib import Path
import numpy as np
import torch
from dual import make_agent,config,save
from train import evaluate
from darrow_env import BatchEnv,Dynamics,load_days
from fcsrl.utils import DeviceConfig


def main(prepared,out):
    DeviceConfig().select_device(-1)
    torch.set_num_threads(1)
    splits,days=load_days(prepared/'data.pkl')
    env=BatchEnv(days,Dynamics(prepared/'surrogate.pkl'))
    agent=make_agent('ppo',0,prepared,config())
    initial,_=evaluate(env,splits['train'],lambda obs:agent.act_det(obs)[:,0])
    rows=[{'reference':'initial_BC_policy',**initial}]
    for scale in (.95,.98,1.):
        result,_=evaluate(env,splits['train'],lambda _:np.clip((env.historical_action()+1)*scale-1,-1,1))
        rows.append({'reference':f'historical_scale_{scale}',**result})
    save(out,{'split':'train','limits':[6.,6.],'references':rows,
         'initial_policy_feasible':initial['both_mean_limits'],
         'any_reference_feasible':any(r['both_mean_limits'] for r in rows),
         'note':'Historical actions depend on recorded operations, not solely on simulated policy observations. Failure to find a witness is not proof of infeasibility.'})
    print(rows,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.prepared,a.out)

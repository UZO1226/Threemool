"""Frozen stochastic e40 environment, preserving the DARROW control contract."""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[3];WORKSPACE=ROOT.parent
sys.path.insert(0,str(ROOT/'Lee/experiments/e38'))
from quantile_dynamics import QuantileDynamics,interpolate,features
from env_dw import VecAerationEnv,OBS_DIM,T_STEPS,AIR_MAX,NH4_MAX,DO_MAX
from surrogate_dw import ZONES


def install_training_fallback(env,data_path):
    import pandas as pd
    training_days=pd.read_pickle(data_path)['days']['train']
    medians=np.median(np.concatenate([env.table.loc[d][ZONES].to_numpy(float) for d in training_days]),axis=0)
    env.fractions=medians/medians.sum()
    for arrays in env.day_arrays.values():
        total=arrays['zone'].sum(-1,keepdims=True)
        arrays['frac']=np.where(total>1,arrays['zone']/np.maximum(total,1),env.fractions[None])


class FrozenDynamics:
    def __init__(self,freeze_path):
        self.freeze_path=Path(freeze_path)
        self.manifest=json.loads(self.freeze_path.read_text())
        for name,digest in {**self.manifest['frozen_files_sha256'],
                **self.manifest.get('frozen_inference_sources_sha256',{})}.items():
            p=Path(name);p=p if p.is_absolute() else WORKSPACE/p
            assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,f'Frozen input changed: {name}'
        os.environ['DARROW_FAST_DYNAMICS']='1'
        self.path=WORKSPACE/self.manifest['environment_run']
        self.model=QuantileDynamics(self.path);self.n_members=5;self.hi=self.model.hi
        self.bank=np.load(self.path/'pit_bank.npz')['pit'];self.bank.setflags(write=False)
        self.alpha=self.manifest['alpha'];assert self.alpha==.5
        assert self.manifest['pit_block_steps']==95 and self.manifest['cost_limit']==12

    def noise(self,member,day,draw=None,training_key=None):
        if draw is not None:
            key=f'e38:{day}:{member}:{draw}'
        else:
            assert training_key is not None;key=f'e42train:{training_key}:{day}:{member}'
        seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],'little')
        rng=np.random.default_rng(seed)
        start=rng.integers(0,len(self.bank[member])//95,size=1)*95+rng.integers(0,1,size=1)
        values=self.bank[member,start[0]+np.arange(95)].copy();values[:,1]=.5
        return values

    def step_with_noise(self,state,exo,zone,previous_zone,members,u):
        x=features(state,exo,zone,previous_zone)
        sampled=interpolate(self.model.quantiles(x,members),u)
        return np.clip(self.alpha*state+(1-self.alpha)*sampled,0,self.hi)


class FrozenQuantileEnv(VecAerationEnv):
    def __init__(self,data_path,dynamics,split='train',n_env=8,seed=0):
        if split not in ('train','val'):raise ValueError('Final test split is reserved')
        super().__init__(data_path,None,split=split,n_env=n_env,seed=seed,dynamics=dynamics)
        # Historical prepared metadata uses the full table. Recompute this
        # fallback from training only without changing the canonical pickle.
        install_training_fallback(self,data_path)
        self.noise_seed=int(seed);self.episode_count=np.full(n_env,-1,dtype=int)
        self.u=np.zeros((n_env,95,5));self.draw=np.full(n_env,-1,dtype=int)
        self.fixed_draws=None

    def _refresh_noise(self,i):
        draw=None if self.draw[i]<0 else int(self.draw[i])
        key=f'{self.noise_seed}:{i}:{self.episode_count[i]}'
        self.u[i]=self.dyn.noise(int(self.member[i]),self.day[i],draw=draw,training_key=key)

    def _pick(self,i):
        k=self._k
        super()._pick(i)
        self.episode_count[i]+=1
        self.draw[i]=-1 if self.fixed_draws is None else self.fixed_draws[k%len(self.fixed_draws)]
        self._refresh_noise(i)

    def set_member(self,member):
        assert np.all(self.t==0),'Member changes are restricted to episode boundaries'
        super().set_member(member)
        for i in range(self.n):self._refresh_noise(i)

    def step(self,action):
        action=np.clip(np.asarray(action,float).reshape(self.n),-1,1)
        air=(action+1)/2*AIR_MAX
        exo=np.stack([self.day_arrays[self.day[i]]['exo'][self.t[i]] for i in range(self.n)])
        frac=np.stack([self.day_arrays[self.day[i]]['frac'][self.t[i]] for i in range(self.n)])
        zone=air[:,None]*frac
        new=self.dyn.step_with_noise(self.state,exo,zone,self.prev_zone,self.member,self.u[np.arange(self.n),self.t])
        self.state,self.prev_air,self.prev_zone=new,air,zone
        self.t+=1
        nh4=(new[:,0]>NH4_MAX).astype(float);do=(new[:,2:].max(-1)>DO_MAX).astype(float)
        truncated=self.t>=T_STEPS;terminal_obs=self._obs()
        reward=-air/AIR_MAX
        info={'air':air.copy(),'v_nh4':nh4,'v_do':do,'done_day':self.day.copy()}
        for i in np.flatnonzero(truncated):self._pick(i)
        return self._obs(),reward,nh4+do,truncated,terminal_obs,info


class TerminalSafeOriginalEnv(VecAerationEnv):
    """Preserve executed air/reward across the historical reset alias."""
    def __init__(self,data_path,dynamics,split='train',n_env=8,seed=0):
        if split not in ('train','val'):raise ValueError('Final test split is reserved')
        super().__init__(data_path,None,split=split,n_env=n_env,seed=seed,dynamics=dynamics)
        install_training_fallback(self,data_path)

    def step(self,action):
        executed=(np.clip(np.asarray(action,float).reshape(self.n),-1,1)+1)/2*AIR_MAX
        obs,_,cost,done,terminal,info=super().step(action)
        info['air']=executed.copy()
        return obs,-executed/AIR_MAX,cost,done,terminal,info


def evaluate(data_path,dynamics,act_batch=None,replay=False,draws=8,split='val',air_offset=0):
    if split not in ('train','val'):raise ValueError('Final test split is reserved')
    import pandas as pd
    days=pd.read_pickle(data_path)['days'][split]
    frozen=isinstance(dynamics,FrozenDynamics)
    if not frozen:draws=1
    cases=[(d,m,r) for m in range(5) for r in range(draws) for d in days]
    cls=FrozenQuantileEnv if frozen else TerminalSafeOriginalEnv
    env=cls(data_path,dynamics,split=split,n_env=len(cases),seed=0)
    env.fixed=[(d,m) for d,m,_ in cases]
    if frozen:env.fixed_draws=[r for _,_,r in cases]
    obs=env.reset();air=np.zeros(env.n);nh4=np.zeros(env.n);do=np.zeros(env.n)
    for t in range(95):
        if replay:
            recorded=np.array([env.day_arrays[env.day[i]]['air'][env.t[i]] for i in range(env.n)])
            action=recorded/AIR_MAX*2-1
        else:action=np.asarray(act_batch(obs)).reshape(env.n)+air_offset
        obs,reward,cost,truncated,terminal,info=env.step(action)
        np.testing.assert_allclose(cost,info['v_nh4']+info['v_do'])
        air+=info['air']*.25;nh4+=info['v_nh4'];do+=info['v_do']
        if t==94:assert truncated.all()
    rows=[{'day':d,'member':m,'noise_draw':r,'air_m3':float(air[i]),'cost':float(nh4[i]+do[i]),
        'nh4_viol_pct':float(nh4[i]/95*100),'do_viol_pct':float(do[i]/95*100)} for i,(d,m,r) in enumerate(cases)]
    summary={key:float(np.mean([row[key] for row in rows])) for key in ('air_m3','cost','nh4_viol_pct','do_viol_pct')}
    summary.update({'reward':-summary['air_m3']/(.25*AIR_MAX),'n_episodes':len(rows),
        'days_within_limit_pct':float(np.mean([row['cost']<=12 for row in rows])*100)})
    return summary,rows

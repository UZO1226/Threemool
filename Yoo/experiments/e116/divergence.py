"""Trace the 2021-12-06 e115 intervention and attribute subsequent policy outputs."""
import argparse
from itertools import permutations
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'e115'))
import replay
from darrow_env import air_to_action

DAY = '2021-12-06'
GROUPS = {'nh4': [0], 'no3': [1], 'do': [2,3,4], 'previous_air': [10]}
GROUP_COLS = ['via_'+k for k in GROUPS]


def trajectory(env, agents, pulse, fixed=None):
    obs = env.reset([(DAY,m) for m in range(5)])
    inputs, flows, states = [], [], []
    for t in range(95):
        inputs.append(obs)
        if t == pulse:
            act = replay.action(agents['uniform'],obs)
        elif fixed is None:
            act = replay.action(agents['weighted'],obs)
        else:
            act = air_to_action(fixed[t])
        obs, _, _, done, info = env.step(act)
        flows.append(info['air'])
        states.append(env.state.copy())
    assert done
    return np.stack(inputs),np.stack(flows),np.stack(states)


def decompose(values):
    accum = np.zeros((*values.shape[1:],4))
    for order in permutations(range(4)):
        mask = 0
        for group in order:
            nxt = mask | (1<<group)
            accum[...,group] += values[nxt]-values[mask]
            mask = nxt
    return accum/24


def attribution(agent, a, b):
    np.testing.assert_array_equal(a[...,5:10],b[...,5:10])
    values = []
    for mask in range(16):
        x = a.copy()
        for bit,cols in enumerate(GROUPS.values()):
            if mask & (1<<bit):
                x[...,cols] = b[...,cols]
        obs = x.reshape(-1,11)
        y = np.concatenate([replay.action_to_air(replay.action(agent,obs[i:i+200])) for i in range(0,len(obs),200)])
        values.append(y.reshape(a.shape[:2]))
    v = np.stack(values)
    phi = decompose(v)
    np.testing.assert_allclose(phi.sum(-1),v[-1]-v[0],atol=1e-8)
    return phi


def smoke():
    # Two interacting inputs split their joint term equally, two dummy groups contribute zero.
    v = np.array([3*(mask & 1) + 2*((mask>>1)&1) + 6*((mask & 3)==3) for mask in range(16)])
    np.testing.assert_allclose(decompose(v),[6,5,0,0])


def first(mask,clock):
    hit = np.flatnonzero(mask)
    return int(clock[hit[0]]) if len(hit) else None


def run(out, original):
    smoke()
    e = replay.e113
    e.base.DeviceConfig().select_device(-1)
    plan = replay.read(replay.ROOT/'plan.json')
    for f,h in plan['input_sha256'].items():
        replay.verify(e.prepared/f,h)
    dates = plan['winter']
    d = dates.index(DAY)
    splits,days = e.base.load_selected(e.prepared/'data.pkl')
    assert splits['val']==dates
    archive,agents = replay.load(1,replay.read(replay.ROOT/'checksums.json'))
    dyn = (e.Dynamics if original else e.FastDynamics)(e.prepared/'surrogate.pkl')
    env = e.BatchEnv(days,dyn)
    a_obs,a_air,a_state = trajectory(env,agents,None)
    for ours,field in ((a_air,'air'),(a_state,'states')):
        np.testing.assert_allclose(ours.swapaxes(0,1),archive['weighted'][field][5*d:5*d+5],atol=2e-6,rtol=0)
    day_rows, times, series = [], [], []
    with np.load(replay.ROOT.parents[1]/'e115/main/branches1.npz') as z:
        keys, hist = z['keys'],z['trace']
    ref = pd.read_csv(HERE.parent/'e115/branches.csv')
    for pulse in (0,7):
        b_obs,b_air,b_state = trajectory(env,agents,pulse)
        _,c_air,c_state = trajectory(env,agents,pulse,a_air)
        selection = (keys[:,0]==d)&(keys[:,1]==2)&(keys[:,3]==pulse)
        np.testing.assert_array_equal(keys[selection,4],np.arange(5))
        np.testing.assert_allclose(np.stack([b_air,b_state[:,:,0],b_state[:,:,2:].max(2)],axis=-1).swapaxes(0,1),
                                   hist[selection,:,:3],atol=2e-6,rtol=0)
        np.testing.assert_allclose(c_air[pulse],b_air[pulse],atol=2e-6,rtol=0)
        np.testing.assert_allclose(c_air[np.arange(95)!=pulse],a_air[np.arange(95)!=pulse],atol=1e-8,rtol=0)
        np.testing.assert_allclose(c_state[:pulse+1],b_state[:pulse+1],atol=2e-6,rtol=0)
        phi = attribution(agents['weighted'],a_obs,b_obs)
        phi[pulse]=0
        direct=np.zeros_like(a_air)
        direct[pulse]=b_air[pulse]-a_air[pulse]
        delta=b_air-a_air
        np.testing.assert_allclose(phi.sum(-1)+direct,delta,atol=.05,rtol=0)
        for m in range(5):
            frame=pd.DataFrame(dict(pulse=pulse,member=m,t=np.arange(95),start_min=np.arange(95)*15,
                                    end_min=(np.arange(95)+1)*15,block=np.arange(95)//24))
            for name,q,s in [('a',a_air,a_state),('b',b_air,b_state),('c',c_air,c_state)]:
                frame[name+'_air']=q[:,m]
                frame[name+'_nh4']=s[:,m,0]
                frame[name+'_maxdo']=s[:,m,2:].max(1)
            frame['direct']=direct[:,m]
            for i,col in enumerate(GROUP_COLS):
                frame[col]=phi[:,m,i]
            for k in ['air','nh4','maxdo']:
                frame['delta_'+k]=frame['b_'+k]-frame['a_'+k]
            series.append(frame)
            for block,sub in [('day',frame),*((str(k),g) for k,g in frame.groupby('block'))]:
                row=dict(pulse=pulse,member=m,block=block)
                for k in ['a_air','b_air','delta_air','direct',*GROUP_COLS]:
                    row[k]=sub[k].sum()/4
                for name in ('a','b','c'):
                    nh=sub[name+'_nh4']
                    row[name+'_nh_steps']=int((nh>4).sum())
                    row[name+'_do_steps']=int((sub[name+'_maxdo']>2).sum())
                    row[name+'_excess']=(nh-4).clip(lower=0).sum()/4
                day_rows.append(row)
        frames=pd.concat(series[-5:])
        contrast=frames.query('member==3').set_index('t').select_dtypes('number')-frames.query('member!=3').groupby('t').mean(numeric_only=True)
        for k,th,after in [('delta_nh4',.1,True),('delta_maxdo',.1,True),('delta_air',250,False)]:
            mask=(np.abs(contrast[k].values)>=th)&(np.arange(95) >= pulse+int(not after))
            clock=np.arange(95)*15+15*int(after)
            when=first(mask,clock)
            times.append(dict(pulse=pulse,metric=k,threshold=th,clock_min=when,
                              signed_value=contrast.loc[np.flatnonzero(mask)[0],k] if when is not None else None))
    frame=pd.concat(series,ignore_index=True)
    results=pd.DataFrame(day_rows)
    for (pulse,m),g in results.groupby(['pulse','member']):
        day=g[g.block=='day'].iloc[0]
        cols=['delta_air','direct',*GROUP_COLS,'a_nh_steps','b_nh_steps']
        np.testing.assert_allclose(g[g.block!='day'][cols].sum(),day[cols].to_numpy(float),atol=1e-7)
        r=ref.query("day==@DAY and seed==1 and origin==2 and member==@m and t==@pulse").iloc[0]
        np.testing.assert_allclose([day.delta_air,day.b_nh_steps-day.a_nh_steps,day.b_do_steps-day.a_do_steps],
                                   r[['air','nh','do']].to_numpy(float),atol=.05,rtol=0)
    out.mkdir(parents=True,exist_ok=False)
    frame.to_csv(out/'paths.csv',index=False)
    results.to_csv(out/'blocks.csv',index=False)
    pd.DataFrame(times).to_csv(out/'onsets.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('--sklearn',action='store_true')
    args=p.parse_args()
    run(args.out,args.sklearn)

"""Replay frozen FCSRL policies with a single risk-gated aeration reduction."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from alarms import ARMS, HERE, ew, probability
sys.path.insert(0,str(HERE.parent/"e144"))
import choices

METRICS = ["air_m3","nh4_min","excess","do_min"]


def sensor_vector(current, previous, hour_ago):
    order = [0,2,3,4,1]  # Environment order is NH4, NO3, DO1, DO2, DO3.
    return np.column_stack((current[:,order], current[:,order]-previous[:,order],
                            current[:,order]-hour_ago[:,order]))


def starts(env, policy, air, state, models):
    size = len(air)
    first = {a:np.full(size,-1) for a in ("ungated",*ARMS)}
    command = {a:np.zeros(size) for a in first}
    initial = env.state.copy()
    past = np.concatenate((initial[:,None,:], state),axis=1).astype(float)
    for t in range(4,92):
        env.t = t
        env.state = state[:,t-1].copy()
        env.prev_air = air[:,t-1].copy()
        env.prev_zone = env.prev_air[:,None]*env.x["frac"][:,t-1]
        actions,win,gap,_ = choices.proposals(policy,env.obs())
        flows = choices.old.action_to_air(actions)
        chosen = flows[np.arange(size),win]
        np.testing.assert_allclose(chosen,air[:,t],rtol=0,atol=.002)
        eligible = (past[:,t,0]<=4)&(gap[:,0]<=.025+1e-12)&(chosen-flows[:,0]>=250)
        x = sensor_vector(past[:,t],past[:,t-1],past[:,t-4])
        for a in first:
            allowed = eligible & (first[a]<0)
            if a != "ungated":
                allowed &= probability(x,models[a])<=models[a]["cutoff"]
            first[a][allowed] = t
            command[a][allowed] = actions[allowed,0]
    return first,command


def replay(env, episodes, policy, original_air, original_states, first, command):
    obs = env.reset(episodes)
    air, states = [], []
    for t in range(env.T):
        action = policy.act_det(obs)[:,0].copy()
        changed = first==t
        action[changed] = command[changed]
        obs,_,_,_,info = env.step(action)
        untouched = (first<0)|(first>t)
        np.testing.assert_array_equal(info["air"][untouched],original_air[untouched,t])
        np.testing.assert_array_equal(env.state[untouched],original_states[untouched,t])
        if changed.any():
            assert (original_air[changed,t]-info["air"][changed]>=250-0.002).all()
        air.append(info["air"])
        states.append(env.state.copy())
    return choices.metrics(np.stack(air,axis=1),np.stack(states,axis=1))


def run(n, seed, model_file, out):
    torch.set_num_threads(1)
    choices.base.DeviceConfig().select_device(-1)
    env,_,episodes,policy,air,state = choices.load(n,seed)
    config = json.loads(model_file.read_text())
    assert config["columns"] == [c+s for s in ("","_d15","_d60") for c in ew.SENSORS]
    # Check the online vector against the same pandas extraction used during fitting.
    sample = np.vstack((env.state[0],state[0,:4])).astype(float)
    ref = ew.features(pd.DataFrame(sample[:,[0,2,3,4,1]],columns=ew.SENSORS)).iloc[-1]
    np.testing.assert_array_equal(sensor_vector(sample[4:], sample[3:4], sample[:1])[0],ref)
    first,command = starts(env,policy,air,state,config["models"][str(n)])
    metrics = {"native":choices.metrics(air,state)}
    for a in first:
        metrics[a] = replay(env,episodes,policy,air,state,first[a],command[a])
        for b in metrics:
            if b not in first or a == b:
                continue
            same = first[a]==first[b]
            np.testing.assert_array_equal(metrics[a][same],metrics[b][same])
    parts = []
    for a,y in metrics.items():
        table = pd.DataFrame(y,columns=METRICS).assign(arm=a,prefix=n,seed=seed,
                    day=[str(d) for d,_ in episodes],member=[m for _,m in episodes],
                    first=first[a] if a in first else -1)
        parts.append(table)
    out.mkdir(parents=True,exist_ok=True)
    pd.concat(parts,ignore_index=True).to_csv(out/f"p{n}_s{seed}.csv",index=False)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("prefix",type=int)
    p.add_argument("seed",type=int)
    p.add_argument("models",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.prefix,a.seed,a.models,a.out)

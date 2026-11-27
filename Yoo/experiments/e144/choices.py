"""Branch actual e135/e139 decisions from identical states across GBM members."""
import argparse
import copy
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e139"))
import folds

old, study, trial, base = folds.old, folds.study, folds.trial, folds.base
ROOT = trial.PREP.parents[1]
TOLS = (0., .025, .05)
PAIR_COLS = ["episode","t","choice","alternative","alternative_air","gap","net","different",
             "nh_gap_min","do_gap_min","old_net"]


def check():
    for folder, name, sha in [
        (ROOT/"e139/main","SHA256SUMS","1c07a2da951ffb0729d0cd4ac80d203b7ce5faabe35f9980eb22f6eec1698ad1"),
        (ROOT/"e135/main","SHA256SUMS","3a5cbbbfdbca2ab487a1cc6a50138d31e4b6a2b063c3234e7b503e8782cc474c"),
        (old.source,"tables/SHA256SUMS","297268876814ab83d214f6866d5ae725eaa9b792ae485a1e0960682eb88122c4")]:
        assert base.sha(folder/name)==sha
        subprocess.run(["sha256sum","--quiet","-c",name],cwd=folder,check=True)
    for name,sha in trial.read(trial.REF/"manifest.json")["input_sha256"].items():
        assert base.sha(trial.PREP/name)==sha


def load(n, seed):
    if n==188:
        prep = trial.PREP
        splits, days = base.load_selected(prep/"data.pkl")
        folder = ROOT/f"e135/main/seed_{seed}"
        rate_path = folder/"rate.pkl"
        agent = old.load_agent("fcsrl",seed)
    else:
        fold, prep = folds.paths(ROOT/"e139/main",n)
        splits, days = folds.load_days(prep/"data.pkl")
        folder = fold/f"evaluated/seed_{seed}"
        rate_path = folder/"risk.pkl"
        agent = trial.factory("fcsrl",seed,prep,base.common.config(66))
        agent.load(fold/f"fcsrl/seed_{seed}/model_final")
    dyn = trial.fast.FastDynamics(prep/"surrogate.pkl")
    older = None
    if n!=108:
        _, previous = folds.paths(ROOT/"e139/main",n-40)
        older = trial.fast.FastDynamics(previous/"surrogate.pkl")
    env = trial.fast.BatchEnv(days,dyn)
    episodes = old.all_episodes(splits["val"],5)
    env.reset(episodes)
    with rate_path.open("rb") as stream:
        policy = study.LocalPolicy(agent,study.FastRate(pickle.load(stream)),0)
    with np.load(folder/"base_risk_1000.npz") as z:
        return env, older, episodes, policy, z["air"], z["states"]


def proposals(policy, obs):
    actions = np.clip(policy.agent.act_det(obs)+study.OFFSETS,-1.,1.)
    features = np.column_stack((np.repeat(obs,5,axis=0),actions.ravel()))
    rates = policy.risk.predict(features).reshape(len(obs),5,2)
    total = rates.sum(axis=2)
    win = total.argmin(axis=1)
    idx = np.arange(len(win))
    gap = np.maximum(np.abs(rates-rates[idx,win,None]).max(axis=2), total-total[idx,win,None])
    unique = np.ones(actions.shape,dtype=bool)
    for j in range(1,5):
        unique[:,j] = (actions[:,j,None]!=actions[:,:j]).all(axis=1)
    unique[idx,win] = False
    return actions, win, gap, unique


def branch(env, dyn, ids, actions, policy):
    # One batch row per distinct starting decision, candidate and evaluator model.
    fork = copy.copy(env)
    fork.dyn = dyn
    parents = ids.repeat(5)
    fork.member = np.tile(np.arange(5),len(ids))
    fork.state = env.state[parents].copy()
    fork.prev_zone = env.prev_zone[parents].copy()
    fork.prev_air = env.prev_air[parents].copy()
    t = env.t
    fork.x = {k:env.x[k][parents,t:t+5] for k in ("exo","frac")}
    fork.t = 0
    nh = np.zeros(len(parents),int)
    do = np.zeros(len(parents),int)
    a = actions.repeat(5)
    for j in range(4):
        obs,_,_,_,info = fork.step(a)
        nh += info["v_nh4"].astype(int)
        do += info["v_do"].astype(int)
        if j<3:
            a = policy.act_det(obs)[:,0]
    return nh.reshape(-1,5),do.reshape(-1,5)


def metrics(air, states):
    nh = states[:,:,0]
    return np.column_stack((air.sum(1)/4,15*(nh>4).sum(1),np.maximum(nh-4,0).sum(1)/4,
                            15*(states[:,:,2:].max(2)>2).sum(1)))


def inspect(env, older, policy, air, states, actor_followup=False):
    followup = policy.agent if actor_followup else policy
    counts = []
    records = []
    for t in range(0,92,4):
        env.t = t
        env.state = states[:,t-1].copy() if t else env.x["state"][:,0].copy()
        env.prev_air = air[:,t-1].copy() if t else env.x["air"][:,0].copy()
        env.prev_zone = env.prev_air[:,None]*env.x["frac"][:,t-1] if t else env.x["zone"][:,0].copy()
        actions,win,gap,unique = proposals(policy,env.obs())
        ids = np.arange(len(win))
        np.testing.assert_array_equal(old.action_to_air(actions[ids,win]),air[:,t])
        high = env.state[:,0]>4
        counts.extend((i,t,bool(v)) for i,v in enumerate(high))
        viable = (gap<=max(TOLS)+1e-12)&unique&high[:,None]
        rows, cols = np.nonzero(viable)
        if len(rows)==0:
            continue
        parents,inverse = np.unique(rows,return_inverse=True)
        joint = np.r_[parents,rows]
        moves = np.r_[actions[parents,win[parents]],actions[rows,cols]]
        nh,do = branch(env,env.dyn,joint,moves,followup)
        k = len(parents)
        # Check the incumbent continuation on the original source member against the archived four-step result.
        orig = env.member[parents]
        if not actor_followup:
            np.testing.assert_array_equal(nh[:k][np.arange(k),orig],(states[parents,t:t+4,0]>4).sum(1))
            np.testing.assert_array_equal(do[:k][np.arange(k),orig],(states[parents,t:t+4,2:].max(2)>2).sum(1))
        def votes(N):
            return (N[:k][inverse]==4).astype(int)-(N[k:]==4).astype(int)
        v = votes(nh)
        old_net = np.full(len(rows),np.nan)
        if older is not None:
            old_nh,_ = branch(env,older,joint,moves,followup)
            old_net = votes(old_nh).sum(1)
        records.extend(zip(rows,np.full(len(rows),t),win[rows],cols,old.action_to_air(actions[rows,cols]),
                           gap[rows,cols],v.sum(1),(v!=0).sum(1),3*(nh[k:]-nh[:k][inverse]).sum(1),
                           3*(do[k:]-do[:k][inverse]).sum(1),old_net))
    return pd.DataFrame(counts,columns=["episode","t","high"]),pd.DataFrame(records,columns=PAIR_COLS)


def run(n,seed,out,actor_followup=False):
    torch.set_num_threads(1)
    base.DeviceConfig().select_device(-1)
    env,older,episodes,policy,air,states = load(n,seed)
    counts,pairs = inspect(env,older,policy,air,states,actor_followup)
    assert len(counts)==23*200
    folder = out/f"p{n}_s{seed}"
    folder.mkdir()
    for name,table in (("anchors",counts),("pairs",pairs)):
        save_table(folder,name,table,episodes)
    if actor_followup:
        return
    best = pairs[(pairs.gap<=.025+1e-12)&(pairs.net>=4)].sort_values(
        ["episode","t","net","alternative_air","alternative"],ascending=[True,True,False,True,True])
    first = best.drop_duplicates("episode").set_index("episode")
    start = np.full(len(episodes),-1)
    rate = np.zeros(len(episodes))
    start[first.index] = first.t
    rate[first.index] = first.alternative_air
    obs = env.reset(episodes)
    air2,states2 = [],[]
    for t in range(95):
        action = policy.act_det(obs)[:,0].copy()
        hit = start==t
        action[hit] = old.air_to_action(rate[hit])
        obs,_,_,_,info = env.step(action)
        unchanged = (start<0)|(start>t)
        np.testing.assert_array_equal(info["air"][unchanged],air[unchanged,t])
        np.testing.assert_array_equal(env.state[unchanged],states[unchanged,t])
        air2.append(info["air"])
        states2.append(env.state.copy())
    delta = metrics(np.stack(air2,axis=1),np.stack(states2,axis=1))-metrics(air,states)
    daily = pd.DataFrame(delta,columns=["air_m3","nh4_min","excess","do_min"])
    daily["start"] = start
    daily["first_air_delta_m3"] = np.where(start>=0,(rate-air[np.arange(200),start])/4,0.)
    daily["episode"] = np.arange(len(episodes))
    save_table(folder,"daily",daily,episodes)


def save_table(folder,name,table,episodes):
    table["day"] = [episodes[i][0] for i in table.episode]
    table["member"] = [episodes[i][1] for i in table.episode]
    table.to_csv(folder/f"{name}.csv.gz",index=False)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    p.add_argument("--check",action="store_true")
    p.add_argument("--prefix",type=int,choices=[108,148,188])
    p.add_argument("--seed",type=int,choices=range(10,15))
    p.add_argument("--actor-followup",action="store_true")
    a=p.parse_args()
    if a.check:
        check()
    else:
        run(a.prefix,a.seed,a.out,a.actor_followup)

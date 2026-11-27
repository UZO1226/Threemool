"""Evaluate a causal two-window sensor gate using one FCSRL trajectory at a time."""
import argparse
import csv
from collections import deque
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e144"))
import choices

METRICS = ["air_m3", "nh4_min", "excess", "do_min"]
RAW = Path("/home/jio/사물/Yoo/data/dynamic-modeling-of-wastewater-treatment-process/train_val_test_online_dataset.csv")


def warmup(path, episodes, env):
    dates = sorted(set(str(day) for day, _ in episodes))
    wanted = set(dates)
    last = deque(maxlen=3)
    history, initial = {}, {}
    with path.open(encoding="utf-8-sig") as stream:
        for record in csv.DictReader(stream, delimiter="\t"):
            time = pd.Timestamp(record["DATETIME"])
            day = str(time.date())
            values = np.array([float(record[k]) for k in ("NH4","DO_1","DO_2","DO_3")])
            if day in wanted and day not in history:
                assert len(last) == 3
                assert [time - stamp for stamp, _ in last] == [pd.Timedelta(minutes=k) for k in (45,30,15)]
                history[day] = np.array([v for _, v in last])
                initial[day] = values
                if day == dates[-1]:
                    break
            last.append((time, values))
    keys = [str(day) for day,_ in episodes]
    np.testing.assert_allclose([initial[k] for k in keys], env.state[:,[0,2,3,4]], rtol=0, atol=1e-6)
    old = np.stack([history[k] for k in keys], axis=1).astype(np.float32)
    return np.stack((old[:,:,0], old[:,:,1:].max(axis=2)),axis=2)


def scores(history):
    x = np.stack(history).astype(float)
    diff = x[-2:].mean(0) - x.mean(0)
    trend = np.minimum(-diff[:,0]/4, diff[:,1]/2)
    level = np.minimum((4-x[-1,:,0])/4, (x[-1,:,1]-2)/2)
    return trend, level


def rollout(env, episodes, policy, past, mode):
    obs = env.reset(episodes)
    rows = np.arange(len(episodes))
    first = np.full(len(rows), -1)
    pulse = first.copy()
    history = deque(past, maxlen=4)
    score = np.full((len(rows),2), np.nan)
    air, states = [], []
    for t in range(env.T):
        history.append(np.column_stack((obs[:,0],obs[:,2:5].max(axis=1))))
        actions, win, gap, unique = choices.proposals(policy,obs)
        flow = choices.old.action_to_air(actions)
        pol = actions[rows,win].copy()
        eligible = unique & (gap <= .025 + 1e-12) & (flow < flow[rows,win,None]-1e-8)
        cut = ((first < 0) & eligible.any(1)) if t in range(0,92,4) else np.zeros(len(rows),bool)
        if cut.any():
            trend, level = scores(history)
            score[cut] = np.column_stack((trend,level))[cut]
            first[cut] = t
            cheap = np.where(eligible,actions,np.inf).argmin(axis=1)
            pol[cut] = actions[rows[cut],cheap[cut]]
        active = (first >= 0) & (t > first) & (t <= first+4) & (pulse < 0)
        if mode == "trend":
            active &= score[:,0] > 0
        elif mode == "level":
            active &= score[:,1] > 0
        elif mode == "once":
            active[:] = False
        change = active & (gap[:,0] <= .025+1e-12) & (flow[rows,win]-flow[:,0] >= 250)
        pol[change] = actions[change,0]
        pulse[change] = t
        obs, _, _, _, info = env.step(pol)
        air.append(info["air"])
        states.append(env.state.copy())
    return np.stack(air,axis=1), np.stack(states,axis=1), first, pulse, score


def run(n, seed, root, out, raw):
    torch.set_num_threads(1)
    choices.base.DeviceConfig().select_device(-1)
    env, _, episodes, policy, base_air, base_state = choices.load(n,seed)
    past = warmup(raw,episodes,env)
    ref = pd.read_csv(root/f"e146/paths/p{n}_s{seed}.csv").sort_values("episode")
    base = choices.metrics(base_air,base_state)
    arms = []
    for mode in ("once","all","level","trend"):
        air, state, first, pulse, score = rollout(env,episodes,policy,past,mode)
        y = choices.metrics(air,state)
        np.testing.assert_array_equal(first,ref["first"])
        np.testing.assert_array_equal([str(d) for d,m in episodes],ref.day)
        np.testing.assert_array_equal([m for d,m in episodes],ref.member)
        if mode == "once":
            once_air, once_state, initial_scores = air,state,score
            np.testing.assert_allclose(y-base,ref[METRICS],rtol=0,atol=1e-7)
        np.testing.assert_array_equal(score,initial_scores)
        before = (pulse[:,None]<0)|(np.arange(env.T)<pulse[:,None])
        np.testing.assert_array_equal(air[before],once_air[before])
        np.testing.assert_array_equal(state[before],once_state[before])
        hit = np.flatnonzero(pulse>=0)
        if len(hit):
            assert ((once_air[hit,pulse[hit]]-air[hit,pulse[hit]])>=250).all()
        table = pd.DataFrame(y,columns=METRICS).assign(prefix=n,seed=seed,mode=mode,episode=ref.episode.values,
                day=ref.day.values,member=ref.member.values,first=first,pulse=pulse,trend=score[:,0],level=score[:,1])
        arms.append(table)
        if mode == "once":
            native = table.copy()
            native["mode"] = "native"
            native[METRICS] = base
            arms.append(native)
    out.mkdir(parents=True,exist_ok=True)
    pd.concat(arms,ignore_index=True).to_csv(out/f"p{n}_s{seed}.csv",index=False)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("prefix",type=int)
    p.add_argument("seed",type=int)
    p.add_argument("root",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("--raw",type=Path,default=RAW)
    a=p.parse_args()
    run(a.prefix,a.seed,a.root,a.out,a.raw)

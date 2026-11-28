"""Fit the e135 factorial conditions at two historical forecast origins."""
import argparse
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e135"))
import report
import study
from darrow_env import load_days
from prepare import fit_prior
from surrogate_dw import train_gbm, ZONES

old = study.old
trial = old.trial
base = old.base
save = base.save
METHODS = ("fcsrl", "fcsrl_actor")


def prepare(root, n):
    src = trial.PREP / "data.pkl"
    assert base.sha(src) == trial.read(trial.REF/"manifest.json")["input_sha256"][src.name]
    source = pd.read_pickle(src)
    dates = source["days"]["train"]
    train, val = dates[:n], dates[n:n+40]
    assert len(dates)==188 and len(val)==40 and train[-1]<val[0]
    table = source["data"].loc[train[0]:val[-1]].copy()
    assert len(table)==96*(n+40)
    training = table.loc[train[0]:train[-1]]
    med = training[ZONES].median()
    dest = root / f"prefix{n}" / "prepared"
    dest.mkdir(parents=True, exist_ok=False)
    splits = dict(train=train,val=val)
    pd.to_pickle(dict(data=table,days=splits,zone_fractions=(med/med.sum()).to_dict(),
                      steps_per_day=96),dest/"data.pkl")
    fit = int(.8*n)
    # train_gbm uses its val list only for a diagnostic, which we keep within the prefix too.
    log = train_gbm(training, dict(train=train,val=train[fit:]),dest/"surrogate.pkl")
    _, days = load_days(dest/"data.pkl")
    torch.manual_seed(2300)
    bc_log = fit_prior(days,train[:fit],train[fit:],dest)
    env = trial.fast.BatchEnv(days,trial.fast.FastDynamics(dest/"surrogate.pkl"))
    slow = trial.fast.BatchEnv(days,trial.fast.Dynamics(dest/"surrogate.pkl"))
    sample = old.all_episodes(train[::35],5)
    for instance in (env,slow):
        instance.reset(sample)
    rng = np.random.default_rng(139)
    for _ in range(95):
        action = rng.uniform(-1,1,len(sample))
        env.step(action)
        slow.step(action)
        np.testing.assert_array_equal(env.state,slow.state)
    save(dest/"manifest.json",dict(train=train,val=val,prior_fit=train[:fit],prior_check=train[fit:],
        hashes={p.name:base.sha(p) for p in dest.iterdir() if p.is_file()},
        gbm_train_diagnostic=log,bc_log=bc_log,commit=subprocess.check_output(
            ["git","rev-parse","HEAD"],cwd=HERE,text=True).strip()))


def paths(root,n):
    fold = root/f"prefix{n}"
    prep = fold/"prepared"
    for name,digest in trial.read(prep/"manifest.json")["hashes"].items():
        assert base.sha(prep/name)==digest
    return fold,prep


def train(root,n,method,seed):
    fold,prep = paths(root,n)
    dest = fold/method/f"seed_{seed}"
    args = argparse.Namespace(out=dest, prepared=prep, updates=66, seed=seed, method=method)
    base.e30.train(args, load_data=load_days, dynamics=trial.fast.FastDynamics, agent_factory=trial.factory)


def evaluate(root,n,seed):
    fold,prep = paths(root,n)
    old.splits,days = load_days(prep/"data.pkl")
    old.env = trial.fast.BatchEnv(days,trial.fast.FastDynamics(prep/"surrogate.pkl"))
    agents = []
    samples = []
    for method in METHODS:
        agent = trial.factory(method,seed,prep,base.common.config(66))
        agent.load(fold/method/f"seed_{seed}"/"model_final")
        agents.append(agent)
        samples.append(study.collect(agent,seed))
    x,y = (np.concatenate(parts) for parts in zip(*samples))
    assert x.shape==(2*n*5*23,12)
    models = study.fit_models(x,y)
    risk = study.FastRate(models)
    np.testing.assert_allclose(risk.predict(x[::17]),study.slow_rates(models,x[::17]),atol=1e-12,rtol=0)
    dest = fold/"evaluated"/f"seed_{seed}"
    dest.mkdir(parents=True,exist_ok=False)
    with (dest/"risk.pkl").open("wb") as f:
        pickle.dump(models,f)
    for arm,method_id,enabled in [("base",0,False),("actor",1,False),("base_risk",0,True),("combined",1,True)]:
        agent = agents[method_id]
        policy = study.LocalPolicy(agent,risk,0) if enabled else agent
        for scale in study.SCALES:
            air,states = old.rollout(policy,old.splits["val"],scale,keep=True)
            path = dest/f"{arm}_{scale}.npz"
            np.savez_compressed(path,air=air,states=states)
            if not enabled and scale==1000:
                ref = trial.read(fold/METHODS[method_id]/f"seed_{seed}"/"metrics.json")
                assert ref["steps"]==401280
                native = report.day_metrics(path).mean(axis=0)
                np.testing.assert_allclose(native[[0,4]], [ref["val_final"][k] for k in ("air_m3","cost")],
                                           atol=1e-8,rtol=0)


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["prepare","train","evaluate"])
    p.add_argument("root",type=Path)
    p.add_argument("prefix",type=int,choices=[108,148])
    p.add_argument("--method",choices=METHODS)
    p.add_argument("--seed",type=int,choices=report.SEEDS)
    a = p.parse_args()
    torch.set_num_threads(1)
    base.DeviceConfig().select_device(-1)
    if a.stage=="prepare":
        prepare(a.root,a.prefix)
    elif a.stage=="train":
        train(a.root,a.prefix,a.method,a.seed)
    else:
        evaluate(a.root,a.prefix,a.seed)

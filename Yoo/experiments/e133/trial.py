"""e50 paired-seed training and independent final checkpoint evaluation."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e113"))
import run as fast

base = fast.base
PREP = Path("/home/jio/fcsrl_src/results/e50/prepared")
REF = PREP.parent / "main"
METHODS = ("fcsrl", "fcsrl_actor", "actor_extra_ppo")
SEEDS = tuple(range(10, 15))


def read(path):
    return json.loads(path.read_text())


def factory(method, seed, prepared, config):
    if method != "actor_extra_ppo":
        return base.common.make_agent(method, seed, prepared, config)
    agent = base.common.make_agent("fcsrl_extra_ppo", seed, prepared, config)
    agent.actor_only_task = True
    return agent


def check():
    for name, digest in read(REF / "manifest.json")["input_sha256"].items():
        assert base.sha(PREP / name) == digest
    splits, days = base.load_selected(PREP / "data.pkl")
    episodes = [(day, m) for day in splits["train"][::38] for m in range(5)]
    slow, quick = [fast.BatchEnv(days, cls(PREP / "surrogate.pkl")) for cls in (fast.Dynamics, fast.FastDynamics)]
    for env in (slow, quick):
        env.reset(episodes)
    rng = np.random.default_rng(133)
    for _ in range(95):
        action = rng.uniform(-1, 1, len(episodes))
        slow.step(action)
        quick.step(action)
        np.testing.assert_array_equal(slow.state, quick.state)
    config = base.common.config(66)
    for method in METHODS:
        agent = factory(method, 0, PREP, config)
        buf, diag = base.e30.collect(agent, config, quick, splits["train"], np.random.default_rng(0))
        rec = agent.update(buf)
        name = "fcsrl_actor" if method == "actor_extra_ppo" else method
        ref = read(REF / "runs" / name / "seed_0/train_log.json")[0]
        for k in ("member", "daily_cost_mean", "raw_action_clip_percent", "reward_mean"):
            np.testing.assert_allclose(diag[k], ref[k], rtol=0, atol=1e-7)
        if method != "actor_extra_ppo":
            for k in ("std", "encoder_sync_kl", "kl"):
                np.testing.assert_allclose(rec[k], ref[k], rtol=0, atol=1e-6)
        if method != "fcsrl":
            assert rec["extra_attempted_steps"] == 50
            for critic in ("reward", "cost"):
                assert rec[f"before_{critic}_error"] == rec[f"after_{critic}_error"]
            assert rec["preserve_fit_samples"] == 2048 and rec["preserve_check_samples"] == 1024
    print("e50 inputs, exact dynamics and initial updates verified", flush=True)


def trial(out, method, seed):
    dest = out / method / f"seed_{seed}"
    args = argparse.Namespace(out=dest, prepared=PREP, updates=66, seed=seed, method=method)
    base.e30.train(args, load_data=base.load_selected, dynamics=fast.FastDynamics, agent_factory=factory)
    agent = factory(method, seed, PREP, base.common.config(66))
    agent.load(dest / "model_final")
    splits, days = base.load_selected(PREP / "data.pkl")
    env = fast.BatchEnv(days, fast.Dynamics(PREP / "surrogate.pkl"))
    metrics = base.evaluate(agent, env, splits["val"], dest, "winter")
    for key, value in read(dest / "metrics.json")["val_final"].items():
        np.testing.assert_allclose(metrics[key], value, rtol=0, atol=1e-8)
    base.save(dest / "winter_metrics.json", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    torch.set_num_threads(1)
    base.DeviceConfig().select_device(-1)
    if args.check:
        check()
    elif args.method is not None and args.seed is not None:
        trial(args.out, args.method, args.seed)
    else:
        parser.error("provide --check or both --method and --seed")

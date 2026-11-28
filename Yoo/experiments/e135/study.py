"""Frozen-policy combination study in the common DARROW environment."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
from pathlib import Path
import pickle
import subprocess

import numpy as np

from risk import FastRate, fit_models, slow_rates

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("e134_replay", HERE.parent / "e134/run.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)

OFFSETS = np.array([0, -.125, .125, -.25, .25])
SCALES = (1000, 800, 900, 1100, 1250)
ARMS = {
    "base": ("fcsrl", None), "actor": ("fcsrl_actor", None),
    "extra": ("actor_extra_ppo", None), "base_risk": ("fcsrl", 0),
    "actor_penalty": ("fcsrl_actor", 1), "combined": ("fcsrl_actor", 0),
}


def collect(agent, seed):
    env = old.env
    episodes = old.all_episodes(old.splits["train"], 5)
    obs = env.reset(episodes)
    rng = np.random.default_rng(135000 + seed)
    features, labels = [], []
    for _ in range(env.T // 4):
        action = agent.act_det(obs)[:, 0]
        action = np.clip(action + rng.choice(OFFSETS, size=len(obs)), -1, 1)
        features.append(np.column_stack((obs, action)))
        counts = np.zeros((len(obs), 2))
        for step in range(4):
            if step:
                action = agent.act_det(obs)[:, 0]
            obs, _, _, _, info = env.step(action)
            counts += np.column_stack((info["v_nh4"], info["v_do"]))
        labels.append(counts / 4)
    return np.concatenate(features), np.concatenate(labels)


class LocalPolicy:
    def __init__(self, agent, risk, penalty):
        self.agent, self.risk, self.penalty = agent, risk, penalty

    def act_det(self, obs):
        a = self.agent.act_det(obs)
        candidates = np.clip(a + OFFSETS, -1, 1)
        features = np.column_stack((np.repeat(obs, len(OFFSETS), axis=0), candidates.ravel()))
        rates = self.risk.predict(features)
        score = rates.sum(axis=1).reshape(candidates.shape) + self.penalty * (candidates - a) ** 2
        return candidates[np.arange(len(a)), score.argmin(axis=1), None]


def run_seed(args):
    out, seed = args
    dest = out / f"seed_{seed}"
    dest.mkdir()
    policies = {method: old.load_agent(method, seed) for method in old.trial.METHODS}
    for method in policies:
        old.restore((method, seed))
    obs = old.env.reset(old.all_episodes(old.splits["train"][:3], 5))
    samples = [collect(policies[m], seed) for m in ("fcsrl", "fcsrl_actor")]
    x, y = (np.concatenate(arr) for arr in zip(*samples))
    assert x.shape == (2 * 188 * 5 * 23, 12)
    models = fit_models(x, y)
    risk = FastRate(models)
    test_x = x[::19]
    np.testing.assert_allclose(risk.predict(test_x), slow_rates(models, test_x), atol=1e-12, rtol=0)
    for penalty in (0, 1):
        agent = policies["fcsrl_actor"]
        a = agent.act_det(obs)
        candidates = np.clip(a + OFFSETS, -1, 1)
        scores = [
            slow_rates(models, np.column_stack((obs, c))).sum(axis=1) + penalty * (c - a[:, 0]) ** 2
            for c in candidates.T
        ]
        expected = candidates[np.arange(len(obs)), np.argmin(scores, axis=0)]
        np.testing.assert_array_equal(LocalPolicy(agent, risk, penalty).act_det(obs)[:, 0], expected)
    with (dest / "rate.pkl").open("wb") as stream:
        pickle.dump(models, stream)
    print("fit", seed, len(x), flush=True)
    for arm, (method, penalty) in ARMS.items():
        agent = policies[method]
        if penalty is not None:
            agent = LocalPolicy(agent, risk, penalty)
        for scale in SCALES:
            air, states = old.rollout(agent, old.splits["val"], scale, keep=True)
            if penalty is None and scale == 1000:
                with np.load(old.source / method / f"seed_{seed}/winter_rollout.npz") as ref:
                    np.testing.assert_array_equal(air, ref["air"])
                    np.testing.assert_array_equal(states, ref["states"])
            np.savez_compressed(dest / f"{arm}_{scale}.npz", air=air, states=states)
        print("finished", seed, arm, flush=True)


def main(out, jobs):
    out.mkdir(parents=True, exist_ok=False)
    (out / "commit").write_bytes(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE))
    for name, digest in old.trial.read(old.trial.REF / "manifest.json")["input_sha256"].items():
        assert old.base.sha(old.trial.PREP / name) == digest
    sums = old.source / "tables/SHA256SUMS"
    assert old.base.sha(sums) == "297268876814ab83d214f6866d5ae725eaa9b792ae485a1e0960682eb88122c4"
    subprocess.run(["sha256sum", "--quiet", "-c", str(sums)], cwd=old.source, check=True)
    with ProcessPoolExecutor(max_workers=jobs, initializer=old.setup) as pool:
        list(pool.map(run_seed, [(out, s) for s in old.trial.SEEDS]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--jobs", type=int, default=5)
    args = parser.parse_args()
    main(args.out, args.jobs)

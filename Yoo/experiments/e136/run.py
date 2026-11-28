"""Test a severity tie-breaker on the frozen e135 controller."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
from sklearn.base import clone

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e135"))
import study
from risk import FastRate

old = study.old
SRC = old.source.parents[1] / "e135/main"


def collect(agent, seed):
    env = old.env
    obs = env.reset(old.all_episodes(old.splits["train"], 5))
    rng = np.random.default_rng(135000 + seed)
    xs, rates, severity = [], [], []
    for _ in range(env.T // 4):
        a = np.clip(agent.act_det(obs)[:, 0] + rng.choice(study.OFFSETS, len(obs)), -1, 1)
        xs.append(np.column_stack((obs, a)))
        counts = np.zeros((len(obs), 2))
        excess = np.zeros(len(obs))
        for j in range(4):
            if j:
                a = agent.act_det(obs)[:, 0]
            obs, _, _, _, info = env.step(a)
            counts += np.column_stack((info["v_nh4"], info["v_do"]))
            excess += np.maximum(env.state[:, 0] - 4, 0) * .25
        rates.append(counts / 4)
        severity.append(excess)
    return np.concatenate(xs), np.concatenate(rates), np.concatenate(severity)


class Policy:
    def __init__(self, agent, risk, severity, mode):
        self.agent, self.risk, self.severity, self.mode = agent, risk, severity, mode
        self.total = self.high = self.ties = self.changed = self.up = 0

    def act_det(self, obs):
        actions = np.clip(self.agent.act_det(obs) + study.OFFSETS, -1, 1)
        x = np.column_stack((np.repeat(obs, 5, axis=0), actions.ravel()))
        scores = self.risk.predict(x).sum(axis=1).reshape(actions.shape)
        idx = scores.argmin(axis=1)
        row = np.arange(len(obs))
        original = actions[row, idx].copy()
        ties = scores == scores[row, idx, None]
        high = obs[:, 0] > 4
        switchable = high & (ties & (actions != original[:, None])).any(axis=1)
        if self.mode != "original" and np.any(switchable):
            if self.mode == "max_air":
                priority = -actions[switchable]
            else:
                active = x.reshape(len(obs), 5, -1)[switchable].reshape(-1, 12)
                priority = np.maximum(self.severity.raw(active), 0).reshape(-1, 5)
            idx[switchable] = np.where(ties[switchable], priority, np.inf).argmin(axis=1)
        chosen = actions[row, idx]
        assert np.all(ties[row, idx]) and np.all(chosen[~high] == original[~high])
        self.total += len(obs)
        self.high += high.sum()
        self.ties += switchable.sum()
        self.changed += (original != chosen).sum()
        self.up += (chosen > original).sum()
        return chosen[:, None]


def one(args):
    out, seed = args
    dest = out / f"seed_{seed}"
    dest.mkdir()
    actor = old.load_agent("fcsrl_actor", seed)
    data = []
    for agent in (old.load_agent("fcsrl", seed), actor):
        x, rates, y = collect(agent, seed)
        x0, rates0 = study.collect(agent, seed)
        np.testing.assert_array_equal(x, x0)
        np.testing.assert_array_equal(rates, rates0)
        data.append((x, y))
    x, y = [np.concatenate(items) for items in zip(*data)]
    assert x.shape == (43240, 12) and np.isfinite(y).all() and min(y) >= 0
    with (SRC / dest.name / "rate.pkl").open("rb") as f:
        original_models = pickle.load(f)
    reg = clone(original_models[0]).fit(x, y)
    with (dest / "severity.pkl").open("wb") as f:
        pickle.dump(reg, f)
    fast = FastRate([reg])
    for block in np.array_split(x, 16):
        np.testing.assert_allclose(fast.raw(block)[:, 0], reg.predict(block), rtol=0, atol=1e-12)
    risk = FastRate(original_models)
    stats = {}
    for mode in ("original", "severity", "max_air"):
        policy = Policy(actor, risk, fast, mode)
        air, states = old.rollout(policy, old.splits["val"], 1000, keep=True)
        if mode == "original":
            with np.load(SRC / dest.name / "combined_1000.npz") as ref:
                np.testing.assert_array_equal(air, ref["air"])
                np.testing.assert_array_equal(states, ref["states"])
        else:
            np.savez_compressed(dest / f"{mode}.npz", air=air, states=states)
        stats[mode] = {k:int(getattr(policy, k)) for k in ("total","high","ties","changed","up")}
        assert policy.total == 19000
    old.base.save(dest / "counts.json", stats)
    print("verified", seed, stats["severity"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "commit").write_bytes(subprocess.check_output(["git", "rev-parse","HEAD"], cwd=HERE))
    for name, digest in old.trial.read(old.trial.REF / "manifest.json")["input_sha256"].items():
        assert old.base.sha(old.trial.PREP / name) == digest
    for path, digest in [(SRC,"3a5cbbbfdbca2ab487a1cc6a50138d31e4b6a2b063c3234e7b503e8782cc474c"),
                         (old.source,"297268876814ab83d214f6866d5ae725eaa9b792ae485a1e0960682eb88122c4")]:
        manifest = path / ("SHA256SUMS" if path == SRC else "tables/SHA256SUMS")
        assert old.base.sha(manifest) == digest
        subprocess.run(["sha256sum","--quiet","-c",str(manifest)], cwd=path, check=True)
    with ProcessPoolExecutor(max_workers=5, initializer=old.setup) as pool:
        list(pool.map(one, [(args.out, s) for s in old.trial.SEEDS]))

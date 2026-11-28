"""Resume the frozen e135 controller after the e137 equal-volume transfers."""
import argparse
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e137"))
import timing

old, study, report = timing.old, timing.study, timing.report
SOURCE = timing.SOURCE.parents[1] / "e137/main"
ARMS = ("original", "early", "late")


def live(starts, t):
    return (starts < 0) | (t > starts)


def rollout(policy, episodes, scheduled, starts):
    env = old.env
    obs = env.reset(episodes)
    air = np.empty_like(scheduled)
    states = []
    for t in range(env.T):
        # Keep the entire inference batch identical in size to the parent policy.
        proposal = old.action_to_air(policy.act_det(obs)[:, 0])
        rate = np.where(live(starts, t), proposal, scheduled[:, t])
        zone = rate[:, None] * env.x["frac"][:, t]
        env.state = env.dyn.step(env.state, env.x["exo"][:, t], zone, env.prev_zone, env.member)
        env.prev_air, env.prev_zone, env.t = rate, zone, t + 1
        obs = env.obs()
        air[:, t] = rate
        states.append(env.state.copy())
    return air, np.stack(states, axis=1)


def verify_inputs():
    np.testing.assert_array_equal(live(np.array([-1, 5, 6]), 5), [True, False, False])
    np.testing.assert_array_equal(live(np.array([-1, 5, 6]), 6), [True, True, False])
    for directory, filename, checksum in [
        (SOURCE, "SHA256SUMS", "a233dec14268ddaf0717d8467e752d40be3fb0080611ba982869d0e152b2597b"),
        (timing.SOURCE, "SHA256SUMS", "3a5cbbbfdbca2ab487a1cc6a50138d31e4b6a2b063c3234e7b503e8782cc474c"),
        (old.source, "tables/SHA256SUMS", "297268876814ab83d214f6866d5ae725eaa9b792ae485a1e0960682eb88122c4"),
    ]:
        manifest = directory / filename
        assert old.base.sha(manifest) == checksum
        subprocess.run(["sha256sum", "--quiet", "-c", str(manifest)], cwd=directory, check=True)
    for name, digest in old.trial.read(old.trial.REF / "manifest.json")["input_sha256"].items():
        assert old.base.sha(old.trial.PREP / name) == digest


def recovery(nh):
    below = np.flatnonzero(nh <= 4)
    return 15 * (below[0] if len(below) else len(nh))


def run(out):
    verify_inputs()
    old.setup()
    episodes = old.all_episodes(old.splits["val"], 5)
    lookup = {(str(day), m): i for i,(day,m) in enumerate(episodes)}
    events = pd.read_csv(SOURCE/"events.csv").query("arm=='original'")
    assert len(episodes)==200 and len(events)==129 and not events.duplicated(["seed","day","member"]).any()
    out.mkdir(parents=True, exist_ok=False)
    (out/"commit").write_bytes(subprocess.check_output(["git","rev-parse","HEAD"], cwd=HERE))
    rows = []
    for seed in report.SEEDS:
        starts = np.full(len(episodes), -1)
        for rec in events.query("seed==@seed").itertuples():
            starts[lookup[rec.day,rec.member]] = rec.k
        src = timing.SOURCE/f"seed_{seed}"
        with np.load(src/"combined_1000.npz") as a, np.load(src/"base_1000.npz") as b:
            expected, _ = timing.anchors(a["states"][:,:,0], b["states"][:,:,0])
        np.testing.assert_array_equal(starts, expected)
        with (src/"rate.pkl").open("rb") as stream:
            risk = study.FastRate(pickle.load(stream))
        policy = study.LocalPolicy(old.load_agent("fcsrl_actor",seed), risk, 0)
        for arm in ARMS:
            with np.load(SOURCE/f"{seed}_{arm}.npz") as z:
                scheduled, fixed = z["air"], z["states"]
            release = np.full_like(starts,-1) if arm=="original" else starts
            air, state = rollout(policy, episodes, scheduled, release)
            same = (starts[:,None]<0) | (np.arange(95)<=starts[:,None])
            if arm=="original":
                same[:] = True
            np.testing.assert_array_equal(air[same], scheduled[same])
            np.testing.assert_array_equal(state[same], fixed[same])
            np.savez_compressed(out/f"{seed}_{arm}.npz", air=air, states=state)
            for i in np.flatnonzero(starts>=0):
                k = starts[i]
                delta = (air[i]-scheduled[i])/4
                added = delta.clip(min=0).sum()
                removed = (-delta).clip(min=0).sum()
                np.testing.assert_allclose(added-removed,(air[i].sum()-scheduled[i].sum())/4,atol=1e-8)
                window = slice(k,k+timing.HORIZON)
                rows.append(dict(seed=seed,day=episodes[i][0],member=episodes[i][1],k=k,arm=arm,
                                 added=added,removed=removed,net=added-removed,
                                 fixed_min=recovery(fixed[i,window,0]),
                                 feedback_min=recovery(state[i,window,0])))
        print("verified seed",seed,flush=True)
    pd.DataFrame(rows).to_csv(out/"events.csv",index=False)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    run(parser.parse_args().out)

"""Locate first substantial changes in the effect of one frozen cut across two model versions."""
import argparse
from itertools import permutations, product
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e147"))
import replay

to_air = replay.choices.old.action_to_air
GROUPS = {"nh4": [0], "no3": [1], "do": [2, 3, 4], "previous_air": [10]}


class Recorder:
    def __init__(self, policy, rows):
        self.policy, self.rows = policy, rows
        self.obs, self.flow, self.actor = [], [], []

    def act_det(self, obs):
        action = self.policy.act_det(obs)
        self.obs.append(obs[self.rows].copy())
        self.flow.append(to_air(action[self.rows, 0]))
        self.actor.append(to_air(self.policy.agent.act_det(obs)[self.rows, 0]))
        return action


def decompose(values):
    phi = np.zeros(4)
    for order in permutations(range(4)):
        mask = 0
        for g in order:
            updated = mask | (1 << g)
            phi[g] += values[updated] - values[mask]
            mask = updated
    return phi / 24


def explain(policy, current, previous, slot, expected):
    np.testing.assert_array_equal(current[5:10], previous[5:10])
    mixed = np.tile(current, (16, 1))
    for mask in range(16):
        for i, cols in enumerate(GROUPS.values()):
            if mask & (1 << i):
                mixed[mask, cols] = previous[cols]
    # Query each mixture in the same row of a full batch as the original decision.
    # This preserves actor inference rounding observed in e147.
    values = np.array([to_air(policy.act_det(np.tile(x, (200, 1)))[slot, 0]) for x in mixed])
    np.testing.assert_allclose(values[[0, -1]], expected, atol=.05, rtol=0)
    return decompose(values)


def first_index(mask):
    hits = np.flatnonzero(mask)
    return int(hits[0]) if len(hits) else None


def run(n, seed, root, dest):
    torch.set_num_threads(1)
    replay.choices.base.DeviceConfig().select_device(-1)
    candidates = pd.read_csv(HERE.parent / "e147/previous_criteria.csv")
    cases = candidates[(candidates.prefix == n) & (candidates.seed == seed)
                       & ((candidates.previous_votes == 5) if n == 148 else (candidates.previous_votes < 5))]
    assert len(cases)
    env, older, episodes, policy, air, states = replay.choices.load(n, seed)
    origin = pd.read_csv(root / f"e146/paths/p{n}_s{seed}.csv", float_precision="round_trip").sort_values("episode")
    archived = pd.read_csv(root / f"e147/paths/p{n}_s{seed}.csv", float_precision="round_trip")
    start = replay.starting_points(env, episodes, policy, origin, air, states)
    rows = [np.flatnonzero(start[-1] == d)[0] for d in cases.decision]
    traces = {}
    for v, model in enumerate([env.dyn, older]):
        env.dyn = model
        for m in range(5):
            outcomes = []
            for arm in range(2):
                rec = Recorder(policy, rows)
                command = start[4+arm]
                outcomes.append(replay.simulate(env, episodes, rec, m, start, command))
                obs, flow, actor = map(np.stack, (rec.obs, rec.flow, rec.actor))
                flow[start[0][rows], np.arange(len(rows))] = to_air(command[rows])
                traces[v, m, arm] = obs, flow, actor
            ref = archived[(archived.evaluator == m) & (archived.version == ["current", "previous"][v])].sort_values("episode")
            np.testing.assert_allclose(outcomes[1]-outcomes[0], ref[replay.METRICS], atol=1e-7, rtol=0)
    events, series = [], []
    for j, row in enumerate(rows):
        t0 = start[0][row]
        identity = dict(prefix=n, seed=seed, decision=start[-1][row], day=origin.day.iat[row], episode=row)
        for mc, mp in product(range(5), repeat=2):
            # From left to right these are current native/cut, then previous native/cut.
            quartet = [traces[v, m, arm] for v, m in enumerate((mc, mp)) for arm in range(2)]
            obs, flow, actor = [np.stack([tr[k][t0:, j] for tr in quartet]) for k in range(3)]
            for arr in obs[1:]:
                np.testing.assert_array_equal(arr[0], obs[0, 0])
                np.testing.assert_array_equal(arr[:, 5:10], obs[0, :, 5:10])
            current, prior = flow[1]-flow[0], flow[3]-flow[2]
            gap = prior-current
            assert gap[0] == 0 and current[0] < 0
            nh_diff = np.max(abs(obs[2:, :, 0]-obs[:2, :, 0]), axis=0)
            substantial = first_index(abs(gap) >= 250)
            small = first_index(abs(gap) > .05)
            nh = first_index(nh_diff >= .1)
            event = dict(**identity, current_member=mc, previous_member=mp, old_air_loss=prior.sum()/4 >= -.05,
                         small_min=None if small is None else 15*small,
                         onset_min=None if substantial is None else 15*substantial,
                         nh_min=None if nh is None else 15*nh, current_m3=current.sum()/4, previous_m3=prior.sum()/4)
            for label, block in zip(["0_1h", "1_3h", "3_6h", "later"], (gap[:4], gap[4:12], gap[12:24], gap[24:])):
                event["m3_" + label] = block.sum()/4
            if substantial is not None:
                t = substantial
                attributed = [explain(policy, obs[k, t], obs[k+2, t], row, flow[[k, k+2], t]) for k in (0, 1)]
                phi = attributed[1]-attributed[0]
                np.testing.assert_allclose(phi.sum(), gap[t], atol=.2, rtol=0)
                actor_gap = (actor[3, t]-actor[2, t])-(actor[1, t]-actor[0, t])
                event.update(gap=gap[t], actor=actor_gap, selection=gap[t]-actor_gap,
                             nh_native=obs[2,t,0]-obs[0,t,0], nh_cut=obs[3,t,0]-obs[1,t,0],
                             top=list(GROUPS)[np.argmax(abs(phi))], **dict(zip(GROUPS, phi)))
            events.append(event)
        # Preserve arm-specific observations, commands, and raw actor for independent checks.
        for (v, m, arm), (obs, flow, actor) in traces.items():
            for t in range(t0, env.T):
                series.append(dict(**identity, version=v, member=m, arm=arm, minute=(t-t0)*15,
                                   flow=flow[t,j], actor=actor[t,j], nh4=obs[t,j,0], no3=obs[t,j,1],
                                   do1=obs[t,j,2], do2=obs[t,j,3], do3=obs[t,j,4], prev_air=obs[t,j,10]))
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(events).to_csv(dest / f"events_{n}_{seed}.csv", index=False)
    pd.DataFrame(series).to_csv(dest / f"traces_{n}_{seed}.csv.gz", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prefix", type=int)
    p.add_argument("seed", type=int)
    p.add_argument("root", type=Path)
    p.add_argument("dest", type=Path)
    a = p.parse_args()
    run(a.prefix, a.seed, a.root, a.dest)

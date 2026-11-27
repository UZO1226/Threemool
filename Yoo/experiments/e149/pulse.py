"""Equalize the effective selector offset for one transition in matched DARROW rollouts."""
import argparse
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e147"))
import replay

M = ["air_3_6h", *replay.METRICS]
to_air = replay.choices.old.action_to_air
to_action = replay.choices.old.air_to_action
EPS = 1e-8


def common_offset(actor, flow):
    requested = flow[0] - actor[0]
    lo, hi = -min(actor), 9000. - max(actor)
    offset = np.clip(requested, lo, hi)
    target = actor + offset
    np.testing.assert_allclose(target-actor, offset, rtol=0, atol=EPS)
    return target, abs(offset-requested) > EPS


class OneStep:
    def __init__(self, policy, rows, pulses, reference=None):
        self.policy, self.rows, self.pulses, self.reference = policy, rows, pulses, reference
        self.flows, self.obs = [], []

    def act_det(self, obs):
        t = len(self.flows)
        command = self.policy.act_det(obs).copy()
        if self.reference is not None:
            active = np.array([t <= self.pulses.get(r, (999, 0))[0] for r in self.rows])
            np.testing.assert_array_equal(obs[self.rows[active]], self.reference[t, active])
        for r, (step, target) in self.pulses.items():
            if step == t:
                command[r, 0] = to_action(target)
                np.testing.assert_allclose(to_air(command[r, 0]), target, atol=EPS, rtol=0)
        self.flows.append(to_air(command[self.rows, 0]))
        self.obs.append(obs[self.rows].copy())
        return command


def run(prefix, seed, root, out):
    torch.set_num_threads(1)
    replay.choices.base.DeviceConfig().select_device(-1)
    ev = pd.read_csv(HERE.parent / "e148/pairs.csv", float_precision="round_trip")
    ev = ev[(ev.prefix == prefix) & (ev.seed == seed)]
    paths = pd.read_csv(root / f"e148/paths/traces_{prefix}_{seed}.csv.gz", float_precision="round_trip")
    trace = paths.set_index(["decision", "version", "member", "arm", "minute"]).sort_index()
    env, older, episodes, policy, air, states = replay.choices.load(prefix, seed)
    origin = pd.read_csv(root / f"e146/paths/p{prefix}_s{seed}.csv", float_precision="round_trip").sort_values("episode")
    archive = pd.read_csv(root / f"e147/paths/p{prefix}_s{seed}.csv", float_precision="round_trip")
    start = replay.starting_points(env, episodes, policy, origin, air, states)
    cases = ev[["decision","day","episode"]].drop_duplicates().sort_values("episode")
    rows = cases.episode.to_numpy()
    np.testing.assert_array_equal(start[-1][rows], cases.decision)
    members = [env.dyn, older]
    baseline, reference, flows = {}, {}, {}
    def drive(v, m, arm, pulses):
        env.dyn = members[v]
        key = (v, m, arm)
        stepper = OneStep(policy, rows, pulses, reference.get(key))
        metrics = replay.simulate(env, episodes, stepper, m, start, start[4+arm])[rows]
        flow = np.stack(stepper.flows)
        flow[start[0][rows], np.arange(len(rows))] = to_air(start[4+arm][rows])
        times = start[0][rows, None] + np.arange(12, 24)
        window = flow[times, np.arange(len(rows))[:, None]].sum(1)/4
        if key not in reference:
            reference[key] = np.stack(stepper.obs)
            flows[key] = flow
        return np.column_stack((window, metrics))
    for v, m, arm in product(range(2), range(5), range(2)):
        baseline[v,m,arm] = drive(v,m,arm,{})
        for j, case in enumerate(cases.itertuples()):
            expected = trace.loc[(case.decision,v,m,arm)].flow
            np.testing.assert_allclose(flows[v,m,arm][start[0][case.episode]:,j], expected, atol=1e-7, rtol=0)
        if arm:
            delta = baseline[v,m,1]-baseline[v,m,0]
            archived = archive[(archive.version == ["current","previous"][v]) & (archive.evaluator == m)].set_index("episode")
            np.testing.assert_allclose(delta[:,1:], archived.loc[rows,replay.METRICS], atol=1e-7, rtol=0)
    cache, results = {}, []
    for mc, mp in product(range(5), repeat=2):
        events = ev[(ev.current_member == mc) & (ev.previous_member == mp)].set_index("episode").loc[rows]
        changed, projected = np.zeros((len(rows),2,2), bool), np.zeros((len(rows),2), bool)
        instant = np.zeros((len(rows),2,2))
        new = {}
        for arm in range(2):
            pulses = [{}, {}]
            for j, e in enumerate(events.itertuples()):
                if pd.isna(e.onset_min):
                    continue
                records = [trace.loc[(e.decision,v,m,arm,int(e.onset_min))] for v,m in enumerate((mc,mp))]
                actors = np.array([r.actor for r in records])
                original = np.array([r.flow for r in records])
                target, projected[j,arm] = common_offset(actors,original)
                for v in range(2):
                    if abs(target[v]-original[v]) > EPS:
                        pulses[v][e.Index] = (start[0][e.Index]+int(e.onset_min)//15, target[v])
                        changed[j,v,arm] = True
                        instant[j,v,arm] = target[v]-original[v]
            for v, m in enumerate((mc,mp)):
                key = (v,m,arm)
                ck = (*key,tuple(pulses[v].items()))
                if ck not in cache:
                    cache[ck] = drive(*key,pulses[v]) if pulses[v] else baseline[key]
                new[v,arm] = cache[ck]
                np.testing.assert_array_equal(new[v,arm][~changed[:,v,arm]], baseline[key][~changed[:,v,arm]])
        for j, e in enumerate(events.itertuples()):
            result = dict(prefix=prefix,seed=seed,decision=e.decision,day=e.day,current_member=mc,previous_member=mp,
                          onset=e.onset_min,old_loss=e.old_air_loss,projected=projected[j].any(),
                          changed=changed[j].any(), changed_current=changed[j,0].any(), changed_prior=changed[j,1].any())
            if pd.notna(e.onset_min):
                residual = e.gap + np.diff(instant[j,1])[0] - np.diff(instant[j,0])[0]
                np.testing.assert_allclose(residual,e.actor,atol=1e-7,rtol=0)
            effects = []
            for v,m in enumerate((mc,mp)):
                b = baseline[v,m,1][j]-baseline[v,m,0][j]
                a = new[v,1][j]-new[v,0][j]
                effects.append((b,a))
                ver = ["current","prior"][v]
                for label, values in [("before",b),("after",a),("cut_change",new[v,1][j]-baseline[v,m,1][j]),
                                      ("native_change",new[v,0][j]-baseline[v,m,0][j])]:
                    result.update({f"{ver}_{label}_{metric}":val for metric,val in zip(M,values)})
            before, after = np.array(effects[1])-np.array(effects[0])
            np.testing.assert_allclose(before[:2],[e.m3_3_6h,e.previous_m3-e.current_m3],atol=1e-7,rtol=0)
            for label, values in [("gap_before",before),("gap_after",after)]:
                result.update({f"{label}_{metric}":val for metric,val in zip(M,values)})
            results.append(result)
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(results).to_csv(out/f"p{prefix}_s{seed}.csv",index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("prefix",type=int)
    p.add_argument("seed",type=int)
    p.add_argument("root",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    run(a.prefix,a.seed,a.root,a.out)

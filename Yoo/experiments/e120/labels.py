"""Create training labels by crossing e118 autumn candidates with all frozen models."""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "e119"))
import agreement as cross

p = cross.prev
r = cross.r


def run(out, seed, smoke):
    e = r.e113
    plan = r.read(r.ROOT / "plan.json")
    for name, digest in plan["input_sha256"].items():
        r.verify(e.prepared / name, digest)
    dates = plan["held"][:32]
    assert len(set(dates)) == 32 and set(dates).isdisjoint(plan["winter"])
    episodes = [(d, m) for d in dates for m in range(5)]
    ix = {ep: i for i, ep in enumerate(episodes)}
    opts = pd.read_csv(cross.SOURCE / "fit_options.csv", float_precision="round_trip")
    refbase = pd.read_csv(cross.SOURCE / "fit_baseline.csv", float_precision="round_trip")
    assert len(opts) == 2246 and not opts.duplicated(cross.KEY).any()
    _, days = e.base.load_selected(e.prepared / "data.pkl")
    e.base.DeviceConfig().select_device(-1)
    agents = r.load(seed, r.read(r.ROOT / "checksums.json"))[1]
    env = e.BatchEnv(days, e.FastDynamics(e.prepared / "surrogate.pkl"))
    saved, _, trigger, _ = p.simulate(env, episodes, agents)
    expected = refbase[refbase.seed == seed].set_index(["day","member"]).loc[episodes]
    np.testing.assert_array_equal(trigger, expected.trigger)
    np.testing.assert_allclose(p.prior.daily(*saved.transpose(2,0,1)), expected[p.METRICS], atol=1e-7, rtol=0)
    flows = saved[:, :, 0]
    states, zones = cross.prefix(env, episodes, flows, saved)
    results = []
    for lag in ([1] if smoke else range(1,5)):
        ref = opts[(opts.seed == seed) & (opts.lag == lag)].copy()
        ref.index = [ix[(v.day, v.member)] for v in ref.itertuples()]
        ids = ref.index.to_numpy()
        when = np.full(len(episodes), -1)
        when[ids] = trigger[ids] + lag
        for shift in ([0,1] if smoke else range(5)):
            args = (env, episodes, agents["weighted"], ref, flows, states, zones, when, shift)
            b, c = (cross.branch(*args, cut=cut) for cut in (False, True))
            t = when[ids]
            np.testing.assert_allclose(b[ids,t,0] - c[ids,t,0], ref.reduction, atol=1e-7, rtol=0)
            prefix = (when[:,None] < 0) | (np.arange(95) < when[:,None])
            np.testing.assert_array_equal(b[prefix], c[prefix])
            delta = p.prior.daily(*c[ids].transpose(2,0,1)) - p.prior.daily(*b[ids].transpose(2,0,1))
            if shift == 0:
                p.prior.compare_traces(b, saved)
                np.testing.assert_allclose(delta, ref[p.METRICS], atol=1e-6, rtol=0)
            frame = ref[cross.KEY].copy()
            frame["model"] = (ref.member + shift) % 5
            frame[p.METRICS] = delta
            results.append(frame)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(results).to_csv(out / f"seed{seed}.csv", index=False, mode="x")
    print(f"Training labels completed for seed {seed}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("seed", type=int, choices=range(5))
    parser.add_argument("--smoke", action="store_true")
    a = parser.parse_args()
    run(a.out, a.seed, a.smoke)

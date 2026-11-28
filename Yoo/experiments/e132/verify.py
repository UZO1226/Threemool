"""Check e132 daily figures and the first eligible decision from stored histories."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from feedback import HERE, KEY, e, load_cases, p, root


def verify(out, dest):
    cases = pd.read_csv(dest / "cases.csv")
    _, days = e.base.load_selected(e.prepared / "data.pkl")
    env = e.BatchEnv(days, e.Dynamics(e.prepared / "surrogate.pkl"))
    e.base.DeviceConfig().select_device(-1)
    checks, all_episodes, all_paths = 0, [], []
    for seed in range(10):
        source = p.r.ROOT if seed < 5 else root / "e127/main"
        agent = e.base.common.make_agent("fcsrl_actor", seed, e.prepared, e.base.common.config(66))
        agent.load(source / "weighted" / f"seed_{seed}/model_final")
        for cohort in ["diagnostic", "frozen"]:
            rows, expected, _ = load_cases(seed, cohort)
            episodes = list(zip(rows.day, rows.member))
            s = cases.query("seed == @seed and cohort == @cohort and origin >= 0")
            s = s.set_index(KEY).loc[pd.MultiIndex.from_frame(rows[KEY])]
            with np.load(out / f"{cohort}{seed}.npz") as z:
                a, b = z["baseline"], z["updated"]
            for tr in (a, b):
                all_episodes.extend(episodes)
                all_paths.append(tr)
            np.testing.assert_allclose(s.air, (b[..., 0]-a[..., 0]).sum(1)/4, atol=1e-7, rtol=0)
            for k, j, threshold in [("nh", 1, 4), ("do", 2, 2)]:
                np.testing.assert_array_equal(s[k], (b[..., j] > threshold).sum(1)-(a[..., j] > threshold).sum(1))
            np.testing.assert_allclose(s.excess, (np.maximum(b[..., 1]-4, 0) -
                                                  np.maximum(a[..., 1]-4, 0)).sum(1)/4, atol=1e-8, rtol=0)
            # Re-query all qualifying decisions up to the first correction for a seed from each set.
            if seed not in (0, 8):
                continue
            obs = env.reset(episodes)
            for t in range(95):
                if t >= 2:
                    can = ((t >= s.origin.to_numpy()+4) & ((s.stamp < 0) | (t <= s.stamp)).to_numpy()
                           & (expected[:, t-1, 0] - expected[:, t-2, 0] >= 250))
                    if can.any():
                        x = obs[can].copy()
                        actual = p.action_to_air(p.r.action(agent, x))
                        x[:, 10] = expected[can, t-2, 0] / p.AIR_MAX
                        alt = p.action_to_air(p.r.action(agent, x))
                        diff = (actual-alt)/2
                        fired = s.stamp.to_numpy()[can] == t
                        np.testing.assert_array_equal(diff >= 1, fired)
                        np.testing.assert_allclose(diff[fired], -4*s.direct_air.to_numpy()[can][fired], atol=.05, rtol=0)
                        checks += can.sum()
                obs, _, _, _, _ = env.step(p.air_to_action(expected[:, t, 0]))
    traces = np.concatenate(all_paths)
    env.reset(all_episodes)
    for t in range(95):
        _, _, _, _, info = env.step(p.air_to_action(traces[:, t, 0]))
        ref = traces[:, t, 1:]
        np.testing.assert_allclose(np.column_stack((env.state[:, 0], env.state[:, 2:].max(1))), ref,
                                   atol=2e-6, rtol=0)
        np.testing.assert_array_equal(info["v_nh4"], ref[:, 0] > 4)
        np.testing.assert_array_equal(info["v_do"], ref[:, 1] > 2)
    print("checked", len(traces), "sklearn trajectories and", checks, "decision queries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--tables", type=Path, default=HERE)
    args = parser.parse_args()
    verify(args.out, args.tables)

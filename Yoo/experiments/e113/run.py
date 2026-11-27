"""Train paired policies with uniform or doubled sampling of rare training dates."""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from fast_gbm import FastPredictor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e112'))
spec = importlib.util.spec_from_file_location("e112_coverage", HERE.parent / "e112/coverage.py")
coverage112 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coverage112)
observations = coverage112.observations
from response import S, onsets
from darrow_env import BatchEnv, Dynamics

base = S.base
prepared = S.PREPARED
arms = ('uniform', 'weighted')


class FastDynamics(Dynamics):
    """Exact finite numeric inference with Lee/code/darrow/fast_gbm.py at 10b477a."""
    def __init__(self, path):
        super().__init__(path)
        self.predict = FastPredictor(self.models)

    def step(self, state, exo, zone, zone_prev, member):
        x = np.column_stack([state, exo, zone / 1000, zone_prev / 1000])
        return np.clip(state + self.predict(x, member), 0, self.hi)


def prepare(out):
    reference = json.loads((prepared.parent / 'main/manifest.json').read_text())
    for name, expected in reference['input_sha256'].items():
        assert base.sha(prepared / name) == expected, name
    splits, days = base.load_selected(prepared / 'data.pkl')
    train, held = splits['train'][:140], splits['train'][140:]
    assert len(held) == 48 and max(train) < min(held) <= max(held) < min(splits['val'])
    x = np.concatenate([observations(days[d]) for d in train])
    scale = np.diff(np.quantile(x, [.25, .75], axis=0), axis=0)[0]
    assert np.all(scale > 0)
    x /= scale
    nn = NearestNeighbors(n_neighbors=96).fit(x)
    events = []
    for i, day in enumerate(train):
        ts = onsets(days[day]['state'][:, 0])
        if not ts:
            continue
        ds, ids = nn.kneighbors(x[95*i + np.array(ts)])
        for t, dist, idx in zip(ts, ds, ids):
            other = np.flatnonzero(idx // 95 != i)[0]
            value = dist[other] / np.sqrt(x.shape[1])
            if not events:
                direct = np.linalg.norm(x - x[95*i+t], axis=1)
                direct[i*95:(i+1)*95] = np.inf
                np.testing.assert_allclose(value, direct.min() / np.sqrt(x.shape[1]))
            events.append((day, t, value))
    frame = pd.DataFrame(events, columns=['day', 'step', 'distance'])
    threshold = frame.distance.quantile(.90)
    frame['rare'] = frame.distance > threshold
    rare = sorted(frame.loc[frame.rare, 'day'].unique())
    assert 0 < len(rare) < len(train) and set(rare) <= set(train)

    # Compare both predictors and full trajectories before any gradient update.
    episodes = [(day, member) for day in train[::20] for member in range(5)]
    envs = [BatchEnv(days, cls(prepared / 'surrogate.pkl')) for cls in (Dynamics, FastDynamics)]
    for env in envs:
        env.reset(episodes)
    rng = np.random.default_rng(113)
    for _ in range(95):
        actions = rng.uniform(-1, 1, len(episodes))
        obs1, r1, c1, _, _ = envs[0].step(actions)
        obs2, r2, c2, _, _ = envs[1].step(actions)
        for v1, v2 in ((envs[0].state, envs[1].state), (obs1, obs2), (r1, r2), (c1, c2)):
            np.testing.assert_array_equal(v1, v2)
    out.mkdir(parents=True, exist_ok=False)
    frame.to_csv(out / 'events.csv', index=False)
    base.save(out / 'plan.json', dict(train=train, held=held, winter=splits['val'], rare=rare,
              threshold=threshold, n_events=len(frame), n_rare=int(frame.rare.sum()),
              input_sha256=reference['input_sha256'], code_commit=subprocess.check_output(
                  ['git', 'rev-parse', 'HEAD'], cwd=HERE, text=True).strip()))
    print('Preflight passed. Rare dates:', len(rare), 'of', len(train), flush=True)


class DayPool(list):
    def __init__(self, dates):
        super().__init__(dates)
        self.draws = Counter()

    def __getitem__(self, i):
        day = super().__getitem__(i)
        self.draws[day] += 1
        return day


def train(out, arm, seed, updates):
    plan = json.loads((out / 'plan.json').read_text())
    train_days = plan['train']
    pool = DayPool(train_days + (plan['rare'] if arm == 'weighted' else []))
    splits, days = base.load_selected(prepared / 'data.pkl')
    assert train_days + plan['held'] == splits['train'] and plan['winter'] == splits['val']
    run = out / arm / f'seed_{seed}'
    args = argparse.Namespace(method='fcsrl_actor', seed=seed, out=run, updates=updates, prepared=prepared)
    base.e30.train(args, load_data=lambda _: ({'train': pool, 'val': splits['val']}, days),
                   dynamics=FastDynamics)
    assert sum(pool.draws.values()) == updates * 64
    base.save(run / 'draws.json', dict(pool.draws))
    agent = base.common.make_agent(args.method, seed, prepared, base.common.config(updates))
    agent.load(run / 'model_final')
    env = BatchEnv(days, Dynamics(prepared / 'surrogate.pkl'))
    for name in ('held', 'winter'):
        metrics = base.evaluate(agent, env, plan[name], run, name)
        if name == 'winter':
            recorded = json.loads((run / 'metrics.json').read_text())['val_final']
            for key in ('air_m3', 'cost', 'nh4_viol_pct', 'do_viol_pct'):
                np.testing.assert_allclose(metrics[key], recorded[key], rtol=0, atol=1e-8)
        base.save(run / f'{name}_metrics.json', metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--arm', choices=arms)
    parser.add_argument('--seed', type=int, choices=range(5), default=0)
    parser.add_argument('--updates', type=int, default=66)
    args = parser.parse_args()
    if args.arm:
        train(args.out, args.arm, args.seed, args.updates)
    else:
        prepare(args.out)

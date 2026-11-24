"""Recompute daily metrics from saved states and apply the prespecified paired rule."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ['air_m3', 'nh4_pct', 'do_pct', 'cost', 'within_pct']
BLOCKS = ('held', 'winter', 'winter_early', 'winter_late')


def read(path):
    return json.loads(path.read_text())


def verify_schedule(plan, arm, seed, root):
    log = read(root / 'train_log.json')
    draws = read(root / 'draws.json')
    assert len(log) == 66
    pool = plan['train'] + (plan['rare'] if arm == 'weighted' else [])
    rng = np.random.default_rng(seed)
    expected = Counter()
    for index, record in enumerate(log, 1):
        assert record['update'] == index and record['member'] == rng.integers(5)
        for _ in range(8):
            expected.update(pool[i] for i in rng.integers(len(pool), size=8))
    assert expected == Counter(draws) and sum(expected.values()) == 4224
    assert set(draws) == set(plan['train'])
    return sum(draws[d] for d in plan['rare']) / 4224


def daily_metrics(path, dates):
    with np.load(path) as raw:
        state = raw['states'].reshape(len(dates), 5, 95, 5)
        air = raw['air'].reshape(len(dates), 5, 95)
    assert np.isfinite(state).all() and np.isfinite(air).all() and np.all((0 <= air) & (air <= 9000))
    nh4 = (state[:, :, :, 0] > 4).sum(axis=2)
    do = (state[:, :, :, 2:].max(axis=3) > 2).sum(axis=2)
    cost = nh4 + do
    values = np.column_stack([(air * .25).sum(axis=2).mean(axis=1), nh4.mean(axis=1) * 100/95,
                              do.mean(axis=1) * 100/95, cost.mean(axis=1), (cost <= 12).mean(axis=1) * 100])
    return pd.DataFrame(values, index=dates, columns=KEYS)


def gate(values):
    return (values.air_m3 < 0) & (values.nh4_pct < 0) & (values.do_pct <= 0)


def report(source, dest):
    plan = read(source / 'plan.json')
    runs, draws, checksums = [], [], {}
    for seed in range(5):
        for arm in ('uniform', 'weighted'):
            root = source / arm / f'seed_{seed}'
            assert read(root / 'metrics.json')['steps'] == 401280
            exposure = verify_schedule(plan, arm, seed, root)
            draws.append(dict(seed=seed, arm=arm, rare_draw_pct=100 * exposure))
            for name in ('held', 'winter'):
                frame = daily_metrics(root / f'{name}_rollout.npz', plan[name])
                metrics = read(root / f'{name}_metrics.json')
                expected = [metrics[k] for k in ['air_m3', 'nh4_viol_pct', 'do_viol_pct', 'cost', 'days_within_limit_pct']]
                np.testing.assert_allclose(frame.mean(), expected, atol=1e-8, rtol=0)
                segments = {name: frame}
                if name == 'winter':
                    segments.update(winter_early=frame.iloc[:20], winter_late=frame.iloc[20:])
                for block, daily in segments.items():
                    runs.append(dict(seed=seed, arm=arm, block=block, **daily.mean().to_dict()))
            for f in sorted([*root.glob('model_final/*'), *root.glob('*_rollout.npz'), root/'train_log.json',
                             root/'effective_config.json', root/'draws.json']):
                checksums[str(f.relative_to(source))] = hashlib.sha256(f.read_bytes()).hexdigest()
    samples = pd.DataFrame(draws).pivot(index='seed', columns='arm', values='rare_draw_pct')
    assert (samples.weighted > samples.uniform).all()
    per_arm = pd.DataFrame(runs)
    paired = per_arm.set_index(['arm', 'block', 'seed'])[KEYS]
    delta = paired.loc['weighted'] - paired.loc['uniform']
    # Percentages are exact transforms of integer counts. Snap roundoff on equality.
    delta = delta.mask(delta.abs() < 1e-10, 0.0)
    delta['joint'] = gate(delta)
    grouped = per_arm.groupby(['block', 'arm'])[KEYS].agg(['mean', 'std'])
    grouped.columns = ['_'.join(col) for col in grouped.columns]
    result = {}
    for block in BLOCKS:
        pairs = delta.loc[block]
        mean = pairs[KEYS].mean()
        passes = int(pairs.joint.sum())
        result[block] = dict(joint_seeds=passes, joint_seed_ids=pairs.index[pairs.joint].tolist(),
                             delta=mean.to_dict(), pass_gate=bool(passes >= 4 and gate(mean)))
    result['adopt'] = result['held']['pass_gate'] and result['winter']['pass_gate']
    dest.mkdir(parents=True, exist_ok=True)
    per_arm.to_csv(dest / 'per_seed.csv', index=False)
    delta.to_csv(dest / 'paired.csv')
    grouped.to_csv(dest / 'summary.csv')
    samples.to_csv(dest / 'sampling.csv')
    (dest / 'decision.json').write_text(json.dumps(result, indent=2) + '\n')
    (source / 'checksums.json').write_text(json.dumps(checksums, indent=2) + '\n')
    print(grouped.round(3).to_string())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('dest', type=Path)
    args = parser.parse_args()
    report(args.source, args.dest)

"""e44와 같은 예산으로 기본 FCSRL과 조건별 NH4 제약을 비교한다."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch
from tqdm import tqdm

from conditional import (config, save, DeviceConfig, load_selected, group_definition, TrackedEnv,
                         Dynamics, collect, make_agent, modules, evaluate, detailed_metrics, all_episodes)


def train(a):
    a.out.mkdir(parents=True, exist_ok=False)
    DeviceConfig().select_device(-1)
    torch.set_num_threads(1)
    splits, days = load_selected(a.prepared / 'data.pkl')
    definition = group_definition(splits, days)
    c = config(a.updates)
    agent = make_agent(a.arm, a.seed, a.prepared, c, definition)
    env = TrackedEnv(days, Dynamics(a.corrected), definition)
    save(a.out / 'effective_config.json', dict(arm=a.arm, seed=a.seed, config=c.to_dict(), groups=definition))
    norm = agent.obs_normalizer.state()
    prior = {k: v.clone() for k, v in agent.prior.state_dict().items()}
    rng, records, start = np.random.default_rng(a.seed), [], time.monotonic()
    for update in tqdm(range(1, c.n_updates + 1), desc=f'e46 {a.arm} s{a.seed}', mininterval=10):
        buf, diag = collect(agent, c, env, splits['train'], rng)
        info = agent.update(buf)
        assert all(np.isfinite(v) for v in info.values())
        assert all(torch.isfinite(p).all() for m in modules(agent) for p in m.parameters())
        if a.arm == 'conditional':
            agent.update_group_constraints(diag)
            lambdas = agent.group_lambdas.tolist()
        else:
            agent.update_constraints([diag['nh4_cost'], diag['do_cost']])
            lambdas = agent.multipliers.tolist()
        records.append(dict(update=update, multipliers=lambdas, **diag, **info))
        elapsed = time.monotonic() - start
        save(a.out / 'train_log.json', records)
        save(a.out / 'progress.json', dict(phase='training', update=update, updates=c.n_updates,
             steps=update * 6080, elapsed_seconds=elapsed, estimated_remaining_seconds=elapsed / update * (c.n_updates - update)))
    assert norm == agent.obs_normalizer.state()
    assert all(torch.equal(v, prior[k]) for k, v in agent.prior.state_dict().items())
    agent.save(a.out / 'model_final')
    results = {}
    for split in ('train', 'val'):
        save(a.out / 'progress.json', dict(phase='evaluating_' + split, steps=c.actual_steps))
        _, _, states, air = evaluate(env, splits[split], lambda obs: agent.act_det(obs)[:, 0])
        summary, rows, members = detailed_metrics(states, air, all_episodes(splits[split], 5))
        np.savez_compressed(a.out / f'{split}_rollout.npz', states=states, air=air)
        save(a.out / f'{split}_episodes.json', rows)
        groups = {}
        for g in (0, 1):
            subset = [r for r in rows if definition['group'][r['day']] == g]
            groups[str(g)] = dict(n_days=len(subset) // 5,
                                  **{k: float(np.mean([r[k] for r in subset])) for k in
                                     ('air_m3', 'nh4_cost', 'do_cost', 'nh4_excess_mg_h_L', 'do_excess_mg_h_L')}) if subset else None
        results[split] = dict(summary=summary, groups=groups, members=members)
    save(a.out / 'metrics.json', dict(arm=a.arm, seed=a.seed, steps=c.actual_steps,
         normalizer_and_prior_frozen=True, results=results, elapsed_seconds=time.monotonic() - start))
    save(a.out / 'progress.json', dict(phase='complete', steps=c.actual_steps))
    print('DONE', a.arm, a.seed, results['val'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=['baseline', 'conditional'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--updates', type=int, default=66)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--corrected', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    train(p.parse_args())

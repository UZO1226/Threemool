"""e63 배분 조건 학습. e60 train_shielded.run 과 같은 순서이며 행동과 수집, 평가만 2차원이다."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import alloc as al  # noqa: E402
t60, se, base, h = al.t60, al.se, al.base, al.h
EVAL = (('val', 'none'), ('val', 'park_e10'), ('val', 'park_e19'), ('train', 'none'), ('train', 'park_e10'))


def run(a):
    manifest = json.loads(a.manifest.read_text())
    assert all(base.sha(p) == v for p, v in manifest['hashes'].items())
    a.out.mkdir(parents=True, exist_ok=False)
    device = 'cuda:0' if a.method == 'fcsrl' else 'cpu'
    c = base.common.config(a.updates)
    learner, collector = al.make_pair(a.method, a.seed, a.prepared, c, device)
    rng = np.random.default_rng(a.seed)
    metadata = dict(method=a.method, seed=a.seed, condition='alloc', prior_mode='on', backend=device, updates=a.updates, train_shield=a.train_shield,
                    w_max=al.W_MAX, back=al.BACK.tolist(), code_commit=manifest['code_commit'], data_sha256=base.sha(a.prepared / 'data.pkl'),
                    model_sha256=base.sha(a.prepared / 'surrogate.pkl'), min_brake_sha256=base.sha(se.HERE / 'min_brake.py'))
    base.save(a.out / 'effective_config.json', dict(**metadata, config=c.to_dict(), cpu_threads=1, action_dim=al.A_DIM,
              gpu_physical_index=1 if device != 'cpu' else None, cost_limit=c.cost_limit, observation_dim=11, test_evaluated=False, prepared=str(a.prepared)))
    splits, days = base.load_selected(a.prepared / 'data.pkl')
    dyn = base.Dynamics(a.prepared / 'surrogate.pkl')
    env = base.BatchEnv(days, dyn)
    train_shield = se.make_shield(a.train_shield, dyn)
    prior = {k: v.detach().cpu().clone() for k, v in learner.prior.state_dict().items()}
    norm = learner.obs_normalizer.state()
    records, start = [], time.monotonic()
    for u in tqdm(range(1, c.n_updates + 1), desc=f'e63 alloc {a.method} s{a.seed}', mininterval=10):
        tick = time.monotonic()
        h.sync_collector(learner, collector)
        h.configure('cpu')
        buf, diag = al.collect(collector, c, env, splits['train'], rng, train_shield)
        collection = time.monotonic() - tick
        tick = time.monotonic()
        info = h.update(learner, buf)
        updating = time.monotonic() - tick
        learner.pid.update(diag['daily_cost_mean'], c.cost_limit)
        learner.lagrg = learner.pid.get_lagrg()
        records.append(dict(update=u, **{'lambda': learner.lagrg}, **diag, **info, collection_seconds=collection, update_seconds=updating))
        h.checkpoint(a.out / 'resume.pt', learner, rng, records, metadata)
        base.save(a.out / 'train_log.json', records)
        elapsed = time.monotonic() - start
        base.save(a.out / 'progress.json', dict(phase='training', update=u, updates=c.n_updates, steps=u * 6080, total_steps=c.actual_steps,
                  elapsed_seconds=elapsed, estimated_remaining_seconds=elapsed / u * (c.n_updates - u), backend=device))
    assert learner.obs_normalizer.state() == norm
    assert all(torch.equal(v.detach().cpu(), prior[k]) for k, v in learner.prior.state_dict().items())
    h.sync_collector(learner, collector)
    h.configure('cpu')
    collector.save(a.out / 'model_final')
    restored = al.make_agent(a.method, a.seed, a.prepared, c)
    restored.load(a.out / 'model_final')
    fp = base.fingerprint(collector)
    assert base.fingerprint(restored) == fp
    act = lambda obs: collector.act_det(obs)  # noqa: E731
    summaries = {}
    for split, name in EVAL:
        base.save(a.out / 'progress.json', dict(phase=f'evaluating_{split}_{name}', steps=c.actual_steps))
        result, rows, states, airs, zones = al.rollout(env, splits[split], act, se.make_shield(name, dyn))
        assert base.fingerprint(collector) == fp
        summaries[f'{split}_{name}'] = result
        base.save(a.out / f'{split}_{name}_episodes.json', rows)
        np.savez_compressed(a.out / f'{split}_{name}_rollout.npz', states=states, air=airs, zones=zones)
    base.save(a.out / 'metrics.json', dict(method=a.method, seed=a.seed, condition='alloc', **summaries, val_final=summaries['val_park_e10'],
              steps=c.actual_steps, normalizer_and_prior_frozen=True, test_evaluated=False, backend=device, policy_fingerprint=fp,
              final_lambda=learner.lagrg, elapsed_seconds=time.monotonic() - start))
    assert all(base.sha(p) == v for p, v in manifest['hashes'].items())
    base.save(a.out / 'progress.json', dict(phase='complete', steps=c.actual_steps, elapsed_seconds=time.monotonic() - start))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['ppo', 'fcsrl'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--updates', type=int, default=66)
    p.add_argument('--train-shield', default='park_e10')
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    run(p.parse_args())

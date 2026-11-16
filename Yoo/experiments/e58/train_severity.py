"""e58 세 조건을 같은 CPU1 수집과 GPU1 갱신에서 실행한다."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from tqdm import tqdm

import severity as s
base, h = s.base, s.h


def run(a):
    manifest = json.loads(a.manifest.read_text())
    assert all(base.sha(p) == v for p, v in manifest['hashes'].items())
    if a.resume:
        assert (a.out / 'resume.pt').exists()
    else:
        a.out.mkdir(parents=True, exist_ok=False)
    c = base.common.config(a.updates)
    learner, collector = s.make_pair(a.method, a.seed, a.prepared, c)
    rng = np.random.default_rng(a.seed)
    metadata = dict(method=a.method, seed=a.seed, prior_mode='on', backend='cuda:0', updates=a.updates,
                    code_commit=manifest['code_commit'], data_sha256=base.sha(a.prepared/'data.pkl'),
                    model_sha256=base.sha(a.prepared/'surrogate.pkl'), auxiliary_weight=s.WEIGHT,
                    window=s.WINDOW, target_scale=s.SCALE)
    records = h.restore(a.out/'resume.pt', learner, rng, metadata) if a.resume else []
    base.save(a.out/'effective_config.json', dict(**metadata, config=c.to_dict(), cpu_threads=1,
              gpu_physical_index=1, test_evaluated=False, prepared=str(a.prepared)))
    splits, days = base.load_selected(a.prepared/'data.pkl')
    env = s.RecordingEnv(days, base.Dynamics(a.prepared/'surrogate.pkl'))
    norm = learner.obs_normalizer.state()
    prior = {k:v.detach().cpu().clone() for k,v in learner.prior.state_dict().items()}
    start = time.monotonic()
    initial_updates = len(records)
    for u in tqdm(range(initial_updates+1, c.n_updates+1), desc=f'e58 {a.method} s{a.seed}', mininterval=10):
        tick = time.monotonic()
        h.sync_collector(learner, collector)
        h.configure('cpu')
        buf, diag = s.collect(collector, c, env, splits['train'], rng)
        collection = time.monotonic()-tick
        tick = time.monotonic()
        with h.gpu_slot(str(learner.dev)):
            diagnostics = s.gradient_diagnostics(learner, buf)
        info = dict(**h.update(learner, buf), **diagnostics)
        updating = time.monotonic()-tick
        learner.pid.update(diag['daily_cost_mean'], c.cost_limit)
        learner.lagrg = learner.pid.get_lagrg()
        records.append(dict(update=u, **{'lambda':learner.lagrg}, **diag, **info,
                            collection_seconds=collection, update_seconds=updating))
        h.checkpoint(a.out/'resume.pt', learner, rng, records, metadata)
        base.save(a.out/'train_log.json', records)
        elapsed = time.monotonic()-start
        base.save(a.out/'progress.json', dict(phase='training', update=u, updates=c.n_updates, steps=u*6080,
                  total_steps=c.actual_steps, elapsed_seconds=elapsed,
                  estimated_remaining_seconds=elapsed/(u-initial_updates)*(c.n_updates-u), backend='gpu1_update'))
    assert norm == learner.obs_normalizer.state()
    assert all(torch.equal(v.detach().cpu(), prior[k]) for k,v in learner.prior.state_dict().items())
    h.sync_collector(learner, collector)
    h.configure('cpu')
    collector.save(a.out/'model_final')
    restored = s.make_agent(a.method, a.seed, a.prepared, c)
    restored.load(a.out/'model_final')
    assert s.fingerprint(restored) == s.fingerprint(collector)
    summaries = {}
    for split in ['val', 'train']:
        base.save(a.out/'progress.json', dict(phase='evaluating_'+split, steps=c.actual_steps))
        before = s.fingerprint(collector)
        summaries[split+'_final'] = base.evaluate(collector, env, splits[split], a.out, split)
        assert before == s.fingerprint(collector)
    base.save(a.out/'metrics.json', dict(method=a.method, seed=a.seed, **summaries, steps=c.actual_steps,
              prior_mode='on', backend='gpu1_update', policy_fingerprint=base.fingerprint(collector),
              full_fingerprint=s.fingerprint(collector), test_evaluated=False,
              normalizer_and_prior_frozen=True, elapsed_seconds=time.monotonic()-start))
    assert all(base.sha(p) == v for p,v in manifest['hashes'].items())
    base.save(a.out/'progress.json', dict(phase='complete', steps=c.actual_steps, elapsed_seconds=time.monotonic()-start))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=s.METHODS, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--updates', type=int, default=66)
    p.add_argument('--prepared', type=Path, default=Path('/home/jio/fcsrl_src/results/e50/prepared'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    run(p.parse_args())

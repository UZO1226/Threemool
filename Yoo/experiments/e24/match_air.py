"""확장과 같은 val 공기량이 되도록 기본 FCSRL 행동에 상수 offset을 더하는 사후 대조."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
import torch
from fcsrl.utils import DeviceConfig
from recovery import make_agent
from env import AerationEnv
from evaluate import batched_rollout, episodes
from ppo_config import PPOConfig


def match(job):
    seed, tag, root, prepared, data, model = job
    torch.set_num_threads(1)
    DeviceConfig().select_device(-1)
    root, prepared = Path(root), Path(prepared)
    base_path = root / 'runs/fcsrl' / f'seed_{seed}'
    ext_path = root / 'runs/fcsrl_recovery' / f'seed_{seed}'
    base = json.loads((base_path / 'cross_evaluation.json').read_text())[tag]['summary']
    ext = json.loads((ext_path / 'cross_evaluation.json').read_text())[tag]['summary']
    agent = make_agent('fcsrl', seed, prepared, PPOConfig(total_steps=200000, n_chunks=2, init_log_std=-1.6))
    agent.load(base_path / 'model_final')
    env = AerationEnv(data, model, split='val'); env.fixed = episodes(env)
    evaluations = []
    cache = {0.: base}

    def objective(offset):
        if offset not in cache:
            sm, _ = batched_rollout(env, env.sur, lambda o: np.clip(agent.act_det(o)[:, 0] + offset, -1, 1), 15.)
            cache[offset] = sm
            evaluations.append({'offset': float(offset), 'air_m3': sm['air_m3'], 'cost': sm['cost']})
        return cache[offset]['air_m3'] - ext['air_m3']

    failure = None
    try:
        brentq(objective, -2., 2., xtol=1e-5, maxiter=40)
    except (ValueError, RuntimeError) as exc:
        failure = str(exc)
    best = min(cache, key=lambda x: abs(cache[x]['air_m3'] - ext['air_m3']))
    matched = cache[best]
    mismatch = 100 * (matched['air_m3'] / ext['air_m3'] - 1)
    return {'seed': seed, 'model': tag, 'extension': ext, 'base': base, 'matched': matched,
            'offset': best, 'air_mismatch_percent': mismatch, 'matched_within_tolerance': abs(mismatch) <= .1,
            'cost_difference': ext['cost'] - matched['cost'],
            'cost_reduction_percent': 100 * (1 - ext['cost'] / matched['cost']) if matched['cost'] else None,
            'solver_note': failure, 'evaluations': evaluations}


def main(a):
    inputs = sorted(a.root.glob('runs/*/seed_*/cross_evaluation.json'))
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    jobs = [(seed, tag, str(a.root), str(a.prepared), str(a.data), str(model))
            for seed in range(5) for tag, model in [('mlp', a.model), ('gbm', a.gbm)]]
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(match, job) for job in jobs]
        for f in as_completed(futures):
            row = f.result(); rows.append(row)
            print('matched', row['seed'], row['model'], 'delta cost', round(row['cost_difference'], 3),
                  'air error %', round(row['air_mismatch_percent'], 5), flush=True)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in before.items())
    a.out.write_text(json.dumps({'input_sha256': before, 'matches': sorted(rows, key=lambda r: (r['model'], r['seed']))}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root', 'prepared', 'data', 'model', 'gbm', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    main(p.parse_args())

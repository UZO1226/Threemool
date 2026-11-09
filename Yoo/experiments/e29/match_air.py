"""각 ablation을 full 보정 정책의 공기량에 맞추는 val 사후 대조."""
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
import torch
from fcsrl.utils import DeviceConfig
from ablation import make_agent
from env import AerationEnv
from evaluate import batched_rollout, episodes
from ppo_config import PPOConfig


def match(job):
    method, seed, tag, root, prepared, data, model = job
    DeviceConfig().select_device(-1)
    torch.set_num_threads(1)
    root, prepared = Path(root), Path(prepared)
    base_path = root / 'runs' / method / f'seed_{seed}'
    ext_path = root / 'runs/fcsrl_preserve' / f'seed_{seed}'
    base = json.loads((base_path / 'cross_evaluation.json').read_text())[tag]['summary']
    ext = json.loads((ext_path / 'cross_evaluation.json').read_text())[tag]['summary']
    agent = make_agent(method, seed, prepared, PPOConfig(total_steps=200000, n_chunks=2, init_log_std=-1.6))
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
    return {'comparator': method, 'seed': seed, 'model': tag, 'extension': ext, 'base': base, 'matched': matched,
            'offset': best, 'air_mismatch_percent': mismatch, 'matched_within_tolerance': abs(mismatch) <= .1,
            'cost_difference': ext['cost'] - matched['cost'],
            'cost_reduction_percent': 100 * (1 - ext['cost'] / matched['cost']) if matched['cost'] else None,
            'solver_note': failure, 'evaluations': evaluations}


def main(a):
    if a.out.exists():
        raise ValueError(f'Refusing to overwrite {a.out}')
    jobs = [(method, seed, tag, str(a.root), str(a.prepared), str(a.data), str(model))
            for method in ('fcsrl', 'fcsrl_actor', 'fcsrl_critic', 'fcsrl_extra_ppo')
            for seed in range(5) for tag, model in [('mlp', a.model), ('gbm', a.gbm)]]
    rows, before, active = [], {}, {}
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=4) as pool:
        while jobs or active:
            status_file = a.root / 'status.json'
            if status_file.exists() and json.loads(status_file.read_text())['phase'] == 'failed':
                raise RuntimeError('Training runner failed')
            for job in jobs.copy():
                if len(active) >= 4:
                    break
                method, seed, tag = job[:3]
                paths = [a.root / 'runs' / m / f'seed_{seed}' for m in (method, 'fcsrl_preserve')]
                if not all((p / 'cross_evaluation.json').exists() for p in paths):
                    if a.wait_for_runs:
                        continue
                    raise FileNotFoundError(f'Incomplete run: {method} seed {seed}')
                for p in paths:
                    for f in [p / 'cross_evaluation.json', *sorted((p / 'model_final').glob('*'))]:
                        before[str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
                active[pool.submit(match, job)] = job
                jobs.remove(job)
            if active:
                completed, _ = wait(active, timeout=2, return_when=FIRST_COMPLETED)
                for future in completed:
                    row = future.result(); rows.append(row); del active[future]
                    print('matched', len(rows), '/40', row['comparator'], row['seed'], row['model'],
                          'delta cost', round(row['cost_difference'], 3),
                          'air error %', round(row['air_mismatch_percent'], 5), flush=True)
            elif jobs:
                time.sleep(2)
            progress = {'completed': len(rows), 'total': 40, 'active': [list(j[:3]) for j in active.values()],
                        'waiting': len(jobs), 'elapsed_seconds': time.monotonic() - started}
            tmp = a.out.with_suffix('.progress.tmp')
            tmp.write_text(json.dumps(progress, indent=2))
            tmp.replace(a.out.with_suffix('.progress.json'))
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in before.items())
    a.out.write_text(json.dumps({'input_sha256': before, 'matches': sorted(rows, key=lambda r: (r['model'], r['comparator'], r['seed']))}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root', 'prepared', 'data', 'model', 'gbm', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--wait-for-runs', action='store_true')
    main(p.parse_args())

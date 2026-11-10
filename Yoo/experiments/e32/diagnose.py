"""e30 저장 정책으로 공기량 차이와 학습/검증 차이를 분리한다. 재학습하지 않는다."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e30'))
from common import config, make_agent, sha, save
from darrow_env import BatchEnv, Dynamics, load_days, evaluate
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import t, ttest_1samp
import torch
from fcsrl.utils import DeviceConfig


def worker(job):
    seed, root, prepared = job
    root, prepared = Path(root), Path(prepared)
    torch.set_num_threads(1)
    DeviceConfig().select_device(-1)
    splits, days = load_days(prepared / 'data.pkl')
    env = BatchEnv(days, Dynamics(prepared / 'surrogate.pkl'))
    policy, reference, train = {}, {}, {}
    for method in ('ppo', 'fcsrl'):
        run = root / 'runs' / method / f'seed_{seed}'
        a = make_agent(method, seed, prepared, config())
        a.load(run / 'model_final')
        policy[method] = a
        reference[method] = json.loads((run / 'metrics.json').read_text())['val_final']
        replay = evaluate(env, splits['val'], lambda x: a.act_det(x)[:, 0])
        assert all(abs(replay[k]-reference[method][k]) < 1e-6 for k in replay)
        train[method] = evaluate(env, splits['train'], lambda x: a.act_det(x)[:, 0])
    target = reference['ppo']['air_m3']
    cache = {0.: reference['fcsrl']}
    def objective(offset):
        if offset not in cache:
            cache[offset] = evaluate(env, splits['val'],
                lambda x: np.clip(policy['fcsrl'].act_det(x)[:, 0]+offset, -1, 1))
        return cache[offset]['air_m3']-target
    error = None
    try:
        brentq(objective, -2., 2., xtol=1e-5, maxiter=30)
    except (ValueError, RuntimeError) as exc:
        error = str(exc)
    best = min(cache, key=lambda x: abs(cache[x]['air_m3']-target))
    mismatch = 100*(cache[best]['air_m3']/target-1)
    assert abs(mismatch) <= .1, (seed, mismatch, error)
    return {'seed': seed, 'original_validation': reference, 'deterministic_train': train,
            'matched_fcsrl': cache[best], 'offset': best, 'air_mismatch_pct': mismatch,
            'cost_difference_fcsrl_minus_ppo': cache[best]['cost']-reference['ppo']['cost'],
            'solver_note': error, 'search': [{'offset': x, **v} for x, v in cache.items()]}


def main(a):
    a.out.mkdir(parents=True, exist_ok=False)
    inputs = [*a.prepared.glob('*'), *a.root.glob('runs/*/seed_*/metrics.json'),
              *a.root.glob('runs/*/seed_*/train_log.json'), *a.root.glob('runs/*/seed_*/model_final/*')]
    hashes = {str(p): sha(p) for p in inputs if p.is_file()}
    data = pd.read_pickle(a.prepared / 'data.pkl')
    distribution = {}
    for split in ('train', 'val'):
        table = pd.concat([data['data'].loc[d] for d in data['days'][split]])
        costs = []
        for day in data['days'][split]:
            d = data['data'].loc[day]
            costs.append(float(((d.nh4.to_numpy()[1:] > 4).astype(float)
                                + (d[['do1','do2','do3']].to_numpy()[1:].max(1) > 2)).sum()))
        distribution[split] = {'days': len(costs), 'observed_cost_mean': float(np.mean(costs)),
            'means': table[['temp','q_in','inf_s_nh','nh4','air']].mean().to_dict()}
    logs = {}
    for method in ('ppo','fcsrl','fcsrl_actor','fcsrl_critic','fcsrl_preserve','fcsrl_extra_ppo'):
        runs = [json.loads(p.read_text()) for p in sorted((a.root/'runs'/method).glob('seed_*/train_log.json'))]
        record = {}
        for key in ('lambda', 'raw_action_clip_percent', 'std', 'encoder_sync_kl',
                    'preserve_before_kl','preserve_after_kl','preserve_accepted'):
            values = [[v[key] for v in run if key in v] for run in runs]
            if all(values):
                record[key] = {'run_mean': [float(np.mean(v)) for v in values],
                               'mean': float(np.mean([np.mean(v) for v in values]))}
        record['max_lambda'] = max(v['lambda'] for run in runs for v in run)
        logs[method] = record
    save(a.out/'audit.json', {'distribution': distribution, 'training_logs': logs,
         'physical_time': {'old_minutes_per_step':5,'new_minutes_per_step':15,
         'gamma_same_physical_discount': .99**3, 'feasibility_same_physical_discount':.9**3,
         'gae_lambda_same_physical_decay':.95**3},
         'partial_observation': {'not_in_policy_obs': ['inf_s_nd','inf_x_nd','inf_s_s','inf_x_s',
                 'current_zone_fractions','previous_zone_air_vector'],
              'interpretation':'Environment transition uses these values. Their contribution to the performance gap remains untested.'}})
    started, rows = time.monotonic(), []
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(worker, (s, str(a.root), str(a.prepared))) for s in range(5)]
        for future in as_completed(jobs):
            row = future.result()
            rows.append(row)
            save(a.out/'progress.json', {'completed':len(rows),'total':5,'elapsed_seconds':time.monotonic()-started})
            print('DONE',row['seed'],'matched delta',row['cost_difference_fcsrl_minus_ppo'], flush=True)
    rows.sort(key=lambda x:x['seed'])
    delta = np.array([r['cost_difference_fcsrl_minus_ppo'] for r in rows])
    half = t.ppf(.975,4)*delta.std(ddof=1)/np.sqrt(5)
    summary = {'matched_fcsrl_minus_ppo_mean':float(delta.mean()),
        'paired_ci95':[float(delta.mean()-half),float(delta.mean()+half)],
        'paired_p':float(ttest_1samp(delta,0).pvalue),
        'matched_air_tolerance_pct': .1, 'post_hoc_diagnostic':True, 'test_used':False,
        'warning':'Constant-offset action matching changes feedback trajectories and is not an optimal equal-resource policy frontier.',
        'train_cost':{m:float(np.mean([r['deterministic_train'][m]['cost'] for r in rows])) for m in ('ppo','fcsrl')},
        'val_cost':{m:float(np.mean([r['original_validation'][m]['cost'] for r in rows])) for m in ('ppo','fcsrl')}}
    assert all(sha(p)==digest for p,digest in hashes.items())
    save(a.out/'results.json', {'summary':summary,'runs':rows,'input_sha256':hashes,
         'source_sha256':{str(HERE/'diagnose.py'):sha(HERE/'diagnose.py')},'elapsed_seconds':time.monotonic()-started})
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root','prepared','out'):
        p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())

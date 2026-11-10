"""고정 정책과 훈련에서 범위를 고정한 BC 공기량 기준을 비교한다."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'e34'))
from audit import metrics
from dual import config, make_agent, METHODS, save
from common import sha
from darrow_env import BatchEnv, Dynamics, load_days, all_episodes, action_to_air
from fcsrl.utils import DeviceConfig


def now():
    return datetime.now(timezone.utc).isoformat()


def replay(env, dates, act):
    obs = env.reset(all_episodes(dates, env.dyn.n_members))
    states, airs = [], []
    clipped = 0
    for _ in range(env.T):
        action = np.asarray(act(obs)).reshape(-1)
        clipped += int((np.abs(action) > 1).sum())
        obs, reward, cost, done, info = env.step(action)
        assert np.allclose(info['air'], action_to_air(action))
        assert np.allclose(env.prev_zone.sum(1), info['air'])
        assert np.array_equal(cost, (env.state[:, 0] > 4).astype(int) + (env.state[:, 2:].max(1) > 2))
        assert np.allclose(reward, -info['air'] / 9000)
        states.append(env.state.copy())
        airs.append(info['air'].copy())
    assert done and env.T == 95
    states, airs = np.stack(states, 1), np.stack(airs, 1)
    result = metrics(states, airs)
    # The source specifies mg/L, without independently confirming nitrogen mass basis.
    result = {k.replace('mg_N_h_L', 'mg_h_L').replace('mg_N_L', 'mg_L').replace('mg_O2_L', 'mg_L'): v
              for k, v in result.items()}
    result['action_clip_pct'] = 100 * clipped / airs.size
    n = (states[:, :, 0] > 4).sum(1)
    d = (states[:, :, 2:].max(2) > 2).sum(1)
    excess = np.maximum(states[:, :, 0] - 4, 0).sum(1) * .25
    episodes = [{'day': day, 'member': int(m), 'air_m3': float(airs[i].sum() * .25),
                 'nh4_cost': int(n[i]), 'do_cost': int(d[i]), 'nh4_excess_mg_h_L': float(excess[i])}
                for i, (day, m) in enumerate(env.episodes)]
    day_n = n.reshape(len(dates), env.dyn.n_members).mean(1)
    result['worst_nh4_dates'] = [{'day': dates[i], 'member_mean_count': float(day_n[i])}
                               for i in np.argsort(-day_n)[:5]]
    return result, episodes


def main(a):
    start = time.monotonic()
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / 'manifest.json').exists():
        raise FileExistsError('Use a fresh output directory.')
    DeviceConfig().select_device(-1)
    torch.set_num_threads(1)
    splits, days = load_days(a.prepared / 'data.pkl')
    assert len(splits['train']) == 188 and len(splits['val']) == 40
    old_audit = json.loads((HERE.parent / 'e34/audit.json').read_text())
    e33_manifest = json.loads((a.runs / 'manifest.json').read_text())
    assert all(sha(a.prepared / k) == v for k, v in e33_manifest['input_sha256'].items())
    inputs = {str(p): sha(p) for p in [*a.prepared.iterdir(), a.corrected] if p.is_file()}
    for method in METHODS:
        for seed in (0, 1):
            for p in (a.runs / 'runs' / method / f'seed_{seed}' / 'model_final').rglob('*'):
                if p.is_file():
                    inputs[str(p)] = sha(p)
    inputs[str(HERE.parent / 'e34/audit.json')] = sha(HERE.parent / 'e34/audit.json')
    train_air = np.concatenate([days[d]['air'][:-1] for d in splits['train']])
    q10, q90 = np.quantile(train_air, [.1, .9])
    radius = min(4500., float(np.ceil((q90 - q10) / 2 / 500) * 500))
    offsets = np.arange(-radius, radius + 1, 500).tolist()
    protocol = {'locked_at': now(), 'train_dates': splits['train'], 'val_dates': splits['val'],
                'offset_rule': 'radius = ceil((train q90 - q10)/2/500)*500, capped at 4500; spacing 500 m3/h',
                'train_flow_q10_q90': [float(q10), float(q90)], 'offsets_m3_h': offsets,
                'baseline_selection': 'Corrected train: minimum air among mean NH4<=6 AND DO<=6. If none: minimum max(NH4/6, DO/6), then air; label infeasible.',
                'quality_comparison': 'All adjacent offset pairs that bracket policy NH4 mean, with both endpoints and policy DO<=6. Linear descriptive interpolation; no extrapolation. Report ALL crossings as a range; not a deployed controller.',
                'dominance': 'Fixed grid point has NH4<=policy NH4, DO<=6 and air<policy air; separately check NH4 severity.',
                'uncertainty': 'Two training seeds reported separately. Paired date block bootstrap (block length 7, 2000 draws); model members averaged within date. Exploratory development data, no population or field claim.',
                'decision': 'No automatic retraining or adoption. Suspend algorithm ranking if environment strongly changes feasibility or ordering.'}
    save(a.out / 'protocol.json', protocol)
    manifest = {'started_at': now(), 'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'source_sha256': {str(p): sha(p) for p in HERE.glob('*.py')},
                'input_sha256': inputs, 'protocol_sha256': sha(a.out / 'protocol.json'),
                'training_performed': False, 'test_evaluated': False,
                'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__, 'threads': torch.get_num_threads()}
    save(a.out / 'manifest.json', manifest)
    bc = make_agent('ppo', 0, a.prepared, config())
    envs = {'legacy': BatchEnv(days, Dynamics(a.prepared / 'surrogate.pkl')),
            'corrected': BatchEnv(days, Dynamics(a.corrected))}
    rows = []
    def run(env_name, split, name, act, **extra):
        result, episodes = replay(envs[env_name], splits[split], act)
        row = {'environment': env_name, 'split': split, 'name': name, **extra, **result}
        rows.append(row)
        save(a.out / f'{split}_{env_name}_{name}_episodes.json', episodes)
        save(a.out / 'rows.json', rows)
        save(a.out / 'progress.json', {'phase': split, 'finished_evaluations': len(rows),
                                     'last': name, 'elapsed_seconds': time.monotonic() - start})
        print(json.dumps({k: row[k] for k in ('environment', 'split', 'name', 'air_m3', 'nh4_cost', 'do_cost')}), flush=True)
        return row
    for offset in tqdm(offsets, desc='Train baseline selection'):
        run('corrected', 'train', f'bc_{offset:+.0f}', lambda obs, k=offset: bc.act_det(obs)[:, 0] + k / 4500,
            kind='bc_offset', offset_m3_h=offset)
    feasible = [r for r in rows if r['both_mean_limits']]
    selected = min(feasible, key=lambda r: r['air_m3']) if feasible else min(rows, key=lambda r: (max(r['nh4_cost'], r['do_cost']), r['air_m3']))
    selection = {'locked_at': now(), 'name': selected['name'], 'offset_m3_h': selected['offset_m3_h'],
                 'train_feasible': bool(feasible), 'train_result': selected, 'protocol_sha256': sha(a.out / 'protocol.json')}
    save(a.out / 'selection.json', selection)
    manifest['selection_sha256_before_validation'] = sha(a.out / 'selection.json')
    save(a.out / 'manifest.json', manifest)
    print('TRAIN SELECTION LOCKED', selection['name'], selection['train_feasible'], flush=True)
    for env_name in envs:
        for offset in tqdm(offsets, desc=f'{env_name} validation baseline'):
            run(env_name, 'val', f'bc_{offset:+.0f}', lambda obs, k=offset: bc.act_det(obs)[:, 0] + k / 4500,
                kind='bc_offset', offset_m3_h=offset)
    hist = run('corrected', 'val', 'recorded_action_replay', lambda _: envs['corrected'].historical_action(), kind='recorded_replay')
    ref = json.loads((HERE.parent / 'e35/comparison.json').read_text())['results']['corrected']['rollout_member_average']
    for key, target in [('NH4', 'nh4_cost'), ('DO', 'do_cost')]:
        assert np.isclose(hist[target], ref['episode_mean_counts'][key]['predicted'], atol=1e-9)
    for method in METHODS:
        for seed in tqdm((0, 1), desc=f'corrected {method}'):
            agent = make_agent(method, seed, a.prepared, config())
            agent.load(a.runs / 'runs' / method / f'seed_{seed}' / 'model_final')
            run('corrected', 'val', f'{method}_s{seed}', lambda obs: agent.act_det(obs)[:, 0], kind='policy', method=method, seed=seed)
    for old in old_audit['rows']:
        if old['method'] == 'initial_BC_policy':
            current = next(r for r in rows if r['environment'] == 'legacy' and r['split'] == 'val' and r['name'] == 'bc_+0')
            for k in ('air_m3', 'nh4_cost', 'do_cost'):
                assert np.isclose(current[k], old[k], rtol=0, atol=1e-6)
            continue
        old = {k.replace('mg_N_h_L', 'mg_h_L').replace('mg_N_L', 'mg_L').replace('mg_O2_L', 'mg_L'): v for k, v in old.items()}
        kind = 'policy' if 'seed' in old else old['kind']
        name = f"{old['method']}_s{old['seed']}" if 'seed' in old else old['method']
        rows.append({**old, 'name': name, 'environment': 'legacy' if kind != 'observed' else 'observed',
                     'split': 'val', 'kind': kind, 'reused_from_e34': True})
        if kind == 'policy':
            src = a.runs / 'runs' / old['method'] / f"seed_{old['seed']}" / 'val_episodes.json'
            save(a.out / f'val_legacy_{name}_episodes.json', json.loads(src.read_text()))
    assert sha(a.out / 'protocol.json') == manifest['protocol_sha256']
    assert sha(a.out / 'selection.json') == manifest['selection_sha256_before_validation']
    assert all(sha(Path(p)) == v for p, v in inputs.items())
    save(a.out / 'rows.json', rows)
    manifest.update(finished_at=now(), elapsed_seconds=time.monotonic() - start, all_checks_passed=True)
    save(a.out / 'manifest.json', manifest)
    save(a.out / 'progress.json', {'phase': 'complete', 'elapsed_seconds': time.monotonic() - start})
    print('DONE', manifest['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--corrected', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())

"""고정된 e44 정책의 날짜 분할과 행동 방식 대조. 학습은 하지 않는다."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / 'e44'))
from command import CommandEnv, Dynamics, make_agent, config, sha, save
from darrow_env import ZONES, STATE, EXO, all_episodes
from fcsrl.utils import DeviceConfig

REPEATS = [45001, 45002, 45003]
KEYS = ('air_m3', 'nh4_cost', 'do_cost', 'nh4_excess_mg_h_L', 'do_excess_mg_h_L')


def load_selected(path):
    data = pd.read_pickle(path)
    splits = {k: list(data['days'][k]) for k in ('train', 'val')}
    assert [len(splits[k]) for k in ('train', 'val')] == [188, 40]
    assert not set(splits['train']) & set(splits['val'])
    assert not set(splits['train'] + splits['val']) & set(data['days']['test'])
    median = np.array([data['zone_fractions'][z] for z in ZONES])
    days = {}
    # 원 pickle에 test가 포함되지만 test 행은 추출하거나 평가하지 않는다.
    for day in splits['train'] + splits['val']:
        g = data['data'].loc[day]
        zones = g[ZONES].to_numpy(float)
        total = zones.sum(1, keepdims=True)
        days[day] = {'state': g[STATE].to_numpy(float), 'exo': g[EXO].to_numpy(float),
                     'frac': np.where(total > 1., zones / np.maximum(total, 1.), median[None, :]),
                     'air': g.air.to_numpy(float), 'zone': zones}
    return splits, days


def freeze(a):
    a.out.mkdir(parents=True, exist_ok=False)
    splits, _ = load_selected(a.prepared / 'data.pkl')
    source = json.loads((a.source / 'manifest.json').read_text())
    hashes = {}
    for rel, expected in source['source_sha256'].items():
        assert sha(ROOT / rel) == expected, rel
        hashes[str(ROOT / rel)] = expected
    for p in [a.prepared / 'data.pkl', a.prepared / 'prior.pt',
              a.prepared / 'normalizer.json', a.corrected, Path(__file__)]:
        hashes[str(p)] = sha(p)
    policies, jobs = [], []
    for method in source['methods']:
        for mode in source['observation_modes']:
            for seed in source['seeds']:
                run = a.source / 'runs' / method / mode / f'seed_{seed}'
                checkpoint = run / 'model_final'
                effective = json.loads((run / 'effective_config.json').read_text())
                assert effective['config'] == config(66).to_dict()
                for p in sorted(checkpoint.iterdir()):
                    if p.is_file():
                        hashes[str(p)] = sha(p)
                for name in ('val_rollout.npz', 'metrics.json', 'effective_config.json'):
                    hashes[str(run / name)] = sha(run / name)
                ident = f'{method}_{mode}_s{seed}'
                policies.append(dict(id=ident, method=method, mode=mode, seed=seed, path=str(run)))
                # val deterministic을 먼저 실행하여 기존 결과 재현을 확인한다.
                for split in ('val', 'train'):
                    for repeat in [None] + REPEATS:
                        jobs.append(dict(policy=ident, split=split, repeat=repeat,
                                         id=f'{ident}_{split}_{repeat or "det"}'))
    protocol = dict(created_at=datetime.now(timezone.utc).isoformat(),
                    code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    prepared=str(a.prepared), corrected=str(a.corrected), source=str(a.source),
                    dates=splits, n_members=5, policies=policies, jobs=jobs, repetitions=REPEATS,
                    input_and_source_sha256=hashes, test_evaluated=False, weight_updates=0,
                    costs='NH4>4; max(DO)>2; 95 post-transition samples; limits 6/6',
                    air_unit='m3/23.75h', selection='all twelve e44 final checkpoints',
                    randomness='SHA256(repeat,day,member) seeded standard Gaussian; common across policies')
    assert len(jobs) == 96
    save(a.out / 'protocol.json', protocol)
    print('FROZEN', len(jobs), 'jobs', flush=True)


def noise(episodes, repeat):
    values = []
    for day, member in episodes:
        key = f'{repeat}|{day}|{member}'.encode()
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little')
        values.append(np.random.default_rng(seed).standard_normal(95).astype(np.float32))
    return np.stack(values, axis=1)


def metrics(states, air, episodes):
    assert states.shape == (len(episodes), 95, 5) and air.shape == (len(episodes), 95)
    assert np.isfinite(states).all() and np.isfinite(air).all()
    assert air.min() >= 0 and air.max() <= 9000
    n = (states[:, :, 0] > 4).sum(1)
    d = (states[:, :, 2:].max(2) > 2).sum(1)
    arrays = dict(air_m3=air.sum(1) * .25, nh4_cost=n, do_cost=d,
                  nh4_excess_mg_h_L=np.maximum(states[:, :, 0] - 4, 0).sum(1) * .25,
                  do_excess_mg_h_L=np.maximum(states[:, :, 2:].max(2) - 2, 0).sum(1) * .25)
    rows = [dict(day=day, member=int(m), **{k: float(v[i]) for k, v in arrays.items()})
            for i, (day, m) in enumerate(episodes)]
    summary = {k: float(v.mean()) for k, v in arrays.items()}
    summary.update(both_mean_limits=bool(n.mean() <= 6 and d.mean() <= 6),
                   both_limits_episode_pct=float(100 * ((n <= 6) & (d <= 6)).mean()),
                   worst_day_nh4_cost=float(n.reshape(-1, 5).mean(1).max()),
                   worst_day_do_cost=float(d.reshape(-1, 5).mean(1).max()))
    members = {str(m): {k: float(v[m::5].mean()) for k, v in arrays.items()} for m in range(5)}
    return summary, rows, members


def fingerprint(agent):
    digest = hashlib.sha256()
    names = ['pi', 'vr', 'vc', 'prior', 'encoder', 'fixed_encoder', 'feasi_head', 'proj_layer', 'post_proj']
    for name in names:
        if hasattr(agent, name):
            for key, val in getattr(agent, name).state_dict().items():
                digest.update((name + key).encode())
                digest.update(val.detach().cpu().numpy().tobytes())
    digest.update(agent.log_std.detach().cpu().numpy().tobytes())
    digest.update(json.dumps(agent.obs_normalizer.state(), sort_keys=True).encode())
    return digest.hexdigest()


def run(a):
    protocol_path = a.out / 'protocol.json'
    p = json.loads(protocol_path.read_text())
    protocol_hash = sha(protocol_path)
    for path, expected in p['input_and_source_sha256'].items():
        assert sha(path) == expected, path
    assert not (a.out / 'status.json').exists(), 'Use a new output directory; no silent resume'
    DeviceConfig().select_device(-1)
    torch.set_num_threads(1)
    splits, days = load_selected(Path(p['prepared']) / 'data.pkl')
    assert splits == p['dates']
    dyn = Dynamics(p['corrected'])
    assert dyn.n_members == 5
    policies = {x['id']: x for x in p['policies']}
    start = time.monotonic()
    current, agent, before = None, None, None
    done = []
    status = dict(phase='running', pid=os.getpid(), started_at=datetime.now(timezone.utc).isoformat(),
                  total=len(p['jobs']), completed=0, protocol_sha256=protocol_hash)
    save(a.out / 'status.json', status)
    try:
        for job in tqdm(p['jobs'], desc='e45 matched evaluation', mininterval=10):
            tick = time.monotonic()
            pol = policies[job['policy']]
            if current != pol['id']:
                if agent is not None:
                    assert fingerprint(agent) == before
                agent = make_agent(pol['method'], pol['seed'], Path(p['prepared']), config(66))
                agent.load(Path(pol['path']) / 'model_final')
                before = fingerprint(agent)
                current = pol['id']
            env = CommandEnv(days, dyn, pol['mode'])
            episodes = all_episodes(splits[job['split']], 5)
            obs = env.reset(episodes)
            assert env.T == 95 and set(d for d, _ in episodes) <= set(p['dates'][job['split']])
            eps = noise(episodes, job['repeat']) if job['repeat'] is not None else None
            if eps is not None:
                assert np.array_equal(eps[:, :1], noise(episodes[:1], job['repeat']))
            states, airs, clipped = [], [], []
            for t in range(95):
                with torch.no_grad():
                    dist = agent.dist(agent._t(agent.obs_normalizer(obs)))
                    raw = dist.mean if eps is None else dist.mean + dist.stddev * agent._t(eps[t, :, None])
                    action = raw.clamp(-1, 1).cpu().numpy()[:, 0]
                    if eps is None and t == 0:
                        np.testing.assert_array_equal(action, agent.act_det(obs)[:, 0])
                    clipped.append(float((raw.abs() > 1).float().mean()))
                obs, reward, cost, terminal, info = env.step(action)
                assert np.allclose(reward, -info['air'] / 9000)
                assert np.array_equal(cost, info['v_nh4'] + info['v_do'])
                states.append(env.state.copy())
                airs.append(info['air'].copy())
            assert terminal and fingerprint(agent) == before
            states, air = np.stack(states, 1), np.stack(airs, 1)
            summary, rows, members = metrics(states, air, episodes)
            reproduced = None
            if job['split'] == 'val' and job['repeat'] is None:
                with np.load(Path(pol['path']) / 'val_rollout.npz') as old:
                    np.testing.assert_allclose(states, old['states'], rtol=0, atol=1e-6)
                    np.testing.assert_allclose(air, old['air'], rtol=0, atol=1e-3)
                previous = json.loads((Path(pol['path']) / 'metrics.json').read_text())['val_final']
                for key in KEYS[:-1]:
                    assert np.isclose(summary[key], previous[key], rtol=1e-8, atol=1e-8), key
                reproduced = True
            path = a.out / 'jobs' / job['id']
            path.mkdir(parents=True, exist_ok=False)
            np.savez_compressed(path / 'rollout.npz', states=states, air=air)
            save(path / 'episodes.json', rows)
            result = dict(**job, summary=summary, members=members, frozen_policy=True,
                          fingerprint=before, e44_reproduced=reproduced,
                          action_clip_pct=float(np.mean(clipped) * 100),
                          std_air_m3_h=float(agent.log_std.exp().detach().mean() * 4500),
                          seconds=time.monotonic() - tick)
            save(path / 'metrics.json', result)
            done.append(result)
            elapsed = time.monotonic() - start
            status.update(completed=len(done), elapsed_seconds=elapsed,
                          estimated_remaining_seconds=elapsed / len(done) * (len(p['jobs']) - len(done)),
                          last_job=job['id'])
            save(a.out / 'status.json', status)
            print('DONE', job['id'], round(result['seconds'], 2), summary, flush=True)
        for path, expected in p['input_and_source_sha256'].items():
            assert sha(path) == expected, path
        assert sha(protocol_path) == protocol_hash
        save(a.out / 'all_metrics.json', done)
        status.update(phase='complete', finished_at=datetime.now(timezone.utc).isoformat(),
                      elapsed_seconds=time.monotonic() - start, estimated_remaining_seconds=0,
                      input_hashes_unchanged=True, test_evaluated=False, weight_updates=0)
        save(a.out / 'status.json', status)
    except BaseException as exc:
        status.update(phase='failed', error=repr(exc), elapsed_seconds=time.monotonic() - start)
        save(a.out / 'status.json', status)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'run'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, default=Path('/home/jio/fcsrl_src/results/e30/prepared'))
    parser.add_argument('--corrected', type=Path, default=Path('/home/jio/fcsrl_src/results/e35/zone_fix/corrected_surrogate.pkl'))
    parser.add_argument('--source', type=Path, default=Path('/home/jio/fcsrl_src/results/e44/main'))
    args = parser.parse_args()
    (freeze if args.action == 'freeze' else run)(args)

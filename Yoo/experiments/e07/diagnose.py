"""학습 없이 저장 정책, 대리 모델과 하루 과제의 불일치를 점검한다."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

CODE = Path(__file__).resolve().parents[2] / 'code/wwtp'
sys.path.insert(0, str(CODE))
import numpy as np
import pandas as pd
import torch
from agents import load_agent
from env import AerationEnv
from evaluate import batched_rollout, episodes, summarize_rows
from ppo_tail import risk_metrics
from surrogate import do_inputs, nh4_inputs, phase_onehot


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False))


def describe(x):
    x = np.asarray(x)
    return {'mean': float(x.mean()), 'median': float(np.median(x)),
            'p95': float(np.quantile(x, .95)), 'max': float(x.max())}


def profile(env, agent, output, name):
    obs, actions, costs = [], [], []
    def observe(t, o, a, cost, air):
        obs.append(o.copy())
        if a is not None:
            actions.append(a.copy())
            costs.append(cost.copy())
    stats, rows = batched_rollout(env, env.sur, lambda o: agent.act_det(o)[:, 0], 15, observe)
    stats.update(risk_metrics(rows))
    o, a, c = np.stack(obs), np.stack(actions), np.stack(costs)
    T, N = c.shape
    values = []
    with torch.no_grad():
        for t in range(T + 1):
            values.append(agent.values(agent._t(agent.obs_normalizer(o[t])))[1].cpu().numpy())
    vc = np.stack(values)
    mc, ft = np.zeros_like(c), np.zeros_like(c)
    future_c, future_f = np.zeros(N), np.zeros(N)
    for t in range(T - 1, -1, -1):
        future_c = c[t] + .99 * future_c
        future_f = np.maximum(c[t], .9 * future_f)
        mc[t], ft[t] = future_c, future_f
    unsafe = c > 0
    current_dominates = np.isclose(ft, c)
    nh4 = o[1:, :, 0] > 4
    do = o[1:, :, 1] > 2
    mask = o[:-1, :, 4] == 1
    start_high = o[0, :, 0] > 4
    report = {'metrics': stats, 'episodes': N,
        'initial_high_nh4_fraction': float(start_high.mean()),
        'cost_initial_low': float(c[:, ~start_high].sum(0).mean()),
        'cost_initial_high': float(c[:, start_high].sum(0).mean()),
        'nh4_cost': float(nh4.sum(0).mean()), 'do_cost': float(do.sum(0).mean()),
        'aerobic_low_do_fraction': float((o[1:, :, 1][mask] < .7).mean()),
        'unsafe_transition_fraction': float(unsafe.mean()),
        'feasibility_immediate_dominates_unsafe_fraction': float(current_dominates[unsafe].mean()),
        'feasibility_immediate_dominates_all_fraction': float(current_dominates.mean()),
        'finite_day_discounted_cost_start': float(mc[0].mean()),
        'critic_cost_start': float(vc[0].mean()),
        'critic_minus_finite_day_return_start': float((vc[0] - mc[0]).mean()),
        'cost_bootstrap_at_day_end': describe(vc[-1]),
        'start_discounted_terminal_contribution': float((.99 ** T * vc[-1]).mean()),
        'gamma_weights': {str(k): .99 ** k for k in (12, 36, 144, 287)},
        'feasibility_weights': {str(k): .9 ** k for k in (12, 36, 144, 287)},
        'by_member': {str(m): summarize_rows([r for r in rows if r['member'] == m], 15)
                      for m in range(env.n_members)}}
    if hasattr(agent, 'encoder'):
        # 평가 자료에서 같은 목표를 실제로 예측하는지 확인한다. 정책을 갱신하지 않는다.
        from fcsrl.utils import DiscDist
        flat_o, flat_a = o[:-1].reshape(-1, 11), a.reshape(-1, 1)
        sample = np.arange(0, len(flat_o), max(1, len(flat_o) // 12000))
        estimates = []
        with torch.no_grad():
            for ids in np.array_split(sample, max(1, len(sample) // 512)):
                ot = agent._t(agent.obs_normalizer(flat_o[ids]))
                pred_z = agent.encoder.zsa(agent.encoder.zs(ot), agent._t(flat_a[ids]))
                logits = agent.feasi_head(pred_z)
                estimates.extend(DiscDist(logits, -2, 4, 63).mean().cpu().numpy().reshape(-1))
        pred, target = np.asarray(estimates), ft.reshape(-1)[sample]
        report['feasibility_prediction_finite_day'] = {
            'mse': float(np.mean((pred - target) ** 2)), 'target_variance': float(target.var()),
            'r2': float(1 - np.mean((pred - target) ** 2) / max(target.var(), 1e-12)),
            'prediction': describe(pred), 'target': describe(target)}
    np.savez_compressed(output / f'{name}.npz', obs=o, action=a, cost=c, vc=vc, finite_feasibility=ft)
    save(output / f'{name}_episodes.json', rows)
    print('PROFILE', name, stats['cost'], report.get('feasibility_prediction_finite_day', {}).get('r2'), flush=True)
    return report


def historical(env):
    X = [env.day_data[d] for d, _ in env.fixed]
    air = np.stack([x['air'] for x in X])
    nh4_true = np.stack([x['nh4'][1:] for x in X]).T
    do_true = np.stack([x['do'][1:] for x in X]).T
    counter = [0]
    states = []
    def action(obs):
        t = counter[0]
        counter[0] += 1
        return np.clip(air[:, t] / 4000 - 1, -1, 1)
    def observe(t, obs, *args):
        states.append(obs[:, :2].copy())
    metrics, rows = batched_rollout(env, env.sur, action, 15, observe)
    predicted = np.stack(states)[1:]
    by_member = {}
    for m in range(env.n_members):
        mask = np.array([i == m for _, i in env.fixed])
        by_member[str(m)] = {'nh4_mae': float(np.abs(predicted[:, mask, 0] - nh4_true[:, mask]).mean()),
            'do_mae': float(np.abs(predicted[:, mask, 1] - do_true[:, mask]).mean()),
            **summarize_rows([r for r in rows if r['member'] == m], 15)}
    return {'predicted': metrics, 'measured_cost': float(((nh4_true > 4).astype(float) + (do_true > 2)).sum(0).mean()),
            'by_member': by_member}


def local_model_checks(env):
    # 동일한 실제 상태에서 공기량만 증가시키는 모델 내부의 반응 점검.
    x = {k: np.concatenate([d[k][:-1] for d in env.day_data.values()]) for k in
         ('nh4', 'do', 'air', 'phase', 'temp', 'q_in', 'hour')}
    use = (x['phase'] == 2) & (x['air'] <= 7000)
    x = {k: v[use] for k, v in x.items()}
    ph = phase_onehot(x['phase'])
    s, c = np.sin(2*np.pi*x['hour']/24), np.cos(2*np.pi*x['hour']/24)
    result = []
    predictions = []
    for m in range(env.n_members):
        do_next, nh4_next = [], []
        for delta in (0, 1000):
            xd = do_inputs(x['do'], x['air'] + delta, ph, x['temp'], x['nh4'], x['q_in'], s, c)
            dn = np.maximum(0, x['do'] + env.sur.do_model.predict(xd, m)[0])
            xn = nh4_inputs(x['nh4'], x['do'], dn, ph, x['temp'], x['q_in'], s, c)
            nn = np.maximum(0, x['nh4'] + env.sur.nh4_model.predict(xn, m)[0])
            do_next.append(dn); nh4_next.append(nn)
        dd, nd = do_next[1] - do_next[0], nh4_next[1] - nh4_next[0]
        predictions.append(np.stack((nh4_next[0], do_next[0]), -1))
        result.append({'member': m, 'samples': len(dd), 'do_delta_mean': float(dd.mean()),
            'nh4_delta_mean': float(nd.mean()), 'do_decreases_fraction': float((dd < -1e-6).mean()),
            'nh4_increases_fraction': float((nd > 1e-6).mean())})
    spread = np.stack(predictions).std(0)
    return {'air_plus_1000': result, 'same_input_between_member_sd_nh4': describe(spread[:, 0]),
            'same_input_between_member_sd_do': describe(spread[:, 1])}


def main(a):
    torch.set_num_threads(1)
    if a.out.exists() and any(a.out.iterdir()):
        raise ValueError(f'Refusing to overwrite {a.out}')
    a.out.mkdir(parents=True)
    envs = {s: AerationEnv(a.data, a.model, split=s) for s in ('train', 'val')}
    for env in envs.values(): env.fixed = episodes(env)
    report = {'started_at': datetime.now(timezone.utc).isoformat(),
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inputs': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.data, a.model)},
        'training_performed': False, 'test_used': False, 'profiles': {}, 'historical': {}, 'model_checks': {}}
    for split, env in envs.items():
        report['historical'][split] = historical(env)
        report['model_checks'][split] = local_model_checks(env)
        for method in ('ppolag', 'ppolag_fcsrl', 'ppolag_fcsrl_tail'):
            for seed in (0, 1):
                run = a.runs / method / f'seed_{seed}'
                agent = load_agent(run, env, cudaid=-1, checkpoint='final')
                key = f'{method}_s{seed}_{split}'
                report['profiles'][key] = profile(env, agent, a.out, key)
                report['profiles'][key]['checkpoint'] = agent.checkpoint_path
                save(a.out / 'diagnostics.json', report)
    env = envs['val']
    report['encoder_swap_diagnostic'] = []
    for seed in (0, 1):
        run = a.runs / 'ppolag_fcsrl' / f'seed_{seed}'
        sources = {kind: load_agent(run, env, cudaid=-1, checkpoint=kind) for kind in ('best', 'final')}
        states = {kind: {k: v.clone() for k, v in agent.fixed_encoder.state_dict().items()}
                  for kind, agent in sources.items()}
        for actor_kind, agent in sources.items():
            for encoder_kind, state in states.items():
                agent.fixed_encoder.load_state_dict(state)
                stats, _ = batched_rollout(env, env.sur, lambda o: agent.act_det(o)[:, 0], 15)
                report['encoder_swap_diagnostic'].append({'seed': seed, 'actor_and_normalizer': actor_kind,
                                                          'encoder': encoder_kind, **stats})
        # 바꾸지 않은 비교군. 표현 학습이 없는 경우의 성능도 함께 기록한다.
        frozen_run = a.runs.parents[1] / 'e05/pilot/runs/ppolag_frozen' / f'seed_{seed}'
        frozen = load_agent(frozen_run, env, cudaid=-1, checkpoint='final')
        report['profiles'][f'ppolag_frozen_s{seed}_val'] = profile(env, frozen, a.out, f'ppolag_frozen_s{seed}_val')
    rows_by_air = {}
    for air in range(0, 8001, 1000):
        metrics, rows = batched_rollout(env, env.sur, lambda obs: np.full(len(obs), air/4000-1), 15)
        rows_by_air[air] = rows
        print('CONSTANT AIR', air, metrics['cost'], flush=True)
    costs = np.array([[r['cost'] for r in rows_by_air[air]] for air in rows_by_air])
    best = costs.min(0)
    report['constant_air_diagnostic'] = {
        'note': 'Best of fixed actions with hindsight is not a deployable policy or a lower bound on achievable cost.',
        'policies': {str(air): summarize_rows(rows, 15) for air, rows in rows_by_air.items()},
        'hindsight_min_cost_mean': float(best.mean()),
        'episodes_without_cost_le15_in_grid': int((best > 15).sum()),
        'hardest_episodes': [{'day': env.fixed[i][0], 'member': int(env.fixed[i][1]), 'minimum_grid_cost': float(best[i]),
                             'costs_by_air': costs[:, i].tolist()} for i in np.argsort(best)[-10:][::-1]]}
    report['completed_at'] = datetime.now(timezone.utc).isoformat()
    save(a.out / 'diagnostics.json', report)
    print('COMPLETE', report['completed_at'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())

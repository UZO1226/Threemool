"""학습 전 점검. 검증 구간에서 모델 오차, critic 오차와 행동을 분리해 기록한다."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

CODE = Path(__file__).resolve().parents[2] / 'code/wwtp'
sys.path.insert(0, str(CODE))
from agents import load_agent
from env import AerationEnv
from evaluate import batched_rollout, batch_act_fn, episodes
from surrogate import Surrogate, do_inputs, nh4_inputs, phase_onehot, time_feats


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def observed_model_audit(data, model):
    d = pd.read_pickle(data)
    r = d['data']
    nxt = r.shift(-1)
    dates = pd.Series(r.index.date, index=r.index).astype(str)
    # 두 시점 모두 검증 구간의 같은 날에 속해야 한다.
    ok = dates.isin(d['days']['val']) & dates.eq(dates.shift(-1))
    x, y = r[ok], nxt[ok]
    s, c = time_feats(x.index)
    ph = phase_onehot(x.phase)
    sur = Surrogate(model)
    dos = np.maximum(0, x.do.to_numpy() + sur.do_model.predict(
        do_inputs(x.do, x.air, ph, x.temp, x.nh4, x.q_in, s, c)))
    nhs = np.stack([np.maximum(0, x.nh4.to_numpy() + sur.nh4_model.predict(
        nh4_inputs(x.nh4, x.do, dos[i], ph, x.temp, x.q_in, s, c), i)[0])
        for i in range(sur.n_members)])
    out = {}
    for name, preds, truth, limit in [('nh4', nhs, y.nh4.to_numpy(), 4.),
                                      ('do', dos, y.do.to_numpy(), 2.)]:
        mu, sd = preds.mean(0), preds.std(0)
        err = truth - mu
        blocks = {}
        for label, mask in [('all', np.ones(len(x), bool)), ('aerobic_low_do', (x.phase == 2) & (x.do < .7)),
                            ('aerobic', x.phase == 2), ('near_limit', np.abs(truth - limit) < .5)]:
            mask = np.asarray(mask)
            if mask.sum() < 2:
                continue
            actual = truth[mask] > limit
            blocks[label] = {'n': int(mask.sum()), 'actual_violation_rate': float(actual.mean()),
                'mae': float(np.abs(err[mask]).mean()), 'mean_sigma': float(sd[mask].mean()),
                'corr_sigma_abs_error': float(np.corrcoef(sd[mask], np.abs(err[mask]))[0, 1]),
                'k_sweep': []}
            for k in (0., .5, 1., 2., 3.):
                upper = mu[mask] + k * sd[mask]
                pv = upper > limit
                blocks[label]['k_sweep'].append({'k': k, 'upper_coverage': float(np.mean(truth[mask] <= upper)),
                    'false_negative_rate': float((~pv[actual]).mean()) if actual.any() else None,
                    'false_positive_rate': float(pv[~actual].mean()) if (~actual).any() else None,
                    'predicted_violation_rate': float(pv.mean())})
        out[name] = blocks
    return out


def discounted_targets(cost, q_mean, horizon, gamma=.99):
    """동일 정책의 h스텝 실제 비용과 끝 상태 Q. 날짜 경계를 넘지 않는다."""
    starts = np.arange(0, len(cost) - horizon + 1, horizon)
    values = np.stack([sum(gamma**j * cost[t+j] for j in range(horizon))
                       + gamma**horizon * q_mean[t+horizon] for t in starts])
    return starts, values


def critic_values(agent, obs, action):
    obs = agent.obs_normalizer(obs)
    with torch.no_grad():
        ot = torch.as_tensor(obs, dtype=torch.float32)
        if hasattr(agent, 'vc'):
            return agent.vc(ot).cpu().numpy()
        at = torch.as_tensor(action[:, None], dtype=torch.float32)
        zs = agent.fixed_encoder.zs(ot)
        zsa = agent.fixed_encoder.zsa(zs, at)
        return agent.cost_critic(ot, zs, at, zsa).cpu().numpy()


def policy_audit(run, data, model, out):
    env = AerationEnv(data, model, split='val')
    env.fixed = episodes(env)
    agent = load_agent(run, env, -1)
    act = batch_act_fn(agent)
    costs, qs, actions, dos = [], {}, [], []
    horizon = 12

    def observe(t, obs, action, cost, air):
        # 12스텝 목표 비교 및 하루 끝 bootstrap 항을 각각 기록한다.
        if t % horizon == 0 or t == env.T:
            qs[t] = critic_values(agent, obs, act(obs) if action is None else action)
        if cost is not None:
            costs.append(cost.copy())
            aerobic = obs[:, 4].astype(bool)
            actions.extend(action[aerobic].tolist())
            dos.extend(obs[aerobic, 1].tolist())

    summ, rows = batched_rollout(env, env.sur, act, 15, observe=observe)
    cost = np.array(costs)
    q_mean = {t: q.mean(1) for t, q in qs.items()}
    starts, targets = discounted_targets(cost, q_mean, horizon)
    mean = np.stack([q_mean[t] for t in starts])
    sig = np.stack([q.std(1, ddof=1) if q.shape[1] > 1 else np.zeros(q.shape[0]) for t, q in ((t, qs[t]) for t in starts)])
    residual = mean - targets
    ucbr = mean + .5 * sig - targets
    discounted = np.sum(cost * (.99 ** np.arange(env.T))[:, None], axis=0)
    tail = .99**env.T * q_mean[env.T]
    m = json.loads((run / 'metrics.json').read_text())
    report = {'tag': m.get('tag', m['method']), 'seed': m['seed'], 'checkpoint': agent.checkpoint_path,
        'validation': summ, 'q0_mean': float(q_mean[0].mean()),
        'discounted_day_cost': float(discounted.mean()), 'discounted_tail_q': float(tail.mean()),
        'q0_minus_day_cost_with_tail': float((q_mean[0] - discounted - tail).mean()),
        'one_hour_q_residual_mean': float(residual.mean()),
        'one_hour_q_residual_mae': float(np.abs(residual).mean()),
        'one_hour_mean_under_target_fraction': float((residual < 0).mean()),
        'one_hour_ucb_under_target_fraction': float((ucbr < 0).mean()),
        'critic_sigma_mean': float(sig.mean()),
        'ucb_below_same_ensemble_max_fraction': float(np.mean(np.stack([
            qs[t].mean(1) + .5 * (qs[t].std(1, ddof=1) if qs[t].shape[1]>1 else 0) < qs[t].max(1)
            for t in starts]))),
        'aerobic_action_lower_clip_fraction': float(np.mean(np.array(actions) <= -.999)),
        'aerobic_action_upper_clip_fraction': float(np.mean(np.array(actions) >= .999)),
        'aerobic_low_do_fraction': float(np.mean(np.array(dos) < .7))}
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{report['tag']}_s{report['seed']}.npz", cost=cost, starts=starts,
                        q_mean=mean, q_sigma=sig, bootstrap_target=targets)
    (out / f"{report['tag']}_s{report['seed']}_episodes.json").write_text(json.dumps(rows))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--roots', type=Path, nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    a.out.mkdir(parents=True, exist_ok=True)
    report = {'split': 'val', 'note': 'Exploratory. Validation data previously used for model selection. Bootstrap targets are not independent ground truth.',
              'inputs': {str(p): sha(p) for p in (a.data, a.model)},
              'source': {str(p): sha(p) for p in [Path(__file__), CODE/'evaluate.py', CODE/'agents.py']},
              'observed_model': observed_model_audit(a.data, a.model), 'policies': []}
    def save():
        tmp = a.out / 'diagnostics.tmp'
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False))
        tmp.replace(a.out/'diagnostics.json')
    save()
    for root in a.roots:
        for path in sorted(root.glob('*/seed_*/metrics.json')):
            row = policy_audit(path.parent, a.data, a.model, a.out/'details')
            report['policies'].append(row)
            save()
            print(row['tag'], row['seed'], 'cost', round(row['validation']['cost'], 2),
                  'Q residual', round(row['one_hour_q_residual_mean'], 3), flush=True)
    report['complete'] = True
    save()


if __name__ == '__main__':
    main()

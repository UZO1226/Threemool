"""정책 결과와 분리해 검증 구간의 실제 관측 대비 환경 오차를 기록한다."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from common import save
from darrow_env import BatchEnv, Dynamics, load_days, all_episodes
from surrogate_dw import build_arrays, GBMDynamics, STATE


def main(prepared, out):
    data = pd.read_pickle(prepared / 'data.pkl')
    splits, days = load_days(prepared / 'data.pkl')
    env = BatchEnv(days, Dynamics(prepared / 'surrogate.pkl'))
    dyn = GBMDynamics(prepared / 'surrogate.pkl')
    x, y = build_arrays(data['data'], splits['val'])
    pred = np.mean([dyn.delta(x, np.full(len(x), m)) for m in range(dyn.n_members)], axis=0)
    r2 = 1 - ((y-pred)**2).sum(0) / ((y-y.mean(0))**2).sum(0)
    env.reset(all_episodes(splits['val'], dyn.n_members))
    actual = np.stack([days[d]['state'][1:] for d, _ in env.episodes])
    trajectories, air = [], []
    for _ in range(env.T):
        _, _, _, _, info = env.step(env.historical_action())
        trajectories.append(env.state.copy())
        air.append(info['air'].copy())
    prediction = np.stack(trajectories, axis=1)
    def quality(z):
        return np.stack([z[:, :, 0] > 4, z[:, :, 2:].max(2) > 2], axis=2)
    truth, guess = quality(actual), quality(prediction)
    confusion = {}
    for j, name in enumerate(('NH4', 'DO')):
        t, p = truth[:, :, j], guess[:, :, j]
        confusion[name] = {'recall_pct': float(100 * (p & t).sum()/max(1, t.sum())),
                           'fpr_pct': float(100 * (p & ~t).sum()/max(1, (~t).sum())),
                           'observed_violation_pct': float(100*t.mean()),
                           'predicted_violation_pct': float(100*p.mean())}
    # Local monotonicity of predicted deltas, not a claim about causal validity or multi-step monotonicity.
    low, high = x.copy(), x.copy()
    low[:, 14:] *= .75
    high[:, 14:] *= 1.25
    local = []
    for m in range(dyn.n_members):
        member = np.full(len(x), m)
        lo, hi = dyn.delta(low, member), dyn.delta(high, member)
        ok = bool((hi[:, 0] <= lo[:, 0]+1e-10).all())
        for j in range(3):
            lo_x, hi_x = x.copy(), x.copy()
            lo_x[:, [14+j, 19+j]] *= .75
            hi_x[:, [14+j, 19+j]] *= 1.25
            lo_j, hi_j = dyn.delta(lo_x, member), dyn.delta(hi_x, member)
            ok &= bool((hi_j[:, 2+j] >= lo_j[:, 2+j]-1e-10).all())
        local.append(ok)
    save(out, {'split': 'val', 'days': len(splits['val']), 'members': dyn.n_members,
         'one_step_delta_r2': dict(zip(STATE, r2.tolist())), 'confusion': confusion,
         'observed_daily_cost': float(truth.sum((1, 2)).mean()),
         'model_replay_daily_cost': float(guess.sum((1, 2)).mean()),
         'historical_air_m3_per_episode': float(np.stack(air).sum(0).mean()*.25),
         'nh4_mae': float(abs(actual[:, :, 0]-prediction[:, :, 0]).mean()),
         'local_monotonicity_all_members': all(local), 'test_evaluated': False,
         'interpretation': 'Conditional surrogate benchmark; no real-plant safety or savings claim.'})
    assert all(local)
    print('DIAGNOSTIC', out, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    main(a.prepared, a.out)

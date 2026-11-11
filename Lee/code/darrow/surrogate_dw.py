"""DARROW 온라인 자료의 상태 전이 대리 모델 (MLP 앙상블).

한 스텝(15분)의 상태 변화 [NH4, NO3, DO1, DO2, DO3] 를 현재 상태, 유입량, 수온, 시각, 총 공기량(이번 스텝과
직전 스텝)으로 예측한다. 학습 구간만 학습에 쓰고 검증 구간은 조기 종료와 평가에만 쓴다.

대리 모델이 알려 주지 못하는 것: 공기량과 NH4 는 같은 시점에서 양의 상관(0.45)이다. 부하가 크면 기존 제어기가
공기를 늘리기 때문이고 인과가 아니다. 그래서 학습 후 "같은 상태에서 공기량만 바꿨을 때" 방향이 물리와 맞는지
(공기를 늘리면 NH4 감소, DO 증가) 별도로 검사하고 결과를 보고한다.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

STATE = ['nh4', 'no3', 'do1', 'do2', 'do3']
EXO = ['q_in', 'temp', 'sin', 'cos', 'inf_s_nh', 'inf_s_nd', 'inf_x_nd', 'inf_s_s', 'inf_x_s']
ZONES = ['air1', 'air2', 'air3', 'air4', 'air5']
N_IN = len(STATE) + len(EXO) + 2 * len(ZONES)
AIR_SCALE = 1000.0


def features(state, exo, zone, zone_prev):
    """state (N, 5), exo (N, 9), zone (N, 5), zone_prev (N, 5) -> (N, 24) 입력"""
    return np.column_stack([state, exo, zone / AIR_SCALE, zone_prev / AIR_SCALE])


def build_arrays(table, days):
    """날짜 목록에서 (입력, 목표 변화량) 쌍을 만든다. 하루 안에서만 이어 붙인다."""
    xs, ys = [], []
    for day in days:
        g = table.loc[day]
        s = g[STATE].to_numpy(float)
        exo, z = g[EXO].to_numpy(float), g[ZONES].to_numpy(float)
        x = features(s[:-1], exo[:-1], z[:-1], np.vstack([z[:1], z[:-2]]))
        xs.append(x)
        ys.append(s[1:] - s[:-1])
    return np.concatenate(xs), np.concatenate(ys)


class Net(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(N_IN, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                               nn.Linear(hidden, len(STATE)))

    def forward(self, x):
        return self.f(x)


class Dynamics:
    """저장된 가중치로 numpy 에서 전이를 계산한다 (환경 안에서 빠르게 쓰려고)."""

    def __init__(self, path):
        z = np.load(path)
        self.xm, self.xs, self.ym, self.ys = z['xm'], z['xs'], z['ym'], z['ys']
        self.hi = z['hi']  # 상태 상한 (학습 구간 최댓값의 1.1배). 연속 예측이 범위를 벗어나 폭주하는 것을 막는다
        self.W = [z[f'W{i}'] for i in range(3)]
        self.b = [z[f'b{i}'] for i in range(3)]
        self.n_members = self.W[0].shape[0]

    def delta(self, x, member):
        """x (N, 11), member (N,) -> 변화량 (N, 5)"""
        h = (x - self.xm) / self.xs
        for i in range(3):
            h = np.einsum('ni,nio->no', h, self.W[i][member]) + self.b[i][member]
            if i < 2:
                h = h * (1 / (1 + np.exp(-np.clip(h, -30, 30))))  # SiLU
        return h * self.ys + self.ym

    def step(self, state, exo, zone, zone_prev, member):
        x = features(state, exo, zone, zone_prev)
        return np.clip(state + self.delta(x, member), 0.0, self.hi)


def train(table, days, out, members=5, epochs=80, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    xt, yt = build_arrays(table, days['train'])
    xv, yv = build_arrays(table, days['val'])
    xm, xs = xt.mean(0), xt.std(0) + 1e-6
    ym, ys = yt.mean(0), yt.std(0) + 1e-6
    nx = lambda a: torch.tensor((a - xm) / xs, dtype=torch.float32)  # noqa: E731
    ny = lambda a: torch.tensor((a - ym) / ys, dtype=torch.float32)  # noqa: E731
    Xt, Yt, Xv, Yv = nx(xt), ny(yt), nx(xv), ny(yv)
    nets, log = [], []
    for m in range(members):
        net = Net()
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        idx = np.random.RandomState(seed * 100 + m).choice(len(Xt), int(0.8 * len(Xt)), replace=False)
        best, best_state, wait = float('inf'), None, 0
        for ep in range(epochs):
            net.train()
            perm = np.random.RandomState(ep + m * 1000).permutation(idx)
            for i in range(0, len(perm), 256):
                b = perm[i:i + 256]
                loss = ((net(Xt[b]) - Yt[b]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float(((net(Xv) - Yv) ** 2).mean())
            if v < best - 1e-5:
                best, wait = v, 0
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                wait += 1
                if wait >= 8:
                    break
        net.load_state_dict(best_state)
        nets.append(net)
        log.append({'member': m, 'epochs': ep + 1, 'val_mse_standardized': best})
    states = np.concatenate([table.loc[d][STATE].to_numpy(float) for d in days['train']])
    arrays = {'xm': xm, 'xs': xs, 'ym': ym, 'ys': ys, 'hi': states.max(0) * 1.1}
    for i, layer in enumerate((0, 2, 4)):
        arrays[f'W{i}'] = np.stack([n.f[layer].weight.detach().numpy().T for n in nets])
        arrays[f'b{i}'] = np.stack([n.f[layer].bias.detach().numpy() for n in nets])
    np.savez(out, **arrays)
    return log


# ---- 단조 제약 모델 (최종 사용) -------------------------------------------------------------------------
# MLP 는 공기량과 NH4 의 교란 상관(부하가 크면 제어기가 공기를 늘림)을 인과로 배워, 공기를 늘리면 NH4 가
# 오히려 올라가는 반응(0.5배에서 0.37, 1.5배에서 3.64)을 냈다. 그대로 쓰면 RL 이 "공기를 끄면 위반이 줄어든다"는
# 잘못된 환경을 배우므로, 물리 사전지식을 구조로 강제한다.
#   NH4 변화량은 모든 구간 공기량(현재, 직전)에 대해 비증가, DO_k 변화량은 해당 구간 공기량에 대해 비감소
# NO3 는 질산화와 탈질이 함께 있어 부호를 강제하지 않는다.
def monotone_constraints(target):
    cst = np.zeros(N_IN, dtype=int)
    z0, zp0 = len(STATE) + len(EXO), len(STATE) + len(EXO) + len(ZONES)
    if target == 0:  # NH4
        cst[z0:z0 + len(ZONES)] = -1
        cst[zp0:zp0 + len(ZONES)] = -1
    elif target in (2, 3, 4):  # DO1, DO2, DO3 -> 같은 번호의 구간 공기량
        k = target - 2
        cst[z0 + k] = 1
        cst[zp0 + k] = 1
    return cst


class GBMDynamics:
    """Dynamics 와 같은 인터페이스 (delta, step, n_members, hi). 모델은 pickle 로 저장한다."""

    def __init__(self, path):
        import pickle
        with open(path, 'rb') as f:
            z = pickle.load(f)
        self.models, self.hi, self.n_members = z['models'], z['hi'], len(z['models'])
        import os
        self.fast = None
        if os.environ.get('DARROW_FAST_DYNAMICS') == '1':
            from fast_gbm import FastPredictor
            self.fast = FastPredictor(self.models)

    def delta(self, x, member):
        if self.fast is not None:
            return self.fast(x, member)
        out = np.zeros((len(x), len(STATE)))
        for m in np.unique(member):
            rows = np.flatnonzero(member == m)
            for j in range(len(STATE)):
                out[rows, j] = self.models[int(m)][j].predict(x[rows])
        return out

    def step(self, state, exo, zone, zone_prev, member):
        x = features(state, exo, zone, zone_prev)
        return np.clip(state + self.delta(x, member), 0.0, self.hi)


def train_gbm(table, days, out, members=5, seed=0):
    import pickle
    from sklearn.ensemble import HistGradientBoostingRegressor as H
    xt, yt = build_arrays(table, days['train'])
    xv, yv = build_arrays(table, days['val'])
    models, log = [], []
    for m in range(members):
        idx = np.random.RandomState(seed * 100 + m).choice(len(xt), int(0.8 * len(xt)), replace=False)
        row, mse = [], []
        for j in range(len(STATE)):
            g = H(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0,
                  monotonic_cst=monotone_constraints(j), early_stopping=True, validation_fraction=0.1,
                  n_iter_no_change=20, random_state=seed * 100 + m + j).fit(xt[idx], yt[idx, j])
            row.append(g)
            mse.append(float(((g.predict(xv) - yv[:, j]) ** 2).mean() / yv[:, j].var()))
        models.append(row)
        log.append({'member': m, 'val_mse_over_var': mse})
    states = np.concatenate([table.loc[d][STATE].to_numpy(float) for d in days['train']])
    with open(out, 'wb') as f:
        pickle.dump({'models': models, 'hi': states.max(0) * 1.1}, f)
    return log


def evaluate(table, days, path, split='val', kind='mlp'):
    """검증 구간으로 한 스텝 정확도, 하루 연속(open-loop) 예측, 공기량 반응을 잰다."""
    dyn = GBMDynamics(path) if kind == 'gbm' else Dynamics(path)
    xv, yv = build_arrays(table, days[split])
    pred = np.mean([dyn.delta(xv, np.full(len(xv), m)) for m in range(dyn.n_members)], axis=0)
    r2 = {k: float(1 - ((pred[:, i] - yv[:, i]) ** 2).sum() / ((yv[:, i] - yv[:, i].mean()) ** 2).sum())
          for i, k in enumerate(STATE)}
    # 하루 연속 예측: 날 시작 상태에서 기록된 공기량을 그대로 넣어 95스텝을 이어서 예측한다
    def rollout(g, zones, m):
        exo = g[EXO].to_numpy(float)
        state = g[STATE].to_numpy(float)[:1].copy()
        traj = [state[0].copy()]
        for t in range(len(g) - 1):
            state = dyn.step(state, exo[t:t + 1], zones[t:t + 1], zones[max(t - 1, 0):max(t - 1, 0) + 1], np.array([m]))
            traj.append(state[0].copy())
        return np.array(traj)

    rows = []
    for day in days[split]:
        g = table.loc[day]
        zones = g[ZONES].to_numpy(float)
        for m in range(dyn.n_members):
            traj = rollout(g, zones, m)
            rows.append({'day': day, 'member': m, 'pred_nh4_mean': float(traj[:, 0].mean()),
                         'true_nh4_mean': float(g.nh4.mean()), 'pred_nh4_viol': float((traj[:, 0] > 4).mean() * 100),
                         'true_nh4_viol': float((g.nh4 > 4).mean() * 100),
                         'pred_do_viol': float((traj[:, 2:].max(1) > 2).mean() * 100),
                         'true_do_viol': float((g[['do1', 'do2', 'do3']].max(axis=1) > 2).mean() * 100),
                         'pred_nh4_mae': float(np.abs(traj[:, 0] - g.nh4.to_numpy()).mean())})
    df = pd.DataFrame(rows)
    per_day = df.groupby('day').mean(numeric_only=True)
    open_loop = {'daily_nh4_mean_corr': float(np.corrcoef(per_day.pred_nh4_mean, per_day.true_nh4_mean)[0, 1]),
                 'nh4_mae_per_step': float(df.pred_nh4_mae.mean()),
                 'nh4_viol_pct_pred_vs_true': [float(df.pred_nh4_viol.mean()), float(df.true_nh4_viol.mean())],
                 'do_viol_pct_pred_vs_true': [float(df.pred_do_viol.mean()), float(df.true_do_viol.mean())]}
    # 공기량 반응: 같은 날들에서 기록된 구간별 공기량에 배율을 곱했을 때 하루 평균 NH4 와 DO3
    response = {}
    for scale in (0.5, 0.75, 1.0, 1.25, 1.5):
        nh4, do3 = [], []
        for day in days[split]:
            g = table.loc[day]
            traj = rollout(g, g[ZONES].to_numpy(float) * scale, 0)
            nh4.append(traj[:, 0].mean())
            do3.append(traj[:, 4].mean())
        response[str(scale)] = {'nh4_mean': float(np.mean(nh4)), 'do3_mean': float(np.mean(do3))}
    mono_nh4 = all(response[a]['nh4_mean'] >= response[b]['nh4_mean'] - 1e-9
                   for a, b in zip(('0.5', '0.75', '1.0', '1.25'), ('0.75', '1.0', '1.25', '1.5')))
    mono_do = all(response[a]['do3_mean'] <= response[b]['do3_mean'] + 1e-9
                  for a, b in zip(('0.5', '0.75', '1.0', '1.25'), ('0.75', '1.0', '1.25', '1.5')))
    return {'split': split, 'one_step_r2': r2, 'open_loop': open_loop, 'air_response': response,
            'air_response_monotone': {'nh4_decreases_with_air': bool(mono_nh4), 'do3_increases_with_air': bool(mono_do)}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out_mlp', type=Path, required=True)
    p.add_argument('--out_gbm', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--members', type=int, default=5)
    a = p.parse_args()
    d = pd.read_pickle(a.data)
    report = {}
    log = train(d['data'], d['days'], a.out_mlp, members=a.members)
    report['mlp'] = {'training': log, 'val': evaluate(d['data'], d['days'], a.out_mlp, 'val', 'mlp')}
    log = train_gbm(d['data'], d['days'], a.out_gbm, members=a.members)
    report['gbm_monotone'] = {'training': log, 'val': evaluate(d['data'], d['days'], a.out_gbm, 'val', 'gbm')}
    a.report.write_text(json.dumps(report, indent=2, allow_nan=False))
    for name, r in report.items():
        v = r['val']
        print('==', name)
        print(' 한 스텝 R2', {k: round(x, 3) for k, x in v['one_step_r2'].items()})
        print(' 하루 연속', {k: (round(x, 3) if isinstance(x, float) else [round(y, 2) for y in x]) for k, x in v['open_loop'].items()})
        print(' 공기량 반응', {k: (round(x['nh4_mean'], 2), round(x['do3_mean'], 2)) for k, x in v['air_response'].items()})
        print(' 단조성', v['air_response_monotone'])


if __name__ == '__main__':
    main()

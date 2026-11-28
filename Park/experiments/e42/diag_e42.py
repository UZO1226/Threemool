"""Park/e42 사전 진단 (Park 작성): 학습 구간 마지막 40일 기록만 본다. 검증 구간은 보지 않는다.

(가) 틀린 차단 (섞음 w = 0.5, 다른 구성원 4개 평균) 의 1 스텝 NH4 오차 (환경 정답 빼기 예측) 가 공기량에 따라 어떻게 바뀌는가.
    같은 기록 상태에서 총 공기량을 1,000 ~ 9,000 으로 바꿔 본다. 환경 정답 = 그 에피소드 구성원의 GBM.
(나) 기록된 공기량으로 하루씩 보정을 돌렸을 때, 모든 공기량 후보에서 보정한 예측이 정답에 얼마나 가까운가.
    보정 없음, 상수 보정 (Park/e31, lam 0.8), 기울기 보정 (이 실험, RLS lam 0.95, b1 [-0.1, 0], p0 1).

    python diag_e42.py --data <darrow_15min.pkl> --gbm <darrow_surrogate_gbm.pkl> --mlp <darrow_surrogate_mlp.npz> --md diag.md
"""
import argparse
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
from darrow_env import Dynamics, load_days  # noqa: E402
from mlp_surrogate import MixedSurrogate, MLPSurrogate  # noqa: E402
from slope_brake import REF, rls_step  # noqa: E402

LEVELS = np.arange(1000.0, 9001.0, 1000.0)
SLOPE = (0.95, -0.1, 0.0, 1.0)
LAM_CONST = 0.8


def nh4(dyn, state, exo, zone, zp):
    """구성원별 다음 NH4 (M, N)."""
    x = np.column_stack([state, exo, zone / 1000.0, zp / 1000.0])
    return np.stack([np.clip(state[:, 0] + dyn.models[m][0].predict(x), 0.0, dyn.hi[0]) for m in range(dyn.n_members)])


def others(P, m):
    return np.delete(P, m, axis=0).mean(axis=0)


def main():
    p = argparse.ArgumentParser()
    for k in ('data', 'gbm', 'mlp', 'md'):
        p.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    splits, day_data = load_days(a.data)
    gbm = Dynamics(a.gbm)
    mix = MixedSurrogate(gbm, MLPSurrogate(a.mlp), 0.5)
    M = gbm.n_members
    days = splits['train'][-40:]
    # 레벨별 오차, 기록 공기량 오차를 하루 단위로 모은다
    lev_err = {n: [] for n in ('gbm', 'mix')}          # (레벨, 스텝) 쌓기
    rmse_acc = {k: [] for k in ('none', 'const', 'slope')}
    b1_end, b0_end = [], []
    ols, rec_u, rec_e = [], [], []   # (다) 기록 공기량과 그때의 오차
    for d in days:
        g = day_data[d]
        T = len(g['state']) - 1
        state, exo, frac = g['state'][:T], g['exo'][:T], g['frac'][:T]
        zone = g['zone'][:T]
        zp = np.vstack([g['zone'][:1], g['zone'][:T - 1]])
        u = zone.sum(axis=1)
        tru_L = np.stack([nh4(gbm, state, exo, L * frac, zp) for L in LEVELS])      # (L, M, T)
        mix_L = np.stack([nh4(mix, state, exo, L * frac, zp) for L in LEVELS])
        tru_u, mix_u = nh4(gbm, state, exo, zone, zp), nh4(mix, state, exo, zone, zp)  # (M, T)
        for m in range(M):
            lev_err['gbm'].append(tru_L[:, m] - np.stack([others(tru_L[i], m) for i in range(len(LEVELS))]))
            pred_L = np.stack([others(mix_L[i], m) for i in range(len(LEVELS))])     # (L, T)
            lev_err['mix'].append(tru_L[:, m] - pred_L)
            e = tru_u[m] - others(mix_u, m)                                           # 기록 공기량에서의 오차 (T,)
            ols.append(np.polyfit((u - REF) / 1000.0, e, 1)[0]); rec_u.append(u); rec_e.append(e)
            # 보정 돌리기: 스텝 t 에서는 t - 1 까지의 오차만 안다
            th, P = np.zeros((1, 2)), np.eye(2)[None] * SLOPE[3]
            b = 0.0
            corr_s, corr_c = np.zeros((len(LEVELS), T)), np.zeros(T)
            for t in range(T):
                if t > 0:
                    phi = np.array([[1.0, (u[t - 1] - REF) / 1000.0]])
                    th, P = rls_step(th, P, phi, e[t - 1:t], *SLOPE)
                    b = LAM_CONST * b + (1 - LAM_CONST) * e[t - 1]
                corr_s[:, t] = th[0, 0] + th[0, 1] * (LEVELS - REF) / 1000.0
                corr_c[t] = b
            err = tru_L[:, m] - pred_L
            rmse_acc['none'].append(err[:, 1:])
            rmse_acc['const'].append((err - corr_c[None])[:, 1:])
            rmse_acc['slope'].append((err - corr_s)[:, 1:])
            b0_end.append(th[0, 0]); b1_end.append(th[0, 1])
    E = {n: np.concatenate(v, axis=1) for n, v in lev_err.items()}               # (L, 전체)
    L = [f'학습 구간 마지막 40일 기록 x 환경 구성원 {M}개. 오차 = 환경 정답 (그 구성원 GBM) 빼기 차단 예측 (다른 구성원 4개 평균), 1 스텝 NH4.', '',
         '## (가) 같은 기록 상태에서 총 공기량을 바꿨을 때 오차 평균', '',
         '| 총 공기량 Nm3/h | ' + ' | '.join(f'{x:,.0f}' for x in LEVELS) + ' |', '| --- |' + ' --- |' * len(LEVELS)]
    for n, lab in (('gbm', '맞는 차단 (GBM, w = 0)'), ('mix', '틀린 차단 (w = 0.5)')):
        L.append(f'| {lab} | ' + ' | '.join(f'{v:+.3f}' for v in E[n].mean(axis=1)) + ' |')
    z = (LEVELS - REF) / 1000.0
    for n, lab in (('gbm', '맞는 차단'), ('mix', '틀린 차단')):
        mean = E[n].mean(axis=1)
        c = np.polyfit(z, mean, 1)
        r2 = 1 - ((mean - np.polyval(c, z)) ** 2).sum() / ((mean - mean.mean()) ** 2).sum()
        L.append('')
        L.append(f'- {lab}: 오차를 공기량 1,000 당 직선으로 맞추면 기울기 {c[0]:+.4f}, 5,000 에서 {c[1]:+.3f}, 결정계수 {r2:.2f}. '
                 f'오차가 양수인 공기량 {", ".join(f"{x:,.0f}" for x in LEVELS[mean > 0]) or "없음"}')
    L += ['', '## (나) 기록 공기량으로 보정을 돌렸을 때 모든 후보 공기량에서의 오차 (틀린 차단)', '',
          f'하루 첫 스텝에 되돌린다. 상수 보정 lam {LAM_CONST} (Park/e31, e41). 기울기 보정 RLS lam {SLOPE[0]}, b1 [{SLOPE[1]}, {SLOPE[2]}], p0 {SLOPE[3]}.', '',
          '| 보정 | RMSE | ' + ' | '.join(f'{x:,.0f} 평균 오차' for x in LEVELS[::2]) + ' |', '| --- | --- |' + ' --- |' * len(LEVELS[::2])]
    for k, lab in (('none', '없음'), ('const', '상수 (e41)'), ('slope', '기울기 (이 실험)')):
        R = np.concatenate(rmse_acc[k], axis=1)
        L.append(f'| {lab} | {np.sqrt((R ** 2).mean()):.3f} | ' + ' | '.join(f'{v:+.3f}' for v in R.mean(axis=1)[::2]) + ' |')
    b1 = np.array(b1_end)
    L += ['', f'- 하루 끝의 b1: 평균 {b1.mean():+.4f}, 중앙값 {np.median(b1):+.4f}, 아래 한계 (-0.1) 에 닿은 비율 {(b1 <= -0.1 + 1e-9).mean() * 100:.0f}%, '
          f'위 한계 (0) 에 닿은 비율 {(b1 >= -1e-9).mean() * 100:.0f}%. 하루 끝의 b0 평균 {np.mean(b0_end):+.3f}',
          f'- 기록 공기량의 하루 안 표준편차 평균 {np.mean([day_data[d]["zone"].sum(axis=1).std() for d in days]):,.0f} Nm3/h']
    ru, re_ = np.concatenate(rec_u), np.concatenate(rec_e)
    true_slope = np.polyfit(z, E['mix'].mean(axis=1), 1)[0]
    L += ['', '## (다) 기록 공기량을 따라 본 오차의 기울기 (식별 가능성)', '',
          '| 기울기 (오차, 공기량 1,000 당) | 값 |', '| --- | --- |',
          f'| (가) 같은 상태에서 공기량만 바꾼 구조적 기울기 | {true_slope:+.4f} |',
          f'| 하루씩 기록 공기량에 회귀한 기울기, 평균 (표준편차) | {np.mean(ols):+.4f} ({np.std(ols):.4f}) |',
          f'| 하루씩 회귀한 기울기가 음수인 비율 | {(np.array(ols) < 0).mean() * 100:.0f}% |',
          f'| 40일 전체를 한꺼번에 회귀한 기울기 | {np.polyfit((ru - REF) / 1000.0, re_, 1)[0]:+.4f} |', '',
          f'- 기록 공기량 평균 {ru.mean():,.0f}, 5% {np.percentile(ru, 5):,.0f}, 95% {np.percentile(ru, 95):,.0f} Nm3/h. 기록 공기량에서의 오차 평균 {re_.mean():+.3f}, 표준편차 {re_.std():.3f}']
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()

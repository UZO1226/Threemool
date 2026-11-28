"""Park/e38 사전 진단 (Park 작성): 학습 구간 마지막 40일 기록에서 차단이 쓰는 예측이 환경 정답과 얼마나 다른가.

환경 정답은 그 에피소드의 GBM 구성원 m 의 1 스텝 예측이다. 차단 예측은 m 을 뺀 나머지 4개 구성원의 평균이다.
지금 대표 차단은 GBM 4개 (같은 계열), 이 실험의 모델 불일치 차단은 (1 - w) x GBM + w x MLP 4개를 쓴다. 검증 구간은 보지 않는다.

    python diag_e38.py --data <darrow_15min.pkl> --gbm <darrow_surrogate_gbm.pkl> --mlp <darrow_surrogate_mlp.npz> --md diag.md
"""
import argparse
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
from darrow_env import Dynamics, load_days  # noqa: E402
from mlp_surrogate import MixedSurrogate, MLPSurrogate  # noqa: E402


def stack(day_data, days):
    S, X, F, Y = [], [], [], []
    for d in days:
        g = day_data[d]
        T = len(g['state']) - 1
        zp = np.vstack([g['zone'][:1], g['zone'][:T - 1]])
        S.append(g['state'][:T]); Y.append(g['state'][1:T + 1]); F.append(g['frac'][:T])
        X.append((g['exo'][:T], g['zone'][:T], zp))
    exo = np.vstack([x[0] for x in X]); zone = np.vstack([x[1] for x in X]); zp = np.vstack([x[2] for x in X])
    return np.vstack(S), exo, zone, zp, np.vstack(F), np.vstack(Y)


def preds(dyn, state, exo, zone, zp):
    """구성원별 다음 상태 (M, N, 5)."""
    x = np.column_stack([state, exo, zone / 1000.0, zp / 1000.0])
    return np.stack([np.clip(state + np.column_stack([dyn.models[m][j].predict(x) for j in range(5)]), 0.0, dyn.hi)
                     for m in range(dyn.n_members)])


def main():
    p = argparse.ArgumentParser()
    for k in ('data', 'gbm', 'mlp', 'md'):
        p.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    splits, day_data = load_days(a.data)
    gbm, mlp = Dynamics(a.gbm), MLPSurrogate(a.mlp)
    state, exo, zone, zp, frac, nxt = stack(day_data, splits['train'][-40:])
    mix = MixedSurrogate(gbm, mlp, 0.5)
    pg, pm, px = preds(gbm, state, exo, zone, zp), preds(mlp, state, exo, zone, zp), preds(mix, state, exo, zone, zp)
    M = gbm.n_members
    rows = []
    for name, P in (('GBM 다른 구성원 4개 (지금 대표 차단)', pg), ('섞음 w = 0.5 다른 구성원 4개 (주 판정)', px), ('MLP 다른 구성원 4개 (w = 1.0, 참고)', pm)):
        e_nh, e_do = [], []
        for m in range(M):
            br = np.delete(P, m, axis=0).mean(axis=0)
            e_nh.append(br[:, 0] - pg[m][:, 0])
            e_do.append(br[:, 2:].max(axis=1) - pg[m][:, 2:].max(axis=1))
        e_nh, e_do = np.concatenate(e_nh), np.concatenate(e_do)
        rec = P.mean(axis=0)
        rows.append(f'| {name} | {np.sqrt((e_nh ** 2).mean()):.3f} | {e_nh.mean():+.3f} | {np.sqrt((e_do ** 2).mean()):.3f} | {e_do.mean():+.3f} '
                    f'| {np.sqrt(((rec[:, 0] - nxt[:, 0]) ** 2).mean()):.3f} |')
    # 공기량 반응: 같은 상태에서 총 공기량 2,000 과 6,000 의 예측 NH4, 최대 DO (구성원 평균)
    resp = []
    for name, dyn in (('GBM (w = 0)', gbm), ('섞음 w = 0.5', mix), ('MLP (w = 1.0)', mlp)):
        lo = preds(dyn, state, exo, 2000.0 * frac, zp).mean(axis=0)
        hi = preds(dyn, state, exo, 6000.0 * frac, zp).mean(axis=0)
        dnh, ddo = hi[:, 0] - lo[:, 0], hi[:, 2:].max(axis=1) - lo[:, 2:].max(axis=1)
        resp.append(f'| {name} | {dnh.mean():+.3f} | {(dnh < 0).mean() * 100:.1f} | {ddo.mean():+.3f} | {(ddo > 0).mean() * 100:.1f} |')
    L = [f'학습 구간 마지막 40일 기록 {len(state)} 스텝 x 환경 구성원 {M}개. 환경 정답 = 그 구성원 GBM 의 1 스텝 예측.', '',
         '| 차단 예측 | NH4 오차 RMSE | NH4 평균 오차 | 최대 DO 오차 RMSE | 최대 DO 평균 오차 | 참고: 실제 기록 대비 NH4 RMSE |',
         '| --- | --- | --- | --- | --- | --- |'] + rows + [
         '', '공기량 반응 (같은 상태에서 총 공기량 2,000 -> 6,000 Nm3/h, 구성원 평균):', '',
         '| 모델 | NH4 변화 평균 | NH4 가 줄어드는 스텝 % | 최대 DO 변화 평균 | 최대 DO 가 느는 스텝 % |',
         '| --- | --- | --- | --- | --- |'] + resp
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()

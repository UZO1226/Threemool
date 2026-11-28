"""Park/e39 사전 진단 (Park 작성): 학습 구간 마지막 40일 기록에서, 측정할 수 없는 유입 변수를 학습 구간 평균으로 바꾸면
차단 예측 (환경 구성원을 뺀 GBM 4개 평균) 이 환경 정답 (그 구성원 GBM) 과 얼마나 달라지는가. 검증 구간은 보지 않는다.

    python diag_e39.py --data <darrow_15min.pkl> --gbm <darrow_surrogate_gbm.pkl> --md diag.md
"""
import argparse
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'code'))
sys.path.insert(0, str(HERE.parent / 'e38'))
from darrow_env import EXO, Dynamics, load_days  # noqa: E402
from diag_e38 import preds, stack  # noqa: E402

HIDDEN = ('inf_s_nd', 'inf_x_nd', 'inf_s_s', 'inf_x_s')   # 정책이 못 보고 처리장에서 실시간으로 재지 않는 유입 변수


def main():
    p = argparse.ArgumentParser()
    for k in ('data', 'gbm', 'md'):
        p.add_argument(f'--{k}', required=True)
    a = p.parse_args()
    splits, day_data = load_days(a.data)
    g = Dynamics(a.gbm)
    mean = np.vstack([day_data[d]['exo'] for d in splits['train']]).mean(0)
    state, exo, zone, zp, frac, _ = stack(day_data, splits['train'][-40:])
    P = preds(g, state, exo, zone, zp)
    L = [f'학습 구간 마지막 40일 기록 {len(state)} 스텝 x 환경 구성원 5개. 환경 정답 = 그 구성원 GBM 의 1 스텝 예측.', '',
         '| 차단 입력 | NH4 오차 RMSE | NH4 평균 오차 | 최대 DO 오차 RMSE | 공기 2,000 -> 6,000 NH4 변화 |', '| --- | --- | --- | --- | --- |']
    for name, cols in (('모두 앎 (지금)', []), (f'{", ".join(HIDDEN)} 를 학습 구간 평균으로 (이 실험)', [EXO.index(k) for k in HIDDEN])):
        e = exo.copy()
        e[:, cols] = mean[cols]
        Q = preds(g, state, e, zone, zp)
        enh = np.concatenate([np.delete(Q, m, 0).mean(0)[:, 0] - P[m][:, 0] for m in range(5)])
        edo = np.concatenate([np.delete(Q, m, 0).mean(0)[:, 2:].max(1) - P[m][:, 2:].max(1) for m in range(5)])
        resp = (preds(g, state, e, 6000.0 * frac, zp).mean(0)[:, 0] - preds(g, state, e, 2000.0 * frac, zp).mean(0)[:, 0]).mean()
        L.append(f'| {name} | {np.sqrt((enh ** 2).mean()):.3f} | {enh.mean():+.3f} | {np.sqrt((edo ** 2).mean()):.3f} | {resp:+.3f} |')
    L += ['', '학습 구간 평균: ' + ', '.join(f'{k} {mean[EXO.index(k)]:.3f}' for k in HIDDEN)]
    Path(a.md).write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()

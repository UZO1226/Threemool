"""Recompute branch residuals from raw forecasts, aggregating by calendar date."""
import argparse
import json
from pathlib import Path

import numpy as np


def daily(values, days):
    return np.mean([values[days == d].mean(0) for d in np.unique(days)], axis=0)


def export(root, dest):
    records = json.loads((root/'summary.json').read_text())
    assert [r['seed'] for r in records] == list(range(5))
    lines = ['# e99 경로별 공기량 예측 오차', '',
             '창 안에서 3~6시간 공기 부피(m³)를 합산했다. 날짜별 평균 뒤 seed에 같은 비중을 준다. '
             'RMSE는 seed마다 제곱근을 취한 뒤 평균한다. 제곱오차 분해 및 상쇄율은 seed별 MSE의 평균을 사용한다.', '']
    for group, title in [('supported', '거리 조건 안'), ('all', '전체 비중복 창')]:
        metrics = []
        for r in records:
            g = r['groups'][group]
            with np.load(root/f"branches{r['seed']}.npz") as z:
                mask = z['supported'] if group == 'supported' else np.ones(len(z['keys']), bool)
                days = z['keys'][mask, 0]
                first = z['first'][mask]
                selected = z['selected'][mask]
                assert (mask.sum(), first.sum(), selected.sum()) == (g['n'], g['candidates'], g['selected'])
                err = z['truth'][mask, None, :] - z['pred'][mask]
                a, b = err[first].transpose(2, 0, 1)
                d = days[first]
                check = dict(mse_a=daily(a*a, d), mse_b=daily(b*b, d), mse_delta=daily((b-a)**2, d),
                             cross=daily(a*b, d), covariance=daily((a-daily(a,d))*(b-daily(b,d)), d),
                             from_a=daily(-err[:, :, 0]*selected[:, None], days),
                             from_b=daily(err[:, :, 1]*selected[:, None], days))
                for key, value in check.items():
                    np.testing.assert_allclose(value, g[key], rtol=1e-12, atol=1e-7)
            metrics.append(g)

        def col(key):
            return np.array([g[key] for g in metrics])
        ma, mb, md, cross, cov = [col(k).mean(0) for k in ('mse_a','mse_b','mse_delta','cross','covariance')]
        lines += [f'## {title}', '', f"전체 {col('n').sum()}개, 감량 후보 {col('candidates').sum()}개, 선택 {col('selected').sum()}개다.", '',
                  '| 시점 | A RMSE | B RMSE | B−A RMSE | 오차 곱 평균 (m⁶) | 공분산 (m⁶) | 경로 MSE 합 중 상쇄율 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        names = ['감량 직전', '1시간', '3시간']
        for i, name in enumerate(names):
            rmse = [np.sqrt(col(k)[:, i]).mean() for k in ('mse_a','mse_b','mse_delta')]
            lines.append(f'| {name} | {rmse[0]:,.2f} | {rmse[1]:,.2f} | {rmse[2]:,.2f} | {cross[i]:,.1f} | {cov[i]:,.1f} | {100*(1-md[i]/(ma[i]+mb[i])):.2f}% |')
        changes = [x[-1]-x[0] for x in (ma, mb, -2*cross, md)]
        lines += ['', '최초에서 3시간까지 A MSE 변화, B MSE 변화, 교차항 변화, 차이 MSE의 순변화(m⁶)는 차례로 '
                  + ', '.join(f'{v:+,.1f}' for v in changes)+'이다.', '',
                  '| 시점 | 기본 A 과대예측 기여 (m³) | 감량 B 과소예측 기여 (m³) | 절감 과대예측 합 (m³) |',
                  '| --- | ---: | ---: | ---: |']
        ca, cb = col('from_a'), col('from_b')
        for i, name in enumerate(names):
            lines.append(f'| {name} | {ca[:,i].mean():+.2f} | {cb[:,i].mean():+.2f} | {(ca+cb)[:,i].mean():+.2f} |')
        lines += ['', '미선택 창은 기여 0으로 두었다. 양수는 절감을 과대예측한 방향이다.', '',
                  '| seed | 선택 창 | 최초 A 기여 | 최초 B 기여 | 3시간 A 기여 | 3시간 B 기여 |',
                  '| ---: | ---: | ---: | ---: | ---: | ---: |']
        for j, g in enumerate(metrics):
            lines.append(f"| {j} | {g['selected']} | {ca[j,0]:+.2f} | {cb[j,0]:+.2f} | {ca[j,-1]:+.2f} | {cb[j,-1]:+.2f} |")
        lines.append('')
    dest.write_text('\n'.join(lines).rstrip()+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--dest', required=True, type=Path)
    args = parser.parse_args()
    export(args.root, args.dest)

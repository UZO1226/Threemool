"""Score the paired frozen forecasts at equal date weight."""
import argparse
import json
from pathlib import Path

import numpy as np

from calibrate import METRICS, metrics


def summarize(archive):
    ids, dq = archive['ids'], archive['dq']
    n = len(archive['dates'])
    result = {}
    rng = np.random.default_rng(106)
    starts = rng.integers(n, size=(2000, (n+6)//7))
    draws = ((starts[:, :, None]+np.arange(7))%n).reshape(2000, -1)[:, :n]
    for label, mask in [('증가', dq>0), ('감소', dq<0)]:
        dates = np.unique(ids[mask])
        values = np.stack([metrics(archive['truth'], p) for p in archive['preds']])
        day = np.full((n, len(values), len(METRICS)), np.nan)
        for d in dates:
            day[d] = values[:, mask & (ids==d)].mean(axis=1)
        averages = day[dates].mean(axis=0)
        paired = (day[:, 1] - day[:, 0])[draws]
        sampled = np.nanmean(paired[np.isfinite(paired[:, :, 0]).any(axis=1)], axis=1)
        intervals = np.quantile(sampled, [.025, .975], axis=0)
        fields = {}
        for k, metric in enumerate(METRICS):
            fields[metric] = dict(base=averages[0, k], candidate=averages[1, k],
                                  delta=averages[1, k]-averages[0, k], ci=intervals[:, k].tolist())
        result[label] = dict(events=int(mask.sum()), days=len(dates), metrics=fields)
    return result


def passed(rows):
    up, down = [rows[k]['metrics'] for k in ('증가', '감소')]
    return bool(all(up[k]['delta'] < 0 for k in ('nh4_3h', 'segment_mae'))
                and all(down[k]['delta'] <= 0 for k in ('nh4_3h', 'segment_mae'))
                and all(x['do_3h']['delta'] <= 0 for x in (up, down)))


def export(root, dest):
    result = {'selection': json.loads((root/'selection.json').read_text())}
    lines = ['# e106 결과', '', '농도 오차는 NH4 mg N/L, 최대 DO mg O₂/L. '
             '차이는 보정에서 기본을 뺀 값이다. 날짜 안 창 평균 후 날짜 동일 가중.', '',
             '| 학습 앞 140일 보정 강도 | 증가 1~3시간 NH4 변화 MAE |',
             '| ---: | ---: |']
    for gain, loss in zip(result['selection']['candidates'], result['selection']['fit_increase_segment_mae']):
        lines.append(f'| {gain:.2f} | {loss:.4f} |')
    labels = ('NH4 15분 MAE', 'NH4 1시간 MAE', 'NH4 3시간 MAE', 'NH4 1~3시간 변화 MAE',
              '최대 DO 3시간 MAE', '실측 NH4 1~3시간 변화', '예측 NH4 1~3시간 변화')
    for name, split in [('confirm', '학습 뒤 48일'), ('winter', '겨울 개발 40일')]:
        with np.load(root/f'{name}.npz') as z:
            np.testing.assert_array_equal(z['gains'], [0, result['selection']['gain']])
            rows = summarize(z)
        result[name] = dict(passed=passed(rows), directions=rows)
        for direction, table in rows.items():
            lines += ['', f"## {split}, 폭기 {direction} ({table['events']}개 창, {table['days']}일)", '',
                      '| 지표 | 기본 | 보정 | 차이 [탐색 95% 구간] |', '| --- | ---: | ---: | ---: |']
            for key, label in zip(METRICS, labels):
                v = table['metrics'][key]
                lo, hi = v['ci']
                lines.append(f"| {label} | {v['base']:.4f} | {v['candidate']:.4f} | "
                             f"{v['delta']:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
    dest.mkdir(parents=True, exist_ok=True)
    (dest/'results.md').write_text('\n'.join(lines)+'\n')
    (dest/'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('dest', type=Path)
    args = parser.parse_args()
    export(args.root, args.dest)

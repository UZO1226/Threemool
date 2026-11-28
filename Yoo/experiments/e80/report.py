"""Check branch archives against the summaries and render the chronological comparisons."""
import argparse
import json
from pathlib import Path

import numpy as np


def report(root, dest):
    result = json.loads((root/'summary.json').read_text())
    assert result['steps'] == [4, 12, 24] and result['metrics'] == ['air', 'nh', 'do', 'excess']
    groups = ['train48_measured', 'winter_measured', 'winter_policy']
    names = ['학습 뒤 48일 실측', '겨울 실측', '겨울 양방향 정책']
    rows = {(r['seed'], r['group']): r for r in result['rows']}
    assert len(result['rows']) == len(rows) == 15
    table = ['# e80 누적 효과 및 날짜별 일관성', '',
             '모든 값은 첫 명령을 증량한 경로에서 기본 명령 경로를 뺀 차이다. '
             '공기는 m³, NH4 및 DO 위반은 15분 스텝 수, NH4 초과 적분은 mg h/L다. '
             '짝 수는 (날짜, 시점, GBM 구성원) 조합 수다. 같은 날짜 안에서 먼저 평균한다. '
             '누적 표의 시간은 첫 명령 시점부터의 경과 시간이다.']
    for group, name in zip(groups, names):
        table += ['', f'## {name}', '',
                  '| seed | 날짜 | 짝 | 시간(h) | 공기 | NH4 위반 | DO 위반 | NH4 초과 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        late = []
        for seed in range(5):
            r = rows[seed, group]
            with np.load(root/f'{seed}_{group}.npz') as z:
                keys = z['keys']
                assert len(np.unique(keys,axis=0)) == len(keys) == r['pairs']
                assert np.isin(keys[:, 1], np.arange(4, 69, 4)).all()
                assert z['totals'].shape == (len(keys), 2, 3, 4)
                assert (np.diff(z['totals'], axis=2) >= -1e-8).all()
                diff = np.diff(z['totals'], axis=1)[:, 0]
                days = np.unique(keys[:, 0])
                assert len(days) == r['days']
                if not len(days):
                    table.append(f'| {seed} | 0 | 0 | | | | | |')
                    continue
                day = np.array([diff[keys[:, 0] == d].mean(axis=0) for d in days])
                weighted = np.average(diff, axis=0, weights=[1/np.sum(keys[:, 0] == d) for d in keys[:, 0]])
                np.testing.assert_allclose(weighted, r['means'], atol=1e-8, rtol=0)
                late_day = day[:, 2] - day[:, 0]
                np.testing.assert_allclose(late_day.mean(0), r['late_mean'], atol=1e-8, rtol=0)
                count = (late_day < -1e-8).sum(0)
                np.testing.assert_array_equal(count, r['late_negative_days'])
                both = int(np.sum((late_day[:, 1] < -1e-8) & (late_day[:, 3] < -1e-8)))
                assert both == r['both_nh_negative_days']
            for h, values in zip([1, 3, 6], weighted):
                table.append(f"| {seed} | {len(days)} | {len(keys)} | {h} | " +
                             ' | '.join(f'{v:+.4f}' for v in values) + ' |')
            late.append(f"| {seed} | {len(days)} | {r['late_mean'][1]:+.4f} | {r['late_mean'][3]:+.4f} | " +
                        f"{count[1]} | {count[3]} | {both} |")
        table += ['', '첫 1시간을 제외하고 6시간까지 추가된 효과를 집계했다. 감소 일수는 해당 날짜의 차이가 음수인 일수다.', '',
                  '| seed | 날짜 | NH4 위반 차 | NH4 초과 차 | 위반 감소 일수 | 초과 감소 일수 | 둘 다 감소 일수 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |', *late]
    (dest/'results.md').write_text('\n'.join(table)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dest', type=Path, required=True)
    args = parser.parse_args()
    report(args.root, args.dest)

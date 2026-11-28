"""Check date-balanced statistics against raw predictions and write tables."""
import argparse
import json
from pathlib import Path

import numpy as np


def by_date(values, day):
    return np.mean([values[day==d].mean(axis=0) for d in np.unique(day)], axis=0)


def export(root, dest):
    records = json.loads((root/'summary.json').read_text())
    assert [r['seed'] for r in records] == list(range(5))
    lines = ['# e97 감량 전 관측의 시간대별 순공기 예측', '',
             '공기 차이는 감량 경로 빼기 기본 경로(m³)다. 음수는 절감이다. 예측 오차는 감량 후보, '
             '선택의 효과는 미선택 창을 포함한 전체 창에서 산출했다. 각 날짜 및 seed의 비중은 같다.', '']
    labels = ['0~1시간','1~3시간','3~6시간']
    for group, title in [('supported', '거리 조건 안'), ('all', '전체 비중복 창')]:
        results = []
        counts = np.zeros(3, dtype=int)
        for r in records:
            g = r['groups'][group]
            with np.load(root/f"interval{r['seed']}.npz") as z:
                mask = z['supported'] if group == 'supported' else np.ones(len(z['keys']),bool)
                p,y = z['pred'][mask],z['truth'][mask]
                first,selected,day = z['first'][mask],z['selected'][mask],z['keys'][mask,0]
                target_days = day[first]
                for source, err, acc in [(p,'rmse','accuracy'), (np.broadcast_to(z['center'],y.shape),'constant_rmse','constant_accuracy')]:
                    mse = by_date(((source-y)**2)[first], target_days)
                    hit = by_date(((source < -1e-8) == (y < -1e-8))[first], target_days)
                    np.testing.assert_allclose([np.sqrt(mse),hit],[g[err],g[acc]],rtol=0,atol=1e-7)
                np.testing.assert_allclose([by_date(t*selected[:,None],day) for t in (p,y)],
                                           [g['pred_effect'],g['actual_effect']],rtol=0,atol=1e-7)
                saved_first = y[:,0] < -1e-8
                saved_final = y.sum(1) < -1e-8
                counts += [(selected & saved_first).sum(), (selected & saved_final).sum(),
                           (selected & saved_first & ~saved_final).sum()]
            results.append(g)
        def mean(key):
            return np.mean([g[key] for g in results],axis=0)
        positive = (np.array([g['skill'] for g in results])>0).sum(0)
        lines += [f'## {title}', '',
                  '| 구간 | 회귀 RMSE | 상수 RMSE | 상수 대비 제곱오차 감소율 | 오차 감소 seed | 절감 방향 적중률 | 상수 방향 적중률 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for i,label in enumerate(labels):
            lines.append(f"| {label} | {mean('rmse')[i]:.2f} | {mean('constant_rmse')[i]:.2f} | "
                         f"{100*mean('skill')[i]:+.2f}% | {positive[i]}/5 | {100*mean('accuracy')[i]:.1f}% | {100*mean('constant_accuracy')[i]:.1f}% |")
        lines += ['', '| e96 선택의 구간별 기여 | 감량 전 예측 | 모의 운전 결과 | 결과 빼기 예측 |',
                  '| --- | ---: | ---: | ---: |']
        p,y = mean('pred_effect'),mean('actual_effect')
        for label,f,t in zip(labels+['6시간 합계'],[*p,p.sum()],[*y,y.sum()]):
            lines.append(f'| {label} | {f:+.2f} | {t:+.2f} | {t-f:+.2f} |')
        lines += ['', f"선택 {sum(g['selected'] for g in results)}건 중 첫 시간에 순절감 {counts[0]}건, "
                  f"6시간에 순절감 {counts[1]}건이었다. 첫 시간에 절감했다가 6시간에 절감을 놓친 건 {counts[2]}건이었다.", '',
                  '| seed | 최초 후보 | e96 선택 | 0~1시간 오차 감소율 | 1~3시간 오차 감소율 | 3~6시간 오차 감소율 |',
                  '| ---: | ---: | ---: | ---: | ---: | ---: |']
        for seed,g in enumerate(results):
            lines.append(f"| {seed} | {g['candidates']} | {g['selected']} | " + ' | '.join(f'{100*s:+.2f}%' for s in g['skill']) + ' |')
        lines.append('')
    dest.write_text('\n'.join(lines).rstrip()+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dest', type=Path, required=True)
    args = parser.parse_args()
    export(args.root, args.dest)

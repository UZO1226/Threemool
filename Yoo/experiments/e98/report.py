"""Audit the landmark comparison by averaging raw predictions within each date."""
import argparse
import json
from pathlib import Path

import numpy as np


def daily(value, day):
    return np.mean([value[day == d].mean(axis=0) for d in np.unique(day)], axis=0)


def export(root, dest):
    rows = json.loads((root / 'summary.json').read_text())
    assert [r['seed'] for r in rows] == list(range(5))
    lines = ['# e98 고정한 3~6시간 순공기 차이의 예측 결과', '',
             'RMSE와 선택 창의 기여 단위는 m³다. 각 수치는 날짜와 seed에 같은 비중을 준다. '
             '제곱오차 감소율은 seed별 기준값에 대해 계산한 후 평균했다.', '']
    for group, title in [('supported','거리 조건 안'), ('all','전체 비중복 창')]:
        results = []
        for rec in rows:
            g = rec['groups'][group]
            with np.load(root/f"landmark{rec['seed']}.npz") as z:
                mask = z['supported'] if group=='supported' else np.ones(len(z['keys']),bool)
                day = z['keys'][mask,0]
                first, selected = z['first'][mask], z['selected'][mask]
                p,y = z['pred'][mask], z['truth'][mask,None]
                mse = daily(((p-y)**2)[first],day[first])
                reference = daily(((z['center']-y)**2)[first],day[first]).item()
                for key, values in [('rmse',np.sqrt(mse)), ('vs_constant',1-mse/reference),
                                    ('vs_initial',1-mse/mse[0]), ('vs_previous',1-mse/mse[[0,0,1]]),
                                    ('accuracy',daily(((p[first]<-1e-8)==(y[first]<-1e-8)),day[first])),
                                    ('pred_effect',daily(p*selected[:,None],day))]:
                    np.testing.assert_allclose(values,g[key],rtol=0,atol=1e-7)
                np.testing.assert_allclose(daily(y[:,0]*selected,day), g['actual_effect'], atol=1e-7,rtol=0)
            results.append(g)
        def column(key):
            return np.array([g[key] for g in results])
        lines += [f'## {title}', '',
                  f"감량 후보 {column('candidates').sum()}개다. 학습 평균의 RMSE는 {column('constant_rmse').mean():.2f} m³, 절감 방향 적중률은 {100*column('constant_accuracy').mean():.1f}%다.", '',
                  '| 예측 시점 | RMSE | 방향 적중률 | 상수 대비 오차 감소율 (seed) | 최초 대비 (seed) | 직전 대비 (seed) |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for i,name in enumerate(['감량 직전','1시간 후','3시간 후']):
            cols = []
            for key in ('vs_constant','vs_initial','vs_previous'):
                vals=column(key)[:,i]
                cols.append(f'{100*vals.mean():+.2f}% ({(vals>0).sum()}/5)' if i or key=='vs_constant' else '기준')
            lines.append(f"| {name} | {column('rmse')[:,i].mean():.2f} | {100*column('accuracy')[:,i].mean():.1f}% | "+' | '.join(cols)+' |')
        lines += ['', '| seed | 후보 | 최초 RMSE | 1시간 RMSE | 3시간 RMSE | 3시간의 최초 대비 오차 감소율 |',
                  '| ---: | ---: | ---: | ---: | ---: | ---: |']
        for seed,g in enumerate(results):
            rmse = ' | '.join(f'{v:.2f}' for v in g['rmse'])
            lines.append(f"| {seed} | {g['candidates']} | {rmse} | {100*g['vs_initial'][-1]:+.2f}% |")
        vals=column('pred_effect').mean(0)
        lines += ['',f"e96 선택 {column('selected').sum()}건을 고정하고 미선택까지 포함한 {column('n').sum()}개 창의 평균 기여를 계산했다. "
                  f"예측은 최초 {vals[0]:+.2f}, 1시간 후 {vals[1]:+.2f}, 3시간 후 {vals[2]:+.2f} m³이고 모의 결과는 {column('actual_effect').mean():+.2f} m³다.",'']
    dest.write_text('\n'.join(lines).rstrip()+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--dest',type=Path,required=True)
    args=parser.parse_args()
    export(args.root,args.dest)

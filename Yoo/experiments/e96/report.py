"""Reconstruct the date-matched control, prediction error and seed comparisons."""
import argparse
import json
from pathlib import Path

import numpy as np


def joint(diff):
    return (diff[..., 0] < -1e-8) & (diff[..., 1:] <= 1e-8).all(axis=-1)


def export(root, dest):
    rows = json.loads((root / 'summary.json').read_text())
    lines = ['# e96 첫 감량 예측과 선택 결과', '',
             '차이는 감량 방식 빼기 기본 FCSRL이다. 창당 공기는 m³, 위반은 15분 스텝, NH4 초과 적분은 mg h/L이다.', '']
    for group, title in [('supported', '거리 조건 안'), ('all', '전체 비중복 창')]:
        results = []
        counts = np.zeros(3, int)
        for rec in rows:
            g = rec['groups'][group]
            with np.load(root / f"prediction{rec['seed']}.npz") as z:
                mask = z['supported'] if group == 'supported' else np.ones(len(z['first']), bool)
                day = z['keys'][mask, 0]
                pred, truth, first = [z[k][mask] for k in ['pred','truth','first']]
                chosen = first & joint(pred)
                counts += [int((chosen & (truth[:, 0] < -1e-8)).sum()),
                           int((chosen & (truth[:, 1:] <= 1e-8).all(1)).sum()), int((chosen & joint(truth)).sum())]
                candidate_days = np.unique(day[first])
                def rmse(estimate):
                    residual = (estimate - truth)**2
                    return np.sqrt(np.mean([residual[first & (day == d)].mean(0) for d in candidate_days], axis=0))
                np.testing.assert_allclose(rmse(pred), g['rmse'])
                np.testing.assert_allclose(rmse(z['train_mean']), g['reference_rmse'])
                for mode in ('random','forecast','known_air','known_quality','oracle','prior'):
                    take = {'random': chosen, 'forecast': chosen, 'known_air': first & (truth[:,0] < -1e-8) & (pred[:,1:] <= 1e-8).all(1),
                            'known_quality': first & (pred[:,0] < -1e-8) & (truth[:,1:] <= 1e-8).all(1),
                            'oracle': first & joint(truth), 'prior': first}[mode]
                    daily = []
                    for d in np.unique(day):
                        ids = day == d
                        if mode == 'random' and (first & ids).any():
                            daily.append(truth[first & ids].mean(0) * chosen[ids].mean())
                        else:
                            daily.append((truth[ids] * take[ids,None]).mean(0))
                    np.testing.assert_allclose(np.mean(daily, axis=0), g['outcomes'][mode]['delta'], atol=1e-8, rtol=0)
            results.append(g)
        lines += [f'## {title}', '', '| 방식 | 감량 횟수 | 공동 개선 창 | 공기 차이 | NH4 위반 | DO 위반 | NH4 초과 | 공동 개선 seed |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for k,label in [('prior','e93 최초 선택'), ('forecast','네 지표 예측'), ('random','같은 날짜 횟수 무작위 기대'),
                        ('known_air','공기만 정답'), ('known_quality','수질만 정답'), ('oracle','모두 정답')]:
            stats = [g['outcomes'][k] for g in results]
            delta = np.array([s['delta'] for s in stats])
            cols = ' | '.join(f'{v:+.5f}' for v in delta.mean(0))
            lines.append(f"| {label} | {sum(s['n'] for s in stats):.0f} | {sum(s['hits'] for s in stats):.2f} | {cols} | {joint(delta).sum()}/5 |")
        forecast = np.array([g['outcomes']['forecast']['delta'] for g in results])
        coin = np.array([g['outcomes']['random']['delta'] for g in results])
        lines += ['', f'예측으로 고른 창 중 공기 감소 {counts[0]}건, 세 수질 비악화 {counts[1]}건, 둘 다 달성 {counts[2]}건이다. '
                  f'무작위 기대 대비 공동 개선은 {joint(forecast-coin).sum()}/5 seed다.', '',
                  '| 예측 항목 | RMSE | 학습 평균의 RMSE | 학습 평균 대비 제곱오차 감소율 | 감소한 seed |',
                  '| --- | ---: | ---: | ---: | ---: |']
        for i,label in enumerate(['공기','NH4 위반','DO 위반','NH4 초과 적분']):
            rm = np.mean([g['rmse'][i] for g in results])
            const = np.mean([g['reference_rmse'][i] for g in results])
            skills = np.array([g['skill'][i] for g in results], float)
            lines.append(f'| {label} | {rm:.4f} | {const:.4f} | {100*np.nanmean(skills):+.2f}% | {(skills>0).sum()}/{np.isfinite(skills).sum()} |')
        lines += ['', '| seed | 최초 후보 | 감량 | 공기 차이 | NH4 위반 | DO 위반 | NH4 초과 |',
                  '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for i,g in enumerate(results):
            s = g['outcomes']['forecast']
            lines.append(f"| {i} | {g['candidates']} | {s['n']:.0f} | " + ' | '.join(f'{v:+.5f}' for v in s['delta']) + ' |')
        lines.append('')
    dest.write_text('\n'.join(lines).rstrip()+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dest', type=Path, required=True)
    args = parser.parse_args()
    export(args.root, args.dest)

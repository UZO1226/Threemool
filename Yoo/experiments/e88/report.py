"""Audit the chronological decisions and render seed-level results from archived scores."""
import argparse
import json
from pathlib import Path

import numpy as np

METHODS = ('raw', 'learned', 'random_features')
NAMES = dict(raw='관측', learned='관측 + FCSRL', random_features='관측 + 무작위 표현', coin='무작위 선택 기대값')


def dominates(delta):
    return bool(delta[0] < -1e-8 and (delta[1:] <= 1e-8).all())


def audit(root, r):
    with np.load(root/f"probe{r['seed']}.npz") as z:
        assert max(z['train_dates']) < min(z['dates'])
        keys, yt, change = z['keys'], z['positive'], z['change']
        train_change = z['train_totals'][:,1] - z['train_totals'][:,0]
        assert ((train_change[:,0]<=-25)&(train_change[:,1:]<=1e-8).all(1)).sum()==r['train_pos']
        np.testing.assert_array_equal(yt, (change[:,0]<=-25)&(change[:,1:]<=1e-8).all(1))
        for j, method in enumerate(METHODS):
            score = z['scores'][j]
            first_choice = score >= r['modes'][method]['threshold']
            take = np.zeros(len(keys),bool)
            included = np.zeros(len(keys),bool)
            pending = {}
            for i in sorted(range(len(keys)), key=lambda k:tuple(keys[k])):
                m = keys[i,2]
                if m in pending:
                    first = pending.pop(m)
                    included[[first,i]] = True
                    take[i] = not take[first]
                else:
                    pending[m] = i
                    take[i] = first_choice[i]
            np.testing.assert_array_equal(included,z['matched'])
            for protocol, mask, selection in [('matched',included,take),('free',np.ones(len(keys),bool),first_choice)]:
                stats = r['modes'][method][protocol]
                assert int((selection&mask).sum())==stats['cuts']
                obtained = change * selection[:, None]
                means = [obtained[mask & (keys[:,0]==d)].mean(0) for d in np.unique(keys[mask,0])]
                np.testing.assert_allclose(np.mean(means,axis=0),stats['delta'],atol=1e-9)
                if stats['cuts']:
                    np.testing.assert_allclose(yt[selection&mask].mean(),stats['precision'])


def report(root, dest):
    records = json.loads((root/'summary.json').read_text())
    for r in records:
        audit(root,r)
    lines = ['# e88 감량 선별 수치','',
             '순절감은 원 명령 빼기 선별 경로의 6시간 공기다. 수질은 선별 빼기 원 명령이다. '
             '아래 평균은 창 안 결과를 적격 날짜별로 평균한 뒤 seed를 동일 가중했다. '
             'AUC는 뒤 48일 전체 적격 창에서 날짜가중 계산했다. 정밀도는 선택한 창 중 공동 개선 표적의 비율이다.','']
    for protocol,title in [('matched','두 후보당 한 번 감량'), ('free','앞 140일 점수 중앙값만 적용')]:
        lines += ['## '+title,'','| 입력 | 감량 횟수 합 | 평균 정밀도 | 평균 AUC | 순절감 m³/창 | NH4 위반 스텝 | DO 위반 스텝 | NH4 초과 mg h/L | 기본 대비 통과 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        deltas = {}
        for method in (*METHODS, 'coin') if protocol=='matched' else METHODS:
            rows = [r['modes']['coin'] if method=='coin' else r['modes'][method][protocol] for r in records]
            deltas[method] = delta = np.array([s['delta'] for s in rows])
            air, nh, do, excess = delta.mean(0)
            auc = '0.500' if method=='coin' else f"{np.mean([r['modes'][method]['auc'] for r in records]):.3f}"
            precision = np.mean([s['precision'] for s in rows])
            lines.append(f"| {NAMES[method]} | {sum(s['cuts'] for s in rows):.0f} | {precision:.1%} | {auc} | "
                         f"{-air:.2f} | {nh:+.4f} | {do:+.4f} | {excess:+.4f} | {sum(map(dominates,delta))}/5 |")
        if protocol=='matched':
            for method in ('raw','random_features'):
                diff = deltas['learned']-deltas[method]
                lines += ['', f"학습 표현 방식에서 {NAMES[method]} 방식의 차이를 뺀 값(공기, NH4, DO, 초과 적분)은 "
                          + ', '.join(f'{v:+.4f}' for v in diff.mean(0))
                          + f"이다. 네 지표 공동 기준은 {sum(map(dominates,diff))}/5, 5개 평균은 {'통과' if dominates(diff.mean(0)) else '미달'}이다."]
        lines += ['', '| seed | 입력 | 감량/후보 | AUC | 순절감 m³/창 | NH4 위반 | DO 위반 | NH4 초과 |',
                  '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for r in records:
            for m in METHODS:
                s=r['modes'][m][protocol]
                air,nh,do,exc=s['delta']
                lines.append(f"| {r['seed']} | {NAMES[m]} | {s['cuts']:.0f}/{s['candidates']} | "
                             f"{r['modes'][m]['auc']:.3f} | {-air:.2f} | {nh:+.4f} | {do:+.4f} | {exc:+.4f} |")
        lines.append('')
    (dest/'results.md').write_text('\n'.join(lines).rstrip()+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--dest',type=Path,required=True)
    args=p.parse_args()
    report(args.root,args.dest)

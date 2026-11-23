"""Render paired winter results and independently recheck selection and attribution totals."""
import argparse
import json
from pathlib import Path

import numpy as np


def num(x):
    return '해당 없음' if x is None else f'{x:.3f}'


def report(root,dest):
    data = json.loads((root/'summary.json').read_text())
    rows = data['rows']
    lines = ['# e82 겨울 회수 진단','',
             '표의 기여는 회수 경로 빼기 증량 단독 경로다. 양수이면 정책 명령을 더 크게 만든다. 첫 명령은 감량 다음 시점이 있는 경우만, 하루 기여는 감량이 없던 날도 포함한다.','']
    for key,title,unit in [('immediate','감량 직후 다음 명령','m³/h'),
                           ('immediate_abs','감량 직후 다음 명령, 절대 기여 평균','m³/h'),
                           ('daily','하루 전체의 기본 명령 부피','m³')]:
        lines += [f'## {title} ({unit})','', '| seed | 직후 짝 | NH4 | NO3 | DO 3개 | 직전 공기 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for r in rows:
            lines.append(f"| {r['seed']} | {r['n_next']} | "+' | '.join(f'{v:+.3f}' for v in r[key])+' |')
        means = np.average([r[key] for r in rows],axis=0,
                           weights=[1 if key=='daily' else r['n_next'] for r in rows])
        lines += ['| 통합 | '+str(sum(r['n_next'] for r in rows))+' | '+' | '.join(f'{v:+.3f}' for v in means)+' |','']
    lines += ['## 감량 직전 판단의 분류 성적','',
              '양성은 회수가 증량 단독보다 하루 공기를 덜 쓰고 세 수질 지표를 모두 유지 또는 개선한 에피소드다. 학습은 188일이며 표의 학습 숫자는 적격 에피소드/양성/해당 날짜다.','',
              '| seed | 학습 | 겨울 적격 | 양성 | 선택 | 정선택 | 정밀도 | 재현율 | AUC |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for i,c in enumerate([r['class_stats'] for r in rows]+[data['pooled']]):
        train = '/'.join(map(str,data['train_counts'][i])) if i<5 else ''
        lines.append(f"| {i if i<5 else '통합'} | {train} | {c['n']} | {c['positive']} | {c['selected']} | "+
                     f"{c['tp']} | {num(c['precision'])} | {num(c['recall'])} | {num(c['auc'])} |")
    lines += ['','## 모든 겨울 날짜의 하루 결과','',
              '| 모드 | 공기 m³/23.75h | NH4 위반 스텝 | DO 위반 스텝 | NH4 초과 mg h/L |',
              '| --- | ---: | ---: | ---: | ---: |']
    for m in ('base','pulse','recover','selected'):
        avg=np.mean([r['metrics'][m] for r in rows],axis=0)
        lines.append('| '+m+' | '+' | '.join(f'{v:.4f}' for v in avg)+' |')
    lines += ['', '선택 회수에서 기본 FCSRL을 뺀 차이다.','','| seed | 공기 | NH4 위반 | DO 위반 | NH4 초과 | 공동 개선 |',
              '| --- | ---: | ---: | ---: | ---: | --- |']
    differences=[]
    for r in rows:
        with np.load(root/f"winter{r['seed']}.npz") as z:
            mask=z['selected']
            episode=np.where(mask[:,None],z['day_recover'],z['day_pulse'])
            np.testing.assert_allclose(episode,z['day_selected'],rtol=0,atol=1e-8)
            np.testing.assert_allclose(z['phi'].mean(0).sum(0)*.25,r['daily'],rtol=0,atol=1e-8)
            dif=(episode-z['day_base']).mean(0)
            np.testing.assert_allclose(dif,np.subtract(r['metrics']['selected'],r['metrics']['base']),atol=1e-8,rtol=0)
            differences.append(dif)
    for i,v in enumerate(differences+[np.mean(differences,axis=0)]):
        win=v[0] < -1e-8 and np.all(v[1:]<=1e-8)
        lines.append(f"| {i if i<5 else '평균'} | "+' | '.join(f'{k:+.4f}' for k in v)+f" | {'충족' if win else '미달'} |")
    (dest/'results.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--dest',type=Path,required=True)
    a=p.parse_args()
    report(a.root,a.dest)

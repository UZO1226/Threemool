"""Independently check paired choices and date-weighted metrics, and render seed tables."""
import argparse
import json
from pathlib import Path

import numpy as np


def daily(x,day):
    return np.mean([x[day==d].mean(axis=0) for d in np.unique(day)],axis=0)


def check(root,records):
    for row in records:
        with np.load(root/f"hour{row['seed']}.npz") as z:
            for group,mask in [('supported',z['supported']),('all',np.ones(len(z['keys']),bool))]:
                s=row['groups'][group]
                a,b,d=z['totals'][mask].transpose(1,0,2)
                day=z['keys'][mask,0]
                w=1/np.bincount(day)[day]
                y=((b[:,1]<=d[:,1]+1e-8)&(b[:,3]<=d[:,3]+1e-8)
                   & ((b[:,1]<d[:,1]-1e-8)|(b[:,3]<d[:,3]-1e-8)))
                assert len(day)==s['n']
                for j,name in enumerate(('level','trend')):
                    r=s[name]
                    score=z['scores'][j,mask]
                    threshold=row['settings'][name]['cutoff']
                    keep=np.ones(len(day),bool) if threshold is None else score>=threshold
                    np.testing.assert_array_equal(keep,z['keep'][j,mask])
                    chosen=np.where(keep[:,None],b,d)
                    assert keep.sum()==r['outcome']['retained']
                    np.testing.assert_allclose(daily(chosen-a,day),r['outcome']['vs_base'],atol=1e-9)
                    np.testing.assert_allclose(daily(chosen-b,day),r['outcome']['vs_keep'],atol=1e-9)
                    assert y.sum()==r['positive'] and (y&~keep).sum()==r['missed']
                    for mask_rate,values,stat in [(y,keep,'recall'),(~y,keep,'fpr'),(keep,y,'precision')]:
                        if mask_rate.any():
                            np.testing.assert_allclose(np.average(values[mask_rate],weights=w[mask_rate]),r[stat])
                    if y.any() and (~y).any():
                        rank=score[y,None]-score[~y]
                        wp=w[y,None]*w[~y]
                        np.testing.assert_allclose((((rank>0)+.5*(rank==0))*wp).sum()/wp.sum(),r['auc'])
                    coin=np.mean([d[day==dt].mean(0)+(b[day==dt]-d[day==dt]).mean(0)*keep[day==dt].mean()
                                  for dt in np.unique(day)],axis=0)
                    np.testing.assert_allclose(coin-daily(a,day),r['matched_coin']['vs_base'],atol=1e-9)


def joint(delta):
    delta=np.asarray(delta)
    return bool(delta[0]<-1e-8 and np.all(delta[1:]<=1e-8))


def render(root,dest):
    data=json.loads((root/'summary.json').read_text())
    check(root,data)
    lines=['# e91 한 시간 뒤 NH4로 추가 폭기 이득 선별','',
           '순절감은 원래 기본 명령 A 빼기 해당 경로의 6시간 공기량이다. 수질은 해당 경로 빼기 A다. '
           '날짜 안에서 후보를 평균한 뒤 날짜와 seed를 동일 가중했다. 유지 횟수와 놓친 양성 건수는 원시 합계다.','']
    for key,title in [('supported','이전 실험의 거리 조건을 만족한 후보'),('all','전체 비중복 후보')]:
        groups=[r['groups'][key] for r in data]
        lines+=['## '+title,'','| 입력 | 평가 수 | 유지 수 | 양성 수 | 놓친 양성 | AUC | 재현율 | 정밀도 | 거짓양성률 | 선별 통과 |',
                '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for mode,name in [('level','농도'),('trend','농도와 상승 속도')]:
            rec=[g[mode] for g in groups]
            auc,recall,prec,fpr=np.mean([[r[k] for k in ('auc','recall','precision','fpr')] for r in rec],axis=0)
            passed=sum(r['auc']>=.6 and r['recall']>=.9 for r in rec)
            lines.append(f"| {name} | {sum(g['n'] for g in groups)} | {sum(r['outcome']['retained'] for r in rec):.0f} | "
                         f"{sum(r['positive'] for r in rec)} | {sum(r['missed'] for r in rec)} | "
                         f"{auc:.3f} | {recall:.1%} | {prec:.1%} | {fpr:.1%} | {passed}/5 |")
        lines+=['','| 1시간 이후 선택 | 순절감 m³/창 | NH4 위반 스텝 | DO 위반 스텝 | NH4 초과 mg h/L | A 대비 공동 개선 | B 대비 공동 개선 |',
                '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for method,name in [('always','추가 공기 유지 B'),('remove','추가 공기 전부 제거 D'),
                            ('level','농도 선별'),('level_coin','농도와 같은 횟수의 무작위 유지'),
                            ('trend','농도와 상승 속도 선별'),('trend_coin','농도와 상승 속도 방식과 같은 횟수의 무작위 유지'),
                            ('oracle','B 대비 공동 개선 여부를 미리 아는 선택')]:
            if method in ('always','remove','oracle'):
                rs=[g[method] for g in groups]
            else:
                rs=[g[method.split('_')[0]]['matched_coin' if method.endswith('_coin') else 'outcome'] for g in groups]
            base=np.array([r['vs_base'] for r in rs])
            air,nh,do,exc=base.mean(0)
            lines.append(f"| {name} | {-air:.2f} | {nh:+.4f} | {do:+.4f} | {exc:+.4f} | "
                         f"{sum(map(joint,base))}/5 | {sum(joint(r['vs_keep']) for r in rs)}/5 |")
        lines+=['','| seed | 입력 | 유지/전체 | 놓친/양성 | AUC | 재현율 | 거짓양성률 | A 대비 순절감 m³ | NH4 위반 변화 | DO 위반 변화 | 초과 적분 변화 |',
                '| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for row,g in zip(data,groups):
            for mode,name in [('level','농도'),('trend','농도와 상승 속도')]:
                m=g[mode]
                air,nh,do,exc=m['outcome']['vs_base']
                lines.append(f"| {row['seed']} | {name} | {m['outcome']['retained']:.0f}/{g['n']} | "
                             f"{m['missed']}/{m['positive']} | {m['auc']:.3f} | {m['recall']:.1%} | {m['fpr']:.1%} | "
                             f"{-air:.2f} | {nh:+.4f} | {do:+.4f} | {exc:+.4f} |")
        lines.append('')
    (dest/'results.md').write_text('\n'.join(lines).rstrip()+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--dest',type=Path,required=True)
    args=p.parse_args()
    render(args.root,args.dest)

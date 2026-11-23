"""Audit cumulative effects and produce tables with identical date and seed weights."""
import argparse
import json
from pathlib import Path

import numpy as np

HOURS=('15분','1시간','3시간','6시간')
STEPS=np.array([0,3,11,23])


def by_date(x,day):
    return np.mean([x[day==d].mean(axis=0) for d in np.unique(day)],axis=0)


def audit(root,records):
    for r in records:
        with np.load(root/f"paths{r['seed']}.npz") as z:
            keys=z['keys']
            a,b,c=z['cumulative'].transpose(1,0,2,3)
            baseline_change=b-a
            extra=b-c
            da,db,dc=[np.diff(path[:,:,0],prepend=0.,axis=1) for path in (a,b,c)]
            np.testing.assert_allclose(dc,np.minimum(da,db),atol=1e-8,rtol=0)
            np.testing.assert_allclose(da[:,0]-db[:,0],125.,atol=1e-8,rtol=0)
            np.testing.assert_array_equal(b[:,0],c[:,0])
            first=np.full(len(keys),25)
            for i,row in enumerate(baseline_change[:,:,0]):
                steps=np.flatnonzero(row>=-1e-8)
                if steps.size:
                    first[i]=steps[0]+1
            for g,ix in [('all',np.ones(len(keys),bool)),('supported',z['supported'])]:
                s=r['groups'][g]
                day=keys[ix,0]
                def verify(arr,ref):
                    np.testing.assert_allclose(by_date(arr[ix],day),ref,atol=1e-9,rtol=1e-12)
                verify(baseline_change[:,STEPS],s['cut_minus_base'])
                verify(first[:,None]<=STEPS+1,s['lost_ever'])
                lost=baseline_change[:,STEPS,0]>=-1e-8
                verify(lost,s['lost_now'])
                verify(lost!=(z['peer_delta'][:,STEPS,0]>=-1e-8),s['mismatch'])
                for label,mask in [('all',ix),('lost',ix&lost[:,-1])]:
                    a_stats=s[label]
                    assert mask.sum()==a_stats['n']
                    np.testing.assert_allclose(by_date(extra[mask][:,STEPS],keys[mask,0]),a_stats['extra'],atol=1e-9)


def render(root,dest):
    records=json.loads((root/'summary.json').read_text())
    audit(root,records)
    lines=['# e90 감량 뒤의 공기와 수질','',
           '누적 순절감은 기본 A 빼기 감량 B, 수질 변화는 B 빼기 A다. 추가 공기 효과는 B 빼기 추가 공기 제거 재생 C다. 날짜 안 평균 후 날짜와 seed에 각각 동일 가중했다.','']
    for group,title in [('supported','e89 지원 범위'),('all','전체 비중복 질의')]:
        rows=[r['groups'][group] for r in records]
        def avg(name):
            return np.mean([r[name] for r in rows],axis=0)
        lines.extend(['## '+title,'','| 시간 | 순절감 m³ | NH4 위반 스텝 | DO 위반 스텝 | NH4 초과 mg h/L | 현재 소진 | 한 번 이상 소진 | 과거 이웃과 절감 방향 불일치 |',
                      '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
        for i,(air,nh,do,exc) in enumerate(avg('cut_minus_base')):
            lines.append(f"| {HOURS[i]} | {-air:.2f} | {nh:+.4f} | {do:+.4f} | {exc:+.4f} | "
                         f"{avg('lost_now')[i]:.1%} | {avg('lost_ever')[i]:.1%} | {avg('mismatch')[i]:.1%} |")
        lines.extend(['','| 추가 공기 효과를 잰 집단 | 시간 | 추가 공기 m³ | NH4 위반 스텝 | DO 위반 스텝 | NH4 초과 mg h/L |',
                      '| --- | --- | ---: | ---: | ---: | ---: |'])
        for sub,name in [('all','모든 창'),('lost','6시간 순절감 소진 창')]:
            v=np.mean([r[sub]['extra'] for r in rows],axis=0)
            for i,(air,nh,do,exc) in enumerate(v):
                lines.append(f"| {name} | {HOURS[i]} | {air:.2f} | {nh:+.4f} | {do:+.4f} | {exc:+.4f} |")
        lines.extend(['','| seed | 후보 | 6시간 소진 건수 | 추가 공기 m³ | NH4 위반 변화 | DO 위반 변화 | NH4 초과 변화 | NH4 이득 / 악화 / 동률 / 상충 |',
                      '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |'])
        for r,s in zip(records,rows):
            row=s['all']
            air,nh,do,exc=row['extra'][-1]
            classes=' / '.join(f'{x:.1%}' for x in row['nh_classes'])
            lines.append(f"| {r['seed']} | {s['n']} | {s['lost']['n']} | {air:.2f} | {nh:+.4f} | {do:+.4f} | {exc:+.4f} | {classes} |")
        lines.append('')
    (dest/'results.md').write_text('\n'.join(lines).rstrip()+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--dest',type=Path,required=True)
    args=p.parse_args()
    render(args.root,args.dest)

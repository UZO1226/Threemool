"""Audit the original branch outcomes and export compact, seed-resolved bounds."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SCOPES={'pooled':'기간 평균 보호','date':'날짜마다 보호','window':'창마다 보호'}


def export(root,source,dest):
    records=json.loads((root/'certificates.json').read_text())
    compact=[]
    lines=['# e92 기본 FCSRL 수질 대비 사후 최대 절감','',
           '순절감은 기본 경로 A 빼기 선택 경로의 공기다. 날짜별 적격 창 평균을 낸 뒤 날짜와 seed를 동일 가중했다. '
           '최대 절감 단위는 m³/6시간 창이다. 불가능은 공기량을 얼마나 써도 저장된 경로로 세 수질을 동시에 지킬 수 없다는 뜻이다.','']
    for rec in records:
        p=source/f"hour{rec['seed']}.npz"
        assert hashlib.sha256(p.read_bytes()).hexdigest()==rec['sha256']
        clean=dict(seed=rec['seed'],sha256=rec['sha256'],groups={})
        with np.load(p) as z:
            for g,ix in [('supported',z['supported']),('all',np.ones(len(z['keys']),bool))]:
                r=rec['groups'][g]
                info={k:r[k] for k in ('n','days','baseline_air','e91_trend')}
                day=z['keys'][ix,0]
                totals=z['totals'][ix]
                w=1/np.bincount(day)[day]
                base=totals[:,0]
                for fam,modes in [('fixed',[1,2]),('optional',[0,1,2])]:
                    info[fam]={}
                    options=totals[:,modes]-base[:,None]
                    best_by_case=options[:,:,1:].min(1)
                    floor=np.average(best_by_case,weights=w,axis=0)
                    np.testing.assert_allclose(floor,r[fam]['pooled']['floors'],atol=1e-10)
                    admissible=(options[:,:,1:]<=0).all(2)
                    blocked=~admissible.any(1)
                    info[fam]['unavoidable_quality_loss_windows']=int(blocked.sum())
                    for scope in SCOPES:
                        report=r[fam][scope]
                        item={k:v for k,v in report.items() if k not in ('selected','failed_units','date_proofs')}
                        if report['status']=='optimal':
                            chosen=np.array(report['selected'])
                            assert np.isin(chosen,modes).all()
                            dx=totals[np.arange(len(day)),chosen]-base
                            measured=np.average(dx,weights=w,axis=0)
                            assert (measured[1:]<=1e-8).all()
                            np.testing.assert_allclose(measured,report['delta'],atol=1e-9)
                            np.testing.assert_allclose(measured[0],report['air_bound'],atol=1e-5,rtol=0)
                            if scope=='window':
                                opt=np.where(admissible,options[:,:,0],np.inf).min(1)
                                np.testing.assert_allclose(dx[:,0],opt,atol=1e-8)
                            if scope=='date':
                                assert all((dx[day==d,1:].mean(0)<=1e-8).all() for d in np.unique(day))
                        elif scope=='window':
                            np.testing.assert_array_equal(np.flatnonzero(blocked),report['failed_units'])
                        elif scope=='date':
                            assert len(report['failed_dates'])>0
                        elif report['method']=='single_metric_lower_bound':
                            assert max(floor)>1e-8
                        info[fam][scope]=item
                    bounds=[r[fam][scope].get('air_bound',np.inf) for scope in SCOPES]
                    assert all(x<=y+1e-5 for x,y in zip(bounds,bounds[1:]))
                for scope in SCOPES:
                    assert r['optional'][scope]['air_bound']<=r['fixed'][scope].get('air_bound',np.inf)+1e-5
                clean['groups'][g]=info
        compact.append(clean)
    for group,title in [('supported','거리 조건 안의 후보'),('all','전체 비중복 후보')]:
        lines+=['## '+title,'','| seed | 후보/날짜 | 첫 감량 필수, 기간 최대 절감 | NH4 위반 최소 차이 | DO 위반 최소 차이 | NH4 초과 최소 차이 | 수질 보호 불가 날짜 / 창 |',
                '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for row in compact:
            g=row['groups'][group]
            f=g['fixed']
            p=f['pooled']
            value='불가능' if p['status']=='infeasible' else f"{-p['delta'][0]:,.2f}"
            nh,do,exc=p['floors']
            dates=len(f['date'].get('failed_dates',[]))
            lines.append(f"| {row['seed']} | {g['n']}/{g['days']} | {value} | {nh:+.5f} | {do:+.5f} | {exc:+.5f} | "
                         f"{dates} / {f['unavoidable_quality_loss_windows']} |")
        lines+=['','각 최소 차이는 다른 지표의 제약을 풀고 한 지표만 최소화했다. 하나라도 양수면 어떤 조합도 해당 수질을 보호할 수 없다. '
                '불가 날짜와 창은 매 날짜 평균 또는 개별 창에서 세 지표를 동시에 지키는 선택이 없는 수다.','',
                '| seed | 첫 감량 생략 가능 시 기간 최대 절감 | 해당 해에서 수질 악화 날짜 수 | 매 날짜 보호 최대 | 각 창 보호 최대 | 각 창 보호 절감률 |',
                '| ---: | ---: | ---: | ---: | ---: | ---: |']
        for row in compact:
            g=row['groups'][group]['optional']
            lines.append(f"| {row['seed']} | {-g['pooled']['delta'][0]:,.2f} | {g['pooled']['worse_days']} | "
                         f"{-g['date']['delta'][0]:,.2f} | {-g['window']['delta'][0]:,.2f} | {g['window']['savings_pct']:.2f}% |")
        lines+=['','| 첫 감량 생략 가능 | 평균 최대 절감 | 평균 절감률 | NH4 위반 차이 | DO 위반 차이 | NH4 초과 적분 차이 | 선택 수 A/B/D |',
                '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for scope,name in SCOPES.items():
            stats=[r['groups'][group]['optional'][scope] for r in compact]
            air,nh,do,exc=np.mean([s['delta'] for s in stats],0)
            counts='/'.join(str(i) for i in np.sum([s['counts'] for s in stats],0))
            lines.append(f"| {name} | {-air:,.2f} | {np.mean([s['savings_pct'] for s in stats]):.2f}% | "
                         f"{nh:+.5f} | {do:+.5f} | {exc:+.5f} | {counts} |")
        lines.append('')
    (dest/'results.md').write_text('\n'.join(lines).rstrip()+'\n')
    (dest/'summary.json').write_text(json.dumps(compact,indent=1,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--source',type=Path,default=Path('/home/jio/fcsrl_src/results/e91/main'))
    p.add_argument('--dest',type=Path,required=True)
    args=p.parse_args()
    export(args.root,args.source,args.dest)

"""e53 고정 여유와 e52 불확실성 여유를 원 제약, 심각도, 최악 날짜로 비교한다."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
KEYS=['air_m3','nh4_cost','do_cost','nh4_excess_mg_h_L','do_excess_mg_h_L','worst_day_nh4_cost','worst_day_do_cost']
LABELS={'k0':'여유 없음','k10':'불확실성 여유','d045':'고정 여유 0.45'}


def read(p):return json.loads(p.read_text())


def decision(fixed,adaptive):
    dn=np.array([y['nh4_cost']-x['nh4_cost'] for x,y in zip(fixed,adaptive)])
    da=np.array([100*(y['air_m3']/x['air_m3']-1) for x,y in zip(fixed,adaptive)])
    ratio=float(100*(np.mean([y['air_m3'] for y in adaptive])/np.mean([x['air_m3'] for x in fixed])-1))
    nh4_route=bool((dn<0).sum()>=2 and dn.mean()<0 and ratio<=2)
    air_route=bool((da<0).sum()>=2 and ratio<=-2 and dn.mean()<=0)
    old_nh4=bool((dn<0).sum()>=2 and dn.mean()<0 and da.mean()<=2)
    old_air=bool((da<0).sum()>=2 and da.mean()<=-2 and dn.mean()<=0)
    return dict(mean_air_change_pct=ratio,mean_paired_air_change_pct=float(da.mean()),
                nh4_route=nh4_route,air_route=air_route,H1=nh4_route or air_route,
                original_report_H1=old_nh4 or old_air,original_report_nh4_route=old_nh4,original_report_air_route=old_air,
                d_nh4=dn.tolist(),d_air_pct=da.tolist(),
                interpretation='탐색 기준 지지' if nh4_route or air_route else '추가 기여 미확인, 고정 여유와의 동등성 입증은 아님')


def main(a):
    a.out.mkdir(parents=True,exist_ok=True)
    assert read(a.e53/'status.json')['phase']=='complete'
    calibration=read(a.calibration/'calibration.json')
    for split,z in calibration['splits'].items():
        with np.load(a.calibration/f'{split}_recorded_replay.npz') as raw:
            base=raw['nh4']>4
            fixed=(raw['nh4']+.45>4)&~base
            adaptive=(raw['nh4']+10*raw['sd']>4)&~base
            z['additional_alarm_intersection']=int((fixed&adaptive).sum())
            z['additional_alarm_union']=int((fixed|adaptive).sum())
            z['additional_alarm_jaccard']=float((fixed&adaptive).sum()/max(1,(fixed|adaptive).sum()))
    rows=[];episodes={}; original_checks=0
    for m in ['ppo','fcsrl']:
        for c in ['k0','k10','d045']:
            for s in range(3):
                root=a.e53 if c=='d045' else a.e52
                path=root/f'{m}_{c}_s{s}'
                r=read(path/'metrics.json');summary=r['summary']
                if c=='d045':original_checks+=len(r['original_metrics_reproduced'])
                rows.append(dict(method=m,condition=c,seed=s,**{k:summary[k] for k in KEYS},
                                 both_limits=bool(summary['nh4_cost']<=6 and summary['do_cost']<=6)))
                episodes[(m,c,s)]=pd.DataFrame(read(path/'episodes.json'))
    frame=pd.DataFrame(rows);frame.to_csv(a.out/'per_seed.csv',index=False)
    output=dict(calibration=calibration,rows=rows,comparisons={},weight_updates=0,test_evaluated=False,
                original_e53_metric_checks=original_checks,analysis='posthoc validation; no selection of delta or kappa')
    date_diffs=[]
    md=['# e53 고정 여유 대조 상세 검산','',
        '공기량 m3/23.75h, NH4와 DO 비용은 95전이 중 위반 횟수, 초과 적분 mg h/L. '
        '같은 seed, 검증 날짜 40개, 대리 모델 구성원 5개, 결정적 최종 정책을 비교한다. '
        '원래 한도는 NH4와 DO 각각 6이며 평가에 안전 여유를 더하지 않는다. 새 학습과 test 평가 없음.', '',
        '표는 seed 평균과 표준편차(ddof=0)다. 주 비교의 차이는 불확실성 여유(kappa 10)에서 고정 여유(delta 0.45)를 뺀 값이다. '
        '집단 평균 공기량 변화와 seed별 변화율 평균을 구분한다. 원 H1과 수정된 집계 H1을 모두 보존한다. '
        'NH4 심각도와 DO 추가 분석은 사후 해석으로 구분한다.', '']
    for m in ['ppo','fcsrl']:
        subset=frame[frame.method==m]
        groups={c:subset[subset.condition==c].sort_values('seed').to_dict('records') for c in LABELS}
        means={c:{k:float(np.mean([r[k] for r in rs])) for k in KEYS} for c,rs in groups.items()}
        sds={c:{k:float(np.std([r[k] for r in rs],ddof=0)) for k in KEYS} for c,rs in groups.items()}
        v=decision(groups['d045'],groups['k10'])
        paired=[]
        for s in range(3):
            f=groups['d045'][s];u=groups['k10'][s]
            d={k:u[k]-f[k] for k in KEYS}
            paired.append(dict(seed=s,**d))
            ar=episodes[(m,'d045',s)];br=episodes[(m,'k10',s)]
            assert ar[['day','member']].equals(br[['day','member']])
            ad=ar.groupby('day').mean(numeric_only=True);bd=br.groupby('day').mean(numeric_only=True)
            for day in ad.index:
                date_diffs.append(dict(method=m,seed=s,day=day,
                    **{k:float(bd.loc[day,k]-ad.loc[day,k]) for k in KEYS if not k.startswith('worst')}))
        h2=[]
        for s in range(3):
            source=Path('/home/jio/fcsrl_src/results')
            f=read(source/f'e53/main/runs/{m}/d0.45/seed_{s}/metrics.json')
            u=read(source/f'e52/main/runs/{m}/k10/seed_{s}/metrics.json')
            h2.append((u['val_final']['nh4_cost']-u['train_final']['nh4_cost'])-(f['val_final']['nh4_cost']-f['train_final']['nh4_cost']))
        v['H2_gap_smaller']=bool(np.mean(h2)<0);v['gap_differences']=h2
        v['nh4_severity_worse_seeds']=sum(p['nh4_excess_mg_h_L']>0 for p in paired)
        v['worst_nh4_worse_seeds']=sum(p['worst_day_nh4_cost']>0 for p in paired)
        v['all_three_means_lower']=all(means['k10'][k]<means['d045'][k] for k in KEYS[:3])
        output['comparisons'][m]=dict(means=means,sds=sds,paired=paired,verdict=v)
        md += [f'## {m.upper()}','', '| 조건 | 공기량 평균 | NH4 평균 | DO 평균 | NH4 초과 적분 | DO 초과 적분 | 최악 날짜 NH4 평균 | 최악 날짜 DO 평균 | 두 평균 한도 충족 seed |',
               '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for c,z in means.items():
            md.append('| '+LABELS[c]+' | '+' | '.join(f'{z[k]:.3f} ± {sds[c][k]:.3f}' for k in KEYS)+f" | {sum(r['both_limits'] for r in groups[c])}/3 |")
        md += ['', '| seed | 공기량 차이 % | NH4 차이 | DO 차이 | NH4 초과 적분 차이 | DO 초과 적분 차이 | 최악 날짜 NH4 차이 | 최악 날짜 DO 차이 |',
               '| --- | --- | --- | --- | --- | --- | --- | --- |']
        for s,d in enumerate(paired):
            md.append(f"| {s} | {v['d_air_pct'][s]:+.3f} | "+' | '.join(f'{d[k]:+.3f}' for k in KEYS[1:])+' |')
        md += ['', f"집단 평균 공기량 변화 {v['mean_air_change_pct']:+.3f}%. seed별 변화율 평균 {v['mean_paired_air_change_pct']:+.3f}%.",
               f"H1 수정 집계: {v['H1']}. 원 집계: {v['original_report_H1']}. H2 격차 감소: {v['H2_gap_smaller']}.",
               f"불확실성 여유가 고정 여유보다 NH4 초과 적분이 큰 seed {v['nh4_severity_worse_seeds']}/3, 최악 날짜 NH4가 큰 seed {v['worst_nh4_worse_seeds']}/3.",
               'H1 지지는 제한된 탐색 기준의 통과이며 실측 안전 보장이나 전체 지표 우월성을 뜻하지 않는다. '
               'H1 미지지는 두 방법의 동등성이나 고정 여유만으로 효과가 완전히 설명됨을 뜻하지 않는다.', '']
    pd.DataFrame(date_diffs).to_csv(a.out/'paired_dates.csv',index=False)
    md += ['## 여유 선정 재검산','', '| 분할 | 실측 NH4 비용 | 여유 없음 재생 | 고정 여유 재생 | 불확실성 여유 재생 | 두 여유의 위반 위치 불일치 % |',
           '| --- | --- | --- | --- | --- | --- |']
    for split,z in calibration['splits'].items():
        md.append(f"| {split} | {z['actual_cost']:.4f} | {z['plain_cost']:.4f} | {z['fixed_cost']:.4f} | {z['adaptive_cost']:.4f} | {z['fixed_vs_adaptive_event_disagreement_pct']:.4f} |")
    md += ['', '기록 운전의 평균 비용을 맞춘 것이며 학습 중 방문 상태나 실제 정책의 공기량을 맞춘 대조는 아니다. '
           '원 e53의 delta를 변경하지 않았다. 검증 날짜는 진단에만 사용했다. 구성원 5개는 독립된 실제 공정 데이터셋이 아니다.', '']
    (a.out/'results.md').write_text('\n'.join(md))
    (a.out/'comparison.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(output['comparisons'],ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--e53',type=Path,required=True);p.add_argument('--e52',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True);p.add_argument('--out',type=Path,default=HERE)
    main(p.parse_args())

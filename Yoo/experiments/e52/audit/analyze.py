"""고정 정책의 날짜, seed, 시간대별 사후 분석. 새 후보 선택용 검증 점수 최적화 없음."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
KEYS=['air_m3','nh4_cost','do_cost','cost','nh4_excess_mg_h_L','do_excess_mg_h_L','worst_day_nh4_cost','worst_day_do_cost']


def read(path):
    return json.loads(path.read_text())


def save(path,x):
    path.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def load(root,exp,ident):
    path=root/exp/ident
    with np.load(path/'rollout.npz') as z:
        arr={k:z[k] for k in z.files}
    return read(path/'metrics.json'),pd.DataFrame(read(path/'episodes.json')),arr


def compare(root,exp,method,seeds,base_name,candidate_name):
    paired=[]; date_rows=[]; pre=[]; sd=[]; extra=[]; step_diff=[]; hour_rows=[]
    all_a=[];all_b=[]; episodes_verified=0
    for s in seeds:
        a,ar,aa=load(root,exp,base_name(s));b,br,ba=load(root,exp,candidate_name(s))
        assert ar[['day','member']].equals(br[['day','member']])
        assert len(ar)==200 and ar.day.nunique()==40
        episodes_verified+=len(ar)
        av,bv=a['summary'],b['summary'];all_a.append(av);all_b.append(bv)
        d={k:bv[k]-av[k] for k in KEYS}
        paired.append(dict(seed=s,baseline=av,candidate=bv,difference=d,
                           air_pct=100*(bv['air_m3']/av['air_m3']-1)))
        ad=ar.groupby('day').mean(numeric_only=True);bd=br.groupby('day').mean(numeric_only=True)
        for day in ad.index:
            row=dict(seed=s,day=day)
            for k in ['air_m3','nh4_cost','do_cost','nh4_excess_mg_h_L','do_excess_mg_h_L']:
                row['d_'+k]=float(bd.loc[day,k]-ad.loc[day,k])
            row['d_cost']=row['d_nh4_cost']+row['d_do_cost']
            date_rows.append(row)
        if exp=='e52':
            pre.append(aa['pre_states'][:,:,0]);sd.append(aa['sd'])
            extra.append(np.maximum(ba['air']-aa['air'],0)*.25)
            step_diff.append((ba['air']-aa['air'])*.25)
            for start in [0,24,48,72]:
                sl=slice(start,min(start+24,95))
                hour_rows.append(dict(seed=s,start_hour=start*.25,end_hour=min(start+24,95)*.25,
                    d_air_m3=float(((ba['air'][:,sl]-aa['air'][:,sl])*.25).sum(1).mean()),
                    d_nh4_cost=float(((ba['states'][:,sl,0]>4).sum(1)-(aa['states'][:,sl,0]>4).sum(1)).mean())))
    frame=pd.DataFrame(date_rows)
    perdate=frame.groupby('day').mean(numeric_only=True)
    group={k:dict(baseline=float(np.mean([x[k] for x in all_a])),candidate=float(np.mean([x[k] for x in all_b]))) for k in KEYS}
    group['air_pct']=100*(group['air_m3']['candidate']/group['air_m3']['baseline']-1)
    costs=np.array([p['difference']['cost'] for p in paired])
    out=dict(experiment=exp,method=method,paired=paired,means=group,
             independent_training_seeds=len(seeds),validation_dates=40,ensemble_members_are_not_independent_datasets=True,
             episodes_aligned=episodes_verified,
             cost_better_seeds=int((costs<0).sum()),leave_one_seed_out_cost_difference=[float(np.delete(costs,i).mean()) for i in range(len(costs))],
             seed_day_counts={k:int((frame['d_'+k]<0).sum()) for k in ['air_m3','cost','nh4_cost','do_cost']},
             seed_day_total=len(frame),date_mean_counts={k:int((perdate['d_'+k]<0).sum()) for k in ['air_m3','cost','nh4_cost','do_cost']},
             all_three_better_seed_days=int(((frame.d_air_m3<0)&(frame.d_nh4_cost<0)&(frame.d_do_cost<0)).sum()),
             all_three_better_dates=int(((perdate.d_air_m3<0)&(perdate.d_nh4_cost<0)&(perdate.d_do_cost<0)).sum()))
    if pre:
        pre=np.stack(pre);sd=np.stack(sd);extra=np.stack(extra);net=np.stack(step_diff)
        total=float(extra.sum()); denom=len(seeds)*200
        risk={}
        for name,mask in [('NH4_le_3',pre<=3),('NH4_3_to_4',(pre>3)&(pre<=4)),('NH4_gt_4',pre>4)]:
            risk[name]=dict(step_pct=float(mask.mean()*100),positive_extra_air_share_pct=float(extra[mask].sum()/total*100),
                            net_extra_m3_per_episode=float(net[mask].sum()/denom))
        cut=np.quantile(sd,[.25,.5,.75])
        out['baseline_pre_nh4_partition']=risk
        out['baseline_sd_quartiles']=dict(cutoffs_mg_L=cut.tolist(),descriptive_validation_quantiles_only=True,
            groups=[dict(quartile=i+1,step_pct=float((np.digitize(sd,cut)==i).mean()*100),
                         positive_extra_air_share_pct=float(extra[np.digitize(sd,cut)==i].sum()/total*100)) for i in range(4)])
        out['air_accounting']=dict(net_extra_m3_per_episode=float(net.sum()/denom),
                                   positive_extra_m3_per_episode=float(extra.sum()/denom),
                                   removed_m3_per_episode=float(np.maximum(-net,0).sum()/denom))
        assert np.isclose(out['air_accounting']['net_extra_m3_per_episode'],group['air_m3']['candidate']-group['air_m3']['baseline'])
        out['time_bins']=hour_rows
    return out,frame,perdate


def main(a):
    a.out.mkdir(parents=True,exist_ok=True)
    for exp in ['e50','e52']:
        assert read(a.root/exp/'status.json')['phase']=='complete'
    results={};checks=0;max_errors={}
    for exp,method,seeds,base,candidate in [
        ('e50','actor',range(5),lambda s:f'fcsrl_s{s}',lambda s:f'fcsrl_actor_s{s}'),
        ('e52','fcsrl',range(3),lambda s:f'fcsrl_k0_s{s}',lambda s:f'fcsrl_k10_s{s}'),
        ('e52','ppo',range(3),lambda s:f'ppo_k0_s{s}',lambda s:f'ppo_k10_s{s}')]:
        ident=f'{exp}_{method}'
        results[ident],frame,perdate=compare(a.root,exp,method,seeds,base,candidate)
        frame.to_csv(a.out/f'{ident}_seed_dates.csv',index=False)
        perdate.to_csv(a.out/f'{ident}_dates.csv')
    for exp in ['e50','e52']:
        for p in (a.root/exp).glob('*/metrics.json'):
            item=read(p)
            checks+=len(item['original_metrics_reproduced'])
            old=read(Path(item['job']['path'])/'metrics.json')['val_final']
            for key in item['original_metrics_reproduced']:
                name=exp+':'+key
                max_errors[name]=max(max_errors.get(name,0.),abs(float(item['summary'][key])-float(old[key])))
    results['verification']=dict(policy_evaluations=27,original_metric_checks=checks,max_metric_absolute_errors=max_errors,
                                 analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),weight_updates=0,test_evaluated=False,
                                 inputs_unchanged=all(read(a.root/e/'status.json')['hashes_unchanged'] for e in ['e50','e52']))
    logs=[]
    for s in range(5):
        log=read(Path('/home/jio/fcsrl_src/results/e50/main/runs/fcsrl_actor')/f'seed_{s}/train_log.json')
        logs.append(dict(seed=s,before_kl=float(np.mean([x['preserve_before_kl'] for x in log])),
                         after_kl=float(np.mean([x['preserve_after_kl'] for x in log])),
                         extra_wall_seconds=float(sum(x['extra_wall_seconds'] for x in log))))
    results['e50_actor_training_diagnostics']=logs
    save(a.out/'analysis.json',results)
    md=['# e50, e52 고정 정책 사후 분석','',
        '새 학습 없이 최종 정책 27개를 각 실험의 원래 Python 환경에서 재평가했다. test는 평가하지 않았다. '
        '공기량은 m3/23.75h, 비용은 95개 전이의 위반 횟수다. 날짜별 수치는 같은 날짜의 구성원 5개를 평균한다.', '',
        '평균 공기량 변화는 평균의 비율로 수정했다. 두 실험의 사전 가설 판정은 수정 전후 동일하다. '
        '이 분석은 기존 결과를 본 뒤 수행한 탐색이며 독립 확인 실험이 아니다.', '']
    for ident,v in results.items():
        if not isinstance(v,dict) or 'paired' not in v:continue
        md += [f'## {ident}', '', '| seed | 공기량 변화 % | NH4 차이 | DO 차이 | 합산 비용 차이 | NH4 초과 적분 차이 | DO 초과 적분 차이 | 최악 날짜 NH4 차이 | 최악 날짜 DO 차이 |',
               '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for p in v['paired']:
            d=p['difference']
            md.append(f"| {p['seed']} | {p['air_pct']:+.2f} | {d['nh4_cost']:+.3f} | {d['do_cost']:+.3f} | {d['cost']:+.3f} | {d['nh4_excess_mg_h_L']:+.3f} | {d['do_excess_mg_h_L']:+.3f} | {d['worst_day_nh4_cost']:+.2f} | {d['worst_day_do_cost']:+.2f} |")
        md += ['', '| 지표 | 기준 seed 평균 | 후보 seed 평균 |', '| --- | --- | --- |']
        for metric in KEYS:
            z=v['means'][metric]
            md.append(f"| {metric} | {z['baseline']:.4f} | {z['candidate']:.4f} |")
        md += ['',f"평균 공기량 변화 {v['means']['air_pct']:+.2f}%. 합산 비용 개선 seed {v['cost_better_seeds']}/{v['independent_training_seeds']}.",
               f"seed 하나씩 제외한 평균 합산 비용 차이: {', '.join(f'{x:+.3f}' for x in v['leave_one_seed_out_cost_difference'])}.",
               f"seed 평균 후 날짜 40개 중 합산 비용 개선 {v['date_mean_counts']['cost']}개, 공기량 개선 {v['date_mean_counts']['air_m3']}개, 세 지표 모두 개선 {v['all_three_better_dates']}개.", '']
        if 'baseline_pre_nh4_partition' in v:
            md += ['| 기준 정책의 행동 전 NH4 | 시간 비중 % | 양의 추가 공기량 비중 % | 순 추가 공기량 m3/episode |','| --- | --- | --- | --- |']
            for k,z in v['baseline_pre_nh4_partition'].items():
                md.append(f"| {k} | {z['step_pct']:.2f} | {z['positive_extra_air_share_pct']:.2f} | {z['net_extra_m3_per_episode']:+.1f} |")
            md += ['', '두 정책은 이후 상태가 달라진다. 같은 날짜, 구성원, 시각의 차이를 분해한 기술 통계다. '
                   '현재 NH4가 낮아도 미래 위반 예방에 공기가 필요할 수 있으므로 낭비나 인과 효과로 해석하지 않는다. '
                   '모델 간 SD는 실측 오차가 아니며 불확실성 보정의 타당성 검증을 대체하지 않는다.', '']
    md += ['## 그림', '', '![Seed별 공기량과 NH4 비용 변화](paired_changes.png)', '', '![e52 시간별 공기량과 NH4 위반](e52_time_profiles.png)', '', '## 재현 검증','',json.dumps(results['verification'],ensure_ascii=False), '',
           '원본 체크포인트, 입력 데이터, 원 평가 결과 파일은 수정하지 않았다. 전체 시계열, 파일 해시, 재평가 상태는 '
           f'`{a.root}`에 있다. 원래 결과와 겹치는 지표는 rtol=1e-9, atol=1e-7로 검산했다. e50 공기량만 합산값 atol=0.001 m3를 허용했으며 위반 비용 허용오차는 완화하지 않았다.', '']
    (a.out/'results.md').write_text('\n'.join(md))
    print(json.dumps({k:{key:v[key] for key in ['means','cost_better_seeds','leave_one_seed_out_cost_difference','date_mean_counts']} for k,v in results.items() if isinstance(v,dict) and 'paired' in v},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,default=HERE)
    main(p.parse_args())

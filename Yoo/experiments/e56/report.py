"""e56 최종 결과를 e50 고정 기준선과 seed별 대응 비교한다."""
import argparse
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
KEYS=['air_m3','nh4_cost','do_cost','cost','nh4_excess_mg_h_L','do_excess_mg_h_L','worst_day_nh4_cost','worst_day_do_cost']


def report(root):
    protocol=json.loads((HERE/'protocol.json').read_text())
    results={};table=['# e56 모방 prior 사용 여부 대조','',
        'e50 환경(scikit-learn 1.9.1), 관측11, 합산 비용 한도12, 401,280전이, 마지막 정책. 개발 val40일 x 구성원5, seed5개. 실제 현장 제어 또는 독립 test 결과가 아니다.','',
        '| 방법 | prior | 공기량 m³/23.75h | NH4 비용 | DO 비용 | 합산 비용 | 합산 평균 한도 충족 seed | NH4 초과 적분 | 최악 날짜 NH4 |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    paired=[]
    for method in protocol['methods']:
        on=[];off=[]
        for seed in protocol['seeds']:
            b=json.loads((Path(protocol['baseline_audit'])/f'{method}_s{seed}'/'metrics.json').read_text())['summary']
            item=json.loads((root/'runs'/method/f'seed_{seed}'/'metrics.json').read_text())
            assert item['steps']==protocol['steps'] and item['prior_mode']=='off'
            a=item['val_final'];on.append(b);off.append(a)
            paired.append(dict(method=method,seed=seed,on={k:b[k] for k in KEYS},off={k:a[k] for k in KEYS},
                               differences={k:a[k]-b[k] for k in KEYS}))
        avg=lambda rs:{k:statistics.mean(x[k] for x in rs) for k in KEYS}
        b,a=avg(on),avg(off)
        nb=sum(x['cost']<=12 for x in on);na=sum(x['cost']<=12 for x in off)
        tolerances={'air_m3':.001,'nh4_cost':1e-9,'do_cost':1e-9}
        dominated=sum(all(x[k]<=y[k]+tolerances[k] for k in tolerances) for x,y in zip(off,on))
        mean_improved=all(a[k]<b[k] for k in tolerances)
        tails_ok=all(a[k]<=b[k]+(1e-8 if 'excess' in k else 1e-9) for k in KEYS[4:])
        passed=mean_improved and dominated>=4 and tails_ok and na>=nb
        results[method]=dict(on=b,off=a,air_mean_ratio_change_pct=100*(a['air_m3']/b['air_m3']-1),
            three_means_lower=mean_improved,nonworsening_primary_seeds=dominated,tail_means_nonworsening=tails_ok,
            feasible_seeds_on=nb,feasible_seeds_off=na,exploratory_joint_screen=passed,
            off_sd={k:statistics.stdev(x[k] for x in off) for k in KEYS},
            on_sd={k:statistics.stdev(x[k] for x in on) for k in KEYS})
        for label,z,count in [('on',b,nb),('off',a,na)]:
            table.append(f"| {method} | {label} | {z['air_m3']:,.1f} | {z['nh4_cost']:.3f} | {z['do_cost']:.3f} | {z['cost']:.3f} | {count}/5 | {z['nh4_excess_mg_h_L']:.3f} | {z['worst_day_nh4_cost']:.2f} |")
    table+=['','## 사전에 고정한 탐색 판정','']
    for method,z in results.items():
        table.append(f"- {method}: 세 평균 감소 {z['three_means_lower']}, 세 지표 비악화 seed {z['nonworsening_primary_seeds']}/5, tail 평균 비악화 {z['tail_means_nonworsening']}, 탐색 선별 {z['exploratory_joint_screen']}. 평균 공기량 변화 {z['air_mean_ratio_change_pct']:+.2f}%.")
    table+=['','## seed별 원 수치','', '| 방법 | seed | prior | 공기량 | NH4 비용 | DO 비용 | NH4 초과 적분 | 최악 날짜 NH4 |', '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for row in paired:
        for mode in ['on','off']:
            z=row[mode];table.append(f"| {row['method']} | {row['seed']} | {mode} | {z['air_m3']:.1f} | {z['nh4_cost']:.3f} | {z['do_cost']:.3f} | {z['nh4_excess_mg_h_L']:.3f} | {z['worst_day_nh4_cost']:.2f} |")
    table+=['','초기화뿐 아니라 학습 내내 더하는 prior를 바꾼 비교다. 환경 오류를 해결한 증거가 아니며 FCSRL의 원 표현 학습은 유지했다.',
            '개발 val을 반복 사용한 탐색이다. 선별 통과만으로 공식 채택하거나 통계적 확증을 주장하지 않는다. 선별 실패를 동등성 증거로 쓰지 않는다. 평균 최악 날짜는 seed마다 가장 나쁜 날짜를 구한 뒤 seed 평균을 낸 값이다.','']
    (root/'summary.json').write_text(json.dumps(dict(protocol=protocol,methods=results,paired=paired),indent=2,ensure_ascii=False)+'\n')
    (root/'results.md').write_text('\n'.join(table))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    report(p.parse_args().root)

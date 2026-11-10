"""NH4 이력 모델의 단기/반복/분류 성능을 분리해 보고한다."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(out):
    c=json.loads((out/'comparison.json').read_text());r=c['results']
    m=json.loads((out/'manifest.json').read_text())
    md=['# e38 NH4 과거 수질 이력 대조', '',
        f"실행 커밋 `{m['code_commit']}`, 소요 {m['elapsed_seconds']:.2f}초. NH4 모델5개 재학습, RL 학습/test 평가 없음.",
        'base=e35 수정 환경, history=같은 NH4 GBM에 과거15/30/45/60분 수질 입력 추가. 다른 상태 모델 가중치는 동일하다.', '',
        '## 예측 오차', '',
        '| 평가 | 센서 | base MAE | history MAE | history 빼기 base 95%구간 |', '| --- | --- | --- | --- | --- |']
    for mode in ('one_step_ensemble_mean','cold','warm'):
        for target in ('nh4','no3','do1','do2','do3'):
            ci=c[mode+'_paired_uncertainty'][target]['paired_7day_block_95ci'] if mode in ('cold','warm') else None
            cs=f'[{ci[0]:+.4f}, {ci[1]:+.4f}]' if ci is not None else '별도 구간 없음'
            md.append(f"| {mode} | {target} | {r['base'][mode]['mae'][target]:.4f} | {r['history'][mode]['mae'][target]:.4f} | {cs} |")
    md += ['', 'MAE 단위mg/L. cold는00시부터95전이, warm은첫1시간 관측 후91전이다. 서로 다른 길이끼리 우열 비교하지 않는다.',
           '1스텝은5개 모델 평균 예측, 반복 예측은모델별오차의평균이다. 구간은날짜7일순환블록2000회 탐색구간으로 다중비교 보정이나 정책seed검정이 아니다.', '',
           '## 위반 분류', '',
           '| 평가 | 대상 | 모델 | recall % | FPR % | 예측 평균 초과 수 | 실제 평균 초과 수 |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for mode in ('cold','warm'):
        for target in ('NH4','DO'):
            for name in ('base','history'):
                q=r[name][mode];cf=q['confusion'][target];counts=q['episode_mean_counts'][target]
                md.append(f"| {mode} | {target} | {name} | {cf['recall_pct']:.3f} | {cf['fpr_pct']:.3f} | {counts['predicted']:.3f} | {counts['observed']:.3f} |")
    md += ['', '예측 위반 감소가 좋은 제어를 뜻하지 않는다. 동일한 기록 행동에 대한 실측과 예측의 분류 일치 문제다.', '',
           '## 공기 변화 구간 NH4 반응', '',
           '| 방향 | 분 후 | 실제 평균 변화 | base 변화 | history 변화 | base MAE | history MAE |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for group in ('increase','decrease'):
        for minutes in (15,60,180):
            def pick(name):
                return next(x for x in r[name]['events'][group]['metrics'] if x['target']=='nh4' and x['minutes']==minutes)
            b,h=pick('base'),pick('history')
            md.append(f"| {group} | {minutes} | {b['observed_mean_delta']:+.4f} | {b['predicted_mean_delta']:+.4f} | {h['predicted_mean_delta']:+.4f} | {b['mae']:.4f} | {h['mae']:.4f} |")
    md += ['', 'e37의132개 구간을 그대로 썼다. 기록 행동/외생 입력의 조건부 재생이며 공기량의 인과 효과가 아니다.', '',
           '## 사전 선별 기준', '', '| 조건 | 통과 |', '| --- | --- |']
    for key,value in c['screen'].items():
        md.append(f'| {key} | {value} |')
    md += ['', f"전체 선별 통과: {c['all_screen_checks_passed']}. 허용폭0의 보수적 점추정 선별이며 실용적 무가치/동등성의 검정이 아니다.",
           '각 지표의 상충을 공개하며 결과를 보고 기준을 완화하지 않는다. 통과하더라도 실제 제어 성과나 현장 안전성 인증이 아니다.',
           '공기량은 기록 운전으로 고정했으므로 이 실험에서 공기 절감이나 공기/NH4/DO 동시 제어 개선을 달성했다고 주장하지 않는다.', '']
    if (out/'ranking.json').exists():
        ranking=json.loads((out/'ranking.json').read_text())
        md += ['', '## 사후 위험 순위 진단', '',
               '아래 판정점은 검증 정답을 보고 선택한 사후 기술적 비교다. 원래 선별 실패를 뒤집거나 임계값을 배포하지 않는다.',
               '| 모델 | AUC | Average precision | 탐지율61.82%의 FPR % | FPR10%의 탐지율 % |',
               '| --- | --- | --- | --- | --- |']
        for name in ('base','history'):
            q=ranking['models'][name]
            md.append(f"| {name} | {q['auc']:.4f} | {q['average_precision']:.4f} | {q['matched_recall'][0]['fpr']*100:.3f} | {q['matched_fpr'][0]['recall']*100:.3f} |")
        ci=ranking['history_minus_base_auc_paired_7day_block_95ci']
        md += ['', f'이력 빼기 기존 AUC 차이의 날짜7일 블록 탐색95%구간: [{ci[0]:.4f}, {ci[1]:.4f}].',
               '위험 순위 정보의 개선은 남아 있지만 예측 농도4를 그대로 적용했을 때의 위험 누락도 남아 있다.',
               '다음 후보는 val 정답으로 판정점을 정하는 대신 train 내부의 모델 학습과 분리한 날짜에서 위험 판정을 보정하는 대조다. 단순 농도 오프셋을 반복 전이에 적용하는 것은 별도 변경이며 이번에 실행하지 않았다.',
               '실제 위반 기준NH4>4는 유지한다. 보정만으로 공기 감소 뒤 수질 상승 누락이나 DO 위반 탐지가 해결된다고 보장하지 않는다.', '']
    (out/'results.md').write_text('\n'.join(md))
    z=np.load(out/'predictions.npz');fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,(group,sign) in zip(axes[0],[('Increase',1),('Decrease',-1)]):
        mask=z['event_dq']*sign>0;actual=z['actual_events'][mask,:,0]
        ax.plot(np.arange(13)*15,(actual-actual[:,:1]).mean(0),color='black',label='Observed')
        for key,color in [('base','tab:blue'),('history','tab:orange')]:
            x=z[key+'_events'][mask,:,:,0].mean(1)
            ax.plot(np.arange(13)*15,(x-x[:,:1]).mean(0),color=color,label=key)
        ax.set(title=f'NH4 after airflow {group.lower()}',xlabel='Minutes',ylabel='Change (mg/L)')
        ax.grid(alpha=.2);ax.legend()
    actual=(z['actual_cold'][:,:,0]>4).sum(1)
    for key,color in [('base','tab:blue'),('history','tab:orange')]:
        pred=(z[key+'_cold'][:,:,:,0]>4).sum(2).mean(1)
        axes[1,0].scatter(actual,pred,color=color,label=key,alpha=.7)
    axes[1,0].plot([0,95],[0,95],':',color='gray');axes[1,0].set(xlabel='Observed NH4 exceedance count',ylabel='Predicted count (member mean)',title='Full-day NH4 violations');axes[1,0].legend()
    keys=['one_step_ensemble_mean','cold','warm'];xx=np.arange(3)
    for shift,key,color in [(-.18,'base','tab:blue'),(.18,'history','tab:orange')]:
        axes[1,1].bar(xx+shift,[r[key][v]['mae']['nh4'] for v in keys],width=.36,label=key,color=color)
    axes[1,1].set_xticks(xx,['One step','Cold 95 steps','Warm 91 steps']);axes[1,1].set(ylabel='NH4 MAE (mg/L)',title='Compare models within each evaluation');axes[1,1].legend()
    fig.suptitle('DARROW: NH4 history input ablation, fixed recorded actions')
    fig.tight_layout();fig.savefig(out/'history_comparison.png',dpi=160)
    print(out/'results.md')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)

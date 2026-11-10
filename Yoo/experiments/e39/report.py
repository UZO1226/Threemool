"""별도 보정 날짜에서 고정한 위험 판정의 개발 검증 결과를 보고한다."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(out):
    c=json.loads((out/'comparison.json').read_text());v=c['validation']
    p=json.loads((out/'protocol.json').read_text());o=json.loads((out/'operating_points.json').read_text());m=json.loads((out/'manifest.json').read_text())
    md=['# e39 날짜 분리 위험 판정 보정 결과','',f"실행 커밋 `{m['code_commit']}`, 소요 {m['elapsed_seconds']:.2f}초. 정책 학습/test 평가는 없다.",
        f"모델 학습140일: {p['fit_dates'][0]} ~ {p['fit_dates'][-1]}. 보정48일: {p['calibration_dates'][0]} ~ {p['calibration_dates'][-1]}. 평가40일: {p['validation_dates'][0]} ~ {p['validation_dates'][-1]}.",
        'e38의188일 모델과 다른 학습 조건이다. 이 실험의 같은140일 모델끼리 비교한다. 반복 사용한 개발val이며 독립 최종시험이 아니다.','',
        '## 검증 전 고정한 판정점','',
        '| 모델 | 프로필 | 확률 경계(이상) | 보정 탐지율 % | 보정 FPR % |','| --- | --- | --- | --- | --- |']
    for name in ('base','history'):
        for mode,x in o['models'][name].items():
            threshold_text='경보 없음' if x['probability_threshold']>1 else f"{x['probability_threshold']:.6f}"
            md.append(f"| {name} | {mode} | {threshold_text} | {x['calibration_recall_pct']:.3f} | {x['calibration_fpr_pct']:.3f} |")
    for mode,title in [('recall80','주 비교: 보정 자료에서 탐지율80% 목표'),('fpr10','보조 비교: 보정 자료에서 FPR10% 이하')]:
        md += ['',f'## {title}','','| 조건 | 검증 탐지율 % | 검증 FPR % | 평균 예측 위반 수 | 실제 위반 수 |','| --- | --- | --- | --- | --- |']
        for name in ('base','history'):
            for setting in ('raw',mode):
                x=v[name]['nh4'][setting]
                md.append(f"| {name}/{setting} | {x['recall_pct']:.3f} | {x['fpr_pct']:.3f} | {x['predicted_count_per_day_member']:.3f} | {x['observed_count_per_day_member']:.3f} |")
        md += ['', '검증의 달성 탐지율/FPR은 보정 날짜 목표와 다를 수 있다. 서로 다른 탐지율을 동일 탐지율 비교라고 부르지 않는다.']
    md += ['', '## 이력 빼기 기존의 대응 차이','', '| 프로필 | 지표 | 차이 %p | 7일 블록95%구간 %p |','| --- | --- | --- | --- |']
    for mode,x in c['history_minus_base'].items():
        for key,q in x.items():
            lo,hi=q['paired_7day_block_95ci_pp'];md.append(f"| {mode} | {key} | {q['difference_pp']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    md += ['', '날짜 단위 탐색 구간이다. 모델5구성원은 독립 정책seed가 아니다. 다중비교 보정이나 독립시험 결과가 아니다.','',
           '## 보정으로 바뀌지 않는 농도/DO 예측','',
           '| 모델 | NH4 MAE | DO1 MAE | DO2 MAE | DO3 MAE | DO recall % | DO FPR % |','| --- | --- | --- | --- | --- | --- | --- |']
    for name in ('base','history'):
        x=v[name];md.append(f"| {name} | {x['mae']['nh4']:.4f} | {x['mae']['do1']:.4f} | {x['mae']['do2']:.4f} | {x['mae']['do3']:.4f} | {x['DO']['recall_pct']:.3f} | {x['DO']['fpr_pct']:.3f} |")
    md += ['', 'MAE 단위mg/L. 보정은 각 모델의 위험 판정만 바꿔 이 수치와 물리 상태 궤적은 보정 전후 같다. 기록 공기도 고정되어 제어 절감 결과가 아니다.','',
           '## 공기 감소 후 실제 NH4 상승 구간의 위반 탐지','',
           '| 모델 | 프로필 | 분 후 | 구간 수 | 위반 표본 수(구성원 반복) | 탐지율 % | FPR % |','| --- | --- | --- | --- | --- | --- | --- |']
    fmt=lambda x:'미정의' if x is None else f'{x:.3f}'
    for x in c['event_diagnostics']:
        if x['group']=='air_decrease_and_nh4_rising':
            md.append(f"| {x['model']} | {x['mode']} | {x['minutes']} | {x['windows']} | {x['positives']} | {fmt(x['recall_pct'])} | {fmt(x['fpr_pct'])} |")
    md += ['', '미래 실제 상승 여부는 평가 부분집합을 나누는 데만 사용했다. 모델/보정 입력에는 포함되지 않는다. 실제 위반 기준NH4>4를 사용하며 상승 자체를 위반으로 바꾸지 않았다.','',
           '## 사전 선별','', '| 조건 | 통과 |','| --- | --- |']
    for key,value in c['screen'].items():md.append(f'| {key} | {value} |')
    md += ['', f"전체 통과: {c['all_screen_pass']}. 통과해도 환경/정책 자동 채택이나 실제 공정 효과 입증은 아니다.",
           '사후 검증 판정점 재선택 없이 보정기/판정점/모델 SHA256을 고정했고 원본 입력도 보존했다.','']
    if (out/'diagnostics.json').exists():
        d=json.loads((out/'diagnostics.json').read_text())
        md += ['', '## 사후 기간/동점 점검', '', '| 구간 | 날짜 수 | 실제 NH4 위반률 % | 평균 수온 | 위반 있는 날짜 수 |', '| --- | --- | --- | --- | --- |']
        for key,q in d['periods'].items():
            md.append(f"| {key} | {q['dates']} | {q['nh4_exceedance_pct']:.3f} | {q['mean_daily_temperature']:.3f} | {q['days_with_any_nh4_exceedance']} |")
        md += ['', '| 모델 | 보정 확률값 종류 | 최고 확률 묶음의 FPR % | 원 연속점수에서 보정 FPR10%의 탐지율 % |', '| --- | --- | --- | --- |']
        for key,q in d['calibration_plateaus'].items():
            md.append(f"| {key} | {q['distinct_calibrated_levels']} | {q['fpr_if_alert_highest_group_pct']:.4f} | {q['raw_score_fpr10_calibration_recall_pct']:.3f} |")
        md += ['', '이력 모델은 최고 확률 묶음만 경보해도 보정FPR10.0774%로 고정10%를 넘었다. 단조 보정이 점수를 같은 확률로 묶었기 때문에 고정 경계로는 그 묶음 일부만 고를 수 없었다.',
               '원 연속점수에서는 보정자료 FPR10%에서 탐지율61.08%가 가능했다. 이는 보조 조건의 무경보 결과가 위험 정보 부재의 증거가 아님을 보여준다. 사후 점검값을 검증 판정점으로 적용하지 않았다.',
               '기간 차이와 확률 동점은 관측된 사실이지만 각각의 검증 실패 기여를 인과적으로 분해한 실험은 아니다.', '']
    (out/'results.md').write_text('\n'.join(md))
    z=np.load(out/'predictions.npz');truth=np.repeat(z['actual'][:,None,:,0]>4,5,axis=1).ravel()
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for name,color in [('base','tab:blue'),('history','tab:orange')]:
        prob=z[name+'_prob'].ravel();edges=np.linspace(0,1,11);px=[];py=[]
        for lo,hi in zip(edges[:-1],edges[1:]):
            mask=(prob>=lo)&((prob<hi) if hi<1 else (prob<=hi))
            if mask.any():px.append(float(prob[mask].mean()));py.append(float(truth[mask].mean()))
        axes[0].plot(px,py,'o-',color=color,label=name)
        for setting,marker in [('raw','o'),('recall80','^'),('fpr10','s')]:
            x=v[name]['nh4'][setting]
            axes[1].scatter(x['fpr_pct'],x['recall_pct'],color=color,marker=marker,s=65,label=name+'/'+setting)
        raw=z[name+'_pred'][:,:,:,0].mean(1)
        act=z['actual'][:,:,0]
        axes[2].plot(np.arange(40),np.abs(raw-act).mean(1),color=color,label=name)
    axes[0].plot([0,1],[0,1],':',color='gray');axes[0].set(xlabel='Mean calibrated probability',ylabel='Observed violation frequency',title='Validation reliability');axes[0].legend()
    axes[1].set(xlabel='False positive rate (%)',ylabel='Recall (%)',title='Frozen operating points');axes[1].legend(fontsize=7)
    axes[2].set(xlabel='Validation day index',ylabel='NH4 MAE (mg/L)',title='Unchanged concentration prediction');axes[2].legend()
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('DARROW: model fit140 / calibration48 / development validation40')
    fig.tight_layout();fig.savefig(out/'calibration.png',dpi=160)
    print(out/'results.md')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)

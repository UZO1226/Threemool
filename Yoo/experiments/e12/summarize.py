"""학습 목적 대조 및 분기 교차 진단을 표와 그림으로 정리한다."""
import argparse
import json
from pathlib import Path
import numpy as np


def main(a):
    trial=json.loads(a.trial.read_text());diagnosis=json.loads(a.diagnosis.read_text())
    assert 'completed_at' in trial and 'completed_at' in diagnosis
    a.out.mkdir(exist_ok=True,parents=True)
    for name,value in [('trial_summary.json',trial),('attribution_summary.json',diagnosis)]:
        (a.out/name).write_text(json.dumps(value,indent=2,allow_nan=False))
    md=['# Yoo/e12 결과','',f"학습 실행 `{trial['commit']}`, 종료 `{trial['completed_at']}` (UTC).",
        f"원인 분리 실행 `{diagnosis['commit']}`, 종료 `{diagnosis['completed_at']}` (UTC).",'',
        '모든 비용은 초기 1시간 뒤 275전이에 대한 값이다. 기존 287전이 정책 성능 및 한도와 비교하지 않는다. 새 모델은 내부 holdout 66일까지만 평가했다. 기존 val 43일의 원인 분리는 e11 모델만 사용하며 test는 쓰지 않았다.','',
        '## 새 모델 내부 holdout의 연속 예측', '',
        '| 군 | seed | 비용 MAE | NH4 MAE | NH4 recall | NH4 precision | NH4 FPR | DO MAE | DO recall | DO FPR |',
        '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in trial['evaluation']:
        if r['split']!='inner_holdout' or r['reset_steps']!=275:continue
        n,d=r['nh4'],r['do']
        md.append(f"| {r['mode']} | {r['seed']} | {r['cost']['daily_mae']:.2f} | {n['mae']:.3f} | {n['step_recall']*100:.1f}% | {n['step_precision']*100:.1f}% | {n['fpr']*100:.1f}% | {d['mae']:.3f} | {d['step_recall']*100:.1f}% | {d['fpr']*100:.1f}% |")
    md += ['', 'FPR은 실제 비위반 중 잘못 위반으로 예측한 비율이다. precision은 예측 위반 중 실제 위반 비율이므로 분모가 다르다. 모든 단기 및 fit 결과, 시작 및 지속시간 지표는 JSON에 보존한다.', '',
           '## 진행 기준', '', '| 후보 | 시작 모델 대비 NH4 recall 변화 | single_mse 대비 변화 | screening | recall80/FPR10 |',
           '| --- | --- | --- | --- | --- |']
    for g in trial['gates']:
        md.append(f"| {g['candidate']} | {g['mean_nh4_recall_gain']['reference']*100:+.2f}%p | {g['mean_nh4_recall_gain']['single_mse']*100:+.2f}%p | {g['screen_passed']} | {g['absolute_recall80_fpr10']} |")
    md+=['','주 후보 multi_boundary의 실패 항목:', '']
    primary=next(g for g in trial['gates'] if g['candidate']=='multi_boundary')
    for row in primary['per_seed']:
        failed=[k for k,v in row.items() if v is False]
        md.append(f"- seed {row['seed']}, {row['base']} 대비: "+', '.join(failed))
    md+=['','## e11 분기 교차 진단', '',
         '새 DO와 새 NH4는 e11 경로 입력 모델의 해당 분기다. 각 분기의 정규화와 경로 입력 유무까지 원래 모델대로 유지했다. 같은 두 분기를 조합한 결과는 원래 모델 출력과 정확히 같았다.', '',
         '| 구간 | DO 분기 | NH4 분기 | NH4 recall 평균 | NH4 편향 평균 | NH4 FPR 평균 |',
         '| --- | --- | --- | --- | --- | --- |']
    for split in ('inner_holdout','val'):
        for d,n in ((False,False),(True,False),(False,True),(True,True)):
            rows=[r for r in diagnosis['rows'] if r['kind']=='crossover' and r['split']==split and r['do_new']==d and r['nh4_new']==n]
            mean=lambda k:np.mean([r['nh4'][k] for r in rows])
            md.append(f"| {split} | {'새' if d else '기존'} | {'새' if n else '기존'} | {mean('step_recall')*100:.2f}% | {mean('bias'):.3f} | {mean('fpr')*100:.2f}% |")
    md+=['','## 실측 상태 주입 진단', '',
         'DO 주입은 현재 및 다음 DO를, NH4 주입은 현재 NH4를 실제 값으로 교체한다. 배포 가능한 예측 성과가 아니며 원인 진단이다.', '',
         '| 대상 | 구간 | 실측 주입 | NH4 recall 평균 | NH4 FPR 평균 | NH4 편향 평균 |',
         '| --- | --- | --- | --- | --- | --- |']
    for label,split in [('e11_with','val'),('e12_multi_boundary','inner_holdout')]:
        for oracle in ('none','do','nh4','both'):
            rows=[r for r in diagnosis['rows'] if r['kind']=='oracle' and r['label']==label and r['split']==split and r['oracle']==oracle]
            mean=lambda k:np.mean([r['nh4'][k] for r in rows])
            md.append(f"| {label} | {split} | {oracle} | {mean('step_recall')*100:.2f}% | {mean('fpr')*100:.2f}% | {mean('bias'):.3f} |")
    md+=['','![위반 탐지와 거짓경보](figures/recall_fpr.png)','']
    (a.out/'results.md').write_text('\n'.join(md))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,5))
    for mode in ('reference','single_mse','single_boundary','multi_mse','multi_boundary'):
        rows=[r for r in trial['evaluation'] if r['split']=='inner_holdout' and r['reset_steps']==275 and r['mode']==mode]
        ax.plot([r['nh4']['fpr']*100 for r in rows],[r['nh4']['step_recall']*100 for r in rows], 'o-',label=mode)
        for r in rows:ax.annotate(str(r['seed']),(r['nh4']['fpr']*100,r['nh4']['step_recall']*100),xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.set_xlabel('NH4 false positive rate (%)');ax.set_ylabel('NH4 recall (%)')
    ax.set_title('Inner holdout, 275-step replay; labels are training seeds')
    ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
    (a.out/'figures').mkdir(exist_ok=True);fig.savefig(a.out/'figures/recall_fpr.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('trial','diagnosis','out'):p.add_argument('--'+key,type=Path,required=True)
    main(p.parse_args())

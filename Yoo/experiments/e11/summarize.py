"""반응조 경로 입력 대조 결과와 동일 날짜의 예측을 기록한다."""
import argparse
import json
from pathlib import Path
import numpy as np


def main(a):
    r=json.loads((a.trial/'results.json').read_text())
    outer=json.loads(a.outer.read_text()) if a.outer else None
    assert 'completed_at' in r and all(r['checks'].values())
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'trial_summary.json').write_text(json.dumps(r,indent=2,allow_nan=False))
    if outer:
        (a.out/'outer_summary.json').write_text(json.dumps(outer,indent=2,allow_nan=False))
    md=['# Yoo/e11 결과','',f"실행 커밋 `{r['commit']}`. 종료 `{r['completed_at']}` (UTC).",'',
        '새 모델 seed 0, 1을 각각 제시한다. 내부 시간 분할과 275전이 평가이며 기존 287전이 정책 성능과 섞지 않는다. e10 뒤 내부 holdout을 재사용한 탐색 결과다. 학습과 내부 판정에는 바깥 val과 test를 사용하지 않았다. 실제 공기량 및 운전 경로를 고정한 재생이므로 새 제어 정책의 인과 효과 검증이 아니다.','']
    if outer:
        md += [f"내부 통과 뒤 설정과 가중치를 고정하여 기존 val 43일에서 한 번 확인했다. 실행 커밋 `{outer['commit']}`, 종료 `{outer['completed_at']}` (UTC). 기존에 관찰한 검증 자료이므로 독립적인 새 평가 자료는 아니다. test는 사용하지 않았다.", '']
    evaluation=r['evaluation']+(outer['evaluation'] if outer else [])
    for split in (('inner_holdout','fit','val') if outer else ('inner_holdout','fit')):
        md += [f'## {split}', '', '| 입력 | seed | 재시작 간격 | NH4 MAE | DO MAE | 예측 비용 | 실측 비용 | 비용 MAE | NH4 recall | 시작 F1 | 최장 구간 MAE (분) |',
               '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for row in evaluation:
            if row['split']!=split:continue
            n,d,c=row['nh4'],row['do'],row['cost']
            md.append(f"| {row['mode']} | {row['seed']} | {row['reset_steps']} | {n['mae']:.3f} | {d['mae']:.3f} | {c['predicted']:.2f} | {c['actual']:.2f} | {c['daily_mae']:.2f} | {n['step_recall']*100:.1f}% | {n['onset_f1']:.3f} | {n['longest_observed_run_mae_minutes']:.2f} |")
        md+=['']
    md+=['## 내부 사전 기준', '',f"통과: `{r['gate']['passed']}`. 평균 비용 MAE 감소율 {r['gate']['mean_cost_mae_improvement_fraction']*100:.2f}%.", '',
         '| seed | 비용 MAE 개선 | 시작 F1 유지 | 최장 구간 오차 유지 | DO 오차 악화 5% 이내 | 한 스텝 NH4 오차 악화 5% 이내 |',
         '| --- | --- | --- | --- | --- | --- |']
    for row in r['gate']['per_seed']:
        md.append('| '+' | '.join(str(row[k]) for k in ('seed','cost_mae_improved','onset_f1_not_worse','longest_run_mae_not_worse','do_mae_within_5pct','one_step_nh4_mae_within_5pct'))+' |')
    if outer:
        md += ['',f"같은 진행 기준의 외부 val 통과: `{outer['gate']['passed']}`. 평균 비용 MAE 감소율 {outer['gate']['mean_cost_mae_improvement_fraction']*100:.2f}%.", '',
               '진행 기준 통과가 실제 안전성 또는 모델 교체 승인은 아니다. NH4 위반 스텝 recall이 낮아지는 등 모든 지표가 개선된 것은 아니며 [최종 판정](README.md)을 함께 본다.']
    md+=['','모든 loss 및 gradient 유한, 같은 초기 가중치와 batch 순서, 저장 복원 일치, 실제 학습망의 미래 실측값 비참조 검사를 통과했다.','',
         '## 예측 예시', '', 'e10과 같은 기준으로 실측 NH4의 가장 긴 위반 구간이 있는 날짜를 고르고 두 seed를 함께 제시한다.', '',
         '![경로 입력 대조](figures/routing_replay.png)', '']
    (a.out/'results.md').write_text('\n'.join(md))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    z=np.load(a.trial/'without_routing_s0_inner_holdout_h275.npz')
    def longest(y):
        changes=np.diff(np.r_[False,y>4,False].astype(int))
        return max(np.flatnonzero(changes==-1)-np.flatnonzero(changes==1),default=0)
    day=int(np.argmax([longest(v[:,0]) for v in z['truth']]))
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    hour=np.arange(13,288)/12
    for seed,ax in enumerate(axes):
        ax.plot(hour,z['truth'][day,:,0],color='black',label='Observed',lw=1)
        for mode in ('without_routing','with_routing'):
            pred=np.load(a.trial/f'{mode}_s{seed}_inner_holdout_h275.npz')['pred']
            ax.plot(hour,pred[day,:,0],label=mode)
        ax.axhline(4,color='gray',ls=':',label='Threshold')
        ax.set_title(f'Seed {seed}');ax.set_ylabel('NH4');ax.legend(fontsize=8);ax.grid(alpha=.2)
    axes[-1].set_xlabel('Hour of recorded day')
    fig.suptitle('Free run after 1-hour initialization: '+r['partition']['inner_holdout'][day])
    fig.tight_layout();(a.out/'figures').mkdir(exist_ok=True)
    fig.savefig(a.out/'figures/routing_replay.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--trial',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--outer',type=Path)
    main(p.parse_args())

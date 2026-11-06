"""위반 사건 진단과 과거 상태 대조를 집계한다."""
import argparse
import json
from pathlib import Path
import numpy as np


def main(a):
    result=json.loads((a.trial/'results.json').read_text())
    audit=json.loads(a.audit.read_text())
    assert 'completed_at' in result and all(result['checks'].values())
    a.out.mkdir(exist_ok=True,parents=True)
    for filename,value in [('trial_summary.json',result),('event_summary.json',audit)]:
        (a.out/filename).write_text(json.dumps(value,indent=2,allow_nan=False))
    md=['# Yoo/e10 결과', '',f"실행 커밋 `{result['commit']}`. 종료 `{result['completed_at']}` (UTC).",'',
        '## 기존 모델의 위반 사건 재현', '',
        'val 43일, 287전이, 구성원 5개 평균이다. 실제 NH4 위반 시작 137개 중 15분 이내 예측 시작과 일대일 매칭되는 수를 집계했다. 시작 경계에서 진행 중인 사건은 제외했다. 이 표는 기존 모델 진단이며 아래 새 모델 내부 holdout과 구간이 다르다.', '',
        '| 모델 | NH4 위반 스텝 recall | precision | 시작 매칭 수 | 시작 F1 | 최장 관측 위반 길이 MAE (분) |',
        '| --- | --- | --- | --- | --- | --- |']
    for mode in ('original','e09_multistep'):
        rows=[r for r in audit['rows'] if r['split']=='val' and r['model']==mode]
        avg=lambda key:np.mean([r['nh4'][key] for r in rows])
        md.append(f"| {mode} | {avg('step_recall')*100:.2f}% | {avg('step_precision')*100:.2f}% | {avg('matched_onsets'):.1f}/137 | {avg('onset_f1'):.3f} | {avg('longest_observed_run_mae_minutes'):.2f} |")
    md+=['','## 새 모델 내부 holdout', '',
         '2023-10-16부터 2023-12-31의 완전한 66일이다. 각 날짜 00:00부터 01:00까지 관측값으로 초기화한 뒤 275전이를 평가한다. 평균 한도 15를 만족하는 정책을 평가하는 실험이 아니다. seed 2개를 각각 제시한다.', '',
         '| 입력 | seed | 재시작 간격 | NH4 MAE | DO MAE | 예측 비용 | 실측 비용 | 날짜별 비용 MAE | NH4 시작 F1 | 최장 위반 길이 MAE (분) |',
         '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in result['evaluation']:
        if r['split']!='inner_holdout':continue
        md.append(f"| {r['mode']} | {r['seed']} | {r['reset_steps']} | {r['nh4']['mae']:.3f} | {r['do']['mae']:.3f} | {r['cost']['predicted']:.2f} | {r['cost']['actual']:.2f} | {r['cost']['daily_mae']:.2f} | {r['nh4']['onset_f1']:.3f} | {r['nh4']['longest_observed_run_mae_minutes']:.2f} |")
    md+=['','## 사전 진행 기준', '',f"통과: `{result['gate']['passed']}`. 평균 연속 예측 비용 MAE 변화는 {result['gate']['mean_cost_mae_improvement_fraction']*100:.2f}% 감소다. 음수는 악화를 뜻한다.", '',
         '| seed | 비용 MAE 개선 | 시작 F1 유지 | 최장 위반 길이 MAE 유지 | DO MAE 악화 5% 이내 | 한 스텝 NH4 MAE 악화 5% 이내 |',
         '| --- | --- | --- | --- | --- | --- |']
    for r in result['gate']['per_seed']:
        md.append('| '+ ' | '.join(str(r[k]) for k in ('seed','cost_mae_improved','onset_f1_not_worse','longest_run_mae_not_worse','do_mae_within_5pct','one_step_nh4_mae_within_5pct'))+' |')
    md+=['','진행 기준은 통계적 유의성 검정이 아니다. 외부 val과 test에는 새 모델을 평가하지 않았다. 세부 사건의 누락, 매칭 수와 지속시간 오차 및 학습 구간 결과는 [JSON](trial_summary.json)에 보존했다.', '',
         '## 예측 예시', '',
         '내부 holdout에서 실측 NH4의 최장 위반 구간이 가장 긴 날짜를 선택했다. 동률이면 먼저 나온 날짜다. 후보에 유리한 날짜나 seed를 골라 제시하지 않고 두 seed를 함께 그린다.', '',
         '![연속 예측 예시](figures/history_replay.png)', '']
    (a.out/'results.md').write_text('\n'.join(md))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    z=np.load(a.trial/'current_repeat_s0_inner_holdout_h275.npz')
    def longest(x):
        diff=np.diff(np.r_[False,x,False].astype(int))
        return max(np.flatnonzero(diff==-1)-np.flatnonzero(diff==1),default=0)
    day=int(np.argmax([longest(y[:,0]>4) for y in z['truth']]))
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    hour=np.arange(13,288)/12
    for seed,ax in enumerate(axes):
        ax.plot(hour,z['truth'][day,:,0],label='Observed',color='black',lw=1)
        for mode in ('current_repeat','history'):
            pred=np.load(a.trial/f'{mode}_s{seed}_inner_holdout_h275.npz')['pred']
            ax.plot(hour,pred[day,:,0],label=mode)
        ax.axhline(4,color='gray',ls=':',label='Threshold')
        ax.set_title(f'Seed {seed}')
        ax.set_ylabel('NH4')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[-1].set_xlabel('Hour of recorded day')
    fig.suptitle('Free run after 1-hour initialization: '+result['partition']['inner_holdout'][day])
    fig.tight_layout()
    (a.out/'figures').mkdir(exist_ok=True)
    fig.savefig(a.out/'figures/history_replay.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--trial',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())

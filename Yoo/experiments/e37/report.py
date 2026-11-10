"""관측 반응 예측, 조건 유사성, 진단 한계를 함께 보고한다."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(out):
    r=json.loads((out/'results.json').read_text())
    m=json.loads((out/'manifest.json').read_text())
    s=json.loads((out/'support_audit.json').read_text())
    p=json.loads((out/'protocol.json').read_text())
    md=['# e37 DARROW 기록 공기 변화 반응 대조', '',
        f"실행 커밋 `{m['code_commit']}`, 평가 {m['elapsed_seconds']:.2f}초. 신규 학습과 test 평가 없음.",
        '기록된 미래 공기/외생 입력을 재생하는 조건부 예측 검사다. 실제 미래 수질은 입력하지 않는다.',
        '원인 효과나 온라인 실시간 예측 성능을 식별한 결과가 아니다. 모든 농도 오차 단위mg/L.', '',
        '## 구간 선택과 자료 지원', '',
        f"train188일에서 큰 변화 기준 {p['large_change_m3_h']:.3f}m³/h, 안정 기준 {p['stable_change_m3_h']:.3f}m³/h를 고정했다.",
        f"val40일의 변화132구간(증가67, 감소65)을 평가했다. 주 기준을 통과한 매칭은 {r['support']['accepted_pairs']}쌍/{r['support']['accepted_event_days']}날짜다.",
        '사전 기술적 충분성 기준30쌍/10날짜를 충족하지 않았다. 매칭된4쌍은 모두 증가 구간이다. 따라서 그 대비로 공기 감소 효과를 주장하지 않는다.', '',
        '| 사후 조건 민감도 | 대응 가능한 변화 구간 |', '| --- | --- |',
        '| 주 기준: 최근접 한 쌍을 고른 뒤1 IQR 검사 | 4 |']
    for k,v in s['events_with_ANY_eligible_crossday_control_by_max_iqr'].items():
        md.append(f'| 모든 다른 날짜 후보 검색, 변수별 최대{k} IQR | {v} |')
    md += ['', '1 IQR에서4와9의 차이는 후보 검색 방식 때문이다. 자료 지원 부족이 유일한 알고리즘의 산물이라고 단정하지 않는다.',
           '2/3 IQR은 결과 확인 후의 보수성 점검이며 사전 성공 기준이 아니다. 넓히면 수가 늘지만 조건 유사성이 약해진다.', '',
           '## 같은 운전 조건이 아닌 증가와 감소 구간', '',
           '| 공기 변화 방향 | 시작 NH4 평균 | 직전1시간 NH4 변화 평균 |', '| --- | --- | --- |']
    for g,x in s['observed_action_context'].items():
        md.append(f"| {g} | {x['initial_nh4_mean']:.3f} | {x['previous_hour_nh4_change_mean']:+.3f} |")
    md += ['', '이 차이는 기존 제어 피드백/운전 조건과의 연관을 보여준다. 공기 변화가 NH4 변화를 유발했다는 증거는 아니다.',
           '모델의 현재 입력에는 이전 수질 추세가 없다. 이력이 빠진 것이 주원인인지는 별도 대조 없이는 확정하지 않는다.', '']
    for group,x in r['response'].items():
        md += [f'## {group}: {x["events"]}구간, {x["days"]}날짜', '',
               '| 센서 | 분 후 | 모델 MAE | 상태 유지 MAE | 차이95%구간 | 방향 균형 정확도 % |',
               '| --- | --- | --- | --- | --- | --- |']
        for q in x['metrics']:
            lo,hi=q['paired_7day_block_95ci']
            md.append(f"| {q['target']} | {q['minutes']} | {q['mae']:.4f} | {q['persistence_mae']:.4f} | [{lo:+.4f}, {hi:+.4f}] | {q['direction_balanced_accuracy_pct']:.1f} |")
    md += ['', '차이는 모델MAE 빼기 상태 유지MAE다. 음수이면 모델이 낫다. 날짜7일 순환블록2000회 탐색 구간이며 다중비교 보정이나 seed 모집단 검정이 아니다.',
           '방향은 감소/유지/증가3종이다. 균형 정확도는 해당 표본에 존재하는 실제 클래스별 재현율 평균이다. 단순히50%와 비교해 우연 수준이라고 단정하지 않는다.', '',
           '## 판정', '',
           '전체 변화 구간의 NH4는15분/1시간/3시간 모두 상태 유지 기준 대비 MAE 차이 구간이0을 포함했다. 개선 입증 실패이며 동등성 입증이 아니다.',
           'DO1/DO2는 특히1시간 조건부 예측에서 도움이 됐다. 전체 환경이 무가치하다는 결론은 아니다.',
           '증가 구간3시간에서는 DO1/DO3 오차가 상태 유지 기준보다 컸다. 전체 평균과 감소 구간의 좋은 점수만으로 숨기지 않는다.',
           '최대 변화 시각 비교는3시간 창 안의 극값이며 공정의 인과적 지연시간이 아니다. NH4는 실제 극값이 창 끝인 비율도31.8%다.',
           '현재 근거로 공기 절감의 NH4/DO 효과를 검증한 환경이라고 주장하거나 새 RL 성과를 확정하지 않는다.',
           '자료 지원 기준을 느슨하게 바꿔 채택하지 않고, 다음 후보는 NH4 전이에 직전 수질 추세를 제공하는 제한적 대조다. 검증된 개선 원인이라는 뜻은 아니며 이번에는 재학습하지 않았다.', '']
    (out/'results.md').write_text('\n'.join(md))
    z=np.load(out/'trajectories.npz')
    n=r['support']['val_events']; actual=z['actual'][:n]; pred=z['pred'][:n].mean(1)
    minutes=np.arange(13)*15
    fig,axes=plt.subplots(2,4,figsize=(15,6),sharex=True)
    for row,(group,sign) in enumerate([('Increase',1),('Decrease',-1)]):
        mask=z['dq'][:n]*sign>0
        for col,(j,label) in enumerate([(0,'NH4'),(2,'DO1'),(3,'DO2'),(4,'DO3')]):
            ax=axes[row,col]
            ax.plot(minutes,(actual[mask,:,j]-actual[mask,:1,j]).mean(0),label='Observed',color='black')
            ax.plot(minutes,(pred[mask,:,j]-pred[mask,:1,j]).mean(0),label='Model mean',color='tab:blue')
            ax.axhline(0,ls=':',color='gray',label='Persistence')
            ax.set(title=f'{group}: {label}',xlabel='Minutes after start',ylabel='Change (mg/L)')
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Recorded trajectories after airflow changes (conditional prediction, not causal effects)')
    fig.tight_layout()
    fig.savefig(out/'response.png',dpi=160)
    print(out/'results.md')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)

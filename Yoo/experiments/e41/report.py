"""고정된 e41 비교와 모든 실패 항목을 표로 기록한다."""
import argparse
import json
from pathlib import Path


def main(out):
    c = json.loads((out/'comparison.json').read_text())
    m = json.loads((out/'manifest.json').read_text())
    labels = {'reference': '동결 이력 기준', 'control': '실측 재학습 대조', 'augmented': '자기 예측 증강'}
    lines = ['# e41 NH4 자기 예측 입력 증강 결과', '',
             f"실행 커밋 `{m['code_commit']}`, 전체 {m['elapsed_seconds']:.2f}초. NH4 GBM 10개 학습, 새 RL 없음.", '',
             '모델 학습 140일, 별도 holdout 48일, 개발 검증 40일. 주 효과는 같은 학습량의 control 대비 augmented다. reference는 기존 e39 이력 모델이다. 실제/예측 NH4>4, max(DO)>2 판정을 고정했다.', '']
    for mode, title in (('cold', '주 평가: 검증 40일, 95전이 재귀 예측'), ('holdout', '별도 학습 내 holdout 48일'), ('warm', '검증 40일, 첫 1시간 실측 후 91전이'), ('one_step', '검증 실측 입력 한 스텝 예측')):
        lines += [f'## {title}', '', '| 모델 | NH4 MAE | NH4 탐지율 % | NH4 FPR % | DO1 MAE | DO2 MAE | DO3 MAE | DO 탐지율 % | DO FPR % |',
                  '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
        for name, rr in c['results'].items():
            v = rr[mode]
            lines.append(f"| {labels[name]} | {v['mae']['nh4']:.4f} | {v['confusion']['NH4']['recall_pct']:.3f} | {v['confusion']['NH4']['fpr_pct']:.3f} | {v['mae']['do1']:.4f} | {v['mae']['do2']:.4f} | {v['mae']['do3']:.4f} | {v['confusion']['DO']['recall_pct']:.3f} | {v['confusion']['DO']['fpr_pct']:.3f} |")
        lines += ['']
    lines += ['## 주 평가의 날짜별 초과 횟수, 강도와 꼬리 오차', '',
              '| 모델 | NH4 초과 수 평균 | 초과 수 MAE | 초과 강도 평균 mg h/L | 초과 강도 MAE | 날짜 NH4 MAE 95분위 |',
              '| --- | --- | --- | --- | --- | --- |']
    for name, rr in c['results'].items():
        v = rr['cold']
        lines.append(f"| {labels[name]} | {v['counts']['NH4']['predicted_mean']:.3f} | {v['counts']['NH4']['mae']:.3f} | {v['nh4_excess_mg_h_L']['predicted_mean']:.3f} | {v['nh4_excess_mg_h_L']['mae']:.3f} | {v['nh4_p95_daily_mae']:.3f} |")
    ref = c['results']['reference']['cold']
    lines += ['', f"실측 NH4 초과 수 평균 {ref['counts']['NH4']['observed_mean']:.3f}, 초과 강도 평균 {ref['nh4_excess_mg_h_L']['observed_mean']:.3f} mg h/L. 수질 상태 MAE 단위는 mg/L이며 현재 비용은 횟수, 강도는 별도 지표다.", '',
              '## 공기 변화 뒤 NH4 반응', '',
              '| 군 | 방향/부분집합 | 분 | 구간 수 | NH4 MAE | 실제 변화 | 예측 변화 | 탐지율 % | FPR % |',
              '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for name, rows in c['events'].items():
        for v in rows:
            recall = v['confusion']['recall_pct']
            fpr = v['confusion']['fpr_pct']
            r = '정의 불가' if recall is None else f'{recall:.3f}'
            f = '정의 불가' if fpr is None else f'{fpr:.3f}'
            lines.append(f"| {labels[name]} | {v['group']} | {v['minutes']} | {v['windows']} | {v['nh4_mae']:.4f} | {v['nh4_actual_delta']:+.4f} | {v['nh4_predicted_delta']:+.4f} | {r} | {f} |")
    lines += ['', 'decrease_rising은 실제 NH4가 0.1 mg/L보다 상승한 공기 감소 구간이다. 미래 상승 조건은 평가 부분집합에만 사용했다. 5구성원은 같은 실제 사건을 공유하며 독립 사건으로 세지 않는다. 기록 행동의 조건부 예측으로 공기량 인과 효과를 입증하지 않는다.', '',
              '## 증강군에서 각 기준을 뺀 차이와 불확실성', '',
              '| 비교 기준 | 지표 | 차이 | 날짜 7일 블록 95% 구간 |', '| --- | --- | --- | --- |']
    for name, v in c['uncertainty_augmented_minus'].items():
        for state, x in v['mae'].items():
            a, b = x['paired_7day_95ci']
            lines.append(f"| {labels[name]} | {state} MAE | {x['difference']:+.4f} | [{a:+.4f}, {b:+.4f}] |")
        for target, rows in v['confusion'].items():
            for metric, x in rows.items():
                a, b = x['paired_7day_95ci_pp']
                lines.append(f"| {labels[name]} | {target} {metric} pp | {x['difference_pp']:+.3f} | [{a:+.3f}, {b:+.3f}] |")
    lines += ['', '2000회 재표집, seed4100. 반복 개발 검증의 탐색적 구간이며 독립 확증/비열등성 검정이 아니다.', '',
              '## 구성원별 NH4 MAE와 학습 시간', '',
              '| 군 | 구성원 0 | 1 | 2 | 3 | 4 | NH4 학습 시간 합계 초 |', '| --- | --- | --- | --- | --- | --- | --- |']
    for name, v in c['results'].items():
        maes = ' | '.join(f'{x:.4f}' for x in v['cold']['nh4_member_mae'])
        seconds = sum(x['seconds'] for x in c['training'] if x['arm'] == name)
        lines.append(f'| {labels[name]} | {maes} | {seconds:.3f} |')
    lines += ['', '동결 기준의 이 표 학습 시간은 0이다. 교사 경로 생성과 전체 평가는 위 전체 실행 시간에 포함된다.', '',
              '## 사전 선별 전 항목', '', '| 기준 | 항목 | 통과 |', '| --- | --- | --- |']
    for name, checks in c['screens_augmented_vs'].items():
        for key, passed in checks.items():
            lines.append(f"| {labels[name]} | {key} | {'예' if passed else '아니오'} |")
    lines += ['', f"전체 선별 통과: {c['all_screen_pass']}. 이 결과만으로 제어 환경을 교체하거나 RL을 재학습하지 않는다. 기존 모델/NO3/DO 가중치/비용 정의는 보존했다. 물리 경로가 달라지므로 DO 가중치가 같아도 재귀 DO 예측은 달라질 수 있다.", '']
    (out/'results.md').write_text('\n'.join(lines))
    print(out/'results.md')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    main(p.parse_args().out)

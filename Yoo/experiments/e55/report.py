"""저온 진단의 작은 결과 파일과 표를 생성한다."""
import argparse
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent


def report(root):
    r=json.loads((root/'results.json').read_text())
    manifest=json.loads((root/'manifest.json').read_text())
    shutil.copyfile(root/'results.json',HERE/'results.json')
    v=r['splits']['val'];labels={'all':'전체','below_train':'학습 최저 수온 미만','inside_train':'학습 수온 범위 안','never_below_day':'95전이 내내 최저 수온 이상인 날짜','no_low_so_far':'현재까지 최저 수온 미만 이력 없음'}
    lines=['# e55 수온 범위와 NH4 예측 오류 진단','',
           f"실행 커밋 `{manifest['commit']}`. 기존 e35 모델과 기록 행동 경로, train188일/개발 val40일. test 평가와 학습 없음. 실행 {r['elapsed_seconds']:.2f}초.",'',
           '| 구간 | 학습 입력 수온 °C | 검증 입력 수온 °C |', '| --- | --- | --- |',
           f"| 최소~최대 | {r['train_input_temp_min']:.3f}~{r['train_input_temp_max']:.3f} | {v['input_temp_min']:.3f}~{v['input_temp_max']:.3f} |",'',
           f"검증 {v['n_transitions']}전이 중 {v['group_steps']['below_train']}전이({100*v['group_steps']['below_train']/v['n_transitions']:.2f}%)가 학습 최저 수온 미만이다. 95전이 내내 최저 수온 이상인 검증 날짜는 {v['never_below_day_count']}/40일이다.",'',
           '## 같은 관측 정답에 대한 예측','',
           '| 예측 방식 | 온도 그룹 | 원 전이 수 | MAE (mg/L) | 예측−실측 평균 | FPR % | 탐지율 % |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for mode,label in [('observed_one_step','실측 상태에서 한 스텝'),('recursive','자기 예측 95전이')]:
        for group in labels:
            z=v['predictions'][mode][group]
            lines.append(f"| {label} | {labels[group]} | {v['group_steps'][group]} | {z['mae']:.3f} | {z['bias']:+.3f} | {z['fpr_pct']:.2f} | {z['recall_pct']:.2f} |")
    lines+=['','마지막 두 그룹은 앞선 그룹과 겹치는 진단이며 합산하지 않는다. 원 전이마다 모델 경로 5개가 있고 이들을 독립 날짜로 세지 않는다. 한 스텝은 매번 실측 상태를 제공한 진단이며 실제 자율 제어 성능이 아니다.','',
            '## 반복 예측 오류의 위치','',
            '| 현재 수온 | 절대오차 합 비중 % | 거짓양성 수 비중 % | 놓친 위반 수 비중 % |',
            '| --- | --- | --- | --- |']
    for group in ['below_train','inside_train']:
        z=v['predictions']['recursive'][group]
        lines.append(f"| {labels[group]} | {z['absolute_error_sum_share_pct']:.2f} | {z['fp_share_pct']:.2f} | {z['fn_share_pct']:.2f} |")
    lines+=['','비중은 오류가 발생한 위치의 기술 통계다. 저온의 인과 기여율이나 저온 수정으로 제거할 수 있는 오류의 상한이 아니다. 현재 온도가 범위 안이어도 과거 저온의 누적 영향이 있을 수 있다. 저온 미경험 날짜 역시 낮은 온도에서의 장기 생물학적 이력을 통제하지 않는다.','',
            '## 수온만 바꾼 모델 반응','',
            f"실제 저온 입력과 그 입력의 수온만 학습 최저값으로 올린 경우의 NH4 변화량 최대 차이: {v['below_train_vs_floor_max_delta_difference']:.12g} mg/L.",'',
            '| 고정 수온 °C | train 입력의 평균 NH4 변화량 | val 입력의 평균 NH4 변화량 |',
            '| --- | --- | --- |']
    for a,b in zip(r['splits']['train']['temperature_response_curve'],v['temperature_response_curve']):
        lines.append(f"| {a['temperature']:.3f} | {a['mean_nh4_delta']:+.4f} | {b['mean_nh4_delta']:+.4f} |")
    z=v['temperature_intervention']
    lines+=['',f"학습 수온 사분위 {z['low']:.3f}→{z['high']:.3f}°C 변경 시 검증 입력에서 NH4 변화량은 평균 {z['high_minus_low_nh4_delta_mean']:+.5f} mg/L 변했다. 감소 {z['decreasing_nh4_pct']:.2f}%, 증가 {z['increasing_nh4_pct']:.2f}%로 방향이 섞인다.",'',
            '동일한 관측 상태, 유입, 공기량에서 수온만 바꾼 한 스텝 모델 민감도다. 관측에 없던 특징 조합이 포함될 수 있고, 실제 질산화 반응률이나 온도의 인과 효과를 직접 추정한 것은 아니다.','',
            '## 검증 및 원 출력','',
            '정답 배열과 e54 저장 정답 일치, train/val 날짜 및 배열 정렬 확인, 온도 그룹의 표본/혼동행렬/절대오차 합 보존, 입력과 코드 해시 불변을 확인했다. 기존 e53 기록 행동 재생과 연결되는 e54 경로를 사용했다.',
            f"전체 입력 해시와 출력: `{root}`.",'']
    (HERE/'results.md').write_text('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    report(p.parse_args().root)

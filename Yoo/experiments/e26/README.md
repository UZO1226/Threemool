# e26 e14 하루 길이 MSE의 외부 시간 구간 확인

2026-11-09 실행 전 계획. 사용자가 승인한 종합 검토의 두 번째 단계다. e25 정책 학습과 분리하며, 이 평가로 진행 중인 정책의 환경을 바꾸지 않는다. 새 모델 학습은 없다.

## 가설과 고정 설정

e14 day_mse는 내부 holdout 66일에서 시작 모델 대비 두 seed 모두 NH4 recall 증가와 FPR 감소를 보였다. 저장 가중치를 고정하고 기존 val 43일(2024년 1~2월)에서도 이 방향이 이어지는지 확인한다. 이 날짜는 다른 과거 실험에서 사용했으므로 독립 최종 test라고 부르지 않는다. test 날짜를 선택하거나 평가하지 않는다.

후보는 e14 day_mse 하나다. e14 day_group 및 day_global은 넣지 않는다. 비교는 e11 with_routing(reference), e12 single_mse, e12 multi_mse(start_multi), e14 short_mse. 각 seed0,1의 기존 마지막 가중치만 사용한다. 선택하거나 재학습하지 않는다. 같은 조건에서 초기 1시간을 관측한 뒤 275전이를 재생한다. 하루 NH4>4,DO>2 판정은 유지하고 recall/FPR, 농도 MAE, 날짜별 비용 MAE를 기록한다. e25의 287전이 정책 비용과 섞지 않는다.

NH4 입력의 자기 예측으로 인한 누적 오차를 확인하기 위해 reset1과 reset275를 평가한다. reset1은 실제 현재 상태를 매번 제공하는 진단이며 배포 가능한 하루 예측 성능이 아니다. 실제 공기량과 경로 재생은 관찰적 모델 검증이며 새 행동의 인과 효과나 RL 정책 성능을 입증하지 않는다.

## 사전 판정

부분 효과 재현: day_mse와 start_multi의 두 seed 각각 NH4 recall이 낮아지지 않고 FPR이 높아지지 않으며, 평균 날짜별 비용 MAE가 5% 이상 낮아야 한다. short_mse와의 비교도 그대로 보고하지만 이 기준의 비교군을 결과에 맞춰 바꾸지 않는다.

환경 교체 적격성: 기존 e12 `gates`를 그대로 호출한다. reference 및 single_mse 대비 오차, recall, precision, FPR, onset/지속시간 기준과 절대 NH4/DO recall80%, FPR10%를 모두 만족해야 한다. 부분 효과만 재현된 경우에는 환경을 교체하지 않는다. 두 판정을 구분하고 실패 원인을 보고한다.

## 출처와 재현

데이터는 Kaggle Avedore의 기존 aved_raw.csv,aved_5min.pkl. 출처 Mendeley DOI10.17632/xmbxhscgpr.1, CC BY4.0. 기존 e11~e14의 load/replay/score/gates를 직접 재사용한다. 원자료/전처리 및 체크포인트, 실제 읽는 소스의 해시를 evaluation.json에 저장하고 실행 후 불변을 확인한다. 미래 NH4/DO 정답을 바꿔도 reset275 예측이 동일한지 검사한다. 데이터 분할/전처리/모델 출처는 e14 README와 같다.

기존 venv, CPU1thread. 실행 명령:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /home/jio/fcsrl_src/.venv/bin/python Yoo/experiments/e26/evaluate_frozen.py \
 --out /home/jio/fcsrl_src/results/e26
```

결과를 보고 seed, 모델 가중치, 판정 기준을 변경하지 않는다. 정책 학습은 e25에서만 진행하며 e26이 통과하기 전 개선 환경과 정책 보정의 결합 실험은 실행하지 않는다.

## 실행 결과

[전체 표](results.md), [가중치 및 입력 해시와 seed별 평가](evaluation.json). 실행 커밋 `edeabfb`에서 새 학습 없이 저장 모델10개를 평가했다. 원자료, 모델 가중치, 소스 해시 불변과 미래 수질 정답 비참조 검사에 통과했다.

주 후보 day_mse는 start_multi 대비 NH4 탐지율 평균44.93%→49.25%로 증가했지만 FPR8.66%→17.75%, 날짜별 비용 MAE45.23→62.59로 악화했다. 비용 MAE가38.38% 증가해 부분 효과 재현 기준 및 기존 교체 적격성 기준 모두 실패했다. 같은 예산 short_mse에 대해서도 평균 탐지율은 낮고 FPR과 비용 MAE가 높았다.

따라서 e14의 내부 시간 구간 개선을 다른 시간 구간에서도 유효한 모델 개선으로 일반화할 수 없다. 시간 구간에 따른 성능 차이가 확인됐지만 온도, 운전 조건 또는 최적화 중 무엇이 원인인지 이번 평가만으로 분리하지 않았다. e25 정책 보정과 결합하거나 기존 환경을 교체하지 않는다. 이 미채택은 하루 길이 학습 일반의 무효를 입증하지 않는다.

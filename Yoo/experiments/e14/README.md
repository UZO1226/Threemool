# Yoo/e14 하루 예측과 조건별 거짓경보 제한

2026-11-06 사전 설정. e13에서는 짧은 학습 창의 평균 FPR이 낮아 제약 승수가 0이 되었지만, 위반 예측을 늘리는 FN proxy 항은 계속 남았다. fit 하루 평균도 취약한 운전 조건의 높은 FPR을 가렸다. 이번에는 FN proxy를 제거하고 실제 평가 길이의 자기 예측과 전체 및 조건별 제약을 나누어 점검한다. FCSRL 표현 학습, PPO, 보상 및 물리 위반 기준은 유지하며 대리환경만 시험한다. 팀 공식 비교가 아닌 개인 탐색이다.

## 사전 대조와 학습 예산

각 seed의 e12 `multi_mse` 마지막 모델에서 시작한다. 모든 군은 같은 날 batch 16, 같은 275개 전이를 300회 학습한다. 하루 초기 1시간까지의 관측 중 01:00 상태를 초기값으로 사용하고 01:05부터 23:55까지 예측한다. 중간 입력에 실제 농도를 사용하는 시점만 대조한다.

| 군 | 실측 상태 재시작 간격 | 거짓경보 제약 |
| --- | --- | --- |
| short_mse | 36스텝 | 없음 |
| day_mse | 275스텝 | 없음 |
| day_global | 275스텝 | 전체 평균 |
| day_group | 275스텝 | 전체 및 조건별 |

주 후보는 사전에 `day_group`으로 고정한다. seed 0, 1의 8개 학습이며 마지막 가중치를 사용한다. 36스텝 대조도 같은 하루의 모든 정답을 사용하므로 관측 수와 갱신 수가 같다. 학습의 상태 방문 분포와 연결된 gradient 길이는 의도적으로 다르다. 초기 모델 동결, e11 경로 모델, e12 single_mse도 평가한다.

## 손실과 제약

기본 목적은 fit 농도 표준편차로 정규화한 연속 MSE와 실측 상태 입력의 한 스텝 MSE다. 한 스텝 계수는 e13에서 fit gradient로 정한 seed별 8.4514486, 11.2222795를 그대로 모든 군에 사용한다. 학습률 0.0001, Adam weight decay 0.00001, gradient clip 1도 유지한다.

FN proxy는 모든 군에서 제거한다. 실제 비위반 정답에 대한 `softplus((예측 농도 - 한도) / fit 변화량 표준편차)`만 FP proxy로 사용한다. 따라서 lambda가 0이면 별도의 위반 판정 손실이 완전히 0이 되고 농도 MSE만 남는다. NH4 > 4, DO > 2라는 물리 기준을 바꾸거나 별도 위험 head를 만들지 않는다.

전체 제약은 실제 비위반 중 예측 위반의 비율(FPR)이다. 조건별 제약은 전체, 운전 단계 0/1/2, fit 수온 하위/중간/상위 1/3의 총 7집단에 대해 NH4와 DO를 각각 계산한다. 온도와 운전 단계의 교차 집단은 표본이 너무 적을 수 있어 이번 학습에는 쓰지 않는다. 같은 표본이 여러 제약에 속할 수 있다.

각 집단의 목표 FPR은 10%, 승수 초기값 0, EMA 0.9, 갱신률 1이며 음수만 0으로 투영하고 상한은 두지 않는다. 하루 batch 안에서 실제 비위반이 64개 미만이면 해당 집단의 proxy 항과 승수 갱신을 모두 건너뛴다. 처음 유효한 관측으로 EMA를 시작하고 이후 부재 시 EMA와 승수를 유지한다. 이 조건은 작은 분모의 불안정을 완화할 뿐 독립 날짜 수나 안전성의 충분조건은 아니다. 집단별 유효 갱신 수를 함께 보고한다.

모델 손실의 제약 항은 활성 집단의 `lambda * FP_proxy`를 전체 집단 수 및 변수 수로 나눠 더한다. day_global은 전체 집단 1개, day_group은 7개를 사용한다. 공통 계수 alpha는 별도의 고정 fit batch에서 모든 집단의 승수가 1이라고 가정한 FP proxy의 gradient norm이 하루 MSE의 0.5배가 되도록 한 번 정한다. 모델과 정규화는 공유하지만 각 제약의 승수는 독립이다. 실제 FPR로 승수를 갱신하고 proxy를 실제 FPR이라고 해석하지 않는다.

이는 e13의 고정 물리 기준용 시험 구현을 수정한 것이며, TFCO 전체 알고리즘의 재현이나 새로운 이론적 보장을 주장하지 않는다. 관련 원리와 구현의 차이는 [e13](../e13/README.md)에 기록했다. 하루 BPTT를 끝까지 연결하고 detach나 중간 실측 주입을 하지 않는다. short_mse의 36스텝 재시작과 한 스텝 MSE에만 실측 상태를 사용한다.

## 데이터와 평가 기준

e12 및 e13과 같은 fit 263일, 제외 1일, 내부 holdout 66일을 사용한다. 원자료 경로 신호와 전처리는 e11 그대로다. 내부 holdout은 반복 관찰한 탐색 자료이며 새 독립 확인으로 부르지 않는다. 모든 학습, 수온 경계와 계수 계산은 fit만 사용한다. 외부 val과 test는 사용하지 않는다. 데이터 및 초기 가중치 해시가 e13 기록과 같은지 검사한다.

275전이 평가의 비용은 NH4 및 DO 위반 스텝 수 합이며 기존 287전이 정책 한도 15와 비교하지 않는다. 평가 재시작 간격은 1, 36, 275다. 기존 e12의 강화된 gate를 유지한다. 각 seed에서 e11 및 single_mse 양쪽 대비 recall 악화 없음, precision 감소 2%p 이내, FPR 증가 1%p 이내, 한 스텝 및 장기 MAE 악화 5% 이내, 비용 및 최장 NH4 위반 길이 MAE 악화 5% 이내, NH4 시작 F1 악화 없음, 평균 NH4 recall 향상 5%p 이상을 요구한다. recall 80% 및 FPR 10% 절대 점검도 별도로 표시한다.

새 tradeoff gate는 시작 다중 스텝 모델과 같은 예산 day_mse 각각에 대해 두 seed 모두 NH4 recall과 FPR이 악화하지 않고, 평균 FPR이 각각 1%p 이상 낮아져야 통과한다. 낮은 recall을 낮은 FPR로 덮지 않는다. 주 후보를 사후 교체하지 않으며, 실패하면 가중치 탐색을 이어가지 않고 미해결 한계를 기록한다. 통과해도 외부 확인 전 실제 환경 교체와 RL 재학습을 자동 진행하지 않는다.

## 검증과 재현

`verify_day.py`는 275스텝 전체 gradient와 36스텝 재시작의 해석적 gradient를 확인한다. 전체 FPR이 10% 미만이어도 작은 집단의 FPR이 높으면 그 집단 승수만 증가하는지, 표본 부재 처리, lambda 0에서 판정 손실이 완전히 0인지, FP proxy가 실제 위반 표본을 밀어 올리지 않는지, recall 악화 후보 거부를 확인한다. 실제 실행은 창 대신 동일 날짜 및 전체 정답의 사용, 미래 실측 비참조, 저장 복원 일치와 모든 gradient 유한을 검사한다.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /home/jio/fcsrl_src/.venv/bin/python Yoo/experiments/e14/verify_day.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /home/jio/fcsrl_src/.venv/bin/python -u Yoo/experiments/e14/day_trial.py \
 --raw /home/jio/사물/Yoo/data/avedore/aved_raw.csv \
 --data /home/jio/사물/Yoo/data/avedore/aved_5min.pkl \
 --e11 /home/jio/fcsrl_src/results/e11/routing_trial \
 --e12 /home/jio/fcsrl_src/results/e12/trial \
 --out /home/jio/fcsrl_src/results/e14/trial
```

기존 Python 3.10.19 venv 및 torch 2.14.1+cu130를 유지하고 CPU 1 thread로 수행한다. 관측 날짜 추출 seed는 14000 + seed, alpha 계수 계산 seed는 14999다. 실행 커밋, 코드 및 입력 해시, 초기 모델과 관측 날짜 해시, 학습 로그 및 집단별 승수를 기록한다. 전체 모델과 로그는 Git 밖의 서버 경로에 저장한다. 비어 있지 않은 결과 폴더는 덮어쓰지 않는다.

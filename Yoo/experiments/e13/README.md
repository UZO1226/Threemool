# Yoo/e13 한 스텝 정확도 및 거짓경보 제한 대조

2026-11-06, 사전 설정. e12의 다중 스텝 모델을 출발점으로 한 스텝 정확도 보완과 거짓경보 제한의 효과를 분리한다. 팀 공식 비교가 아닌 개인 탐색이며, FCSRL 표현 학습과 PPO, 보상, 물리적 위반 기준(NH4 > 4, DO > 2)은 유지한다. 이번 시험의 대상은 대리환경 예측 모델이고 정책 학습은 아니다.

## 가설과 대조

e12 내부 holdout에서 FPR을 사후에 10%로 맞춰 비교한 NH4 recall은 기존 31.87%, 다중 스텝 37.67%, 다중 스텝 및 균형 위반 손실 38.12%다. 구분 능력 개선의 대부분이 다중 스텝 학습에서 나왔다는 탐색 근거다. 이는 같은 관찰 구간에서 판정 기준을 움직인 진단이고 배포 성능이나 독립 확인 결과가 아니다. 실제 농도 기준을 바꾸지 않는다.

각 seed의 e12 `multi_mse` 마지막 체크포인트에서 네 군을 같은 순서의 창으로 300회 추가 학습한다. 추가 학습만으로 달라지는 영향을 통제하기 위해 `multi`도 다시 학습한다. 주 후보는 사전에 `multi_anchor_rate`로 고정한다.

| 군 | 36스텝 자기 예측 MSE | 한 스텝 실측 입력 MSE | 비율 제약 손실 |
| --- | --- | --- | --- |
| multi | 사용 | 없음 | 없음 |
| multi_anchor | 사용 | 사용 | 없음 |
| multi_rate | 사용 | 없음 | 사용 |
| multi_anchor_rate | 사용 | 사용 | 사용 |

시작 모델 동결(`start_multi`), e11 경로 모델(`reference`), e12 한 스텝 모델(`single_mse`)도 평가한다. 학습군끼리 초기 모델, 관측 창, 갱신 수, optimizer 설정이 같다. 계산량은 추가 손실 때문에 다르다. 기존 모델을 덮어쓰지 않는다.

## 손실과 제약 구현

36스텝 MSE와 한 스텝 MSE는 같은 정답을 fit 농도 표준편차로 나누어 계산한다. 한 스텝 보완은 모델 복제가 아닌 실측 정답 오차 손실이다. 한 스텝 손실의 계수는 고정한 fit batch에서 gradient norm이 연속 MSE와 같도록 한 번 정한다.

비율 손실은 `z = (예측 농도 - 물리 한도) / fit 변화량 표준편차`에 대해 실제 위반 조건의 `softplus(-z)`와 실제 비위반 조건의 `softplus(z)`를 각각 평균한다. 모델 목적은 `MSE_multi + beta * MSE_one + alpha * mean(FN_proxy + lambda * FP_proxy)`다. 해당 군에서 사용하지 않는 항은 제외한다. alpha는 초기 lambda 1일 때 비율 손실 gradient norm이 연속 MSE의 0.5배가 되도록 fit batch에서 한 번 정한다. 두 제약의 lambda는 독립이다.

lambda는 실제 36스텝 창의 FPR과 목표 10%의 차이로 갱신한다. EMA 0.9, 갱신률 1, 초기값 1, 음수만 0으로 투영하며 상한은 없다. proxy를 실제 FPR로 오해해 갱신하지 않는다. 비위반 표본이 없는 변수의 갱신은 건너뛴다. 실제 위반이 없는 변수의 FN 손실은 0이다. 큰 gradient는 norm 1로 자르고 비유한 값은 실패로 처리한다.

이 방법은 [TFCO의 미분 가능한 proxy와 실제 비율을 구분하는 원리](https://github.com/google-research/tensorflow_constrained_optimization) 및 [Cotter 등 논문](https://arxiv.org/abs/1809.04198)을 참고한 고정 임계값용 PyTorch 시험 구현이다. 논문의 전체 proxy-Lagrangian 알고리즘, 확률적 모델 혼합, 수렴 보장을 재현했다고 주장하지 않는다. TFCO 저장소는 2026-04-19에 보관 상태가 되었으며 설치하거나 현재 venv를 바꾸지 않는다. 이전에 검토한 [implicit rate-constrained optimization](https://arxiv.org/abs/2107.10960)의 학습 가능한 임계값은 이 시험에 사용하지 않는다.

10%는 탐색 목표이지 안전 인증이나 팀 합의 기준이 아니다. 36스텝 창의 FPR 제약을 만족해도 275스텝 연속 예측의 FPR이 보장되지 않는다. 이 차이를 fit와 holdout의 재시작 간격 1, 36, 275 평가로 확인한다. 재가중 sigmoid는 보정된 위반 확률이 아니다.

## 데이터와 고정 설정

- e12와 같은 원자료 및 전처리, train 330일 중 fit 263일, 제외 1일, 내부 holdout 66일이다. 원자료 경로 신호는 e11 방식으로 입력한다. [e11 기록](../e11/README.md), [e12 기록](../e12/README.md).
- 내부 holdout은 반복 관찰한 탐색 구간이다. 새로운 독립 검증이라고 부르지 않는다. 계수 및 lambda 학습은 fit만 사용한다. 외부 val과 test는 이번 실행에 사용하지 않는다. 별도의 사후 농도 보정이나 임계값 보정은 수행하지 않는다.
- seed 0, 1, batch 128, 36전이, 300갱신, Adam lr 0.0001, weight decay 0.00001, clip 1, 마지막 가중치를 사용한다. 창 추출 seed는 13000 + seed, 계수 계산 seed는 13999다.
- 초기 1시간 이후 275전이를 평가한다. 비용은 두 제약 위반 스텝 수의 합이다. 기존 287전이 정책 비용 및 한도 15와 직접 비교하지 않는다.
- 기존 Python 3.10.19 venv, torch 2.14.1+cu130, CPU 1 thread. 데이터, 모델, 전체 로그는 Git에 넣지 않고 서버에 저장한다. 입력, 코드, 초기 및 최종 모델, 관측 창의 SHA256과 실행 커밋을 결과 JSON에 남긴다.

## 사전 진행 기준

e12의 강화된 gate를 그대로 적용한다. 각 seed에서 e11 기준 및 e12 single_mse 양쪽 대비 두 제약의 recall 악화 없음, precision 감소 2%p 이내, FPR 증가 1%p 이내, 한 스텝 및 연속 농도 MAE 악화 5% 이내를 요구한다. 비용 및 최장 위반 길이 MAE 악화 5% 이내, NH4 시작 F1 악화 없음, 평균 NH4 recall 향상 5%p 이상도 유지한다. 별도로 두 제약 모두 recall 80% 및 FPR 10%를 점검한다.

새 tradeoff gate는 시작 다중 스텝 모델 및 같은 예산의 multi 양쪽에 대해 각 seed에서 NH4 recall과 FPR이 나빠지지 않으면서, 평균 FPR이 각각 1%p 이상 감소하는 조건이다. 이 조건만 좋아지고 한 스텝 정확도나 DO가 악화한 후보도 채택하지 않는다. 주 후보 실패 후 다른 군을 사후 주 후보로 바꾸지 않는다. 모든 gate를 통과해도 외부 확인 전 실제 환경 모델 교체나 RL 재학습을 자동 진행하지 않는다.

## 실행

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /home/jio/fcsrl_src/.venv/bin/python Yoo/experiments/e13/verify.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /home/jio/fcsrl_src/.venv/bin/python -u Yoo/experiments/e13/experiment.py \
 --raw /home/jio/사물/Yoo/data/avedore/aved_raw.csv \
 --data /home/jio/사물/Yoo/data/avedore/aved_5min.pkl \
 --e11 /home/jio/fcsrl_src/results/e11/routing_trial \
 --e12 /home/jio/fcsrl_src/results/e12/trial \
 --out /home/jio/fcsrl_src/results/e13/trial
```

결과 폴더가 비어 있지 않으면 실행을 거부한다. 초기 점검은 분모, 누락 위반 및 거짓경보 gradient 방향, 표본 부재, lambda 방향, 36스텝 전체의 gradient, recall 악화 후보 거부를 확인한다. 실제 학습에서는 미래 실측 비참조, 모델 저장 복원 완전 일치, 모든 gradient 유한, 초기 모델 및 관측 창 일치를 확인한다.

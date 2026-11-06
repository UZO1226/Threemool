# Yoo/e13 한 스텝 정확도 및 거짓경보 제한 대조

**2026-11-06 19:55:53 KST, 학습 8개와 조건별 원인 진단 완료. 모든 후보 미채택.** e12의 다중 스텝 모델을 출발점으로 한 스텝 정확도 보완과 거짓경보 제한의 효과를 분리했다. 팀 공식 비교가 아닌 개인 탐색이며, FCSRL 표현 학습과 PPO, 보상, 물리적 위반 기준(NH4 > 4, DO > 2)은 유지했다. 이번 시험의 대상은 대리환경 예측 모델이고 정책 학습은 아니다. 아래 설정과 진행 기준은 학습 전 커밋으로 고정했다.

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

## 실행 결과

학습 커밋 `78f90be556a25cc0b7aedc903ebc81b345735993`, 19:51:23부터 19:53:23 KST까지 약 120초다. 조건별 진단 커밋 `d2087ce831e0427eb07a7126d6869568c2ccc909`, 19:55:53 KST에 완료했다. 진단의 첫 실행은 같은 이름의 과거 summarize 모듈을 불러오는 import 충돌로 학습 없이 중단됐다. 로컬 모듈 import 순서를 수정한 뒤 실행했으며 학습 가중치와 설정은 바꾸지 않았다.

내부 holdout 66일, 275전이, seed 2개 평균이다. 같은 예산의 multi와 비교해야 추가 학습 효과를 보완 항의 효과로 혼동하지 않는다.

| 군 | NH4 recall | NH4 FPR | 한 스텝 NH4 MAE | 날짜별 비용 MAE |
| --- | --- | --- | --- | --- |
| 시작 다중 스텝 모델 동결 | 46.23% | 14.32% | 0.293 | 40.35 |
| 다중 스텝 추가 학습 | 51.21% | 15.79% | 0.298 | 39.88 |
| 한 스텝 보완 추가 | 49.79% | 15.40% | 0.294 | 40.20 |
| 비율 제약 추가 | 62.21% | 21.53% | 0.300 | 49.04 |
| 두 보완 동시 적용 | 60.04% | 20.86% | 0.295 | 49.61 |

한 스텝 보완은 같은 예산 multi보다 한 스텝 오차를 조금 낮췄지만, 시작 모델의 0.293이나 e11 기준 0.276으로 회복하지 못했다. 비율 제약 추가는 탐지율과 거짓경보를 함께 늘렸다. 주 후보와 나머지 후보 모두 사전 gate를 통과하지 못했으며 새 환경 모델로 채택하지 않았다. 독립 검증이나 실제 제어 성능 개선으로 주장하지 않는다.

## 왜 거짓경보 제약이 작동하지 않았는가

주 후보의 마지막 36스텝 학습 창에서 NH4 FPR은 seed 0, 1 각각 2.71%, 3.70%로 목표 10%보다 작았다. 두 seed와 두 제약 모두 최종 lambda가 0이 되었다. 거짓경보 proxy의 계수는 0으로 내려가고 누락 위반을 줄이는 FN proxy 항은 남는 구조다. 학습 제약을 계산한 범위에서는 갱신 방향이 맞지만, 이를 내부 holdout의 하루 제약으로 해석하면 안 된다.

실제 fit 하루 연속 FPR도 평균 7.51%로 10%보다 낮았다. 따라서 단지 lambda 갱신 길이를 하루로 바꾸기만 하면 해결된다는 결론도 근거가 부족하다. 같은 모델의 내부 holdout 하루 FPR은 20.86%였다. 학습 평균이 취약한 조건과 평가 조건의 차이를 가리고 있다.

| 주 후보의 조건 | fit NH4 FPR | 내부 holdout NH4 FPR |
| --- | --- | --- |
| 전체 | 7.51% | 20.86% |
| fit 수온 하위 1/3 구간 | 13.95% | 34.23% |
| 운전 단계 0 | 30.81% | 47.43% |
| 운전 단계 2 | 10.01% | 26.56% |

수온 구간은 fit에서 13.1691, 17.9184도로 나눴다. 온도 3구간 x 운전 단계 3구간의 실제 비위반 구성 비율을 holdout에 맞추면 fit FPR은 7.51%에서 9.59%가 된다. 실제 holdout 20.86%와는 여전히 11.27%p 차이가 남는다. 이 거친 조건 분할에서는 단순 구성 비율 차이가 전체 간격의 작은 부분만 설명한다. 같은 구간 안의 시점, 유입 부하, 상태와 관측되지 않은 변수 차이는 아직 분리하지 못했다. 실제 인과 원인이나 특정 누락 변수를 확정하지 않는다.

주 후보 NH4 편향은 예측 첫 36스텝 +0.350에서 145~275스텝 +1.076으로 커졌다. 실제 입력을 쓰는 한 스텝 오차와 자기 예측을 반복한 오차가 여전히 다르다. 같은 FPR 10%에서 비교한 사후 NH4 recall은 multi 39.46%, 주 후보 40.65%로, 고정 기준 recall의 큰 상승 대부분을 구분 능력 개선으로 해석할 수 없다. 이 ROC 비교는 동일 holdout에서 임계값을 움직인 진단이고 실제 판정 기준은 변경하지 않았다.

## 판정과 다음 보완 대상

이번 시험은 학습 평균 FPR 제약이 실제 검증 FPR을 제어하지 못하는 실패를 확인했다. 다음 설계에서는 fit 안에서도 이미 10%를 넘는 운전 조건을 분리해 제약을 평가하고, 실제 사용 길이의 연속 예측 오차를 함께 다뤄야 한다. 온도와 단계별 단순 재가중만으로 해결된다고 가정하지 않는다. 적은 표본의 집단에 지나치게 큰 가중치를 주지 않도록 표본 수와 날짜별 편차도 확인해야 한다. 반복 관찰한 holdout을 새 독립 확인 자료로 재분류하지 않는다.

FCSRL + PPO와 기존 환경 모델은 유지했다. 추가 후보를 자동 탐색하거나 외부 val, test, 정책 학습을 진행하지 않았다. 위반 기준, recall 유지 조건과 FPR 허용치를 결과에 맞춰 완화하지 않았다. 이 시험을 FCSRL 확장 성공 성과로 세지 않는다.

## 산출물과 재생성

[전체 표 및 그림](results.md), [학습 요약](trial_summary.json), [조건별 편향과 ROC](diagnostics.json), [학습 및 검증 조건 차이](condition_summary.json). 서버 가중치 및 전체 로그는 `/home/jio/fcsrl_src/results/e13`에 있다. 학습 손실 계수는 seed 0, 1에서 anchor 8.4514, 11.2223, rate 0.022168, 0.036881이며 fit gradient norm으로만 정했다.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /home/jio/fcsrl_src/.venv/bin/python Yoo/experiments/e13/audit_conditions.py \
 --raw /home/jio/사물/Yoo/data/avedore/aved_raw.csv \
 --data /home/jio/사물/Yoo/data/avedore/aved_5min.pkl \
 --trial /home/jio/fcsrl_src/results/e13/trial \
 --out /home/jio/fcsrl_src/results/e13/conditions.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /home/jio/fcsrl_src/.venv/bin/python Yoo/experiments/e13/summarize.py \
 --trial /home/jio/fcsrl_src/results/e13/trial \
 --data /home/jio/사물/Yoo/data/avedore/aved_5min.pkl \
 --conditions /home/jio/fcsrl_src/results/e13/conditions.json \
 --out Yoo/experiments/e13
```

재현 시 학습과 조건 진단은 새 출력 경로를 사용한다. 집계는 저장된 모델 및 예측 해시와 전체 지표를 재확인한다. 조건 진단도 fit 전체 재생 지표가 원래 학습 보고서와 정확히 같은지 확인했다. 모든 점검을 통과했다.

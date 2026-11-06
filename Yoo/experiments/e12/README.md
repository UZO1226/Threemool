# Yoo/e12 위반 누락과 연속 예측 목적의 수정

**2026-11-06 19:30:49 KST, 학습 8개와 원인 분리 완료.** 주 후보는 탐지율을 높였으나 거짓경보 증가 등으로 강화한 기준을 통과하지 못했다. 새 모델의 외부 val 평가와 RL 재학습은 하지 않았다.

## 목적과 사전 설정

e11은 비용 MAE가 줄어도 NH4 위반 recall이 악화했다. 평균 오차를 줄이는 목표와 위반을 놓치지 않는 목표의 차이, 한 스텝 학습과 연속 예측 사용의 차이를 분리해서 고친다. FCSRL 원본 표현 학습, PPO, 보상과 제약 정의는 유지하며 대리 모델 시험만 수행한다. 팀 공식 비교가 아닌 개인 탐색이고 RL 개선 성과로 세지 않는다.

e11의 경로 입력 모델 seed 0, 1에서 시작해 아래 4군을 각각 추가 학습한다. 동결된 시작 모델도 별도 기준으로 평가한다.

| 군 | 상태 입력 | 손실 |
| --- | --- | --- |
| single_mse | 매 스텝 실제 상태 | 농도 MSE |
| single_boundary | 매 스텝 실제 상태 | 농도 MSE 및 균형 위반 BCE |
| multi_mse | 36스텝 동안 자기 예측 | 농도 MSE |
| multi_boundary | 36스텝 동안 자기 예측 | 농도 MSE 및 균형 위반 BCE |

주 후보는 사전에 `multi_boundary`로 정한다. 나머지는 원인 분리를 위한 대조이며 유리한 군을 사후 선택하지 않는다. 다중 스텝 원리는 [Benechehab 등](https://arxiv.org/abs/2402.03146), 이진 손실은 [PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)를 참고한다. 논문 전체 방법의 재현이나 보정된 위반 확률 모델이라고 주장하지 않는다.

## 손실과 예산

263일 fit, 1일 제외, 66일 내부 holdout은 e10과 e11 그대로다. 내부 holdout을 반복 관찰한 탐색이라는 한계를 명시한다. 원래 val과 test는 학습이나 이 시험의 설정 결정에 사용하지 않는다. 입력과 가중치는 e11 경로 모델을 유지한다.

- 네 군 모두 같은 날 안에서 시작 시점 12부터 뽑은 36전이(3시간) 창을 batch 128, 300갱신 학습한다. 두 seed의 군 간 초기 모델과 창 순서가 같다. 한 스텝 군도 같은 36개 정답을 사용하므로 관측 수와 갱신 수가 같다.
- MSE는 예측 NH4 및 DO와 실측의 차이를 fit 농도 표준편차로 나눠 제곱한다. e11 추가 학습 없이 유지한 모델과, 같은 새 손실 척도로 추가 학습한 single_mse를 둘 다 기준으로 삼는다.
- 위반 BCE의 label은 실측 NH4 > 4, DO > 2다. logit은 `(예측 농도 - 한도) / fit 농도 변화량 표준편차`다. 별도 분류 head를 만들어 농도와 판정이 모순되게 하지 않는다.
- 양성 가중치 `1/(2p)`, 음성 가중치 `1/(2(1-p))`를 쓰며 p는 fit의 위반 비율이다. 실제 위반 누락과 거짓경보를 모두 손실에 포함한다. 재가중한 sigmoid 출력을 보정된 실제 확률로 해석하지 않는다.
- BCE 계수는 seed별 사전에 고정한 fit batch에서 한 스텝 MSE gradient norm의 0.5배가 되도록 계산하고 고정한다. 동일 seed의 두 BCE 군은 같은 계수를 쓴다. holdout으로 계수를 고르지 않는다.
- Adam lr 0.0001, weight decay 0.00001, gradient clip 1, 마지막 가중치 사용. 기존 venv, CPU 1 thread, 패키지 변경 없음.

평가는 초기 1시간 뒤 275전이, 재시작 간격 1, 36, 275다. 비용은 NH4 > 4 및 DO > 2의 스텝 수 합이지만 기존 287전이 정책 한도와 직접 비교하지 않는다.

## 수정한 진행 기준

각 seed에서 주 후보를 시작 모델 및 single_mse 양쪽과 비교한다. 두 제약 모두 recall 악화 없음, precision 감소 2%p 이내, 거짓양성률(FPR) 증가 1%p 이내, 농도 MAE와 한 스텝 MAE 악화 5% 이내를 요구한다. 비용 MAE와 NH4 최장 관측 위반 길이 MAE도 악화 5% 이내, NH4 시작 F1 악화 없음을 요구한다. NH4 recall은 두 기준 각각에 대해 seed 평균 5%p 이상 높아져야 한다.

위 조건은 외부 확인을 진행할 기준이다. 별도로 두 seed의 두 제약 모두 recall 80% 이상 및 FPR 10% 이하인지 표시한다. 이 수치는 시험 후보의 최소 점검 기준이며 실제 공정 안전 인증이나 팀 합의 기준이 아니다. 모든 단계를 통과해도 이 시험만으로 기존 환경 모델을 교체하지 않는다. 위반을 항상 예측하는 모델은 FPR과 precision 조건에서 제외한다.

## 악화 원인의 분리

학습 대조를 끝낸 뒤 e11에서 경로 입력 추가 전후의 DO 및 NH4 모델을 2 x 2로 교차 결합해 재생한다. 각 부분의 입력 정규화와 경로 입력 유무도 해당 원래 모델 것을 유지한다. 같은 두 부분을 결합하면 원래 저장 모델과 일치하는지 검사한다. 이는 모델 부품 교체 진단이며 실제 공정의 인과 효과는 아니다.

NH4 현재 상태 또는 DO 현재 및 다음 상태를 실측값으로 바꾸는 진단도 수행한다. 해당 결과는 배포 가능한 예측 성능으로 세지 않는다. 비용, recall, FPR, 농도 편향 및 0 농도 발생 비율을 함께 보고한다. 상호작용이 있으므로 한 부품의 영향만으로 실제 공정의 단독 원인을 확정하지 않는다.

## 재현

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /home/jio/fcsrl_src/.venv/bin/python -u Yoo/experiments/e12/trial.py \
 --raw /home/jio/사물/Yoo/data/avedore/aved_raw.csv \
 --data /home/jio/사물/Yoo/data/avedore/aved_5min.pkl \
 --pretrained /home/jio/fcsrl_src/results/e11/routing_trial \
 --out /home/jio/fcsrl_src/results/e12/trial
```

실행 전 `--preflight-only`로 누락 위반에는 농도를 높이는 gradient, 거짓경보에는 낮추는 gradient가 나오는지 확인한다. 실제 실행은 창 및 초기 모델 해시, 모든 gradient 유한, 저장 복원 일치를 확인한다. 실행 커밋과 해시, 계수, 데이터 비율, 시간은 결과 JSON에 남긴다. 전체 모델과 로그는 서버 결과 경로에 둔다. 입력 자료와 전처리는 [e02](../e02/README.md), e11 경로 보완을 따른다.

## 결과

학습 실행 커밋 `58c8094a37d4b37480b80e18ca4d78114dd05001`, 19:28:12부터 19:30:16 KST까지 약 124초다. 원인 분리 실행 커밋 `59ab1d9355fd19283ab7d0ce2f0da22c9cfeb119`, 19:30:49 KST에 완료했다. fit 위반 비율은 NH4 13.01%, DO 2.76%다. BCE 계수는 seed 0에서 0.028496, seed 1에서 0.039581이며 holdout으로 조정하지 않았다.

다음은 내부 holdout 66일, 275전이 연속 예측의 seed 2개 평균이다. recall과 FPR은 분모가 각각 실제 위반과 실제 비위반이므로 서로 다른 비율이다.

| 군 | NH4 recall | NH4 FPR | 날짜별 비용 MAE |
| --- | --- | --- | --- |
| 시작 모델 동결 | 30.12% | 8.57% | 42.42 |
| 한 스텝 MSE 추가 학습 | 28.25% | 8.62% | 43.17 |
| 한 스텝 MSE 및 위반 손실 | 44.38% | 19.51% | 57.53 |
| 연속 MSE | 46.23% | 14.32% | 40.35 |
| 연속 MSE 및 위반 손실 | 51.85% | 16.67% | 42.70 |

위반 손실만 추가하면 탐지가 늘지만 거짓경보와 비용 오차도 커졌다. 연속 MSE는 NH4와 DO의 장기 농도 오차를 낮추고 탐지를 늘렸지만 FPR이 증가했다. 두 수정을 함께 적용하면 시작 모델보다 NH4 recall이 21.73%p 높아지지만 FPR도 8.10%p 높아져 미채택이다. seed 1의 DO recall 저하, 일부 한 스텝 NH4 오차 증가도 남았다. 주 후보를 사전에 고정했으며 다른 군의 유리한 지표만 골라 채택하지 않았다.

강화한 판정은 비용 오차가 좋아도 recall이 나쁘면 통과시키지 않는다. 항상 위반을 예측하는 후보도 FPR 조건으로 탈락시킨다. 개선된 검사는 구현했지만 실제 예측 품질 문제가 완전히 해결됐다고 보고하지 않는다. 모든 상세 지표와 실패 항목은 [결과 표](results.md), [학습 요약](trial_summary.json)에 있다.

## 어느 분기가 앞선 악화에 크게 작용했는가

기존 val 43일에서 e11의 반응조 경로 입력 추가 전후 분기를 교차 결합했다. 초기 상태, 실제 공기량과 운전 경로는 같고 각 분기의 가중치, 정규화 및 경로 입력 유무만 바뀐다.

| DO 분기 | NH4 분기 | NH4 recall 평균 |
| --- | --- | --- |
| 기존 | 기존 | 36.54% |
| 새 | 기존 | 31.26% |
| 기존 | 새 | 20.11% |
| 새 | 새 | 19.41% |

**이 검증 구간에서 탐지율 하락은 NH4 예측 분기의 변경에 더 크게 연결된다.** NH4 분기만 교체해도 대부분의 하락이 나타났다. 두 교체 순서를 평균하는 기여 분해에서는 총 하락 중 약 83%가 NH4 쪽, 17%가 DO 쪽으로 배분된다. 이는 상호작용을 순서 평균으로 나눈 모델 내부 진단이지 실제 공정 인과 비율이 아니다. 내부 holdout에서는 DO 쪽의 평균 기여가 더 컸으므로 모든 운전 조건에 같은 비율을 일반화하지 않는다.

추가로 e11 새 모델의 NH4 현재값을 매번 실측으로 교체하면 val recall이 19.41%에서 94.22%로 회복되고 FPR은 1.72%에서 1.08%다. 이는 NH4 예측값이 다음 입력으로 다시 들어가는 누적 경로를 끊었을 때의 변화다. DO 현재 및 다음값만 실측으로 교체하면 recall은 77.88%가 되지만 FPR도 36.55%로 커졌다. DO 모델만 정확해지면 해결된다는 결론은 맞지 않으며 NH4 모델의 DO 입력 반응과 자기 예측 누적을 함께 봐야 한다.

새 e12 주 후보도 내부 holdout에서 NH4 현재값을 실측으로 주면 recall 87.90%, FPR 1.93%지만 자기 예측을 이어 쓰면 recall 51.85%, FPR 16.67%다. 학습 목적을 바꾸어도 장기 결합 오차가 남았다는 근거다. 실측 주입은 배포 가능한 정책이나 모델 성능으로 보고하지 않는다. [전체 교차 및 실측 주입 결과](attribution_summary.json)에 seed별 수치를 보존했다.

이 결과로 확인한 것은 앞선 val 악화의 주요 변경 부위와 누적 경로다. NH4 분기의 입력, 가중치, 정규화 변화 중 하나만 단독 원인으로 확정한 것은 아니며, 실제 유입 NH4 부하 등 관측되지 않은 변수의 효과도 이번 시험으로 규명하지 않았다.

## 검증과 산출물

초기 가중치 및 창 순서 해시 일치, 모든 loss 및 gradient 유한, 저장 후 출력 완전 일치를 확인했다. `verify.py`는 36스텝 전체의 gradient가 연결되는지 해석 가능한 누적 모델로 검증했다. 낮은 recall을 비용 오차 감소로 덮는 후보 및 항상 위반을 예측하는 후보가 탈락하는지도 확인했다. 교차 진단에서 같은 두 분기를 합친 결과는 원래 저장 모델과 정확히 같았다.

전체 가중치와 로그는 `/home/jio/fcsrl_src/results/e12`에 있다. 기존 모델을 덮어쓰거나 FCSRL 및 PPO를 재학습하지 않았다. 이번 시도는 하나의 고정 설정이며 성능이 좋은 계수를 찾기 위한 추가 탐색은 하지 않았다.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /home/jio/fcsrl_src/.venv/bin/python Yoo/experiments/e12/verify.py

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /home/jio/fcsrl_src/.venv/bin/python Yoo/experiments/e12/attribute.py \
 --raw /home/jio/사물/Yoo/data/avedore/aved_raw.csv \
 --data /home/jio/사물/Yoo/data/avedore/aved_5min.pkl \
 --pretrained /home/jio/fcsrl_src/results/e11/routing_trial \
 --trial /home/jio/fcsrl_src/results/e12/trial \
 --out /home/jio/fcsrl_src/results/e12/attribution.json

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /home/jio/fcsrl_src/.venv/bin/python Yoo/experiments/e12/summarize.py \
 --trial /home/jio/fcsrl_src/results/e12/trial/results.json \
 --diagnosis /home/jio/fcsrl_src/results/e12/attribution.json \
 --out Yoo/experiments/e12
```

결과 경로가 이미 존재하면 학습 및 진단을 거부하므로 재현 시 새 경로를 사용한다.

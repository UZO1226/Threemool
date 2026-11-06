# Yoo/e12 위반 누락과 연속 예측 목적의 수정

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

# e42: 정상/위반 조건부 NH4 예측기 분리

## 질문과 선택 이유

e41의 자기 예측 입력 증강은 정상 NH4 오차를 줄였지만 실제 위반 구간 농도를 더 낮게 예측했다. 과거 e12/e13의 위반 손실/가중치 추가도 탐지율과 FPR을 함께 늘렸으므로 이번에는 손실 계수를 바꾸지 않는다. 하나의 회귀기가 두 구간을 모두 맞히는 구조를 정상/위반 조건부 회귀기와 입력 기반 혼합기로 나눈다. 상승 반응은 별도 평가하며 상승 여부를 추론 입력이나 미래 정답 선택기로 쓰지 않는다.

[Jacobs 등, Adaptive Mixtures of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.html)의 분담 예측 원리를 참고한다. 원 논문의 학습 알고리즘을 재현한 것이 아니라 구간을 관측 정답으로 정의한 GBM 대조다. [scikit-learn 이진 GBM](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)의 log loss와 단조 제약을 사용한다. 설치된 sklearn 1.7.2 API를 확인했고 패키지는 바꾸지 않는다.

FCSRL 표현 학습, PPO와 제어 목표는 그대로다. DARROW에서 환경 모델의 다음 NH4 예측만 대조하며, 새 알고리즘 성과나 공기 절감 성과로 세지 않는다.

## 실행 전에 고정한 구조

입력은 e41의 상태/외생입력/공기량/수질 이력 44개다. 같은 fit140일, 같은 실측/합성 1:1 자료, 같은 구성원별 80% 행 ID를 사용한다. 합성 정답도 e41과 같은 실측 경로 복귀용이며 변경 상태에서 측정한 물리 정답은 아니다.

- 혼합기 g(x): 다음 실제 NH4>4 여부를 균형 가중치 없는 이진 log loss로 학습한다.
- 정상 예측기 mu0(x): 다음 실제 NH4가 4 이하인 행만으로 다음 농도의 MSE를 학습한다.
- 위반 예측기 mu1(x): 다음 실제 NH4가 4 초과인 행만으로 다음 농도의 MSE를 학습한다.
- 다음 농도: `(1-g) * clip(mu0,0,4) + g * clip(mu1,4,fit_hi)`.

실제 정답은 학습 구간 분할과 손실에만 쓰고 평가 때는 x만 사용한다. 위험 여부를 실제 정답으로 골라 예측하지 않는다. 전문가 출력의 구간 제한은 조건부 평균의 지지 범위를 반영한다. 이후 판정은 기존과 같은 예측 NH4>4이며 g>0.5를 별도 비용으로 사용하지 않는다. g의 확률 정확도는 보장하지 않으며 AUC/Brier/실제 정상과 위반에서의 평균을 진단으로 공개한다.

공기 단조 제약은 세 예측기에 모두 유지한다. 고정된 다른 입력에서 공기가 증가하면 g, mu0, mu1은 비증가하고 mu1>=mu0이므로 혼합 농도도 비증가다. 실제 학습 모델에서 검산한다. 이것은 기존 모델 제약의 유지이지 관측 자료에서 공기량의 인과 효과가 확인됐다는 뜻이 아니다.

조건부 평균을 합치는 구조여서 위험 확률이 낮으면 높은 농도 예측이 여전히 평균에 묻힐 수 있다. 이 구조가 희귀 위반을 자동으로 해결한다고 가정하지 않는다. 위험 모델을 따로 학습해도 실제 입력으로 그 구간을 구분하지 못하면 실패할 수 있다.

## 대조군과 평가

| 군 | 역할 | 구성원별 트리 반복 수 |
| --- | --- | --- |
| control | e41 실측 재학습 모델 동결, 기존 성능 유지 기준 | 기존 T |
| augmented | e41 단일 증강 모델 동결, 정상/FPR 개선 보존 기준 | 기존 T |
| direct | 같은 증강 자료에서 다음 절대 농도 단일 회귀 | T |
| capacity | 같은 절대 농도 단일 회귀, 총 트리 수 대조 | 3T |
| split | 혼합기와 정상/위반 회귀기 | 각각 T, 합계 3T |

T는 e41 각 구성원의 기존 반복 수다. 조기 종료와 설정 탐색 없이 마지막 모델을 사용한다. 새 학습기는 구성원별 5개, 총 25개다. split과 capacity의 합계 트리 수는 같지만 전문가별 표본 수가 달라 전체 행-트리 연산량이나 벽시계 시간이 동일하지 않다. 두 값을 함께 공개한다. 단일 절대 농도 direct가 있어 변화량에서 절대값으로 정답 표현을 바꾼 효과와 분리 구조 효과를 구분한다.

NO3/DO 가중치와 clipping, recorded 공기/외생입력은 e41 그대로다. fit140, holdout48, val40, test41 미사용을 유지한다. 주 평가 val40 cold95, 별도 holdout48, warm91, 한 스텝 실측 입력과 e37 공기 변화132구간을 공개한다. 실제/예측 NH4>4와 maxDO>2를 바꾸지 않으며 새 보정이나 임계값 탐색은 없다.

주 비교는 split 대 direct/capacity다. 기존 성능 유지도 split 대 control에서 확인한다. e41의 16개 항목(NH4 MAE/초과 횟수/강도, NH4/DO 탐지/FPR, DO1/2/3 MAE, 양 방향 반응 오차)을 그대로 보고한다. 추가로 정상 MAE/FPR이 augmented보다 나쁘지 않은지, 위반 MAE가 control보다 나쁘지 않은지 확인한다. 0 허용폭 점추정치 선별이지 실용적 동등성 검정은 아니다. 날짜 7일 블록 bootstrap 2000회로 차이의 불확실성을 보고한다. 5구성원은 독립 RL seed가 아니다.

이미 관찰한 개발 검증을 재사용하므로 독립 확증이 아니다. 추가 학습, 구간 경계, 가중치나 사후 혼합 비율 선택은 하지 않는다. 실험 결과로 환경을 자동 교체하거나 RL을 시작하지 않는다. 공기량이 고정된 조건부 재생이므로 제어 효율은 별도 문제다.

## 실행

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/jio/fcsrl_src/.venv/bin/python Yoo/experiments/e42/preflight.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE \
/home/jio/fcsrl_src/.venv/bin/python Yoo/experiments/e42/run.py \
  --prepared /home/jio/fcsrl_src/results/e30/prepared \
  --source /home/jio/fcsrl_src/results/e41/main \
  --events /home/jio/fcsrl_src/results/e37/main \
  --out /home/jio/fcsrl_src/results/e42/main
```

기존 Python3.10.19/numpy1.23.5/sklearn1.7.2 환경, CPU1스레드다. 새 output 폴더만 허용한다. 모델/대용량 예측은 runtime에 두고 작은 결과/코드/설정/해시를 Git으로 관리한다. 팀 공통 조건 변경이 아닌 개인 대조다.

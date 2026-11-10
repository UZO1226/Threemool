# e40: DARROW 연속 NH4 점수 판정 대조

## 실행 전 고정한 질문

e39의 isotonic 확률 동점 때문에 선택할 수 없었던 판정 경계를 원래 연속 예측값으로 선택하면, 검증에서도 탐지율과 거짓양성률의 상충이 줄어드는가?

변경은 위험 판정 경계를 고르는 입력 한 가지다. e39의 저장된 기존/이력 모델 예측을 재사용한다. 모델 140일 학습, 48일 보정, 40일 검증 분할, 기록 행동과 외생입력, 예측 농도, NH4>4 실제 위반 정의, DO, 비용 한도와 공기량은 유지한다. 재학습과 새 RL은 없다. 과거 Avedore/BSM1은 사용하지 않는다.

보정 날짜에서만 두 규칙의 경계를 구한다. 주 조건은 탐지율 80% 이상 중 거짓양성률이 가장 낮은 경계, 보조 조건은 거짓양성률 10% 이하 중 탐지율이 가장 높은 경계다. 같은 탐지 수에서는 오경보가 적은 경계를 고른다. 원 점수의 동점은 함께 판정한다. 경보 없음은 별도 규칙으로 저장하며, 확률 전용 1 초과 상수를 농도 경계로 사용하지 않는다. 경계와 해시를 저장한 다음 검증 예측을 읽는다.

각 모델에서 연속값과 e39 확률 판정을 비교하고, 별도로 연속값을 쓰는 기존/이력 모델을 비교한다. 점추정치의 탐지율이 낮지 않고 거짓양성률이 높지 않으며 하나 이상 좋아야 두 지표 동시 우위다. 목표 80%/10%의 검증 달성 여부도 따로 공개한다. 날짜 7일 블록 bootstrap 2000회로 차이의 불확실성을 보고한다. 모델 구성원 5개는 독립 정책 seed가 아니다.

이번 설계는 이미 본 e39 결과에서 나온 후속 대조다. 경계를 보정 자료에서만 고정해도 반복 사용한 검증 40일이 독립 최종 시험이 되지는 않는다. test41일은 사용하지 않는다. 검증 결과로 경계를 다시 고르거나 두 프로필 중 유리한 것만 선택하지 않는다. 이 대조로 상충이 남으면 경계 탐색을 반복하지 않고 NH4 반응 예측 문제로 돌아간다.

## 실행과 재현

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/jio/fcsrl_src/.venv/bin/python Yoo/experiments/e40/test_threshold.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/jio/fcsrl_src/.venv/bin/python Yoo/experiments/e40/run.py \
  --source /home/jio/fcsrl_src/results/e39/main \
  --out /home/jio/fcsrl_src/results/e40/main
```

기존 Python 3.10.19/numpy 1.23.5 환경에서 CPU 1스레드로 실행한다. 모델 추가 학습은 없고 저장된 40일 x 5구성원 x 95전이와 e39 공기 변화 구간을 평가한다. 원 데이터는 Kaggle dynamic-modeling-of-wastewater-treatment-process의 DARROW이며 e30 준비 자료와 e39 분할을 그대로 사용한다. 조건부 기록 재생이지 공기량의 인과 효과나 온라인 예측 성능이 아니다. 팀 공통 평가 조건 변경이 아닌 개인 진단이다.

# e41: 자기 예측 상태를 학습에 추가하는 NH4 전이 대조

## 목적과 기존 실험에서 남은 문제

DARROW e38에서 수질 이력은 NH4 오차를 낮췄으나 공기 감소 뒤 상승 반응과 위반 탐지는 충분하지 않았다. e39/e40에서 확률 보정과 연속 점수 경계를 시험했지만 탐지율/FPR 동시 개선을 얻지 못했다. 이제 경계를 고정하고 전이 학습 입력을 바꾼다. 공기량/NH4/DO 동시 개선이라는 제어 목표를 위한 환경 점검이며 FCSRL 자체의 신규 확장 성과가 아니다.

코드에서 확인한 사실은 기존 NH4 GBM이 실측 상태와 실측 과거 수질에서 다음 변화량을 학습한다는 점이다. 재귀 사용에서는 예측 상태와 예측 이력을 받는다. 이 차이가 상승 누락의 단독 원인이라고 확인된 것은 아니다. 이번 시험은 그 차이를 줄이는 학습이 도움이 되는지 확인한다.

[Venkatraman 등, AAAI 2015 Data as Demonstrator](https://ojs.aaai.org/index.php/AAAI/article/view/9590)는 모델이 만든 입력을 다음 실측 상태와 연결해 학습 자료에 추가한다. 여기서는 그 원리를 GBM에 단일 반복으로 적용한다. 원 논문의 전체 알고리즘이나 이론 보장을 재현했다고 주장하지 않는다. 기록 행동을 따라간 예측을 실측 경로로 복귀시키는 합성 정답이며, 달라진 상태에서 측정한 물리적 정답이나 새로운 행동의 인과 효과는 아니다.

과거 다른 데이터의 e09/e12/e13에서도 다중 스텝 학습이 평균 오차만 줄이거나 탐지율과 FPR의 상충을 남겼다. 이를 DARROW의 성능 근거로 쓰지 않고, 평균 오차 외에 탐지/오경보/DO를 함께 봐야 한다는 설계 교훈으로 반영한다.

## 실행 전에 고정한 대조

| 항목 | 설정 |
| --- | --- |
| 자료 | Kaggle dynamic-modeling-of-wastewater-treatment-process, DARROW e30 준비 자료 |
| 분할 | e39와 같은 학습 140일, 별도 holdout 48일, 개발 검증 40일. test41일 미사용 |
| 기준 | e39 수질 이력 모델과 44개 입력 유지 |
| 합성 입력 | 학습 날짜의 기존 모델로 3시간(12전이) 예측. 구간 시작에만 실측과 과거 이력을 주고 이후 자기 예측 사용 |
| 창 | 매일 t=0,12,...84의 비중첩 구간, 마지막은 11전이 |
| 합성 정답 | 같은 시각의 다음 실측 NH4에서 현재 예측 NH4를 뺀 값 |
| reference | e39 이력 모델 동결 |
| control | 실측 입력/정답을 두 번 넣어 NH4 모델 재학습 |
| augmented | 실측과 합성 입력/정답을 1:1로 넣어 NH4 모델 재학습 |
| 학습량 대조 | 두 새 군의 구성원별 동일 80% 행 ID, 같은 총 행 수와 모델 구조, 난수, 반복 수 |
| 반복 수 | 기존 각 NH4 모델의 n_iter_를 사전 고정. 두 새 군 모두 조기 종료 없음 |
| 변경 모델 | 군별 NH4 GBM 5개씩, 총 10개. 나머지 NO3/DO 모델 20개와 clipping 고정 |
| 판정 | 예측 NH4>4, max(DO)>2. 보정기/경계 최적화 추가 없음 |
| 실행 환경 | 기존 Python 3.10.19, numpy 1.23.5, sklearn 1.7.2, CPU 1스레드 |

실측 기준 군의 자료를 복제해 총 표본 수와 트리 반복 수를 맞춘다. 기존 동결 모델과는 조기 종료/재학습 설정 차이가 있으므로 주 효과는 augmented 대 control이다. reference 대비 유지 여부도 별도로 확인한다. 벽시계 시간까지 동일한 것은 아니므로 실제 시간을 보고한다. 합성 자료는 같은 학습 날짜로 학습된 교사 모델에서 만들며 독립 표본으로 세지 않는다.

주 평가는 검증 40일의 95전이 재귀 예측이다. holdout48, 한 스텝 실측 입력, 첫 1시간 실측 뒤 91전이도 별도 공개한다. e37과 같은 공기 변화132구간에서 양 방향 15/60/180분 NH4 오차와 상승 구간 탐지를 평가한다. 기록된 미래 공기량/외생입력은 조건부 경로 재생에 쓰며 온라인 미래 입력을 안다고 주장하지 않는다.

사전 선별은 augmented가 control 대비 NH4 MAE를 낮추고, NH4 초과 횟수/강도 MAE, NH4/DO recall/FPR, DO1/2/3 MAE, 양 방향 각 시간의 NH4 반응 오차를 악화시키지 않는지다. reference 대비 같은 항목도 공개한다. 0 허용폭의 점추정치 선별은 작은 악화를 실용적 무가치로 단정하는 검정이 아니다. 날짜 7일 블록 bootstrap 2000회로 불확실성을 함께 보고하며 5구성원을 독립 정책 seed로 세지 않는다. 반복 사용한 검증이므로 독립 확증이 아니다.

한 번의 증강만 실행한다. 결과를 보고 반복 수/혼합 비율/창 길이/위반 경계를 다시 고르지 않는다. 원래 환경을 덮어쓰거나 새 RL을 자동 실행하지 않는다. 공기량은 기록 그대로여서 이번 실험 자체로 절감 성과가 나오지는 않는다.

## 실행

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/jio/fcsrl_src/.venv/bin/python Yoo/experiments/e41/preflight.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE \
/home/jio/fcsrl_src/.venv/bin/python Yoo/experiments/e41/experiment.py \
  --prepared /home/jio/fcsrl_src/results/e30/prepared \
  --source /home/jio/fcsrl_src/results/e39/main \
  --events /home/jio/fcsrl_src/results/e37/main \
  --out /home/jio/fcsrl_src/results/e41/main
```

출력 폴더가 있으면 실행하지 않는다. 대용량 모델, 예측, 학습 입력 배열은 runtime에 보관하고 작은 설정/집계/코드를 Git으로 관리한다. 입력/코드/후보 SHA256과 실행 커밋을 기록한다. 팀 공통 평가 조건 변경이 아닌 개인 대조다.

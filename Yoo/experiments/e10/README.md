# Yoo/e10 위반 사건 진단과 과거 상태 입력 대조

## 목적

e09에서 평균 농도 오차 감소가 날짜별 위반 비용 정확도 개선으로 이어지지 않았다. 실제 위반의 시작과 지속시간을 먼저 측정하고, 최근 상태 이력이 빠진 것이 원인 후보인지 확인한다. 원본 FCSRL 표현 손실, PPO, 환경 코드 및 기존 모델은 바꾸지 않는다. 이 시험은 대리 환경 진단이며 새로운 FCSRL 확장 성과가 아니다. 팀 공식 비교 조건은 아직 합의되지 않았고 개인 탐색으로 수행한다.

과거 출력 입력과 연속 예측 구분은 [SysIdentPy Neural NARX 구현](https://sysidentpy.org/user-guide/how-to/create-a-narx-neural-network/)과 [검증 문서](https://sysidentpy.org/book/9-Validation/)를 참고했다. 패키지를 설치하거나 전체 구현을 복사하지 않았으며, 기존 DO 이후 NH4 예측 순서를 유지하는 작은 PyTorch 대조 모델을 구현했다. 해당 라이브러리 전체를 재현했다고 주장하지 않는다.

## 사전 고정한 실험

기존 데이터 `aved_5min.pkl`의 train 330일만 새 모델 시험에 사용한다. 학습 263일(2022-10-06부터 2023-10-12), 제외 1일(2023-10-15), 내부 holdout 66일(2023-10-16부터 2023-12-31)로 나눈다. 실제 날짜 간격도 확인한다. 새 모델과 정규화는 앞의 263일만 사용하며 기존 사전 학습 가중치를 사용하지 않는다. 내부 holdout 결과를 보고 epoch 또는 설정을 고르지 않는다. 바깥 val과 test는 새 모델 시험에 사용하지 않는다. 별도 기존 모델 사건 진단만 기존 train 및 val에서 수행한다.

| 항목 | 설정 |
| --- | --- |
| 비교군 | `current_repeat`, 현재 NH4 및 DO를 과거 상태 입력 자리에 반복 |
| 후보 | `history`, 5, 15, 30, 60분 전 NH4 및 DO |
| 공통 입력 | 기존 DO와 NH4 각각 10개 입력에 상태 8개 추가 |
| 구조 | 각 모델 ReLU 64 x 2, 두 군의 파라미터 개수와 초기 가중치 동일 |
| 결합 | 공기량은 DO에 입력, NH4는 예측 DO 다음값을 입력받음. 두 모델을 함께 학습 |
| 목표 | 두 농도 다음값 MSE를 학습 구간의 농도 변화량 표준편차 제곱으로 나눠 평균 |
| 학습 | seed 0, 1, 60 epochs, batch 1024, Adam lr 0.001, weight decay 0.00001, gradient clip 1 |
| 제어 | seed별 같은 샘플 순서, 마지막 가중치 사용. 기존 모델 대비가 아닌 같은 예산의 새 모델끼리 대조 |
| 실행 | 기존 Python 3.10.19 venv, CPU 1 thread. 패키지와 torch 변경 없음 |

양쪽 모두 각 날짜의 처음 13개 관측값(00:00부터 01:00)을 초기 조건으로 준다. 이후 01:05부터 23:55까지 275전이를 평가한다. 1스텝, 12스텝, 나머지 하루 전체를 예측하며 재시작 지점에서만 직전 13개 실측 상태를 다시 준다. 연속 예측 중 과거 상태도 자기 예측값으로 갱신한다. 실측 과거를 매 스텝 계속 주는 평가와 혼동하지 않는다. 실제 공기량과 외생 변수는 기록대로 재생한다.

비용 정의는 기존 NH4 > 4 및 DO > 2 위반 스텝 수 합이다. 그러나 275전이 비용은 기존 287전이 하루 비용 또는 한도 15와 직접 비교하지 않는다. 공기량을 최적화하는 정책 학습을 수행하지 않는다.

## 지표와 진행 기준

- NH4와 DO 각각 스텝별 precision, recall, F1, 총 위반 스텝 수 및 날짜별 오차를 측정한다.
- 위반 시작은 false에서 true로 바뀌는 시점이다. 시작 시점 차이가 3스텝(15분) 이내인 쌍을 가까운 순서로 일대일 매칭한다. 이는 최적 매칭 보장을 주장하는 알고리즘이 아니다. 시작 경계에서 이미 진행 중인 사건은 onset 집계에서 제외한다.
- 매칭된 사건의 시작 시간 MAE와 지속시간 MAE를 구분한다. 지속시간은 양쪽 끝이 관측된 사건만 포함하며 매칭 수를 함께 보고한다. 매칭된 일부 사건의 작은 오차만으로 전체 개선이라 주장하지 않는다.
- 누락 사건도 반영하도록 날짜별 최장 관측 위반 구간 길이 MAE와 날짜별 비용 MAE를 함께 측정한다. 최장 구간에는 관측 경계에서 잘린 구간도 포함된다.
- 후속 외부 검증 진행 기준: 두 seed 모두 내부 holdout 연속 예측 비용 MAE 개선, 평균 비용 MAE 5% 이상 감소, NH4 onset F1 및 최장 위반 길이 MAE 악화 없음, DO MAE 및 한 스텝 NH4 MAE 악화 5% 이내. 5%는 사전에 정한 실용적인 진행 기준이며 유의성 검정이 아니다.

통과하지 않으면 해당 후보를 채택하거나 외부 val에 맞춰 재조정하지 않는다. 통과하더라도 독립적인 실제 공정 검증이나 FCSRL 개선 성과로 해석하지 않는다. 기존 학습 자료에 대한 이전 연구의 관찰이 있었으므로 내부 분할도 완전히 새로운 데이터의 증거는 아니다. 미래 부하, 관측 잡음, 행동과 부하의 교란 문제는 이력 추가만으로 해소된다고 가정하지 않는다.

## 실행 및 검증

`history_trial.py --preflight-only`는 사건 지표의 알려진 예제, 과거 인덱스, 연속 예측의 미래 실측 비참조를 검사한다. 실제 실행에서 동일 파라미터 개수 및 초기값, 같은 batch 해시, 모든 loss와 gradient 유한, 저장 및 복원 출력을 확인한다.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /home/jio/fcsrl_src/.venv/bin/python -u Yoo/experiments/e10/history_trial.py \
 --data /home/jio/사물/Yoo/data/avedore/aved_5min.pkl \
 --out /home/jio/fcsrl_src/results/e10/history_trial

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /home/jio/fcsrl_src/.venv/bin/python -u Yoo/experiments/e10/audit_events.py \
 --data /home/jio/사물/Yoo/data/avedore/aved_5min.pkl \
 --original /home/jio/fcsrl_src/results/e02/surrogate/surrogate.json \
 --candidate /home/jio/fcsrl_src/results/e09/model_trial/multi_step12.json \
 --out /home/jio/fcsrl_src/results/e10/event_audit.json
```

데이터 출처와 전처리는 [e02 기록](../e02/README.md)을 따른다. 코드 커밋, 입력 해시, 날짜 목록, 버전 및 시간은 결과 JSON에 남긴다. 전체 모델과 예측 배열은 서버 결과 폴더에 보존하고 Git에는 요약만 올린다. 재실행 시 비어 있는 별도 결과 경로를 사용한다.

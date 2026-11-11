# e45 기존 정책의 동일 조건 평가

## 목적과 고정 조건

e44에서 NH4 제약 실패가 학습 날짜에도 남는지, 개발 날짜와 행동 실행 방식에 따라 달라지는지 구분한다. [문헌 검토](../../analysis/FCSRL_EXTENSION_REVIEW_20261111.md)의 첫 판별 작업이다. 학습, 파라미터 변경, head 분리 구현은 하지 않는다.

- 기준 코드: main 0cf9d26. e44 최종 정책 12개 전부 사용, 정책 seed 0/1. 선택적 제외 없음.
- 데이터: Kaggle dynamic-modeling-of-wastewater-treatment-process의 DARROW, e30 준비 자료. 학습 188일과 개발 val 40일 전체를 결과 확인 전에 protocol.json에 고정. test 41일의 행은 평가용으로 추출하지 않으며 test 평가 없음.
- 환경: e35 corrected_surrogate.pkl, 모델 구성원 5개를 모든 날짜에 동일하게 적용. 구성원 5개는 독립 학습 seed가 아니다.
- 행동: 결정적 평균 정책과 동일 정규분포에서 표본을 뽑는 확률 정책. 확률 평가는 고정 반복 ID 45001, 45002, 45003. 날짜/모델/반복 ID로 정규 난수를 고정하며 모든 정책에 같은 표준 난수를 사용한다. 반복 3회는 추가 학습 seed가 아니다.
- 12정책 x 2날짜 분할 x (결정적 1회 + 확률적 3회) = 96평가. 총 54,720 에피소드, 5,198,400 전이. CPU 한 스레드, 순차 실행.
- 행동 범위 0~9000 m³/h. 평가 95전이, 공기량은 .25*sum(Q) m³/23.75h. 실제 kWh 미측정.
- 비용은 NH4>4와 max(DO)>2인 전이 수, 각 평균 한도 6/6. 두 초과 적분, 최악 날짜, 모델 구성원별 결과와 행동 clipping도 보관한다.
- FCSRL Encoder, dynamics/feasibility 손실, EMA, PPO/PID, Actor 보정과 입력은 기존 체크포인트 그대로. 현재는 업데이트를 전혀 호출하지 않는다.

원 FCSRL 코드 커밋 d882b91054d52c7a63a18a21f7db67bbbed82b63, Apache 2.0. 구현 상속 관계와 원본 대비 적용 변경은 e44에 기록되어 있다. 공식 프로토콜v1의 합계12/원환경과 다른 Yoo 개인 진단이며 공통 프로토콜 변경이나 채택 결정이 아니다.

## 검증과 해석

각 정책의 결정적 val 평가를 먼저 실행하여 e44 원 수질 경로와 공기 경로, 집계를 재현해야 나머지 평가를 진행한다. 정책 가중치와 normalizer의 전후 fingerprint, 입력과 checkpoint/소스 해시를 확인한다. 각 step의 reward/cost와 상태 기반 재집계도 대조한다. 난수는 배치 길이 및 정책 순서에 독립적이며 그 성질을 확인한다. NaN이나 재현 실패가 생기면 중단한다.

train과 val 모두 실패하면 제약 학습의 문제가 학습 날짜에서도 남는다는 근거다. train만 통과하면 날짜별 분포 차이를 우선 검토한다. 확률 정책만 통과하면 평균 행동 실행의 영향을 우선 검토한다. 어느 경우도 단독 인과 증명이나 feasibility head 분리 효과의 증명이 아니다. 여러 양상이 섞이면 혼합 결과로 보고한다.

개발 날짜는 반복 사용되었으므로 확증 자료가 아니다. 결과를 보고 날짜, 한도, 반복 ID를 바꾸지 않는다. 전체 결과를 확인하고 후속 실험의 진행 또는 보류를 결정하며 모듈을 자동으로 누적하지 않는다. 세 지표 동시 감소 및 tail 악화 없음이라는 기존 채택 기준을 완화하지 않는다.

## 실행

환경은 기존 `/home/jio/fcsrl_src/.venv`의 Python 3.10.19, torch 2.14.1+cu130, numpy 1.23.5, scikit-learn 1.7.2다. 패키지 변경 없음.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE \
/home/jio/fcsrl_src/.venv/bin/python Yoo/experiments/e45/evaluate.py freeze \
  --out /home/jio/fcsrl_src/results/e45/main

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE \
/home/jio/fcsrl_src/.venv/bin/python -u Yoo/experiments/e45/evaluate.py run \
  --out /home/jio/fcsrl_src/results/e45/main
```

runtime protocol.json에 실행 코드 커밋, 날짜, 입력/코드/checkpoint 해시를 저장한다. status.json은 완료 수, 소요 시간, 남은 예상 시간을 기록한다. 각 job의 전체 경로는 jobs 아래 rollout.npz, episodes.json, metrics.json에 보존하며 큰 자료는 Git에 넣지 않는다. 로그는 `/home/jio/fcsrl_src/results/e45/main.log`다.

이 문서 작성 시점에는 평가 전이며 결과는 없다.

# Yoo/e05 PPO 기반 FCSRL 적용

## 목적과 범위

원본 FCSRL의 동역학 일관성과 feasibility 일관성 학습을 유지하면서, e03에서 안정적으로 학습된 PPO-Lag 기반으로 폐수처리 대리 환경에 적용한다. FCSRL 논문은 이미 PPO-Lag와 TD3-Lag를 모두 사용하므로 PPO 전환 자체를 새로운 확장이라고 주장하지 않는다. 이 단계는 확장 전 적용 기준선의 확인이며, UCB나 새로운 표현 손실은 추가하지 않는다.

개인 탐색 실험이다. 팀 공통 평가 조건은 아직 빈 항목이 있으므로 팀 공식 비교 또는 채택 결과가 아니다. 기존 e01부터 e04 결과를 교체하지 않는다.

## 원본과의 연결

- 원본: [czp16/FCSRL](https://github.com/czp16/FCSRL), 커밋 `d882b91054d52c7a63a18a21f7db67bbbed82b63`, Apache-2.0. `Yoo/code/wwtp/FCSRL_LICENSE`에 라이선스를 보관한다.
- [논문 5.2.2](https://arxiv.org/html/2405.11718v2)는 PPO-Lag와 TD3-Lag 양쪽에 FCSRL을 결합한다.
- `ppo_fcsrl.py`는 원본 `Encoder`, 분포형 feasibility head, projection 구조를 사용한다. 원본 `PPOLagReprAgent.train_encoder`를 직접 호출한다. 동역학 손실 계수 0.5, feasibility 손실 계수 1, z 차원 64, encoder 은닉층 256 x 2, 63개 bin과 범위 [-2, 4], 학습률 0.0003을 유지한다.
- 원본 PPO 코드의 한 단계 잠재 전이 예측을 사용한다. 논문 전체 실험의 다단계 설정을 완전히 재현했다고 주장하지 않는다. feasibility 목표의 max 재귀와 할인율 0.9를 유지하되, 병렬 환경마다 따로 계산하고 reset 경계에서 다음 날로 이어지지 않게 한다.
- 고정 encoder를 정책과 가치망 입력에 사용한다. PPO rollout 갱신이 끝나면 원본과 같이 tau=0.05로 EMA 갱신한다. 정책 최적화 중에는 고정 encoder를 변경하지 않는다. EMA 직전과 직후 정책 KL도 기록한다.
- 정책과 보상 및 비용 가치망은 e03 PPO의 64 x 2 tanh MLP를 유지하고, FCSRL에서만 정규화 관측에 z를 연결한다. 원본의 EncodedActorProb 및 EncodedCritic과 세부 입력층 구조는 다르다. frozen 대조군은 FCSRL군과 완전히 같은 구조와 초기화에서 표현 학습만 끈다.
- PPO는 e03의 raw Gaussian 행동 log probability, 상태 독립 log std, 정규화한 Lagrangian advantage와 GAE를 유지한다. 표현의 동역학 학습에는 환경에 전달한 clipped 행동을 사용한다. 원본 PPO의 잘린 행동 확률 계산을 복원하지 않는다.

## 비교 구성과 고정 조건

| 방법 | 정책 입력 | 표현 학습 | 역할 |
| --- | --- | --- | --- |
| ppolag | 관측 | 없음 | 기존 PPO-Lag 비교군 |
| ppolag_fcsrl | 관측 + z | 원본 PPO-FCSRL 손실 | 기법 적용 |
| ppolag_frozen | 관측 + z | 초기값 고정 | 구조 크기와 학습 효과 분리 |

세 방법 모두 seed 0, 1과 목표 400,000 전이(실제 404,096)를 사용한다. 8개 환경 x 1,148 스텝, 44회 PPO 갱신, 갱신당 최대 10 epoch, 미니배치 512, clip 0.2, target KL 0.02, 초기 log std -0.5, 보상 및 비용 할인율 0.99, GAE lambda 0.95다. actor 학습률 0.0003, 가치망 학습률 0.001, gradient norm 상한 0.5를 사용한다. 표현 학습의 갱신 횟수는 해당 PPO 미니배치 갱신과 같다.

학습 및 보고 비용 한도는 모두 15, PID 이득은 (0.02, 0.005, 0.01), 승수 상한은 기존 PPO와 같은 20이다. 초기 상태 샘플링, PID anti-windup, 환경 불확실성 벌점, 비용 정의, 하루 종료 처리를 변경하지 않는다. 원본 PPO와 달라진 안정화 및 학습 설정은 두 PPO 비교 방법에 공통 적용한다.

매 실행 5회, 검증 43일 x 대리 모델 구성원 5개인 215에피소드를 고정 평가한다. 검증 평균 비용 15 이하인 모델 중 보상이 가장 큰 모델을 선택한다. 만족 모델이 없으면 비용이 가장 작은 모델을 선택하고 `best_feasible=false`로 표시한다. 모든 검증 시점 모델과 마지막 모델을 저장한다. test는 읽거나 평가하지 않는다. 검증 자료는 이미 탐색에 사용되어 독립 확증 자료가 아니다.

## 환경과 데이터

기존 e02와 같은 Avedore 5분 관측 자료, 전처리 및 날짜 분할을 사용한다. 데이터는 `/home/jio/사물/Yoo/data/avedore/aved_5min.pkl`, 학습용 MLP 앙상블은 `/home/jio/fcsrl_src/results/e02/surrogate/surrogate.json`이다. 학습 330일, 검증 43일이며 원본 입력 SHA256은 실행 manifest에 남긴다.

한 에피소드는 하루 287전이다. 11차원 상태, 호기 단계에서만 공기량 0~8,000 Nm3/h를 제어하는 행동 1개를 사용한다. 보상은 -공기량/8,000이다. 단계별 비용은 NH4 > 4 mg N/L와 DO > 2 mg O2/L의 지시함수 합이다. 비용은 0, 1, 2이므로 원본 이진 비용에 대한 안전 확률 해석을 그대로 적용하지 않는다. 보고 비용은 하루 비용 합의 평균, 공기량은 Nm3/일이며 전력량이 아니다. 하루 종료는 기존과 같은 truncation이며 마지막 상태 가치로 bootstrap한다.

## 검증과 실행

원본 손실과 gradient 갱신 일치, 원본 feasibility 재귀와 환경별 일치, reset 경계 차단, raw 행동 확률, 표현 학습 중 정책 입력 고정, EMA 후 정책 입력 반영, frozen encoder 불변, 체크포인트 복원을 검사한다. 기존 PPO update와의 비트 단위 일치 및 실제 데이터의 짧은 학습도 확인한다. 자세한 결과는 `verification.json`에 기록한다.

```bash
PYTHONPATH=/home/jio/fcsrl_src/FCSRL \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/jio/fcsrl_src/.venv/bin/python -m unittest discover -s Yoo/code/wwtp/tests -v

/home/jio/fcsrl_src/.venv/bin/python Yoo/experiments/e05/run_pilot.py \
  --data /home/jio/사물/Yoo/data/avedore/aved_5min.pkl \
  --model /home/jio/fcsrl_src/results/e02/surrogate/surrogate.json \
  --out /home/jio/fcsrl_src/results/e05/pilot
```

GPU가 다른 작업에 사용 중이므로 CPU 스레드 1개로 두 실행씩 진행한다. Python 및 패키지 버전은 `verification.json`, 실제 실행 커밋, 입력과 소스 해시, 전체 명령은 실행 폴더 `manifest.json`에 남긴다. 실패하면 나머지 실행도 중지한다. `status.json`과 각 실행의 `progress.json`에 진행 상태와 추정 잔여 시간을 기록한다. 완료 시 `results.json`을 생성한다. 모델과 전체 로그는 Git에 넣지 않는다.

## 상태와 해석 기준

구현과 사전 검증을 완료했다. 단위 검증 10개가 통과했으며, 세 방법 각각 실제 자료 9,184전이 학습과 전체 검증을 완료했다. FCSRL 동일 seed 반복의 모든 가중치와 검증 결과가 일치했고, 기존 PPO update는 변경 전과 비트 단위로 일치했다. 짧은 학습 결과를 성능 우열로 해석하지 않는다.

2026-11-06 16:19:04 KST, 실행 커밋 `98fb8b8d0386c8dd2d6a8c3b26ebbea4773965e2`로 6개 비교 실행을 시작했다. runner PID는 3820591이었다. **16:30:56 KST에 모두 exit code 0으로 완료했고 남은 실행은 없다.** 완료 상태는 `/home/jio/fcsrl_src/results/e05/pilot/status.json`에 있다. 최초 실행 설정과 소스 및 입력 해시는 `launch.json`에 보관한다.

이 시험은 PPO 기반 FCSRL의 작동과 비교 가능성을 확인한다. seed 2개의 짧은 시험으로 우월성이나 확장의 성공을 주장하지 않는다. FCSRL군이 raw PPO 및 frozen 대조에 비해 어떤 효과를 보이는지 먼저 확인하고, 그 뒤 확장 변경 하나를 정한다. 성능이 낮으면 실패 결과도 보존하며 다른 목적함수로 바꿔서 성공으로 처리하지 않는다.

## 완료 결과

[전체 결과](results.md), [개별 실행](pilot_results.json), [집계 및 갱신 진단](summary.json).

동일 기준 선택 모델에서 PPO-Lag의 평균 공기량과 비용은 40,222 Nm3/일과 15.17, FCSRL+PPO-Lag는 38,527과 13.41, 표현 고정 대조는 30,710과 14.14였다. 비용 한도를 만족한 seed는 각각 1/2, 2/2, 2/2다. 기본 PPO seed 1은 한도를 만족한 평가 시점이 없어 최소 비용 모델을 남겼으며 `best_feasible=false`다.

FCSRL이 기본 PPO보다 선택 모델 평균에서 낮은 비용과 공기량을 보였지만, 표현 고정 대조보다 공기량이 많았다. 표현 학습 자체의 우월성은 확인되지 않았다. FCSRL 마지막 모델은 seed 0 비용 13.44, seed 1 비용 18.80으로 평균 16.12였다. 따라서 중간 모델의 제약 만족을 마지막까지 유지했다고 주장할 수 없다.

모든 실행에서 최대 lambda는 3.06 미만으로 상한 20에 닿지 않았다. 이번 실패를 승수 상한 부족으로 설명할 근거는 없다. FCSRL 고정 encoder EMA 직전과 직후 정책 KL의 최댓값은 seed별 약 0.00164, 0.00152였다. 이 측정만으로 모든 학습 불안정을 배제할 수는 없다.

각 실행의 44회 PPO 및 PID 갱신, 5회 검증, 에피소드 215개 중복 없는 집계, 실제 실행 소스 해시를 확인했다. 집계는 아래 명령으로 재생성한다. 학습 코드를 바꾸거나 test를 추가 평가하지 않는다.

```bash
python3 Yoo/experiments/e05/summarize.py \
  --runs /home/jio/fcsrl_src/results/e05/pilot \
  --out Yoo/experiments/e05
```

현재까지는 PPO 기반 FCSRL 적용 및 대조 시험을 완료한 상태다. 추가 확장은 구현하지 않았으며, 다음 검토 대상은 표현 학습이 실제로 주는 이득과 후반 제약 이탈이다. 원본 기법과의 연결을 유지한 채 검토한다.

# Yoo/e03 PPO 비교군, UCB 확장과 승수 상한 점검

Yoo/e02의 폭기 제어 환경에 PPO Lag 비교군과 FCSRL + UCB 비용 critic을 추가한다. 학습에 쓰지 않은 GBM 대리 모델에서도 정책을 평가하고, TD3 계열의 Lagrange 승수 상한 0.75가 결과를 제한했는지 상한 5 실험으로 점검한다. 팀 공통 PROTOCOL은 아직 합의되지 않았으므로 개인 실험 조건의 탐색적 비교다.

초기 제안은 [initial_plan.md](initial_plan.md)에 보관했다. 실제로 수행하지 않은 SAC, 환경 불확실성 벌점, DO 안전층을 이번 결과에 포함하지 않는다.

## 확인할 질문

1. PPO Lag가 FCSRL 및 TD3 Lag와 비교해 공기량과 제약 준수에서 어떤 결과를 내는가?
2. 같은 승수 상한에서 UCB 비용 critic이 기본 FCSRL보다 제약 준수를 개선하는가?
3. 상한을 0.75에서 5로 올리면 결과가 개선되고, UCB의 추가 효과가 드러나는가?

상한 변경 실험은 초기 평가 결과를 본 뒤 추가한 사후 분석이다. 같은 평가 구간을 다시 사용하므로 독립적인 확증 실험으로 해석하지 않는다.

## 데이터와 환경

출처, 전처리와 분할은 [e02 기록](../e02/README.md)을 따른다. Avedøre 운전 기록을 5분 단위로 묶고, 학습 330일, 검증 43일, 평가 60일로 나눈다. 데이터 원본과 전처리 결과는 `Yoo/data/avedore/`에 있다.

- 상태는 NH4, DO, 운전 단계, 수온, 유입량, 시각, 단계 경과 시간과 직전 공기량의 11차원이다.
- 행동은 호기 단계 공기량 0~8000 Nm3/h다. 다른 단계에는 공기량을 0으로 둔다.
- 보상은 매 스텝 `-air/8000`이다. 공기량 합은 Nm3/일이며 kWh가 아니다.
- 비용은 NH4 > 4.0이면 1, DO > 2.0이면 1을 더한 값이다. DO는 상한 제약이며 하한은 없다.
- 하루는 287개 전이로 구성된다. 하루 비용 한도는 15다. 두 위반이 겹치면 스텝 비용은 2다.
- 학습은 e02에서 저장한 MLP 앙상블을 사용한다. `pess_k=0`, `unc_pen=0`이며 환경의 비관적 비용과 불확실성 보상 벌점은 이번 실행에 사용하지 않았다.
- 최종 평가는 60일 x 모델 구성원 5개 = 300 에피소드다. 한도 만족 비율도 이 날짜와 구성원 조합 기준이며 서로 독립인 300일이 아니다.
- 실제 운전 기준선은 기록된 공기량을 대리 모델에 재생한 결과다. 그 비용을 실측 수질의 위반 비용과 혼동하지 않는다.

## 비교 구성

모든 학습 seed는 0, 1, 2, 3, 4다. TD3, TD3 Lag, FCSRL은 e02의 저장된 정책을 재사용한다.

| 결과 tag | 구성 | 승수 상한 | 본 실행 |
| --- | --- | --- | --- |
| td3 | TD3 Lag에서 승수 상한을 0으로 제한 | 0 | e02 5개 |
| td3lag | 원본 FCSRL에서 표현 학습을 끔 | 0.75 | e02 5개 |
| fcsrl | FCSRL + TD3 Lag | 0.75 | e02 5개 |
| fcsrl_ucb | FCSRL의 비용 critic을 5개 앙상블과 UCB로 변경 | 0.75 | e03 5개 |
| ppolag | 이점 정규화와 PID를 사용하는 자체 PPO Lag | 20 | e03 5개 |
| fcsrl_ml5 | FCSRL에서 승수 상한만 변경 | 5 | e03 추가 5개 |
| fcsrl_ucb_ml5 | FCSRL + UCB에서 승수 상한만 변경 | 5 | e03 추가 5개 |

TD3 계열은 25 epoch x 2500 gradient step, 수집 길이 32와 warmup 16에피소드다. PPO는 목표 2,000,000개 전이, 실제 2,002,112개 전이(8환경 x 1148스텝 x 218회 수집)다. 환경 경험량은 약 2M로 맞췄지만 네트워크, 갱신 횟수와 손실의 척도까지 동일한 비교는 아니다.

검증은 학습 중 25회, 매번 무작위 10에피소드다. 원본 트레이너의 선택 규칙을 유지했다. 아직 한도를 지키는 모델이 없으면 최신 모델로 교체하고, 한도를 지키는 모델을 얻은 뒤에는 한도 안에서 보상이 개선될 때 저장한다. 최종 표의 선택 모델은 이 규칙으로 고른 모델이며, 마지막 모델의 결과도 별도로 제시한다.

## UCB 확장의 범위

`agents.py`는 원본 `fcsrl/agent/td3_lag_repr.py`를 상속한다. 비용 critic을 2개에서 5개로 늘리고, 목표 비용 Q와 actor 손실의 비용 Q를 `평균 + 0.5 x 표준편차`로 계산한다. 원본은 목표값에 두 critic의 최대값, actor 손실에는 평균을 사용한다.

이는 비용 critic 불일치를 사용한 확장이다. 환경 대리 모델의 예측 불확실성을 직접 제약에 넣은 방법은 아니다. critic의 표준편차가 실제 오차를 나타내는지는 아직 검증하지 않았다. 또한 critic 수와 목표값 집계 방식도 함께 바뀌었으므로 이번 비교만으로 표준편차 항 하나의 효과를 분리할 수 없다. `평균 + 0.5 x 표준편차`가 원본의 최대값보다 항상 보수적인 것도 아니다.

원본 FCSRL은 `czp16/FCSRL`, 커밋 `d882b91054d52c7a63a18a21f7db67bbbed82b63`, Apache-2.0 라이선스다. 라이브러리 원본은 수정하지 않았다. 확장 동기는 [CAL 논문](https://arxiv.org/abs/2401.14758)의 비용 Q 과소평가 문제이며 CAL 전체 알고리즘의 재현은 아니다.

## PPO 수정과 실제 설정

`ppo_lag.py`는 정규분포에서 뽑은 행동을 버퍼에 그대로 저장하고, 환경 입력만 [-1, 1]로 자른다. 확률 비율은 저장된 원래 행동으로 계산한다. 보상 이점은 표준화하고 비용 이점은 평균만 뺀 뒤 `(A_r - lambda A_c)/(1 + lambda)`로 결합한다. 이 결합식은 [OmniSafe PPOLag 소스](https://omnisafe.readthedocs.io/en/latest/_modules/omnisafe/algorithms/on_policy/naive_lagrange/ppo_lag.html)와 같다. 승수는 FCSRL의 PID 갱신기를 사용하므로 OmniSafe 구현 전체와 동일하지 않다.

실제 본 실행 설정은 다음과 같다. 이전 `ppo_wwtp.yaml`에는 원본 PPO 설정이 남아 있었지만 자체 구현은 대부분을 사용하지 않았다. 현재 파일은 실제 사용한 기본값을 명시하며 모든 항목을 실행에 연결한다.

| 항목 | 실제 기본값 |
| --- | --- |
| 정책 및 각 가치망 | 은닉층 64 x 2, tanh |
| 정책 표준편차 | 상태와 무관한 학습 파라미터, 초기 log std -0.5 |
| 정책 및 가치 학습률 | 0.0003, 0.001 |
| gamma, GAE lambda | 0.99, 0.95 |
| rollout | 8환경 x 1148스텝 |
| rollout당 최적화 | 최대 10회, 미니배치 512 |
| PPO clip, KL 조기 종료 기준 | 0.2, 0.02 |
| gradient norm 상한 | 0.5 |
| PID KP, KI, KD | 0.02, 0.005, 0.01 |
| 비용 한도, 승수 상한 | 15, 20 |

같은 PID 계수를 쓴다고 동일한 제약 갱신 조건은 아니다. PPO는 32일분을 모은 rollout마다 한 번 갱신하고 TD3 계열은 수집 중 종료된 에피소드가 있을 때 갱신한다. 이점과 Q의 척도도 달라 PPO와 TD3의 lambda 수치를 직접 비교하지 않는다.

초기 행동의 자르기 전 공기량 표준편차는 약 `exp(-0.5) x 4000 = 2426` Nm3/h다. 상태에 따라 바뀌는 원본 분산 대신 이를 학습하며, 실제 환경 행동은 범위에 맞게 잘린다.

### 원본 PPO 실패 기록

보관 경로는 `/home/jio/fcsrl_src/results/e03/failed_ppolag_orig/`다. seed 1 로그에는 NaN으로 인한 예외가 남아 있다. seed 0과 2 로그는 학습 중간에 끝나며, 해당 로그만으로 정상 종료나 NaN 종료를 확인할 수 없다. 따라서 이전 설명의 "두 seed가 NaN 종료"를 확정된 사실로 사용하지 않는다.

원본은 잘린 행동에 정규분포의 밀도를 계산하고 비율의 지수에 별도 제한이 없었다. 이는 수치 불안정의 가능한 경로지만, 행동 처리, 표준편차, 손실 정규화와 네트워크를 함께 바꿨으므로 각 변경의 인과 효과를 분리한 실험은 아니다.

### 재현성 수정과 검증

- PPO의 `--total_steps`는 실제 환경 경험량 목표를, `--eval_chunks`는 검증 횟수를 바꾼다. 마지막 rollout은 끝까지 수집하므로 실제 전이 수를 별도로 저장한다.
- 이전에 PPO에서 학습량을 바꾸지 못했던 `--epochs`, `--grad_steps`는 오류로 안내한다. 이 두 옵션은 TD3 계열에만 사용한다.
- 실행 디렉터리에 `effective_config.json`, 각 PPO 모델 디렉터리에 `config.json`을 남긴다. 평가 시 저장된 신경망 크기로 복원한다. 기존 체크포인트는 당시 기본값으로 읽는다.
- 기존 기본값의 한 번 업데이트를 수정 전 소스와 비교했으며 정책망, 두 가치망과 log std가 정확히 일치했다.
- 설정 단위 테스트 3개를 통과했다. 별도의 작은 실제 데이터 실행에서 CLI 덮어쓰기, 574개 전이 학습, 16차원 은닉층과 best/final 모델 재로딩을 확인했다. 이 smoke 실행은 본 실험 결과에 넣지 않는다.
- MLP와 GBM, FCSRL과 PPO 각각 2에피소드씩 순차 평가와 배치 평가를 대조했다. 비용은 같았고 공기량 최대 절대 차이는 0.001 Nm3/일 미만이었다.
- 이 수정으로 이미 진행 중이던 TD3 본 실행을 재시작하거나 학습 계산을 바꾸지 않았다.

## 평가 전용 GBM과 한계

GBM은 학습 구간에서 만든 HistGradientBoostingRegressor 5개 앙상블이다. 구성원마다 학습 쌍 80%를 뽑고, DO 변화량의 공기량 방향과 NH4 변화량의 다음 DO 방향에 단조 제약을 둔다. 정책 학습에는 사용하지 않는다. 입력과 인과 구조는 MLP와 같다.

평가 구간의 일평균 NH4는 실측 1.626 mg/L, MLP 약 1.45, GBM 1.272다. GBM도 NH4를 낮게 예측한다. 두 모델에서 좋은 결과가 나와도 공통 편향이나 정책의 모델 오차 의존성을 배제할 수 없다. 낮은 DO 영역의 예측 정확도, N2O, 실공정 안전성은 확인되지 않았다.

검증 10에피소드는 모델 선택에 불안정할 수 있다. 선택 모델의 검증 비용이 낮더라도 고정된 평가 300에피소드의 비용이 한도를 넘는 사례가 있다. 후속 실험에서는 평가 구간을 더 보면서 모델을 고르는 대신 고정된 검증 날짜와 더 많은 검증 에피소드를 사전에 정해야 한다.

## 결과 파일과 상태

추가 실험 완료 후 생성되는 `results.md`에는 두 대리 모델의 선택 모델 결과와 MLP의 마지막 모델 결과를 함께 넣는다. `training_audit.md`에는 seed별 선택 epoch, 검증 비용, 평가 비용과 마지막 승수를 남긴다. 표준편차는 기존 e02와 같은 ddof=0이며, 통계 검정은 학습 seed 5개를 단위로 하는 Welch t 검정이다. 다중 비교 보정 없는 탐색적 p값이다.

기존 TD3 실행은 마지막 모델의 MLP 평가 수치는 저장했지만 마지막 가중치를 항상 저장하지 않았다. 따라서 마지막 모델 GBM 평가의 누락은 그대로 보고하며, 선택 모델을 마지막 모델로 대신하지 않는다. 앞으로 실행하는 TD3는 `model_final/`도 저장한다.

결과 해석은 전체 seed 평가가 끝난 뒤 이 절에 추가한다.

## 재현 명령과 산출물

저장된 모델의 전체 재평가 및 집계:

```bash
PY=/home/jio/fcsrl_src/.venv/bin/python
R=/home/jio/fcsrl_src/results
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" Yoo/experiments/e03/finalize_e03.py \
  --e02 "$R/e02/runs" --e03 "$R/e03/runs" \
  --data Yoo/data/avedore/aved_5min.pkl \
  --mlp "$R/e02/surrogate/surrogate.json" \
  --gbm "$R/e03/surrogate_gbm/surrogate_gbm.pkl" \
  --out "$R/e03/review_20261106" --wait
"$PY" Yoo/experiments/e03/audit_training.py \
  --runs "$R/e02/runs" "$R/e03/runs" --out "$R/e03/review_20261106"
```

새로운 결과 폴더에서 학습을 재현할 때:

```bash
"$PY" Yoo/experiments/e03/run_e03.py \
  --surrogate "$R/e02/surrogate/surrogate.json" --out "$R/e03_reproduction/runs" \
  --methods ppolag fcsrl_ucb --jobs 5
"$PY" Yoo/experiments/e03/run_e03.py \
  --surrogate "$R/e02/surrogate/surrogate.json" --out "$R/e03_reproduction/runs" \
  --methods fcsrl_ml5 fcsrl_ucb_ml5 --jobs 5
```

- 실행 장치는 서버 .32의 RTX 3090 두 장이다. 재평가는 CPU와 스레드 수 1을 사용한다.
- 데이터, 모델, 로그는 Git에 넣지 않는다. 원본 결과는 `/home/jio/fcsrl_src/results/e02/`, `/home/jio/fcsrl_src/results/e03/`에 보관한다.
- 재평가의 `manifest.json`은 데이터, 대리 모델, 정책 가중치, 정규화 통계와 평가 코드의 SHA256을 기록한다.
- 본 실행은 `yoo/e03-uncertainty`의 미커밋 변경으로 시작됐다. 실행 당시의 정확한 코드 커밋을 소급해 주장하지 않는다. 설정 수정 전 소스를 `/home/jio/fcsrl_src/results/e03/source_before_config_fix_20261106/`에 보존했으며, 이 역시 실행 시점이 아닌 사후 캡처다.
- 재현성 수정 구현 커밋은 `12e05ec`이다. 이는 본 실행 후의 수정이며 학습 실행 커밋이 아니다. 패키지는 `environment.json`, 기본값 갱신 대조는 `config_parity_check.json`, 평가 대조는 `evaluation_parity.json`, 수정 전 소스 해시는 `pre_fix_source_manifest.json`에 기록한다.

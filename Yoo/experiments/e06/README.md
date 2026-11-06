# Yoo/e06 FCSRL + PPO에 하루 비용 CVaR 항을 추가하는 확장 후보

## 상태와 목적

2026-11-06, 사전 코드 검증 완료. seed 0, 1의 4방법 시험을 준비했다. 정식 팀 비교나 최종 성능 실험이 아닌 개인 탐색 실험이다. `Compare/PROTOCOL.md`의 공동 합의는 아직 없다. 실제 실행 커밋, 입력과 소스 SHA256, 시작 시간은 실행 폴더의 `manifest.json`에 기록한다.

e05에서 학습 평균 비용은 한도 15보다 낮아도 검증 비용은 높았다. 마지막 FCSRL 정책의 학습 비용은 8.30, 11.98이고 검증 비용은 13.44, 18.80이었다. 초기 NH4가 4보다 큰 날은 학습의 15.45%, 검증의 39.53%였다. 학습의 초기 NH4 두 집단 비용을 검증 구성비로 재가중하면 13.49, 17.80으로 검증값에 가까워졌다. 이는 구성비 차이의 영향을 지지하지만 인과관계의 증명은 아니다. 근거는 `e05_train_val_final.json`, `e05_initial_mix.json`이다.

가설은 학습 평균에서 가려진 높은 하루 비용에 추가 경사를 주면 검증의 큰 비용을 줄일 수 있다는 것이다. 원본 FCSRL 표현 학습은 유지하고 정책 갱신에 위험 항 하나만 추가한다. PPO 전환 자체는 논문에 이미 있는 적용이며 새 확장으로 주장하지 않는다.

## 기존 연구와 실제 구현의 관계

- [Tamar, Glassner, Mannor, Optimizing the CVaR via Sampling, AAAI 2015](https://arxiv.org/abs/1404.3862): 분위수를 baseline으로 쓰는 likelihood-ratio CVaR 경사를 근거로 삼았다. 하루 비용의 upper tail 방향으로 구현했다.
- [TRC 공식 구현](https://github.com/rllab-snu/Trust-Region-CVaR/tree/5066569d7e802cc8552c8c9a0dc04abfb81ec914): 비용 평균과 분산을 학습해 정규근사 CVaR와 trust region을 사용하는 기존 구현이다. 이번에는 PPO와 원본 표현 학습을 유지하기 위해 이 optimizer나 비용 분산 critic을 옮기지 않았다.
- [WCSAC 논문](https://ojs.aaai.org/index.php/AAAI/article/view/17272), [공식 구현](https://github.com/AlgTUDelft/WCSAC): 평균 비용과 별도로 tail risk를 다루는 선행 연구다. 이 전체 알고리즘을 재현했다고 주장하지 않는다.

`ppo_tail.py`는 위 CVaR 표본 경사 원리를 직접 구현한 확장 후보다. 기존 기법의 완전한 이식이나 새로운 CVaR 이론의 제안은 아니다. FCSRL은 Apache 2.0 원본 `czp16/FCSRL` 커밋 `d882b91054d52c7a63a18a21f7db67bbbed82b63`의 `PPOLagReprAgent.train_encoder`를 e05와 동일하게 직접 호출한다. 원본 저장소는 수정하지 않았다. encoder, feasibility 손실, dynamics 손실, EMA 및 PPO 기반 적용에서의 차이는 [e05 기록](../e05/README.md)을 따른다.

## 갱신식과 주의점

한 번의 수집은 길이 287인 완결된 하루 32개다. 각 날의 비용 합을 C, 상위 비율 q=0.2, 표본 80% 분위수를 v로 둔다.

```text
CVaR20 = v + mean(max(C - v, 0)) / q
S_i = max(C_i - v, 0) / (q * 15)
A_base = (normalized(A_r) - lambda * centered(A_c)) / (1 + lambda)
A_used(i,t) = A_base(i,t) - beta * S_i / (1 + lambda)
beta = 0.03
```

S_i는 해당 날의 모든 timestep에 같은 값으로 들어간다. 비용 GAE에 tail 가중치를 곱하는 방식이 아니다. 원본 비용 critic, 평균 비용 15를 추적하는 PID, 보상 정규화는 유지한다. tail 안에서는 v 외의 baseline을 다시 빼지 않는다. 평균 비용 한도 15는 CVaR 한도가 아니며, 여기의 위험 항은 별도의 고정 가중치 soft penalty다.

동일 길이 T의 trajectory를 timestep 평균으로 갱신하므로, clipping 전 기존 정책에서 tail 추가 경사는 원래 하루 비용 CVaR 경사의 `beta / (T * 15 * (1 + lambda))` 배에 해당한다. 원본 보상과 비용 이점의 정규화도 있으므로 이를 원래 단위의 보상에서 CVaR를 단순히 0.03배 빼는 목적함수라고 해석하지 않는다. PPO clipping, 표본 분위수 및 여러 갱신 epoch를 사용하는 local surrogate이며 정확한 전역 CVaR 최적화나 안전 보장은 없다.

표본 tail 질량은 6.4개여서 잡음이 있다. 경계 표본의 분수 질량을 포함해 CVaR를 계산한다. 비용이 이산값이므로 동률 때문에 양의 초과 비용을 가진 날 수는 6보다 적을 수 있다. 잘린 rollout이나 길이가 다른 episode는 즉시 실패 처리한다. 하루 위험 점수에는 bootstrap을 넣지 않고 실제 하루 합을 쓴다. 기존 GAE의 time-limit bootstrap은 그대로 유지한다.

## 고정 비교 조건

| 비교군 | 원본 FCSRL 표현 학습 | CVaR 추가 항 |
| --- | --- | --- |
| PPO-Lag | 없음 | 없음 |
| PPO-Lag + tail | 없음 | beta 0.03 |
| FCSRL + PPO-Lag | 있음 | 없음 |
| FCSRL + PPO-Lag + tail | 있음 | beta 0.03 |

- seed 0, 1, 각 404,096전이(44회 갱신), CPU 1 thread씩 최대 4개 병렬.
- 환경, 자료, 시간 분할과 PPO 설정은 e05와 같다. 학습 330일, 검증 43일, 미사용 test 60일. 매 평가 동일한 검증 43일 x MLP 구성원 5개, 총 215 episode. 총 5회 평가.
- 데이터는 `/home/jio/사물/Yoo/data/avedore/aved_5min.pkl`, MLP는 `/home/jio/fcsrl_src/results/e02/surrogate/surrogate.json`이다. SHA256을 실행 manifest에 보존한다.
- 단계 비용은 `1[NH4 > 4] + 1[DO > 2]`, 하루 비용은 287단계 합이다. 논문의 binary 비용 가정과 차이가 있다. 보상은 음의 정규화 공기량, 보고 공기량은 Nm3/일이며 전력량이 아니다.
- gamma 0.99, GAE 0.95, actor 학습률 0.0003, critic 0.001, PPO clip 0.2, 10 epoch, minibatch 512, KL 0.02, gradient norm 0.5. logstd 초기값 -0.5. 환경 8개, horizon 1148.
- PID (0.02, 0.005, 0.01), 평균 비용 한도 15, 승수 상한 20. 초기 상태 샘플링과 PID를 바꾸지 않는다.
- 선택 기준도 e05와 같다. 검증 평균 비용 15 이하인 시점 중 최고 보상. 만족한 시점이 없으면 최소 비용을 남기고 실패로 표시. CVaR로 선택하지 않는다. 선택 정책과 마지막 정책을 모두 보고한다.
- Python 3.10.19, torch 2.14.1+cu130, numpy 1.23.5, gymnasium 0.28.1, pandas 2.3.3. 기존 venv와 torch를 변경하지 않았다.

확장의 효과는 같은 표현 조건에서 tail on/off를 먼저 비교한다. 평균 비용, CVaR20, 공기량과 seed별 방향을 함께 보고한다. FCSRL + tail과 PPO + tail도 비교해 개선을 FCSRL 표현 학습 덕분으로 잘못 돌리지 않는다. seed 2개로 유의성이나 수렴을 주장하지 않는다.

## 사전 점검과 beta 선택

기존 10개를 포함한 단위 검증 15개가 통과했다. CVaR 경계 질량, 하루 경계, 분석적으로 계산 가능한 정책 경사, 잘못된 입력 거부, beta 0일 때 기본 PPO 및 FCSRL의 갱신과 모든 가중치의 동일성을 확인했다. 실제 자료에서도 위험 항이 없는 두 비교군의 첫 갱신 가중치가 e05와 정확히 같았다.

beta 0.1의 첫 수집에서 추가 위험 신호 RMS가 기본 이점 RMS의 1.23배(PPO), 1.13배(FCSRL)였다. 과도한 초기 신호를 피하기 위해 0.03으로 낮췄다. beta는 검증 성능이 아니라 학습 buffer의 추가 신호가 기본 신호의 절반 미만이라는 사전 점검 기준으로 정했다. 0.03으로 두 방법 x 두 seed, 각 18,368전이를 실행해 첫 비율 0.22~0.37, 두 번째 0.01~0.15를 확인했다. 저장 모델 복원과 전체 검증 재평가가 정확히 일치했다. pilot에서는 0.03을 고정하고 결과를 본 뒤 변경하지 않는다.

요약은 `verification.json`, 원본 기록은 `/home/jio/fcsrl_src/results/e06/smoke`와 `smoke_weight003`에 있다.

## 한계

CVaR 항은 큰 비용의 꼬리를 줄이는 수단이지 초기 상태 분포 차이 자체를 제거하는 수단은 아니다. 학습에 없는 상황으로 일반화한다는 보장도 없다. 초기 비호기 구간은 행동이 무시되며, 그 구간에서만 비용 15를 넘는 날도 존재한다(`e05_uncontrollable_prefix.json`). 따라서 모든 날의 비용을 15 이하로 만드는 것은 이 환경에서 불가능한 경우가 있다. 평균 한도 달성 가능성과 구분한다.

실제 공정 안전성, GBM 검증이나 최종 test 성능을 주장하지 않는다. 공기량을 늘려 tail 비용을 줄였는지 반드시 함께 보고하며, 개선되지 않은 결과도 남긴다.

## 실행

```bash
PYTHONPATH=/home/jio/fcsrl_src/FCSRL /home/jio/fcsrl_src/.venv/bin/python -m unittest discover -s Yoo/code/wwtp/tests
/home/jio/fcsrl_src/.venv/bin/python Yoo/experiments/e06/run_pilot.py \
  --data /home/jio/사물/Yoo/data/avedore/aved_5min.pkl \
  --model /home/jio/fcsrl_src/results/e02/surrogate/surrogate.json \
  --out /home/jio/fcsrl_src/results/e06/pilot
```

전체 로그와 모델은 위 실행 폴더에 보존하고 Git에는 설정과 작은 결과 요약만 넣는다. 진행률은 각 실행의 `progress.json`, 전체 종료 상태는 `status.json`으로 확인한다.

# Yoo/e08 학습한 표현의 중간 고정 대조

## 가설과 사전 고정 조건

e07에서 encoder만 바꿔도 같은 정책망의 행동과 비용이 달라졌다. 그 효과가 실제 학습 과정에서도 나타나는지 확인한다. 원본 FCSRL의 feasibility 및 dynamics 표현 손실, PPO-Lag, 환경과 비용 정의는 유지한다. CVaR 항이나 정책 KL 보정은 이 비교에 추가하지 않는다.

| 비교군 | 처음 22회 갱신 | 나머지 22회 갱신 |
| --- | --- | --- |
| continuous | 원본 표현 학습 및 EMA | 원본 표현 학습 및 EMA 계속 |
| freeze_half | continuous와 동일 | encoder, target encoder 및 표현 보조망 가중치 고정 |

총 44회 중 절반인 22회 직후(202,048전이)에 고정한다. e07에서 검증으로 선택한 encoder 시점을 재사용하지 않는다. 이후 평가 결과에 따라 경계를 바꾸지 않는다. 두 비교군 모두 처음부터 다시 학습하고 고정 직전의 모든 저장 가중치, 관측 정규화 및 난수 상태가 일치하는지 확인한다.

정책망과 보상 및 비용 가치망은 후반부에도 학습한다. 관측 정규화도 기존처럼 계속 갱신한다. 따라서 입력 변환까지 완전히 고정하는 비교는 아니며, 표현망 가중치의 갱신 여부만 바꾼다. 처음부터 무작위 표현을 고정한 e05와 다른 대조다. 이 고정 자체를 최종 확장 기법으로 채택하지 않는다.

## 학습 및 평가

- seed 0, 1, 비교군 2개로 총 4개. 실행마다 404,096전이, 44회 갱신. CPU 1 thread씩 4개 병렬.
- 원본 FCSRL `czp16/FCSRL`의 Apache 2.0 커밋 `d882b91054d52c7a63a18a21f7db67bbbed82b63`. 수정 없이 원본 `PPOLagReprAgent.train_encoder`를 사용한다. e05와 같은 표현 구조, 손실 가중치 및 EMA다.
- 데이터 `/home/jio/사물/Yoo/data/avedore/aved_5min.pkl`, MLP `/home/jio/fcsrl_src/results/e02/surrogate/surrogate.json`. 학습 330일, 검증 43일. test는 사용하지 않는다.
- 검증 43일 x MLP 구성원 5개를 같은 순서로 5회 평가한다. 평균 비용 15 이하인 시점 중 최고 보상으로 선택하고, 만족 시점이 없으면 최소 비용 시점을 실패 표시와 함께 남긴다.
- 주요 비교는 마지막 정책의 비용과 공기량이며 중간 선택 정책, CVaR20과 seed별 변화도 보고한다. 고정 후의 변화가 한 seed에서만 좋으면 일관된 개선으로 주장하지 않는다. 공기량을 늘려 비용을 줄였는지도 함께 본다.
- 비용은 5분마다 `1[NH4 > 4] + 1[DO > 2]`를 더한 하루 합이다. reward는 음의 정규화 공기량, 공기량 단위는 Nm3/일이다. 평균 한도 15와 PID (0.02, 0.005, 0.01), 승수 상한 20을 유지한다.
- gamma 0.99, GAE 0.95, actor 학습률 0.0003, critic 0.001, PPO clip 0.2, 10 epoch, minibatch 512, KL 0.02, gradient norm 0.5. 초기 logstd -0.5, 환경 8개, horizon 1148.
- Python 3.10.19, torch 2.14.1+cu130, numpy 1.23.5, gymnasium 0.28.1. 기존 venv와 설치를 유지한다.

팀 공통 PROTOCOL 합의 전 개인 탐색 실험이다. e07에서 확인한 대리 모델의 실제 수질 과소예측은 해결되지 않았다. 여기의 차이를 실제 공정 안전성 개선으로 해석하지 않는다. 모델을 동시에 바꾸면 표현 고정의 효과를 분리할 수 없어 이번 비교에서는 유지한다.

## 코드와 검증

`train_ppo.py`는 e05 수집 및 PPO 계산을 유지하고, `scheduled_agent.py`가 지정 시점 이후 원래 구현의 `frozen` 분기를 사용한다. 고정 후 모든 표현 모듈의 SHA256이 유지되는지 매 갱신 확인한다. 경계 체크포인트는 두 비교군 모두 저장한다. checkpoint는 평가용이며 optimizer와 환경 상태를 포함한 학습 재시작 기능은 아니다.

단위 검증은 경계 전 기존 구현과 모든 가중치 및 갱신 지표 일치, 경계 후 표현 고정과 정책 학습 지속, 저장 및 복원, 잘못된 경계 거부를 확인한다. 실제 데이터의 짧은 실행은 2회 갱신 중 1회 뒤 고정해 경계를 통과하도록 구성한다. 이 짧은 실행의 성능으로 경계를 선택하지 않는다.

```bash
export PYTHONPATH=/home/jio/fcsrl_src/FCSRL
/home/jio/fcsrl_src/.venv/bin/python -m unittest discover -s Yoo/experiments/e08 -p 'test_*.py'
/home/jio/fcsrl_src/.venv/bin/python Yoo/experiments/e08/run_pilot.py \
  --data /home/jio/사물/Yoo/data/avedore/aved_5min.pkl \
  --model /home/jio/fcsrl_src/results/e02/surrogate/surrogate.json \
  --out /home/jio/fcsrl_src/results/e08/pilot
```

입력과 소스 SHA256, 실행 커밋과 명령은 runtime `manifest.json`에 기록한다. 전체 로그와 모델은 `/home/jio/fcsrl_src/results/e08/pilot`에 보존하고 Git에는 작은 결과와 재현 스크립트만 넣는다.

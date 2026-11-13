# e57: 새 보조망 이전의 고정 표현 진단

**완료. 새 네트워크의 RL 학습은 시작하지 않는다.** 실제 DARROW 기록에서 기존 FCSRL 표현과 원 관측이 현재 NH4 및 다음 한 시간 NH4 초과 적분을 얼마나 예측하는지 검사했다. 학습된 표현은 선형 예측에 도움이 됐지만 원 관측의 비선형 대조와 지속성 기준이 더 정확했다. 정책 입력에는 원 관측이 직접 남아 있었다. 따라서 단순한 농도 정보 소실을 새 Encoder나 보조망으로 해결해야 한다는 근거는 확보하지 못했다.

[결과 표](results.md), [요약](summary.json), [실행 전에 기록한 조건](protocol.json), [팀 선행 실험과 중복 대조](../../analysis/TEAM_EXPERIMENT_REVIEW_20261113.md).

## 절차와 선행 실험

실행 조건은 결과 전에 커밋했으나, Lee/Park의 선행 문서 확인은 실행 뒤에 했다. 사용자의 지적 뒤 일시 정지를 시도했지만 32.1초의 진단이 이미 완료된 상태였다. 이후 추가 실험을 실행하지 않았다. 이 순서의 오류를 사전 팀 검토로 표현하지 않는다.

- Park/e03은 BSM1에서 F 위험 점수와 원 관측 로지스틱 회귀를 비교했다. 위험 예측 표현의 유용성을 단순 관측 기준과 비교하는 질문이 겹친다.
- Yoo/e24는 Avedore에서 앞으로 한 시간의 평균 이진 위반 비용을 예측하는 보조 head를 이미 추가했다. 두 평가 모델 모두에서 개선이 확정되지 않았다.
- 이번에는 DARROW, 고정 z64 표현을 직접 읽는 probe, 연속 농도 초과 적분이라는 차이가 있다. 정책 확장의 신규성 또는 기존 실패의 해결로 주장하지 않는다.

## 보존한 구조와 확인한 사실

`Yoo/code/wwtp/ppo_fcsrl.py:policy_input`은 정규화 관측11과 고정 Encoder의 z64를 이어 붙인다. Actor와 두 Critic 모두 이 입력을 쓴다. 그러므로 z에서 어떤 정보를 읽기 어렵다고 해도 전체 정책 입력에서 그 정보가 사라진 것은 아니다. 이번에는 NH4 정규화 잘림도 없었다.

원 표현의 dynamics/feasibility 손실, EMA, prior, 정책, PID, 비용과 대리 모델은 수정하지 않았다. 기존 FCSRL final checkpoint seed0~4는 읽기만 했다. 새 정책 또는 대리 모델 학습은 0회다. Ridge와 작은 비선형 회귀기만 관측 기록의 일부 날짜에서 학습했다.

현재 비용은 `I(NH4>4)+I(max(DO)>2)`이므로 같은 DO에서 NH4가 4.1이든 10이든 즉시 비용은 같다. 초과 적분은 평가 지표지만 별도 정책 목적이 아니다. 이는 코드상 목적과 평가 지표의 차이이며, e56 악화의 주원인을 입증한 인과 결과는 아니다. 추가 head만으로 이 목적 차이가 해결되지는 않는다.

## 데이터와 probe 설계

- Kaggle DARROW. e50 prepared의 같은 원자료, 정규화 및 prior. 원 FCSRL `d882b91`, 기존 Apache-2.0 설치를 사용했다.
- 기존 train188일을 날짜순 fit112, tune38, probe holdout38로 나눴다. 개발 val40일은 별도 보고만 하고 후보/계수를 선택하지 않았다. test41일 행 추출/평가 없음.
- 기존 Encoder는 train188일 전체로 정책 학습을 했다. 이번 holdout은 probe에만 미학습 날짜다. 사전학습 Encoder 자체의 학습 밖 일반화라고 해석하지 않는다.
- 입력: 현재 관측11과 현재 기록 공기 행동. 이전 공기는 시점0에서 기록 첫 행동, 이후 기록 직전 행동이다. 미래 관측과 미래 행동은 입력하지 않는다.
- 목표: 현재 NH4 농도, 다음 4개 관측의 `max(NH4-4,0)` 합에 0.25시간을 곱한 값. 각 날짜 시점0~91, 총92개 창만 사용한다. 날짜를 넘는 창은 없다.
- 동일 seed의 학습 전 Encoder와 학습 후 Encoder를 비교한다. raw, z, initial z, raw+z, raw+initial z에 Ridge를 맞춘다. 표준화는 fit만 사용하고 alpha 0.01/1/100 중 tune MAE로 고른 뒤 다시 맞추지 않는다.
- 원 관측 HGB 비선형 대조는 100회, 깊이3, 학습률0.05 등 고정 조건을 쓴다. 지속성은 현재 NH4가 한 시간 유지된다고 보는 기준이다. 모든 seed에서 raw 대조 데이터는 같으므로 반복 수치를 독립 재현으로 세지 않는다.

이 검사는 기록 운전 아래의 예측이다. 다른 공기를 넣었을 때의 수질 정답, 정책 행동의 인과 효과, 전체 하루 제어 성능을 제공하지 않는다. NH4 한 시간 예측만 검사하므로 DO를 포함한 모든 안전 정보의 충분성을 판단할 수도 없다.

## 판단

원 관측+학습 표현의 선형 probe는 holdout에서 원 관측 선형 probe보다 약29% 낮은 MAE를 보였다. 무작위 초기 표현을 붙인 대조보다도 낮았다. 반면 원 관측 비선형 회귀기와 단순 지속성은 더 정확했다. 이는 학습 표현에 신호가 있지만 기존 관측 자체도 이미 강한 정보를 담는다는 제한된 관찰이다. 낮은 probe 오차가 좋은 행동 결정을 보장하지 않으며, probe의 표현력 한계도 남는다.

따라서 보조 head 본 학습을 자동으로 시작하지 않는다. 다음 설계는 팀의 비용 변경, 위험 벌점, 위험 날짜 표집 실험과 중복되지 않는 질문을 먼저 정해야 한다. 단순히 더 큰 MLP, GRU 또는 Transformer를 쓰는 것을 해결책으로 확정하지 않는다.

## 실행과 재현

실행 코드 `39eae82`, 실행 경로 `/home/jio/fcsrl_src/results/e57_representation_probe_20261113/`. 원 manifest에 데이터/정규화/prior, 5개 final checkpoint, 실제 import 소스의 SHA256, 코드 커밋, 날짜 목록과 환경을 기록했다. 입력과 checkpoint 해시는 종료 때 다시 비교했다. 원 결과에는 각 날짜 MAE도 있고 Git 요약에는 전체 MAE/R2 및 선택 alpha를 남겼다.

Python은 기존 `.venv-skl191`, scikit-learn 1.9.1, PyTorch 기존 cu130. GPU 없이 CPU1스레드, 5개 고정 Encoder 순차 검사. 학습된 정책 가중치 불변, 원 관측 prefix 일치, 미래 창 인덱스와 하루 경계, 분할 중복 없음, 유한 예측을 확인했다.

```bash
OMP_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' /home/jio/fcsrl_src/.venv-skl191/bin/python Yoo/experiments/e57/probe.py --out <새 경로>
python3 Yoo/experiments/e57/report.py --source /home/jio/fcsrl_src/results/e57_representation_probe_20261113 --out Yoo/experiments/e57
```

새 실험 전 [Yoo 작업 지침](../../AGENTS.md)에 따라 Lee/Park의 최신 실험을 먼저 읽는다.

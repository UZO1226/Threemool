# Yoo/e17: BSM1 평균 수질 제약의 짧은 학습 시험

## 목적과 사전 고정 조건

e16에서 연결한 물리 환경에서 기존 PPO, FCSRL+PPO와 CVaR 확장 후보를 실제로 갱신할 수 있는지 확인한다. 각 agent의 구현은 변경하지 않는다. 기존 Avedore 적용 및 한계 분석은 보존한다. 이번 4회 갱신은 구현과 수치 안정성을 확인하기 위한 개인 탐색 시험이며 수렴이나 기법 우월성을 입증하는 본 실행이 아니다.

| 항목 | 사전 설정 |
| --- | --- |
| 방법 | PPO, PPO+CVaR, FCSRL+PPO, FCSRL+PPO+CVaR |
| seed | 0, 1 |
| 학습량 | 각 23,040전이, 5,760전이 x 4갱신 |
| 수집 | 4환경 x 1,440전이, 갱신마다 완결된 하루 20개 |
| 학습 데이터 | 14일 기본 폭기로 조건화한 상태에서 건기 0일부터 6일까지의 하루 시작 상태와 유입. 날 선택 RNG는 동일 seed의 방법끼리 동일 |
| 평가 | 학습 전과 마지막 정책을 같은 상태에서 14일 연속 운전, 후반 7일을 집계. 건기 및 우기 |
| 선택 | 마지막 정책만 보고. 중간 평가로 checkpoint 또는 설정을 선택하지 않음 |
| 미사용 | 폭풍 시나리오 정책 평가, Avedore val/test, 5개 seed 본 학습 |
| 행동과 관측 | e16의 11차원 관측, 5분마다 KLa5 0~240 day^-1. 내부 적분 1분 |
| 초기 표준편차 | log_std=-1.5, KLa 단위 약 26.8 day^-1. 모든 군 동일 |
| PPO | actor 3e-4, critic 1e-3, gamma .99, GAE .95, clip .2, 10 epoch, batch 512, KL 조기 종료 .02, gradient clip .5 |
| FCSRL | e05의 원본 encoder 학습 메서드, 동역학 및 feasibility 손실, EMA .05, feasibility discount .9, symlog 63구간 [-2,4] 유지 |
| CVaR | e06의 상위 20% 표본 CVaR 항, 가중치 .03. 갱신당 tail 질량 4개 |
| 기준 제어 | 고정 KLa5=84, 240 및 목표 DO5=2 mg/L의 PI |

건기는 반복 운전 조건이고 우기도 e16에서 이미 관찰했다. 독립 최종 test라고 부르지 않는다. 학습은 기준 운전의 하루 시작 상태로 reset하지만, 평가는 후보 정책 자체가 만든 상태를 14일 연속 이어간다. 하루 reset의 편의가 연속 운전 성능으로 이어지는지 별도로 확인한다.

## 평균 수질 비용과 단위

물리 기준은 방류 유량 가중 평균 NH4 <= 4 mg N/L이다. 원본 BSM1 기술 보고서의 평균 수질 평가 정의를 따른다. NH4 순간 초과 시간은 별도로 보고하며 임의의 5% 한도를 쓰지 않는다.

각 5분 구간 내부의 NH4 농도를 n, 방류 유량을 q라고 하면 먼저 `load_t = mean(n*q)/(4*Qref)`, `flow_t = mean(q)/Qref`를 기록한다. Qref=18446 m3/day다. 하루 수집이 끝난 후 `cost_t = load_t / mean_day(flow_t)`로 정규화한다. 따라서 `sum_day(cost_t)/288 = flow_weighted_NH4/4`가 정확히 성립한다. 미래 유량을 관측에 넣거나 행동을 고르는 데 사용하지 않는다.

학습의 하루 비용 한도는 288이다. 이는 새로운 농도 허용치가 아니라 4 mg N/L를 합계 비용 단위로 나타낸 것이다. 하루별 평균 비용의 기대 제약이므로 모든 날의 기준 만족을 보장하지 않는다. 일별 농도 비율 평균과 7일 전체 유량 가중 평균도 구분해 보고한다.

기존 PID 계수 (.02, .005, .01)에 15/288을 곱한다. 같은 상대 제약 오차에 대한 기존 PID 응답을 유지하는 단위 환산이며, 결과를 본 뒤 승수 상한을 바꾸지 않는다. lambda 상한은 20이다. CVaR cost_scale도 288로 바꾸며 .03 가중치는 유지한다. actor 비용 advantage는 실제 정규화 비용으로 계산한다. 환경이 달라 이전 실험과 같은 최적화 문제가 되지는 않는다.

하루 끝은 유한한 에피소드의 terminal이다. reward/cost 가치 bootstrap을 0으로 만들고 feasibility 재귀도 종료한다. 실행 행동은 [-1,1]로 제한하지만 log probability는 제한하기 전 Gaussian 행동으로 계산한다.

보상은 전체 폭기 에너지의 음수이며 기본 폭기 5분 에너지로 나눈다. kWh/day, 혼합 에너지, 평균 NH4, 순간 초과율, 일별 NH4 CVaR20을 보고한다. 총질소, COD, BOD5, TSS도 진단용으로 기록한다. NH4만으로 전체 수질이나 공정 안전성이 입증됐다고 하지 않는다.

## 정규화와 안정성 확인

관측 정규화는 훈련용 건기 0~6일의 기본 제어 관측만으로 계산한 뒤 고정한다. 초기 통계에서 상수인 특징이 다른 행동을 모두 clip하지 않도록 11개 특징의 표준편차 하한을 [1, .5, 1, 1, .5, .2, 5, .1, .5, .5, .25]로 고정한다. 관측의 TSS/1000, 유량/10000 등 사전 단위 변환을 적용한 뒤의 척도다. 모든 군에 같은 통계를 사용한다.

학습 전 `verify.py`에서 다음을 검사한다.

- 농도 및 유량 집계 항등식, 유량 단위 변경에 대한 불변성
- 상대 제약 오차가 같은 PID 응답을 내는지
- terminal에서 GAE 및 feasibility가 다음 에피소드로 넘어가지 않는지
- 낮은 폭기와 높은 폭기에서 방류 유량이 동일한지, 실제 하루 비용 집계가 물리 지표와 일치하는지
- seed 999의 별도 5,760전이로 FCSRL+CVaR 실제 최적화 1회. 비교 seed에 포함하지 않음

각 비교 실행에서는 행동 및 관측 clipping 비율, gradient norm과 clipping 비율, 전체 갱신 KL, encoder 동기화 KL, feasibility 구간 이탈, lambda와 tail 항 크기를 기록한다. NaN, 비유한 경사나 ODE 경고가 발생하면 실행을 실패 처리한다. 검증 또는 학습 결과를 보고 추가 파라미터 탐색으로 바꾸지 않는다.

## PI 기준의 범위

IWA `BSM1_v1/BSM1/reginit.m`의 DO PI 계수 K=25, Ti=.002일, Tt=.001일, offset=144, 목표 2를 사용해 1분마다 포화 및 anti-windup 적분을 계산한다. 순환 유량은 모든 군에서 고정이다. 정확한 센서와 actuator를 가정하는 이산 구현이며, 원본의 잡음 및 지연과 NO3 제어까지 포함한 closed-loop 전체를 재현한 것은 아니다. RL 행동 간격은 5분이므로 PI는 별도의 운전 참고 기준이다.

## 데이터와 재현

시뮬레이터, IWA 원본 유입 ZIP, 패키지 고정 버전 및 라이선스는 [e16 기록](../e16/README.md)을 따른다. ZIP SHA256은 `f1b7683596512f2e432bb616d20fdc85ffdef3006e27a170b562789d30a050aa`다. 원본 FCSRL은 `d882b91054d52c7a63a18a21f7db67bbbed82b63`이며 수정하지 않는다. 기존 torch 2.14.1+cu130과 numpy 1.23.5를 유지하고 CPU 한 스레드로 실행한다.

```bash
export BSM1_ARCHIVE=/home/jio/fcsrl_src/results/e16/BSM1.zip
export PYTHONPATH=/home/jio/fcsrl_src/bsm-deps-e16
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

/home/jio/fcsrl_src/.venv/bin/python -u Yoo/experiments/e17/prepare.py \
  --steady /home/jio/fcsrl_src/results/e16/preflight_v2/steady_state.npz \
  --out /home/jio/fcsrl_src/results/e17/prepared_v2
/home/jio/fcsrl_src/.venv/bin/python -u Yoo/experiments/e17/verify.py \
  --prepared /home/jio/fcsrl_src/results/e17/prepared_v2 \
  --out /home/jio/fcsrl_src/results/e17/verification.json
/home/jio/fcsrl_src/.venv/bin/python -u Yoo/experiments/e17/run_pilot.py \
  --prepared /home/jio/fcsrl_src/results/e17/prepared_v2 \
  --out /home/jio/fcsrl_src/results/e17/pilot
```

출력 디렉터리는 새 경로를 사용해야 한다. 준비 실행 커밋은 `69e1a57`이며 metadata.json에 전체 커밋과 실행 소스 해시가 기록돼 있다. 최초 준비 실행은 수질 보조 지표 API 호출을 점검하다 중단했고, 수정 후 준비를 다시 수행했다. 비교 학습 실행 커밋과 소스 해시는 pilot/manifest.json에 고정한다. 원시 전이, 준비 상태, 체크포인트와 전체 로그는 서버 경로에 보관하고 작은 요약만 Git에 기록한다.

진행률은 `pilot/status.json`, 개별 실행의 `progress.json` 및 tqdm 로그에서 확인한다. 준비와 검증이 통과한 뒤에만 비교 학습을 실행한다. 현재 단계는 결과를 보고 채택하기 위한 본 실행이 아닌 사전 학습 시험이다.

# e46 시작 수온 조건별 NH4 제약

## 가설과 변경 범위

e45에서는 동일 평가 조건에도 학습 날짜와 개발 날짜의 NH4 격차가 남았다. 추가 사후 집계에서 FCSRL padded seed1은 학습 전체 NH4 평균4.999지만 가장 추운 일평균 수온25% 날짜에서는11.536이었다. 전체 평균 제약이 취약한 조건을 가릴 수 있다는 단서다. 이 수치는 원인을 확정하지 않으며 일평균 수온을 미래에 알고 운전할 수 있다는 뜻도 아니다.

이번에는 **에피소드 시작 때 측정한 수온**으로 학습188일의 하위25% 경계를 정한다. 이 경계 이하를 cold, 나머지를 normal로 고정한다. 하루 내내 같은 시작 조건 그룹을 적용하고 미래 수온은 그룹 정의에 사용하지 않는다. 경계는 개발/테스트 날짜를 보지 않고 학습 날짜만으로 계산한다. 일평균을 사용한 앞선 진단의25%와 같은 날짜 집합이라고 가정하지 않는다.

대조는 기본 FCSRL과 조건별 NH4 FCSRL, seed0/1의4개다. 같은 초기 가중치와 환경 날짜 난수, 각66회 x6080=401280전이로 처음부터 학습한다. 기존 e44 정책을 이어 학습하지 않는다.

| 항목 | baseline | conditional |
| --- | --- | --- |
| NH4 제약 | 전체 날짜 평균 <=6 | cold 평균 <=6, normal 평균 <=6 |
| DO 제약 | 전체 날짜 평균 <=6 | 동일 |
| 승수 | NH4, DO의2개 PID | cold NH4, normal NH4, 전체 DO의3개 PID |
| 비용 Critic | NH4/DO의2개 | 그대로2개 |
| 표현 | 원 FCSRL dynamics/합계 feasibility, EMA | 동일 |
| 관측 | e44 padded: 원11개+0다섯 개 | 동일 |
| Actor 보정, 새 feasibility head | 없음 | 없음 |

정책과 환경 구조를 유지하고 제약의 조건별 집계와 정책 손실의 승수 적용만 바꾼다. 정책에 조건 ID나 새 센서를 추가하지 않는다. 추론 시 승수나 미래 날짜 정보는 필요하지 않다. 이는 기존 평균 제약보다 강한 학습 목표이며 동일 목적함수의 단순 파라미터 튜닝으로 주장하지 않는다.

## 수식과 구현

학습 날짜를 균등 추출하는 기존 방식은 유지한다. 그룹 g의 학습 날짜 비율 p_g를 고정하고, J_N,g를 해당 그룹의 에피소드 NH4 비용 평균으로 둔다. 양의 p_g로 재매개화한 Lagrangian의 비용 항을 다음처럼 구성한다.

```text
sum_g p_g * lambda_g * (J_N,g - 6) + lambda_DO * (J_DO - 6)
```

따라서 각 전이의 NH4 advantage에는 에피소드 그룹의 lambda_g를 적용한다. 기존의 reward advantage 정규화와 cost advantage 중심화를 유지한 결합은 다음과 같다.

```text
(A_r - lambda_group * A_N - lambda_DO * A_DO)
    / (1 + sum_g p_g * lambda_g + lambda_DO)
```

분모는 모든 전이에 같은 값이다. 그룹마다 reward 크기가 달라지는 추가 효과를 피하고, 두 NH4 승수가 같으면 기존 결합식으로 돌아간다. p_g로 가중한 dual을 사용하는 재매개화이며 그룹확률 역수를 실수로 빠뜨린 식이 아니다. PPO/GAE와 PID를 사용하는 실용적 근사이고 이론적 안전 보장을 주장하지 않는다. 원본의 할인/에피소드 비용 차이는 이번에 변경하지 않는다.

`conditional.py`의 업데이트는 e33 DualMixin.update를 복사하여 advantage 결합 부분만 바꿨다. 두 Critic의 목표/손실과 원 `PPOLagReprAgent.train_encoder` 호출, EMA, PPO clipping/KL 중단은 유지한다. 새 그룹 Critic은 만들지 않는다. 두 비용 advantage와 reward의 스케일은 기존 e44와 같다.

원 PID gains와 상한20을 그대로 사용한다. 그룹별 비용은 완전한95전이 에피소드 합으로 계산하며 그 그룹의 에피소드 수로 평균낸다. 해당 업데이트에서 관측되지 않은 그룹은 PID 내부 상태까지 갱신하지 않는다. λ 초기값은 모두0이며 처음1회는 기존과 같은 업데이트가 된다. 희소 그룹의 표본 수/비용/승수를 매번 기록한다.

## 고정 데이터와 평가

- Kaggle DARROW, e30 준비물의 train188/val40/test41 분할 유지. 기존 e31 계절 혼합을 반복하거나 개발 날짜를 학습에 넣지 않는다.
- e35 corrected_surrogate.pkl의 구성원5개 고정. 새 환경 학습 없음. 각 학습 수집에서 구성원 선택도 기존과 같다.
- 총 공기량0~9000 m³/h, 기존 기록 zone 분배, reward=-Q/9000 유지.95전이=23.75h, 공기 부피=.25*sum(Q). 실제 kWh 미측정.
- NH4>4 및 max(DO)>2 표본 수를 비용으로 사용. 농도 임계값과 기존 전체 평균 평가 한도6/6을 바꾸지 않는다.
- 최종 정책만 train188/val40 각각 모든 모델5개로 결정적 평가한다. 조건별 평균, 초과 적분과 최악 날짜도 저장한다. test 행은 평가용으로 추출하지 않으며 test 평가 없음.
- 원 코드 커밋 d882b91054d52c7a63a18a21f7db67bbbed82b63, Apache 2.0. 기존 venv Python3.10.19, torch2.14.1+cu130, numpy1.23.5, sklearn1.7.2 유지. CPU 한 스레드 작업2개씩.

공통 프로토콜v1과 다른 Yoo 개인 탐색이며 공통 평가 조건과 공식 채택을 변경하지 않는다. FCSRL 적용 및 제한적 확장 검증이고 조건별 제약 자체의 신규성을 단정하지 않는다. 학습 범위 밖의 저수온에 대한 개선이나 실제 공정 안전을 보장하지 않는다.

## 사전 검사와 판정

미래 수온 교란에도 그룹 불변, 초기 가중치 일치, e44 수집 경로 정확 일치, 완전한 에피소드64개 그룹 비용 독립 집계, 같은 승수의 advantage 일치, 승수0일 때 원 업데이트의 모든 가중치 정확 일치, 조건별 PID 및 미관측 그룹 skip, Encoder/EMA 실제 갱신, prior/normalizer 동결, checkpoint 복원, 승수 활성화 뒤 유한한 갱신을 검사한다. 사전 검사는 train만 사용한다. 실패하면 본 실행을 시작하지 않는다.

본 실행 중 NaN, 실행 오류 또는 source/input 해시 변화가 있으면 중단한다. 완료 후 저장 경로를 독립 재집계하며 baseline의 val 경로가 e44의 같은 seed와 재현되는지도 확인한다.

사전 지정 추가 검증 후보 조건은 각 seed에서 **val 공기/NH4/DO 모두 감소, 두 초과 적분 및 최악 날짜 NH4/DO 악화 없음, val 전체 평균6/6 만족, train의 두 조건 NH4 평균 각각6 이하**다. 두 seed 모두 통과해도 확증이나 공식 채택은 아니다. 그룹별 개선과 공기/DO 악화의 상충을 숨기거나 기준을 사후 완화하지 않는다. 추후 확률적 평가와 추가 seed는 탐색 결과를 보고 별도 계획으로 정한다.

## 실행

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE \
/home/jio/fcsrl_src/.venv/bin/python -u Yoo/experiments/e46/preflight.py \
  --prepared /home/jio/fcsrl_src/results/e30/prepared \
  --corrected /home/jio/fcsrl_src/results/e35/zone_fix/corrected_surrogate.pkl \
  --out /home/jio/fcsrl_src/results/e46/preflight

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE \
/home/jio/fcsrl_src/.venv/bin/python -u Yoo/experiments/e46/run.py \
  --prepared /home/jio/fcsrl_src/results/e30/prepared \
  --corrected /home/jio/fcsrl_src/results/e35/zone_fix/corrected_surrogate.pkl \
  --verification /home/jio/fcsrl_src/results/e46/preflight/verification.json \
  --out /home/jio/fcsrl_src/results/e46/main
```

runtime manifest.json에 실제 코드 커밋, 경계/그룹별 날짜 수, 입력/소스 해시, 실행 명령을 기록한다. status.json은 전체 진행, 각 실행의 progress.json과 tqdm 로그는 갱신 수와 예상 시간을 기록한다. 전체 로그는 `/home/jio/fcsrl_src/results/e46/main.log`, 큰 checkpoint와 경로는 runtime에만 보관한다. 이 문서 작성 시점에는 사전 검사 전이며 최종 결과는 없다.

## 사전 검사 완료

실행 커밋 `f7d026e`,105.73초에 모든 사전 검사를 통과했다. [검사 원본](verification.json). 두 군의 초기 가중치, 원 수집 경로, 승수0에서의 전체 업데이트 가중치가 정확히 일치했다. 동일 NH4 승수의 기존 advantage 환원,64개 에피소드의 그룹별 독립 집계, Encoder/EMA 갱신, prior/normalizer 동결, 조건별 승수 활성화와 미관측 그룹 skip, checkpoint 복원을 확인했다. 미래 수온을 교란해도 그룹 정의가 그대로였다. 검사에는 train만 사용했다.

확정 경계는 시작 수온18.2049575434°C, 학습은 cold47일/normal141일이다. 이 경계를 그대로 개발 날짜에 적용하면 cold39일/normal1일이다. 일반 조건의 개발 성과는 하루뿐이므로 안정적 일반화 근거로 해석하지 않는다. 경계나 그룹 수를 개발 결과에 맞춰 바꾸지 않는다. 사전 검사의3회 진단 업데이트는 폐기하고 본 실행은 같은 초기 정책에서 새로 시작한다.

## 본 실행 시작

실행 커밋 `a8a5e0f7e95211b8a21cd62263a6efc02b014a56`, 시작 UTC `2026-11-12T05:42:20.302035+00:00` (한국 시각14시42분경), 감독 PID 3923824. baseline/conditional seed0을 먼저 실행 중이며 이어 seed1을 실행한다. source/input/검사 해시는 runtime manifest.json에 고정했다. 완료 후 report.py를 자동 실행한다. 진행 상태는 runtime status.json과 각 run의 progress.json을 따른다.

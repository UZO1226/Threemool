# e33: NH4와 DO의 평균 제약을 분리하는 DARROW 적용 시험

후속 정정(e35): 이 실험이 사용한 e30 대리 환경의 DO-공기 구간 단조 제약에 대응 오류가 확인됐다. DO1/2/3의 직접 폭기 입력은 air3/4/5인데 air1/2/3으로 제약했다. 기존 정책 결과는 해당 환경 안의 역사적 비교이며 실제 공정 안전이나 효율의 증거로 사용하지 않는다. [원출처와 수정 대조](../e35/README.md).

## 목적

e31에서 DO 초과를 줄이면 NH4 위반 증가가 합산 비용에 가려지는 현상을 확인했다.
e32에서는 e30의 PPO/FCSRL 비용 차이가 공기량을 맞추면5.133에서1.387로 줄고, FCSRL이 학습 구간보다 검증 구간에서 크게 악화됨을 확인했다.
사용자가 후속 실행을 지시해, 먼저 비용을 합쳐 제약하는 방식만 분리한다. 새로운 알고리즘 기여를 주장하는 실험이 아니다.

## 사전 고정 조건

데이터/분할/전이 모델/관측/행동/초기 운전 정책/정규화는 e30을 그대로 사용한다.
원자료는 사용자 지정 Kaggle DARROW이며 준비물은 `/home/jio/fcsrl_src/results/e30/prepared`다.
관측11차원, 총 공기량0~9000 Nm³/h, 기록 구간 배분, 학습188일/검증40일/보류41일을 유지한다.
계절 혼합 분할의 e31 결과와 직접 비교하지 않는다. test를 평가하지 않는다.

| 항목 | 기존 e30 | e33 |
| --- | --- | --- |
| 정책 제약 | NH4 위반 + DO 초과의 평균 합 ≤12 | NH4 평균 ≤6 이고 DO 평균 ≤6 |
| 비용 가치 출력 | 합계1개 | 같은 은닉 구조에서 NH4/DO 출력2개 |
| PID 승수 | 합계1개 | NH4/DO 각각1개 |
| FCSRL dynamics loss | 원본 함수 | 유지 |
| FCSRL feasibility 표적 | 두 이진 비용 합의 할인 max 재귀 | 유지 |
| 표현 EMA와 Actor 보정 | e30/e29 방식 | 유지 |

한도6/6은 **학습 구간 실제 기록**의 평균 NH4 위반5.622340, DO 초과5.026596을 각각 올림한 값이다.
검증 정책 결과로 한도를 고르지 않았으며 법적 기준이나 공정 운영자의 승인 한도가 아니다.
95전이(23.75시간)당 위반 횟수의 기대값을 제약한다. 모든 날/모든 순간 제약 만족을 보장하는 방식은 아니다.
각 한도가6이므로 합계 한도12와 숫자 합은 같지만 허용되는 정책 집합은 더 좁다. 난도가 같은 실험이라고 주장하지 않는다.

정책 이점은 `(A_r - lambda_N A_N - lambda_D A_D) / (1 + lambda_N + lambda_D)`다.
보상 이점만 표준화하고 각 비용 이점은 평균만 뺀다. 각 PID는 .02/.005/.01, 상한20이다.
NH4 비용이9이고 DO 비용이0이면 NH4 승수만 증가한다. DO의 여유가 NH4 초과를 상쇄하지 못한다.
feasibility 보조 학습에는 e30의 합산 위험 표적을 유지하므로, 원본 표현 학습을 제거하거나 두 목표에 맞춰 새 손실을 만들지 않는다.
따라서 이 시험은 정책 제약 적용 패키지(두 가치 출력과 두 PID)의 효과를 보며 PID만의 기여를 분리하지는 않는다.

## 구성과 판정

PPO Lag, FCSRL + PPO Lag, FCSRL + Actor 보정 x seed0,1 = 신규6개.
각401,280전이(8환경 x760 x66), 마지막 정책을 검증40일 xGBM5개에서 평가한다.
e30 동일 seed의 원 합산 조건도 NH4/DO 각각6 기준으로 다시 집계한다. 기존 결과를 새 제약 학습 결과로 취급하지 않는다.
두 평균 한도를 동시에 만족해야 제약 충족으로 센다. 합계12 이하만으로 성공 처리하지 않는다.
공기량과 NH4/DO 비용을 모두 보고, 두 제약 충족 상태에서의 공기량을 우선 비교한다.
seed2개는 탐색이며 자동 채택/통계적 우위/현장 안전성을 주장하지 않는다. 실패 시 원인 항목과 비용을 그대로 보존한다.
다음 단계인 관측 정보 보완과 시간 할인 조정은 이번 학습에 섞지 않는다.

## 구현과 검증

- `dual.py`: e30/e29 에이전트를 재사용하고 비용 가치 출력과 PID/이점 결합만 분리한다.
- `train.py`: e30 수집 경계와 GAE를 유지하며 각 비용별 GAE를 계산한다. 원래 합산 비용은 표현 학습 표적에 남긴다.
- `verify.py`: 비용 분해, PID 독립성, 이점 수식, raw log probability, 시간 제한 경계, 저장/복원과 Actor-only 보정의 비활성 head 불변을 검사한다.
- 초기 승수0에서 e30 대비 actor, log std, encoder/EMA 및 난수열의 첫 갱신이 정확히 같음을 확인한다. 이후 정책 경로가 달라지는 것은 제약 변경의 결과다.
- `feasibility_check.py`: 학습 날짜만 써 초기 정책과 기록 공기량 재생을 점검한다. 가능한 예를 못 찾았다고 수학적으로 불가능하다고 결론내리지 않는다.
- `run.py`는 e30 실행기를 사용해 실패 시 중단하고 `report.py`로 자동 집계한다.

원본 FCSRL은 `czp16/FCSRL`, `d882b91054d52c7a63a18a21f7db67bbbed82b63`, Apache-2.0이며 수정하지 않는다.
자체 PPO/FCSRL 결합과 초기 정책은 `Yoo/code/wwtp`, e23/e25/e29/e30을 사용한다. 기준 main `6c2035a`.
예측 환경의 DO 낮은 탐지율과 공기량 외삽 문제는 해결된 것이 아니다. 비용 설계 수정만으로 환경 신뢰도가 좋아졌다고 해석하지 않는다.
CPU는2개 프로세스로 실행하며 각 torch thread1을 명시한다. e30은 DeviceConfig가 thread수를 다시 바꿨으므로 실행 시간의 공정한 대조로 사용하지 않는다.

## 실행

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE
PY=/home/jio/fcsrl_src/.venv/bin/python
DATA=/home/jio/fcsrl_src/results/e30/prepared
OUT=/home/jio/fcsrl_src/results/e33
$PY Yoo/experiments/e33/verify.py --prepared "$DATA" --out "$OUT/verification.json"
$PY Yoo/experiments/e33/feasibility_check.py --prepared "$DATA" --out "$OUT/feasibility_check.json"
$PY Yoo/experiments/e33/run.py --prepared "$DATA" --out "$OUT/smoke" --seeds 1 --updates 1 --jobs 2
$PY Yoo/experiments/e33/run.py --prepared "$DATA" --out "$OUT/main" --jobs 2
```

실행 상태와 해시는 `main/status.json`, `main/manifest.json`, 각 실행은 `runs/<method>/seed_<n>/progress.json`에 남긴다.
진행률 표시: `python3 Yoo/experiments/e30/status.py /home/jio/fcsrl_src/results/e33/main`.
끝나면 `main/results.md`, `main/summary.json`을 자동 생성한다.

## 완료 결과

**2026-11-10 15:35:42 KST 신규6개와 평가/집계를 모두 정상 완료했다.** 각401,280전이, 모든 종료 코드0. 현재 실행 중인 e33 작업은 없다.
실행 코드 `f91571be92e90ffb00b36c586b4e2d554c7f7471`. [전체 결과](results.md), [seed별 수치](summary.json).

| 방법 | 기존 NH4 비용 평균 | 분리 후 NH4 비용 평균 | 분리 후 DO 비용 평균 | 공기량 변화 | 두 평균 한도 충족 seed |
| --- | --- | --- | --- | --- | --- |
| PPO | 10.50 | 7.13 | 1.91 | +3.90% | 1/2 |
| FCSRL | 19.65 | 15.83 | 2.06 | +17.31% | 1/2 |
| FCSRL + Actor 보정 | 9.88 | 6.95 | 1.68 | +2.43% | 0/2 |

비교는 e30의 같은 seed0,1이며 기존 세 방법 모두 분리 기준6/6으로 재집계하면0/2다.
DO는 신규6개 모두 평균 한도6을 만족했다. 남은 실패 항목은 NH4다.
기본 FCSRL의 NH4 비용은 seed0에서27.73, seed1에서3.93으로 크게 갈렸다. Actor 보정은7.65/6.24로 격차가 작지만 둘 다6을 초과했다.
PPO는8.27/5.995로 한 seed만 통과했다. 5.995의 통과를 여유 있는 만족이라고 해석하지 않는다.
세 방법의 평균 NH4 비용은 낮아졌지만 공기량도 증가했다. 같은 공기량 효율 개선을 입증한 결과가 아니며 seed2개의 탐색 결과다.
한도 분리로 목적의 상쇄는 막았지만, 안정적인 NH4 제약 만족과 검증 구간 일반화 문제는 남아 있다. 모델 신뢰도 문제도 해결되지 않았다.

초기 BC 정책의 학습 비용은 NH4=3.420, DO=2.363으로 두 한도를 만족했다. 이는 모델의 학습 구간에서 실현 가능한 예이며, 검증 구간이나 실제 공정의 안전 보장은 아니다.
다음 원인 분리는 사전에 정한 순서대로 현재 관측 누락을 점검하는 것이다. 이번 결과만 보고 제약 한도를 완화하거나 여러 파라미터를 동시에 바꾸지 않는다. 후속 학습은 아직 시작하지 않았다.

# Yoo/e21: 설정값 결정 주기 30분과 빠른 lambda 반응

## 목적

e20 은 7일 연속 학습으로 하루 reset 착취를 대부분 없앴지만, 수질 기준을 못 지켰고(건기 NH4 2.99 ~ 3.12, 기준 2.69 이하) 시간대별 배분도 배우지 못했다(경계 대비 에너지 −1.0 ~ +0.5%). e21 은 두 원인을 겨냥한다.

1. 탐색: 5분마다 독립인 탐색 잡음은 몇 시간 단위로 반응하는 공정에서 평균으로 묻힌다. 설정값 결정 주기를 30분으로 늘린다(PI 는 1분 그대로). 하루 결정 수가 288 에서 48 로 줄어 같은 탐색 잡음이 공정에 더 오래 작용한다
2. 제약: e20 에서 lambda 가 100회째에도 오르는 중이었다. PID 반응을 e18 환산의 5배에서 8배로 높인다

## e20 대비 바꾼 것 (나머지는 같다)

| 항목 | e20 | e21 |
| --- | --- | --- |
| 설정값 결정 주기 | 5분 | 30분 (`--interval 6`, 같은 설정값을 5분 x 6 유지) |
| 갱신당 결정 수 | 4환경 x 2,016 | 4환경 x 336 (시뮬레이터 스텝은 같은 8,064) |
| lambda 반응 | 5배 | 8배 |

- 결정 주기를 바꿔도 비용 단위는 같다. 30분 구간의 load 는 5분 구간 load 의 합, 유량은 평균으로 모아 하루 정규화 비용의 합이 5분 결정과 같다 (상수 설정값 2.0 의 2일 비용 합이 소수점 끝까지 같음을 확인)
- gamma 0.99 는 결정 단위라 유효 지평이 약 50시간으로 길어진다
- 평가에서도 정책은 30분마다만 다시 계산한다

비용 한도, 제어 구조, 같은 출발점, 7일 연속 학습, 평가와 비교 기준은 e20 과 같다.

## 단계와 판정 기준 (결과를 보기 전에 정함)

e20 과 같다.

- 1단계: PPO-Lag, FCSRL + PPO-Lag x seed 0, 1. 건기 주 평가에서 4개 중 3개 이상이 NH4 를 DO PI 의 1.05배(2.69) 이하로 지키면 2단계로 간다
- 2단계: seed 2, 3, 4 를 더해 e19 의 기준 1 ~ 4 를 seed 5개로 판정한다

## 실행

```bash
export BSM1_ARCHIVE=/home/jio/fcsrl_src/results/e16/BSM1.zip
export PYTHONPATH=/home/jio/fcsrl_src/bsm-deps-e16:$PWD/Yoo/code/wwtp
PY=/home/jio/fcsrl_src/.venv/bin/python
cd Yoo/experiments/e18
$PY run.py --prepared /home/jio/fcsrl_src/results/e19/prepared --out /home/jio/fcsrl_src/results/e21/stage1 \
    --tanks 3 4 5 --pid_mult 8 --continuous_days 7 --interval 6 --updates 100 --seeds 0 1 --jobs 4 --desc "e21 1단계"
$PY ../e19/report.py --prepared /home/jio/fcsrl_src/results/e19/prepared --main /home/jio/fcsrl_src/results/e21/stage1 --out ../e21
```

## 결과

### 1단계 (2026-11-08 19:05 ~ 20:03 KST, 4개 정상 종료)

| 방법 | seed | 건기 폭기 kWh/일 | 건기 NH4 | 경계 대비 에너지 | 학습 날짜 하루 reset NH4 | 7일 연속 NH4 | 마지막 lambda |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PPO-Lag | 0 | 3442.5 | 2.602 | +0.28% | 2.495 | 2.591 | 2.13 |
| PPO-Lag | 1 | 3359.4 | 2.564 | −2.90% | 2.501 | 2.543 | 2.34 |
| FCSRL + PPO-Lag | 0 | 3502.6 | 2.474 | −1.22% | 2.439 | 2.462 | 1.30 |
| FCSRL + PPO-Lag | 1 | 3485.9 | 2.472 | −1.74% | 2.444 | 2.471 | 0.61 |

진행 조건(4개 중 3개 이상이 NH4 2.69 이하) 통과. 4개 모두 통과했다. 2단계(seed 2, 3, 4)를 바로 시작했다 (20:03 KST).

- 30분 결정 주기로 바꾼 뒤 처음으로 시간대별 배분 효과가 보였다 (경계 대비 음수 3개)
- FCSRL + PPO-Lag 는 lambda 가 0.6 ~ 1.3 에서 안정되고 학습 rollout NH4 도 한도 안(2.44 ~ 2.47)이다. PPO-Lag 는 lambda 가 아직 오르는 중이고 rollout NH4 가 2.56 ~ 2.63 으로 한도 2.49 를 조금 넘는다

<!-- STAGE2 -->


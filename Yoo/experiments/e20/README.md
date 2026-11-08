# Yoo/e20: 7일 연속 에피소드 학습으로 하루 reset 착취 제거

## 목적

e19 에서 하루씩 reset 하고 하루 끝을 종료로 처리하는 학습이, 오늘 덜 폭기해 생긴 피해를 다음 날로 넘기는 정책을 만든다는 것을 확인했다(같은 학습 날짜에서 하루 reset NH4 2.80 ~ 3.21, 7일 연속 3.42 ~ 4.06). e20 은 학습 자체를 연속 운전으로 바꿔 이 착취를 없앤 뒤 e19 와 같은 기준으로 판정한다.

## e19 대비 바꾼 것 (나머지는 같다)

| 항목 | e19 | e20 |
| --- | --- | --- |
| 학습 에피소드 | 하루 (건기 0 ~ 6일 중 무작위, DO PI 운전 상태로 reset) | 7일 연속 (건기 0일 시작 상태에서 0 ~ 6일) |
| 하루 끝 | 종료 (가치 0) | 이어지는 시간 |
| 에피소드 끝 | 종료 | 시간 제한. 마지막 상태의 가치로 이어 붙인다 |
| 갱신당 전이 | 4환경 x 1,440 = 5,760 | 4환경 x 2,016 = 8,064 (환경마다 7일 한 번) |
| 학습량 | 100회, 576,000전이 | 100회, 806,400전이 |
| lambda 반응 | e18 환산의 3배 | 5배. e19 에서 100회째에도 lambda 가 오르는 중이었다 |

비용 한도(179.467, 학습 날짜에서 설정값 2.0 의 평균 NH4 2.493), 제어 구조(3, 4, 5번 조 DO PI, 공통 설정값), 같은 출발점, PPO 설정, value_scale 100, 평가 방식, 비교 기준(e19 준비 폴더)은 e19 와 같다.

학습 날짜가 7일 하나뿐이라 4개 환경이 같은 유입을 보고 탐색 잡음만 다르다. 평가는 학습하지 않은 건기 7 ~ 13일과 우기다.

## 단계와 판정 기준 (결과를 보기 전에 정함)

- 1단계: PPO-Lag, FCSRL + PPO-Lag x seed 0, 1. 건기 주 평가에서 4개 중 3개 이상이 NH4 를 DO PI 의 1.05배 이하로 지키면 2단계로 간다. 아니면 원인을 먼저 본다
- 2단계: seed 2, 3, 4 를 더해 seed 5개로 e19 와 같은 기준 1 ~ 4 를 판정한다

e19 기준(건기 주 평가, seed 5개):
1. 수질 유지: 평균 NH4 가 DO PI 의 1.05배 이하
2. 개선: 경계 대비 에너지 평균 < 0, 95% 신뢰구간 상한 < 0
3. 규칙 대비: 경계 대비 에너지가 유입 부하 비례 규칙(−3.01%)보다 작다
4. FCSRL 효과: 두 방법의 경계 대비 에너지와 NH4 를 Welch t 검정으로 비교, p < 0.05 일 때만 차이로 쓴다

추가 진단: 마지막 정책의 학습 날짜 하루 reset 평가와 7일 연속 평가를 함께 기록해, 둘의 차이가 e19 보다 줄었는지 본다.

## 실행

```bash
export BSM1_ARCHIVE=/home/jio/fcsrl_src/results/e16/BSM1.zip
export PYTHONPATH=/home/jio/fcsrl_src/bsm-deps-e16:$PWD/Yoo/code/wwtp
PY=/home/jio/fcsrl_src/.venv/bin/python
cd Yoo/experiments/e18
$PY run.py --prepared /home/jio/fcsrl_src/results/e19/prepared --out /home/jio/fcsrl_src/results/e20/stage1 \
    --tanks 3 4 5 --pid_mult 5 --continuous_days 7 --updates 100 --seeds 0 1 --jobs 4 --desc "e20 1단계"
$PY ../e19/report.py --prepared /home/jio/fcsrl_src/results/e19/prepared --main /home/jio/fcsrl_src/results/e20/stage1 --out ../e20
```

## 결과

<!-- RESULTS -->

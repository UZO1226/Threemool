# e61: 학습 비용 한도를 묶이는 2.0 으로 낮춘 학습 중 차단

**2026-11-13 20:21 KST 본 학습 시작.** [사전 조건](protocol.json), [실행기](train_limit.py), [검사](check.py), [실행](run.py), [집계](report.py).

## 질문

e60 은 학습 중 차단으로 PPO-Lag 79,360 m³/23.75h, 합산 비용 4.81 에 도달했지만 10개 정책 모두 라그랑주 승수가 끝까지 0 이었다. 학습 수집의 하루 비용 (마지막 10갱신 평균 1.81 ~ 3.28) 이 한도 12 보다 훨씬 낮아 비용 신호가 정책에 전달되지 않았다. e61 은 한도를 2.0 으로 낮춰 승수가 실제로 움직이게 하고, 차단 아래 남은 DO 위반과 공기량이 함께 줄어드는지 본다.

## 팀 선행 실험과의 차이

- Park/e12 B 는 한도를 6 으로 낮췄지만 학습 비용 2.16 ~ 4.42 보다 높아 한 번도 묶이지 않았다. Park 문서에도 한도 효과는 시험되지 않았다고 적혀 있다.
- Park/e13 은 한도 대신 차단 개입 횟수에 비용을 줘 승수를 살렸다. 의존은 줄었지만 공기 +7 ~ 11%, DO 위반 증가였다. e61 은 개입이 아니라 실제 NH4 와 DO 위반에 압력을 준다.
- Park/e14, e15 의 학습 중 양방향 차단과 여유 표는 바꿔 끼운 것보다 낫지 않았다. 그래서 학습 중 차단 종류는 바꾸지 않는다.

## 조건

e60 과 한 가지만 다르다. 학습 PID 의 비용 한도 12 를 2.0 으로 바꾼다. 이 값은 `learner.pid.update(..., c.cost_limit)` 에만 쓰인다. 코드 (`e60/train_shielded.py` 수정 없음), 데이터, 대리 모델, 학습 중 차단 park_e10, 모방 prior, 장치 (CPU1 수집, PPO CPU1, FCSRL GPU1 갱신), seed 0~4, 401,280전이, 평가 (검증 40일 x 구성원 5개, 차단 없음, park_e10, park_e19) 는 같다. 평가 지표의 한도는 12 그대로다. test 는 평가하지 않는다.

2.0 은 결과 전에 하나만 정했다. e60 학습 비용 약 2.5 보다 낮아 묶이고, 학습 날짜 차단 아래 NH4 비용 바닥 약 1.2 보다 0.8 위라 DO 를 줄이면 지킬 수 있는 값이다. 바닥보다 낮으면 지킬 수 없는 제약이 되어 승수가 상한 20 까지 커진다. 다른 한도 값은 탐색하지 않는다.

## 결과 전에 고정한 판정

[protocol.json](protocol.json) 에 적었다.

- J0 결속: 승수 최댓값이 0.05 를 넘은 seed 가 4/5 미만이면 한도가 묶이지 않은 것으로 보고 J1 을 판정 불가로 둔다.
- J1 주 비교: 같은 seed 의 e61 minus e60 (val park_e19). 공기와 합산 비용이 모두 낮은 seed 4/5 이상이고 두 평균이 모두 낮으면 개선, 비용만 낮고 공기가 늘면 교환.
- J2: Park e19 H1 (80,748 / 4.98) 보다 두 평균이 낮고 seed 4/5 이상이면 Park 단일 정책 대표를 넘는 탐색 후보. 짝 비교가 아니다.
- J3: Park e22 H1 앙상블 (79,633 / 4.68) 대비 보고. J4: 차단 의존. J5: 세 지표와 tail 공개.

## 사전 검사 (통과)

`/home/jio/fcsrl_src/results/e61/check/checks.json`, 실행 코드 `f74ce91` (검사 결과 저장 수정 포함).

| 검사 | PPO-Lag | FCSRL |
| --- | --- | --- |
| 바꾼 설정이 한도 외 e60 설정과 같음 | 통과 | 통과 |
| 한도 12 로 e60 학습 고리를 돌리면 seed 0 의 첫 두 갱신 기록이 시간 항목 외 모두 같음 | 통과 | 통과 (GPU 갱신 포함) |
| 한도 2.0 의 첫 갱신은 승수 외 같고 승수 > 0 | 0.0738 | 0.1466 |
| 한도 2.0 의 두 번째 갱신은 e60 과 다름 | 통과 | 통과 |
| 실제 명령줄 1갱신 학습, 저장, 다섯 평가, 설정 한도 2.0, 승수 일치 | 589초 | 648초 |

검사 과정의 실수 세 가지를 기록한다. 학습이나 결과 계산은 바뀌지 않았다.

1. 첫 작성본의 실행기가 e60 실행 함수가 읽는 `resume` 인자를 넘기지 않았다. 코드 검토에서 발견해 실행 전에 고쳤다.
2. 첫 검사 실행 (`check_v1_jsonfail`) 은 계산을 마쳤지만 numpy 참거짓 값을 JSON 으로 저장하지 못했다. 값 변환을 넣고 처음부터 다시 실행했다.
3. 두 번째 실행은 평가 개수 검사식이 metrics 의 문자열 항목 `train_shield` 를 평가로 세어 6개로 실패 표시를 했다. 실제 평가는 5개다. 검사식만 고치고 저장된 값으로 다시 판정했으며, 원래 판정은 `passed_original` 로 남겼다.

## 실행

```bash
cd Yoo/experiments/e61
export OMP_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
/home/jio/fcsrl_src/.venv-skl191/bin/python check.py --out <새 검사 경로>
/home/jio/fcsrl_src/.venv-skl191/bin/python -u run.py --check <검사 경로>/checks.json --out <새 본 실행 경로> --jobs 5
```

본 실행 `/home/jio/fcsrl_src/results/e61/main/`, tmux `e61`. 실행 중에는 검사 해시 목록의 파일 (e59 `shield_eval.py`, `min_brake.py`, e60 `train_shielded.py`, 이 폴더의 `train_limit.py` 등) 을 고치지 않는다. 완료되면 `results.md`, `summary.json` 을 이 폴더에 자동 생성한다.

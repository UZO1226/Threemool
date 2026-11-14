# e63: 총 공기와 구간 배분을 함께 정하는 정책

**2026-11-14 19:00 KST 본 학습 시작.** [사전 조건](protocol.json), [배분 행동과 평가](alloc.py), [학습](train_alloc.py), [교차 평가](eval_cross.py), [검사](check.py), [실행](run.py), [집계](report.py), [준비물](prepare.py).

## 출발점

e62 에서 차단 아래 NH4 비용의 대부분은 하루 시작 상태에서 이어진 위반이었고, 총 공기만으로는 더 줄일 여지가 거의 없었다. 같은 총 공기에서 배분을 뒤 구간 쪽으로 옮기면 DO 수정 모델 합산 비용이 31.85 에서 7.49 로 줄었다. 이는 사후 관찰이었고 e62 사전 규칙은 결함 때문에 중단 판정이었다. 진행은 사용자가 결정했다.

## 설계

| 항목 | 내용 |
| --- | --- |
| 환경 | e50 준비물의 데이터, prior, 정규화 + e51 DO 대응 수정 대리 모델. 공통 v1.1 모델은 DO1/2/3 제약이 air1/2/3 에 잘못 걸려 배분 학습에 쓰지 않는다 |
| total 조건 | e60 과 같은 1차원 행동. `e60/train_shielded.py` 를 수정 없이 준비물만 바꿔 실행 |
| alloc 조건 | 행동 [총 공기, 배분 기울기]. w = 0.5 x clip(a1, 0, 1), 배분 = (1 - w) x 기록 + w x [0, 0, .2, .3, .5]. 초기 평균은 기록 배분. 차단은 정책 배분으로 필요 공기를 계산 |
| 공통 | 학습 중 차단 park_e10, 모방 prior on, 401,280전이, 마지막 정책, PPO-Lag 와 FCSRL, seed 0~4 |
| 평가 | 수정 모델 검증 40일 x 구성원 5개, 차단 없음, park_e10, park_e19. 공통 v1.1 모델 val park_e19 교차 평가. test 없음 |

W_MAX 0.5 와 목표 배분은 e62 에서 coverage 0.979 였던 값이며 탐색하지 않는다.

## 결과 전에 고정한 판정

[protocol.json](protocol.json). J1 수정 모델 val park_e19 에서 alloc minus total 을 seed 번호끼리 비교해 공기와 비용이 모두 낮은 seed 4/5 이상이고 두 평균이 모두 낮으면 개선. J2 공통 모델 교차 평가에서 같은 규칙이며, J1 만 개선이면 이득이 DO 수정 모델에 기댄 것으로 적는다. J3 실행 구간 공기 coverage 0.95 미만이면 지원 범위 밖 의존. J4 배분 사용량, J5 NH4 시작 이월과 제어 구간 분해. e60, e61, Park 결과는 모델과 행동 공간이 달라 우열로 쓰지 않는다.

## 사전 검사 (통과)

`/home/jio/fcsrl_src/results/e63/check/checks.json`, 실행 코드 `dbce4b6`.

| 검사 | 결과 |
| --- | --- |
| T1 준비물 해시, 기록 재생 (검증 31.85, 학습 9.578) 이 e62 수정 모델 값과 정확히 같음 | 통과 |
| T2 a1 <= 0 이면 기록 배분, a1 = 1 이면 w = 0.5, 행 합 1 | 통과 |
| T3 새 평가 함수가 1차원 정책에서 e59 평가 함수와 같은 값 (차단 없음 35.645, park_e19 3일) | 통과 |
| T4 a1 = -1 고정 2차원 행동은 같은 총 공기 1차원 행동과 같은 상태, 공기, 구간 공기 | 통과 |
| T5 차단이 정책 배분을 봄: 공기 0 정책 + park_e10 3일, 기록 배분 공기 62,613, 뒤 구간 절반 47,188 m³ (스텝 86% 에서 실행 공기 다름) | 통과 |
| T6 2차원 초기 정책의 총 공기 축이 같은 seed 1차원 정책과 정확히 같고 배분 축 0 | 통과 |
| T7 네 조건 실제 명령줄 1갱신 학습, 저장, 다섯 평가, 교차 평가 두 개 | 통과 |

T5 에서 차단만 쓰는 제어기도 뒤 구간 배분이면 같은 3일에 공기가 25% 적게 들었다. 3일, 진입 검사 값이며 결과로 쓰지 않는다.

## 실행

```bash
cd Yoo/experiments/e63
/home/jio/fcsrl_src/.venv-skl191/bin/python prepare.py --out <준비물 경로>
export OMP_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
/home/jio/fcsrl_src/.venv-skl191/bin/python check.py --prepared <준비물 경로> --out <검사 경로>
/home/jio/fcsrl_src/.venv-skl191/bin/python -u run.py --check <검사 경로>/checks.json --prepared <준비물 경로> --out <본 실행 경로> --jobs 5
```

본 실행 `/home/jio/fcsrl_src/results/e63/main/`, tmux `e63`. 학습 20개 뒤 교차 평가 20개와 집계를 자동으로 한다. 실행 중에는 검사 해시 목록의 파일을 고치지 않는다.

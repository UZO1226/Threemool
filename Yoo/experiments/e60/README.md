# e60: 학습 중 최소 개입 차단을 켠 Yoo PPO-Lag / FCSRL

**2026-11-13 사전 검사 완료, 본 학습 실행 중.** [사전 조건](protocol.json), [학습 코드](train_shielded.py), [검사](check.py), [실행](run.py), [집계](report.py).

## 질문

e59 는 차단 없이 학습한 Yoo 정책에 Park 의 최소 개입 차단을 실행 시점에만 걸었다. e60 은 Park/e12 A 와 같이 1스텝 target 3.5 차단을 학습 중에도 켠다. 정책은 차단이 지켜 주는 범위에서 공기를 줄이는 법을 배우고, 평가에서는 e19 차단(양방향, 여유 표, 회복 검사, 충돌 조정)과 결합한다. Park 의 같은 환경 대표(e19 H1 80,748 / 4.98, e22 H1 79,633 / 4.68)와 e59 의 정책 없는 모델 제어기를 넘는지 본다.

## 조건

| 항목 | 내용 |
| --- | --- |
| 환경 | e50 공통 v1.1, scikit-learn 1.9.1, 관측 11, 비용 NH4>4 + max(DO)>2, 학습 한도 12 |
| 방법 | PPO-Lag, FCSRL + PPO-Lag. 모방 prior on, 초기 log std -1.6. seed 0~4, 401,280전이, 마지막 정책 |
| 학습 중 차단 | e59 에 복사한 `min_brake.py` 의 1스텝, 다른 구성원 4개 평균, target 3.5 (park_e10). PPO 확률비는 정책이 뽑은 행동, 보상, 비용, FCSRL 표현 학습은 실행 행동 |
| 장치 | e56 hybrid. CPU1 수집, PPO CPU1 갱신, FCSRL GPU1 갱신 |
| 평가 | 검증 40일 x 구성원 5개, 결정적 행동. 차단 없음, park_e10, park_e19. 학습 날짜는 없음과 park_e10. test 없음 |

판정 J1~J5 는 [protocol.json](protocol.json) 에 결과 전에 적었다. 요점은 Park e19 H1 보다 공기와 비용이 모두 낮은 seed 4/5 이상, e59 공기 0 + park_e19 대비 정책 기여, 같은 seed 의 e59 대비 학습 중 차단 효과, 차단 의존(차단 없음 비용과 승수), 세 지표 공개다.

## 사전 검사

`/home/jio/fcsrl_src/results/e60/check/checks.json` (통과). 차단 없는 수집 buffer 는 e30 `collect` 와 모든 배열이 정확히 같다. 차단을 켜면 첫 시점 정책 표본은 같고, 실행 공기는 정책 공기 이상이며, 첫 수집의 하루 비용이 11.02 에서 4.41 로, 개입은 3.0% 다.

첫 검사(`check_v1_smoke`)는 차단으로 상태가 달라진 뒤의 표본까지 같기를 요구하는 검사식 오류로 실패 표시가 났다. 비교를 첫 시점으로 고친 뒤 buffer 검사만 다시 돌렸다. 학습 코드는 바뀌지 않았다. 첫 검사의 1갱신 진입 학습은 두 방법 모두 끝까지 돌았고 저장, 복원, 다섯 평가를 마쳤다 (PPO 1,068초, FCSRL 840초). 이 수치는 결과에 쓰지 않는다.

## 실행

```bash
cd Yoo/experiments/e60
export OMP_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
/home/jio/fcsrl_src/.venv-skl191/bin/python check.py --out <새 검사 경로>
/home/jio/fcsrl_src/.venv-skl191/bin/python -u run.py --check <검사 경로>/checks.json --out <새 본 실행 경로> --jobs 5
```

본 실행 `/home/jio/fcsrl_src/results/e60/main/`, tmux `e60`. 완료되면 `results.md`, `summary.json` 을 이 폴더에 자동 생성한다.

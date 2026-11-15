# e65: e63 배분 정책의 seed 5 ~ 9 재현

**2026-11-16 03:20 KST 완료. PPO-Lag 재현, FCSRL 재현 실패.** [결과 표](results.md), [요약](summary.json). [사전 조건](protocol.json), [검사](check.py), [실행](run.py), [집계](report.py).

## 질문

e63 과 e64 에서 배분 정책은 seed 0 ~ 4 의 5/5 에서 총 공기만 정책보다 공기와 비용을 함께 낮췄다. 같은 학습을 seed 5 ~ 9 로 다시 해 이 결과가 seed 운인지 확인한다.

## 조건

seed 만 바꾼다. 학습은 e63 의 `train_alloc.py` 와 e60 의 `train_shielded.py` 를 그대로 부르고, 준비물 (DO 수정 모델), 학습 중 차단 park_e10, 배분 상한 0.5, 예산 401,280전이, 마지막 정책이 같다. 평가는 e64 와 같은 일별 실행 상한, park_e19, DO 수정 모델과 공통 v1.1 모델이다. test 는 쓰지 않는다.

팀 확인: Park/e28 (11-15 병합) 은 막을 수 없는 NH4 스텝에서 차단의 여러 스텝 계획을 다룬다. 배분 행동과 다른 변경이며 이 실험은 Yoo 자체 재현이다.

## 판정 (결과 전 고정)

J1 DO 수정 모델 상한 평가에서 seed 5 ~ 9 의 공기와 비용이 모두 낮은 seed 4/5 이상, 두 평균 모두 낮음이면 재현. J2 seed 0 ~ 9 합산 8/10 이상 (보고). J3 공통 모델 같은 규칙 (보고).

## 사전 검사 (통과)

`/home/jio/fcsrl_src/results/e65/check/checks.json`. e63 사전 검사에서 고정한 학습 코드와 준비물 해시가 모두 같다. 결과 경로 인자를 더한 e64 평가 코드가 e64 의 배분 PPO seed 0 상한 평가를 공기량, 비용, 잘린 비율까지 정확히 재현했다.

## 실행

```bash
cd Yoo/experiments/e65
OMP_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' /home/jio/fcsrl_src/.venv-skl191/bin/python check.py --out <검사 경로>
/home/jio/fcsrl_src/.venv-skl191/bin/python -u run.py --check <검사 경로>/checks.json --out <본 실행 경로>
```

본 실행 `/home/jio/fcsrl_src/results/e65/main/`, 실행 코드 `ebb8bfd`, tmux `e65`. 학습 20개 (동시 5) 뒤 상한 평가 40개 (동시 7) 와 집계를 자동으로 한다.

## 결과 (일별 실행 상한, park_e19)

| 방법 | 모델 | seed | 총 공기만 | 총 공기 + 배분 | 둘 다 낮은 seed | 판정 |
| --- | --- | --- | --- | --- | --- | --- |
| PPO-Lag | DO 수정 | 5 ~ 9 | 75,399 / 4.80 | 59,349 / 3.26 | 5/5 | 재현 |
| PPO-Lag | DO 수정 | 0 ~ 9 | 74,432 / 4.81 | 59,549 / 3.16 | 10/10 | 10 seed 이득 유지 |
| PPO-Lag | 공통 v1.1 | 0 ~ 9 | 76,067 / 5.24 | 67,194 / 3.28 | 10/10 | 10 seed 이득 유지 |
| FCSRL | DO 수정 | 5 ~ 9 | 77,206 / 4.68 | 71,129 / 4.06 | 3/5 | 재현 실패 |
| FCSRL | DO 수정 | 0 ~ 9 | 77,832 / 4.71 | 70,385 / 3.74 | 8/10 | 10 seed 이득 유지 |
| FCSRL | 공통 v1.1 | 0 ~ 9 | 77,531 / 5.09 | 74,570 / 4.01 | 6/10 | 10 seed 기준 미달 |

값은 공기량 m³/23.75h / 합산 비용, seed 평균.

1. **PPO-Lag 는 새 seed 에서 그대로 재현됐다.** seed 5 ~ 9 모두 공기와 비용이 함께 낮고, 10 seed 합산에서도 두 모델 10/10 이다. 평균 공기 -14,883 m³ (-20.0%), 비용 -1.65 (DO 수정 모델).
2. **FCSRL 은 seed 5 ~ 9 에서 사전 기준 미달 (3/5).** 평균은 공기 -6,077, 비용 -0.62 로 낮지만 두 seed 에서 공기 또는 비용이 늘었다. 배분 사용량도 작다 (w 평균 0.063, PPO 0.108). 10 seed 합산은 DO 수정 모델 8/10 으로 기준을 넘고 공통 모델 6/10 으로 미달이다.
3. FCSRL 의 배분 이득은 PPO-Lag 보다 작고 seed 에 민감하다. FCSRL 표현 학습이 2차원 행동에서 배분 축을 덜 쓰는 이유는 이번 실험에서 분리하지 않았다.

e66 의 주 구성은 e64 규칙으로 이미 정했으며, 사전 조건대로 FCSRL 결과를 해석할 때 이 seed 민감도를 함께 적는다.

# e65: e63 배분 정책의 seed 5 ~ 9 재현

**2026-11-15 22:04 KST 본 학습 시작.** [사전 조건](protocol.json), [검사](check.py), [실행](run.py), [집계](report.py).

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

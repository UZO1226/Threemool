# Yoo (유지오)

유지오 의 개인 작업 공간이다. `yoo/<주제>` 브랜치에서만 수정한다 (RULES/BRANCH_RULES.md 4절).

| 위치 | 내용 |
| --- | --- |
| `analysis/` | 논문 분석 노트 |
| `code/` | 구현 코드 (FCSRL 재현, 환경, 확장 기법) |
| `experiments/<실험ID>/README.md` | 실험 기록. 실험 ID 는 e01, e02 처럼 이 폴더 안에서 겹치지 않게 붙인다 |
| `RESULT.md` | 결과 비교용 요약 |

## 구현체 구동 결과 (Yoo/e01)

FCSRL 공개 구현체(czp16/FCSRL)를 수정 없이 돌려 논문 Table 4 (FCSRL + TD3 Lag, Safety Gymnasium 6개 과제 x seed 5개)를 재현했다. 자세한 기록은 `experiments/e01/README.md`.

| Task | 논문 reward | 재현 reward | 재현 cost (한도 10) | 판정 |
| --- | --- | --- | --- | --- |
| PointGoal1 | 24.4 ± 1.4 | 24.4 ± 1.5 | 15.1 ± 7.8 | reward 재현 |
| PointButton1 | 10.5 ± 2.3 | 10.7 ± 1.5 | 8.2 ± 1.4 | 재현 |
| PointPush1 | 11.8 ± 4.5 | 9.9 ± 4.7 | 12.2 ± 3.0 | 편차 안 |
| PointGoal2 | 13.5 ± 2.6 | 10.1 ± 3.7 | 10.1 ± 7.6 | 낮음, 단 통계적으로 구분 안 됨 (p = 0.17) |
| CarGoal1 | 27.6 ± 2.0 | 26.8 ± 4.1 | 8.6 ± 3.1 | 재현 |
| CarButton1 | 3.6 ± 1.3 | 2.9 ± 1.4 | 8.2 ± 2.9 | 편차 안 |

재현 값은 저장된 모델을 서로 다른 맵 50개로 다시 평가한 seed 평균 ± 표준편차다.

- 원본 코드의 문제 두 가지를 찾았다. (1) 병렬 환경이 모두 같은 맵을 받는다 (seed 설정 코드가 gymnasium 에서 실행되지 않음). (2) 그래서 epoch 시험 값이 에피소드 하나이고, 그 값으로 고른 "best" 모델의 로그 값은 실제보다 2 ~ 8 부풀려진다
- PointGoal2 가 낮은 이유: 평가 잡음은 아니고(맵 50개로 다시 재도 10.1), seed 사이 편차가 크다 (6.5 ~ 16.7). 보수적인 정책으로 수렴한 seed 2개가 평균을 끌어내렸고, 이 편차로는 논문 값과의 차이가 통계적으로 뚜렷하지 않다

## 실행 환경

- 서버 .32 (i9-10900K, RTX 3090 x2), Python 3.10.19 venv `/home/jio/fcsrl_src/.venv`
- torch 2.14.1+cu130, safety-gymnasium 1.0.0, gymnasium 0.28.1, mujoco 2.3.3, numpy 1.23.5, pandas, scikit-learn
- 원본 FCSRL: `/home/jio/fcsrl_src/FCSRL` (커밋 d882b91, 수정 없음, `pip install -e` 로 설치)
- 설치 명령은 `experiments/e01/README.md` 에 있다

## 현재 상태

**구현체 구동 완료.** 원본 FCSRL 재현(e01)과 폭기 제어 환경 적용(e02)을 마쳤다. 현재는 e03에서 PPO Lag 비교군과 UCB 비용 critic 확장을 평가하고, Lagrange 승수 상한의 영향을 추가 실험하고 있다.

아래 상태는 2026-11-06 13:17 (한국 시간) 기준이다.

| 실험 | 내용 | 상태 |
| --- | --- | --- |
| e01 | FCSRL 원본 재현 (Safety Gymnasium 6개 과제 x seed 5개) | 완료. PointGoal2 를 뺀 5개 과제가 논문 편차 안 |
| e02 | FCSRL 을 Avedøre 폭기조 데이터 기반 폭기 제어에 적용, TD3 Lag 와 TD3 와 실제 운전 비교 | 완료. 대리 모델 위에서 공기량 약 45% 절감, 비용 실제 운전 22.9 에서 15 안팎. FCSRL 과 TD3 Lag 차이는 seed 편차 안. `experiments/e02/README.md` |
| e03 | PPO Lag 비교군, FCSRL + UCB 비용 critic 확장, 평가 전용 GBM 교차 평가 | 진행 중. PPO Lag와 FCSRL + UCB는 각각 seed 5개 학습 완료. e02 15개와 e03 10개 정책의 MLP 및 GBM 평가 완료 |
| e03 추가 실험 | FCSRL과 FCSRL + UCB의 Lagrange 승수 상한을 0.75에서 5로 높여 비교 | 진행 중. 각 seed 5개 중 FCSRL 3개, FCSRL + UCB 2개 완료. 나머지 5개 학습 중 |

현재 확인하는 질문은 승수 상한이 제약 학습을 제한했는지, 상한을 높이면 UCB가 도움이 되는지다. 추가 실험이 끝나기 전에는 원인을 확정하지 않는다. GBM 교차 평가는 다른 대리 모델에서 결과를 확인하는 절차이며, 실제 공정의 안전성이나 모델 오차 의존성 해소를 입증한 것은 아니다.

다음 할 일: 추가 실험 완료 후 MLP 및 GBM 평가, 결과 표와 그림 생성, e03 README와 `RESULT.md` 정리, PR 작성.

실행 및 집계 스크립트는 `experiments/e03/run_e03.py`, `experiments/e03/summarize_e03.py`에 있고, 학습 로그와 평가 결과는 `/home/jio/fcsrl_src/results/e03/`에 있다. `RESULT.md`의 기존 "확장 계획"은 초기 제안이며, 현재 진행 상황은 위 표를 따른다.

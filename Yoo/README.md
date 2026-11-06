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

**구현체 구동 완료.** 원본 재현(e01), 폭기 제어 적용(e02), PPO 비교군과 UCB 확장 및 승수 상한 추가 실험(e03)을 마쳤다. 아래는 2026-11-06 14:23 (한국 시간) 기준이다.

| 실험 | 내용 | 상태 |
| --- | --- | --- |
| e01 | 원본 FCSRL 구동과 재현, PointGoal2 진단 | 완료. [실험 기록](experiments/e01/README.md) |
| e02 | FCSRL 폭기 제어 적용, TD3 Lag와 TD3 비교 | 완료. [실험 기록](experiments/e02/README.md) |
| e03 | PPO Lag, FCSRL + UCB, 승수 상한 5 추가 실험 | 학습 20개 완료. e02 15개와 합친 35개 정책의 MLP 및 GBM 재평가 완료. [실험 기록](experiments/e03/README.md) |

e03의 전체 결과와 재현성 점검을 정리했다. 선택 모델과 마지막 모델을 구분해 [RESULT.md](RESULT.md)에 요약했다. 기존 TD3의 마지막 가중치가 없는 경우 GBM 재평가 누락을 명시했다.

다음 단계는 고정된 검증 조건과 대리 모델 오차 점검이다. UCB 효과와 승수 상한 가설의 판정은 [e03 결과](experiments/e03/results.md)를 따른다. 실제 공정 안전성을 입증한 결과는 아니다.

로그와 대용량 산출물은 `/home/jio/fcsrl_src/results/e03/`에 있다. 재평가와 집계는 `experiments/e03/finalize_e03.py`로 재현한다.

## 후속 실험 e04

2026-11-06: 학습 전 코드와 설정 점검, 기존 정책 35개의 전체 검증 진단을 완료했다. 무작위 소수 에피소드로 모델을 고르는 문제와 불확실성 추정의 한계를 확인했다. 다음은 고정 검증에서 FCSRL 학습 한도 15 대 12를 seed 2개씩 비교하는 작은 시험 실행이다. [점검과 실행 기록](experiments/e04/README.md), [진단 결과](experiments/e04/results.md).

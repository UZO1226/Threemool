# e59: Park 모델 예측 차단을 Yoo 정책에 적용해 정책의 기여를 분리

**2026-11-13 사전 검사 완료, 본 평가 실행 중.** 새 학습은 없다. [사전 조건](protocol.json), [검사 코드](check.py), [평가 코드](shield_eval.py), [실행](run.py), [집계](report.py).

## 왜 이 실험인가

Yoo e30~e58 의 정책 변형(출력 보정, 조건별 제약, NH4 여유 세 종류, prior 제거, 보조 손실)은 모두 e50 환경에서 공기량과 NH4 비용의 한 교환선 위를 움직였다. 같은 환경에서 그 선을 벗어난 결과는 Park 의 실행 시점 최소 개입 차단뿐이다 (Park/e19 H1 80,748 m³, 비용 4.98; Park/e22 H1 79,633, 4.68). 차단은 대리 모델을 실행 시점에 직접 써서 다음 스텝 NH4 를 여유 아래로 유지하는 가장 작은 공기량까지만 올린다.

이 실험은 Park 의 차단을 Yoo 파이프라인에 그대로 옮긴 뒤 두 가지를 묻는다.

1. 차단 아래에서 RL 정책이 무엇을 기여하는가. 공기 0 정책 + 차단은 정책 없이 모델만으로 움직이는 제어기다. 이 값보다 정책 + 차단이 공기량과 비용을 함께 낮추지 못하면, 이 차단 아래에서 정책은 기여가 없다.
2. Yoo 의 기존 정책(모방 prior, FCSRL, Actor 보정, severity)이 차단과 결합했을 때 어디에 놓이는가.

Park/e10 은 차단 없이 학습한 Park e08 FCSRL 정책에 실행 시점 차단을 걸어 90,747 / 5.46 을 얻었고, 차단과 함께 학습한 e12 A 는 84,031 / 5.51 이었다. 정책 없는 모델 제어기 값은 Park 기록에 없다. 이 값이 다음 단계를 가른다.

## 조건

| 항목 | 내용 |
| --- | --- |
| 환경 | e50 공통 v1.1, scikit-learn 1.9.1 대리 모델, 관측 11, 비용 NH4>4 + max(DO)>2, 한도 12 |
| 평가 | 검증 40일 x 구성원 5개 = 200 에피소드, 결정적 행동, 마지막 정책. test 행은 추출하거나 평가하지 않음 |
| 정책 | e50 PPO-Lag, FCSRL + PPO-Lag, FCSRL + Actor 보정 (모두 모방 prior on, CPU4), e58 FCSRL + severity (GPU1). seed 0~4 |
| 고정 행동 | 기록 운전, 공기 0 정책 (차단이 전부 정함) |
| 차단 none | 없음. 원 실험 metrics.json 의 공기량과 비용을 정확히 재현해야 한다 |
| 차단 park_e10 | 1스텝, 다른 구성원 4개 평균, target 3.5, 올리기만. Park/e10, e12 학습에 쓴 설정 |
| 차단 park_e19 | 위에 양방향(trim), 여유 표 P1, DO 1.8, 회복 검사 (3, 3.5), 충돌 조정 0.1. Park/e19 H1 과 e22 가 쓴 설정 |

차단 코드는 `Park/code/min_brake.py` (main `bbaefe2`, sha256 `1f25865141dafe3f3bf4ce199b8d1b9a625f07d60d663844acb20e8cf4195c69`) 를 수정 없이 복사했다 (협업 규칙 4절, 출처 경로와 커밋 기록). `Yoo/experiments/e30/darrow_env.py` 는 `Park/code/darrow_env.py` 와 바이트 단위로 같아 그대로 동작한다. Park 의 정책 가중치는 쓰지 않는다.

## 결과 전에 고정한 판정

- J1 정책 기여: 방법별로 (정책 + park_e19) 가 (공기 0 + park_e19) 보다 공기량과 비용이 모두 낮은 seed 수. 4/5 이상이면 기여, 0/5 이면 기여 없음, 그 사이는 불확정.
- J2 차단 효과: 같은 seed 에서 park_e19 minus none 의 공기량 변화율과 비용 변화. 보고만 한다.
- J3 교환선: none 조건 20개 정책의 (공기량, NH4 비용) 에 맞춘 직선 대비 차단 조건의 잔차.
- J4 Park 대비: 공기 0 + park_e19 와 비용이 가장 낮은 방법의 정책 + park_e19 를 Park e10, e12 A, e19 H1, e22 H1 과 나란히 적는다. 학습 조건이 달라 우열로 해석하지 않는다.
- 다음 단계 규칙: 공기 0 + park_e19 가 Park e22 H1 을 공기량과 비용 모두에서 넘으면 정책 없이 모델 제어기만으로 Park 대표를 넘는 것이므로 다음은 모델 예측 제어기 개선이다. 넘지 못하면 다음은 Park e12 방식(1스텝 target 3.5 차단을 학습 중에 켬)으로 Yoo FCSRL/PPO 를 학습하는 e60 이다.

결과를 본 뒤 seed, 조건, 차단 설정을 바꾸거나 빼지 않는다. 차단 설정 탐색은 하지 않는다.

## 사전 검사 (통과)

`/home/jio/fcsrl_src/results/e59/check/checks.json`.

- 복사한 차단 코드와 환경 코드가 Park 원본과 sha256 동일.
- 정답 모델(환경과 같은 구성원) 1스텝 target 3.5 차단을 공기 0 정책에 걸면, 필요 공기량이 9,000 미만인 검증 스텝 17,803개에서 실제 다음 NH4 가 3.5 를 넘은 경우 0. target 무한대이면 개입 0. Park/e10 설계 점검과 같은 검사다.
- 차단 없는 평가가 네 방법 seed 0 의 metrics.json 비용을 정확히, 공기량을 상대 1e-8 안에서 재현. FCSRL 두 조건은 e50 당시 집계의 부동소수 합산 순서 때문에 공기량이 1e-4 m³ 수준에서 다르다.
- 공기 0 정책 + park_e19 한 평가는 약 10.5분 (매 스텝 개입하는 최악 경우).

## 실행

```bash
cd Yoo/experiments/e59
export OMP_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
/home/jio/fcsrl_src/.venv-skl191/bin/python check.py --out <새 검사 경로>
/home/jio/fcsrl_src/.venv-skl191/bin/python -u run.py --check <검사 경로>/checks.json --out <새 본 실행 경로> --jobs 6
```

본 실행 `/home/jio/fcsrl_src/results/e59/main/`. 완료되면 `results.md` 와 `summary.json` 을 이 폴더에 자동 생성한다. 각 평가의 에피소드별 수치와 상태 배열은 실행 폴더에 남긴다.

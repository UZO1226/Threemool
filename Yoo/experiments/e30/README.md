# e30: 사용자 지정 DARROW 데이터에서 FCSRL과 출력 보정 재검증

## 목적과 상태

2026-11-10 사용자가 새로 제공한 원자료와 Kaggle URL을 확인했다. **본 데이터는 DARROW다.**
기존 Avedore 결과(e02~e15, e23~e29)는 다른 처리장의 탐색 기록으로 보존하며 본 데이터 결과로 부르지 않는다.
FCSRL의 dynamics/feasibility 표현 학습은 유지하고, e25와 e29에서 시험한 출력 보정의 효과를 새 데이터에서 검증한다.
데이터, 전이 모델, 초기 운전 정책과 정규화를 새로 만들었다. 이전 데이터의 가중치를 이식하지 않는다.

현재: 데이터 준비, 환경 진단, 구현 검사와 6조건 시험 실행(각 6,080전이)을 완료했다. **2026-11-10 09:22 KST 본 학습 30개를 시작했다.** 2개씩 실행 중이며 본 학습 결과는 아직 없다.
본 실행 코드 커밋은 `8b1bf00`, 정확한 코드와 입력 해시는 `main/manifest.json`에 기록했다.
실행 상태 원본: `/home/jio/fcsrl_src/results/e30/`.

## 원자료와 분할

- [Kaggle 대회 원자료](https://www.kaggle.com/competitions/dynamic-modeling-of-wastewater-treatment-process/data), 사용자 URL 확인 2026-11-10.
- 원본 위치: `Yoo/data/dynamic-modeling-of-wastewater-treatment-process/`.
- 학습 입력: `train_val_test_online_dataset.csv`, `train_val_test_dynamic_influent.csv`.
- 온라인 25,921행, 15분 간격. 완전한 날짜 269개를 시간 순서로 분할한다.
- 학습 188일: 2021-05-06~2021-11-09. 검증 40일: 2021-11-10~2021-12-19. 보류 41일: 2021-12-20~2022-01-29.
- `evaluation_online_dataset.csv`에는 NH4, DO, NO3 정답이 없다. 제출용 파일을 정답 있는 평가 자료처럼 쓰지 않는다. lab 파일도 이번 환경 학습에 쓰지 않는다.
- e30에서 보류 구간 정책 평가를 하지 않는다. 팀 선행 실험에는 그 구간의 환경 진단 기록이 있으므로 팀 전체에서 한 번도 보지 않은 데이터라고 주장하지 않는다.
- 14.4분 유입 성분을 15분 격자로 선형 보간한다. 이는 제공된 외생 입력을 재생하는 조건이며 실시간 관측 가능성이나 인과적 제어 유효성을 보장하지 않는다.
- 6개 원본 파일 SHA256, 실제 사용 파일, 날짜 목록은 `prepared/data_manifest.json`에 저장한다. 데이터 원본은 Git에 올리지 않는다.

## 코드 출처와 변경

팀 코드 기준: `8decad7`.

| e30 파일 | 출처 | 변경 |
| --- | --- | --- |
| `prep_darrow.py` | `Lee/code/darrow/prep_darrow.py` | 비율 대체값을 전체 자료 대신 학습 날짜에서 계산 |
| `surrogate_dw.py` | `Lee/code/darrow/surrogate_dw.py` | 복사본, GBM 학습 함수만 실행 |
| `darrow_env.py` | `Park/code/darrow_env.py` | 복사본, 하루 끝 공기량 덮어쓰기 문제가 없는 명시적 reset 방식 |
| `common.py`, `train.py` | Yoo/e23, e25, e29 알고리즘 및 수집 방식 | DARROW 11차원 상태와 1차원 총 공기량 환경으로 연결 |

원본 FCSRL은 `czp16/FCSRL`, 커밋 `d882b91054d52c7a63a18a21f7db67bbbed82b63`, Apache-2.0이다.
`Yoo/code/wwtp/FCSRL_LICENSE`에 라이선스를 보존한다. 원본의 encoder 학습을 그대로 호출하며 원본 저장소는 수정하지 않는다.
PPO 기반 FCSRL 자체를 새로운 알고리즘이라고 주장하지 않는다. 확장은 EMA 이후 출력 보정이다.

## 환경과 비교 조건

상태는 NH4, NO3, DO 3개, 유량, 수온, 시각 sin/cos, 유입 NH4, 직전 총 공기량의 11차원이다.
행동은 총 공기량 0~9,000 Nm³/h이며 5개 구간에 해당 시각 기록 비율로 배분한다.
구간별 독립 제어가 아니며 기록 비율을 외생 입력으로 제공하는 조건부 시뮬레이션이다.
초기 직전 공기량은 첫 기록 값으로 둔다. 이후에는 실제로 실행한 이전 행동을 사용한다.

보상은 `-공기량 / 9000`, 비용은 `I(NH4 > 4) + I(max(DO1,DO2,DO3) > 2)`다.
DO 상한은 이 연구의 비용 정의이며 보편적인 법적 수질 기준이 아니다.
하루 96개 관측에서 95개 전이를 실행한다. 공기량은 0.25시간을 곱한 **23.75시간 합계**이며 24시간으로 보정하지 않는다.
비용 한도 12는 팀 DARROW 기준이다. 새 모델의 학습 구간 기록 운전 재생 평균을 반올림해 12인지 검사한다.
이전 Avedore의 5분 간격 비용 한도 15와 수치를 직접 비교하지 않는다.

| 조건 | 목적 |
| --- | --- |
| PPO Lag | 표현 학습 없는 비교군 |
| FCSRL + PPO Lag | 기본 적용 |
| FCSRL + Actor 보정 | 정책 변화 보정 효과 |
| FCSRL + Critic 보정 | 가치 변화 보정 효과 |
| FCSRL + Actor/Critic 보정 | e25 확장 |
| FCSRL + 추가 PPO | 추가 최적화 예산 대조 |

모든 조건은 새 데이터에서 만든 같은 기록 운전 정책을 초기값으로 쓴다.
초기 정책은 학습 188일 중 앞 150일로 40 epoch 학습하고 나머지 38일은 점검에만 쓴다.
고정 정규화도 앞 150일에서 계산한다. 기록 행동 벌점은 추가하지 않는다.
초기 log std -1.6, 8환경 x 760전이 x 66갱신 = **401,280전이**, seed 0~4, 총 30개다.
PPO 10 epoch, minibatch 512, clip .2, KL .02, actor/critic 학습률 3e-4/1e-3,
PID .02/.005/.01, 승수 상한 20, EMA .05를 공통으로 쓴다.
보정은 e29와 동일한 50회 추가 최적화와 보정용 fit/check 분할을 유지한다.
전이 수와 초기 조건을 통제하지만 실행 시간은 다르므로 실제 소요 시간도 저장한다.
팀의 초기 log std -.5 및 초기 정책과 차이가 있어 팀 정책 숫자와 직접 우열을 주장하지 않는다.

모든 최종 평가는 **마지막 정책**, 검증 40일 x GBM 구성원 5개에 고정한다. 중간 검증으로 정책을 고르지 않는다.
seed가 통계 반복 단위이며 200개 에피소드를 독립 학습 200회로 세지 않는다.
평균/표준편차, 평균 비용 한도 충족 seed, NH4/DO 위반률, 공기량, 실행 시간을 기록한다.
학습 종료 후 `report.py`가 표와 seed 대응 비용 차이 및 Holm 보정 결과를 만든다.
공기량이 다르면 비용 감소만으로 효율 개선을 채택하지 않는다. 후속 같은 공기량 대조가 필요하다.

## 환경 진단 결과와 해석 범위

학습 날짜만 사용해 단조 제약 GBM 5구성원 x 5상태를 새로 학습했다. scikit-learn 1.7.2.
NH4 변화량은 공기량 증가에 비증가, DO 변화량은 대응 구간 공기량 증가에 비감소하도록 강제한다.
센서와 폭기 구간의 대응 및 단조성은 팀 환경에서 채택한 모델 가정이며 원자료만으로 인과성을 입증하지 못한다.
MLP의 잘못된 공기량 반응이 팀 선행 진단에서 보고돼 이번 비교 환경은 단조 GBM으로 고정했다.

검증 구간 기록 공기량 재생 결과:

| 지표 | 실제 관측 | 모델 재생 |
| --- | --- | --- |
| 95전이 비용 합 | 17.55 | 27.365 |
| NH4 위반 비율 | 17.316% | 26.937% |
| DO 위반 비율 | 1.158% | 1.868% |

NH4 탐지율 48.60%, FPR 22.40%. DO 탐지율 1.36%, FPR 1.87%다.
한 스텝 NH4 변화량 R² 0.241, NH4 연속 예측 MAE 2.383이다.
**특히 DO 위반을 놓치는 문제가 크므로 이 환경을 신뢰할 수 있는 실제 제어 검증 환경이라고 주장하지 않는다.**
제약 단조성 검사는 통과했지만 정확도와 실제 안전성 검증을 대신하지 않는다.
정책 비교는 동일 대리 환경 안의 알고리즘 검증이며 결과가 좋아도 현장 안전/절감 또는 확장 채택으로 자동 연결하지 않는다.
환경 진단 원본은 `environment_diagnostic.json`에 보존한다.

## 실행 및 산출물

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/jio/fcsrl_src/.venv/bin/python
OUT=/home/jio/fcsrl_src/results/e30
$PY Yoo/experiments/e30/prepare.py --raw Yoo/data/dynamic-modeling-of-wastewater-treatment-process --out "$OUT/prepared"
$PY Yoo/experiments/e30/verify.py --prepared "$OUT/prepared" --out "$OUT/verification.json"
$PY Yoo/experiments/e30/diagnose.py --prepared "$OUT/prepared" --out "$OUT/environment_diagnostic.json"
$PY Yoo/experiments/e30/run.py --prepared "$OUT/prepared" --out "$OUT/smoke" --updates 1 --seeds 1 --jobs 2
$PY Yoo/experiments/e30/run.py --prepared "$OUT/prepared" --out "$OUT/main" --jobs 2
```

검증과 smoke가 모두 통과해 본 실행을 시작했다. 출력 경로가 이미 있으면 덮어쓰지 않는다.
실행 커밋, 코드 및 입력 해시는 `main/manifest.json`, 전체 상태는 `main/status.json`,
각 실행 진행률과 ETA는 `main/runs/<method>/seed_<n>/progress.json`, tqdm은 `main/<method>_s<n>.log`다.
진행률 확인: `python3 Yoo/experiments/e30/status.py /home/jio/fcsrl_src/results/e30/main`.
종료 후 `main/results.md`, `main/summary.json`이 자동 생성된다. 실패하면 실패 로그와 상태를 남기고 대기 작업을 중단한다.
검증 범위는 원시 행동 log probability, 시간 제한 경계, terminal 공기량 보존, 모든 조건의 같은 초기 행동,
FCSRL encoder/EMA/RNG 유지, 비활성 head 불변, 추가 PPO KL 제한, 저장/복원 및 정규화/초기 정책 고정이다.

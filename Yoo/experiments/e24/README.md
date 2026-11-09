# e24 Kaggle FCSRL 회복 지속시간 보조 학습

## 실행 전 가설과 판정 기준

2026-11-09, 학습 전에 고정한 계획이다. 수행계획서의 Kaggle Avedore 데이터로 PPO-Lag, FCSRL + PPO-Lag, 회복 지속시간 예측을 추가한 FCSRL을 비교한다. BSM1은 사용하지 않는다. e23 행동 벌점은 사용하지 않는다.

기존 FCSRL의 max 형태 feasibility 목표는 현재 위반 중인 상태의 지속시간 차이를 충분히 구분하지 못할 수 있다. 원본 dynamics 및 feasibility 손실을 유지하고, 같은 잠재 상태에서 앞으로 12전이(1시간)의 평균 위반 비용을 예측하는 MSE 손실만 추가한다. 시간과 가중치 1은 학습 전 고정한다. 평균 비용은 0~2 범위이며 회복 완료 시각 자체의 라벨은 아니다. 정책 입력에 새 head의 출력을 추가하지 않는다.

최종 정책(seed 0~4, 마지막 checkpoint)의 탐색적 성공 기준은 아래 조건을 모두 만족하는 것이다. 결과를 보고 seed나 기준을 바꾸지 않는다.

1. MLP와 GBM 각각 확장 평균 비용이 15 이하이고, 개별 seed 4/5 이상이 평균 비용 한도 15를 만족한다.
2. 각 seed와 평가 모델에서 기본 FCSRL에 상수 행동 offset을 더한 대조군을 만들고 확장의 평균 공기량과 0.1% 이내로 맞춘다. 같은 공기량 대조군 대비 확장의 평균 비용 감소가 10% 이상이다.
3. 각 모델에서 seed별 비용 차이(확장 빼기 같은 공기량 대조군)의 paired t 95% 신뢰구간 상한이 0 미만이다. 독립 표본 단위는 seed 5개이며 215개 에피소드를 독립 seed처럼 취급하지 않는다.

모든 판정은 두 모델 모두에 요구한다. 공기량 offset은 검증 데이터에서 공기량만 맞추는 사후 진단용 대조군이다. 배포 정책이나 최적 Pareto 경계가 아니다. 이미 여러 번 확인한 val과 5개 seed를 사용하므로 독립 확증 실험 또는 새로운 유의성 보장으로 해석하지 않는다. test는 선택과 평가에 사용하지 않는다.

## 보존 요소와 구현

- 원본 https://github.com/czp16/FCSRL, 커밋 `d882b91054d52c7a63a18a21f7db67bbbed82b63`, Apache-2.0. 코드 출처는 `Yoo/code/wwtp/FCSRL_LICENSE` 및 `recovery.py`에 명시했다.
- PPO 기반 FCSRL 적용은 e05 및 e23과 같다. e24는 원 논문 환경 재현이 아니라 Kaggle 폭기 환경에 적용한 FCSRL 표현 학습의 확장이다.
- `recovery.py`: 기존 encoder, dynamics, feasibility 손실을 보존하고 64→64→1 head를 추가한다. 다음 12개 비용이 완전히 관측된 경우만 사용하며 에피소드 reset을 넘지 않는다. 마지막 비용 전이에서 종료하는 창은 허용한다. 추론에는 head가 필요 없다.
- 원본 손실 `.5 * dynamics + feasibility`에 `1 * recovery_mse`를 더해 같은 optimizer로 한 번 갱신한다. 미래 비용은 학습 목표로만 사용한다. 행동 선택 시 미래 관측을 사용하지 않는다.
- head 생성은 별도 RNG 문맥에서 실행해 기존 망과 초기 행동을 바꾸지 않는다. 보조 head만 학습하고 encoder 연결을 끊은 검사에서는 기존 망의 갱신이 원본과 bit 단위로 같다.
- e23과 동일한 고정 운전 기록 prior, 고정 관측 정규화, 초기 log std -1.6, PID, 한도 15, 학습량, 분할, 마지막 checkpoint 비교를 유지한다. reward shaping beta는 모든 조건 0이다.
- 학습 목표는 실제 공정이 아니라 학습용 MLP 대리 환경에서 얻은 비용이다. 보조 목표만으로 대리 모델 편향을 해결한다고 주장하지 않는다.

## 데이터와 환경

Kaggle `jueyetao/wastewater-treatment-dataset`, 원 출처 Mendeley DOI `10.17632/xmbxhscgpr.1`, CC BY 4.0. 전처리는 기존 e23과 동일하다.

| 파일 | SHA256 |
| --- | --- |
| `Yoo/data/avedore/aved_raw.csv` | b57bc042a394d84da9fc9620aae5b90f029885dbafa14d3ce6ed437a0b005a2e |
| `Yoo/data/avedore/aved_5min.pkl` | 3a01d5995e49d3b4f3fd42240a4579d1be991629ef70243435422f797d789315 |
| 학습 MLP `e02/surrogate/surrogate.json` | d25dbe3ea1a8e33b17f0a2d0f11a5983fe6c4a6d81a1d0db851b530386658117 |

train 330일(2022-09-15~2023-12-31), val 43일(2024년 1~2월), test 60일(2024년 3~6월). e23 prior는 train 앞 264일로 학습했고 나머지 66일은 prior 점검에 사용했다. RL은 train 330일을 사용한다. val은 43일 x 대리 모델 member 5개 = 215에피소드다. 평균과 표준편차는 이 에피소드 평균을 낸 seed 5개 사이에서 계산한다.

5분 간격, 하루 287전이. 11차원 관측, 호기 단계 공기량 0~8000 Nm³/h. reward는 매 전이 `-air/8000`, cost는 `I(NH4>4)+I(DO>2)`, 하루 비용 한도 15. 공기량은 하루 적분 Nm³/일, 비용은 하루 위반 비용 합이다. 비용 15 이하가 모든 시간의 수질 안전을 뜻하지 않는다. 공동 채택 조건이 아닌 Yoo 개인 실험이며 원본 Safety Gymnasium 환경과 구분한다.

## 사전 검사

`verification.json`과 `diagnosis.json`에 원자료를 보관한다. 학습 날짜에서 e23 FCSRL의 결정론적 궤적을 재생했을 때, 위반 중인 완전한 창의 99.755%, 99.974%(seed 0, 1)는 할인 max 진단 목표가 현재 비용과 같았다. 이는 하루 끝 값을 0으로 둔 진단이며 실제 학습의 부트스트랩 목표와 동일하다고 단정하지 않는다. 비용 1로 포화된 창 중 약 48~50%는 다음 1시간 중 일부에서 비용이 낮아졌다. 영구 회복 또는 시간 내 완전 회복 비율은 아니다. e07에서 관찰한 max 목표 한계를 이번 정책에서도 확인한 것이다.

초기 행동 동일, 미래/현재 행동 라벨 누수 방지, raw action log probability, prior와 정규화 고정, 유한 갱신, 저장 및 재로딩, reset 마스크, 보조 연결 제거 시 기존 갱신 일치 검사에 통과했다. 사전 배치에서 보조 encoder 경사 norm / 원본 norm은 0.645였다. 모든 학습 단계에서 같은 비율이라는 의미는 아니다.

## 실행과 재현

3조건 x seed 5개, 각 202,048전이(요청 200,000, 22회 rollout 갱신), 학습 중 val 2회. 모든 비교는 마지막 checkpoint로 고정한다. e23의 PPO 및 FCSRL seed 0, 1의 동일 조건 실행 4개를 재사용하고 나머지 11개를 새로 학습한다. 재사용 체크포인트와 요약의 해시는 실행 manifest에 고정한다. 기본 경로의 코드 및 RNG를 바꾸지 않았고, 재사용 결과를 새 학습이라고 세지 않는다.

서버 .32, CPU 병렬 4작업, 작업당 torch/BLAS thread 1. Python 3.10.19, torch 2.14.1+cu130, numpy 1.23.5, scipy 1.15.3. 기존 venv와 설치를 유지한다. 실행 코드 커밋, 입력과 소스 해시는 `/home/jio/fcsrl_src/results/e24/pilot/manifest.json`에 기록한다. 대용량 체크포인트와 로그는 Git에 올리지 않는다.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/jio/fcsrl_src/.venv/bin/python -u Yoo/experiments/e24/run.py \
  --data /home/jio/사물/Yoo/data/avedore/aved_5min.pkl \
  --model /home/jio/fcsrl_src/results/e02/surrogate/surrogate.json \
  --gbm /home/jio/fcsrl_src/results/e03/surrogate_gbm/surrogate_gbm.pkl \
  --prepared /home/jio/fcsrl_src/results/e23/prepared \
  --reuse /home/jio/fcsrl_src/results/e23/pilot/runs \
  --out /home/jio/fcsrl_src/results/e24/pilot
```

학습 후 같은 venv와 thread 설정으로 `match_air.py`의 `--root`에 e24/pilot, `--prepared`에 e23/prepared, 위 data/model/gbm과 `--out /home/jio/fcsrl_src/results/e24/matched_air.json`을 전달한다. 요약과 사후 검사 결과는 완료 후 이 폴더에 추가한다.

## 해석 한계

e23에서 같은 실제 운전 행동의 val 비용은 실측 70.953, MLP 재생 26.791, GBM 재생 28.391이었다. 두 모델 모두 실제 위반을 크게 과소평가한다. 따라서 이번 두 모델 평가가 좋아져도 실제 공정 절감과 안전성의 증거가 되지 않는다. 짧은 동일 예산에서 확장 효과를 구분하는 실험이며 장기 수렴 비교는 아니다.

기법 근거: [FCSRL 원 논문](https://proceedings.mlr.press/v235/cen24b.html), [보조 학습의 일반적 근거인 UNREAL](https://arxiv.org/abs/1611.05397). UNREAL 재현이나 논문의 원래 목적함수 변경으로 주장하지 않는다.

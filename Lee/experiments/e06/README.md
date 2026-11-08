# Lee/e06: Kaggle 폐수처리 데이터(Avedøre)에서 feasibility 점수를 정책 학습에 쓰는 확장

## 이 실험이 하는 일

수행계획서의 "Kaggle 폐수처리 공정 데이터로 에너지 절감과 수질 제약(NH4, DO)을 함께 고려하는
Safe RL 폭기 제어"를 그대로 따른다. 데이터, 환경, 평가는 Yoo 가 만든 것을 수정 없이 쓰고
(Yoo/e02, e05, e06), 원래 FCSRL 적용 결과보다 좋아지는 확장 한 가지를 시험한다.

Lee/e01 에서 e05 는 BSM1 물리 시뮬레이터에서 했고, 그 환경은 Kaggle 데이터가 아니며 하루 reset
학습의 결함이 있다 (Yoo/e19 가 확인). 그래서 이 실험이 Lee 의 주 확장 실험이다.

## 데이터와 환경 (Yoo 와 같다)

| 항목 | 내용 |
| --- | --- |
| 데이터셋 | Kaggle `jueyetao/wastewater-treatment-dataset` (원출처 Hansen 외, Mendeley Data, CC BY 4.0, 덴마크 Avedøre 처리장) |
| 내려받은 날 | 2026-11-08, Kaggle 공개 API. `dl.zip` sha256 `2058148f...c2e5ec`, `aved_raw.csv` sha256 `b57bc042...a2e` 모두 Yoo 의 기록과 일치 |
| 전처리 | Yoo/code/wwtp/prep.py 를 그대로 실행. 학습 330일, 검증 43일, 평가 60일로 Yoo/e02 와 같은 결과 |
| 환경 | Yoo 의 AerationEnv. 하루 287 스텝, 행동은 호기 단계의 공기량, 보상은 공기량의 음수, 비용은 NH4 > 4 와 DO > 2 지시값의 합 |
| 대리 모델 | Yoo/experiments/e02/surrogate.json (MLP 앙상블, 저장소에 있는 가중치를 그대로 사용, 다시 학습하지 않음) |
| 구현체 | 원본 FCSRL 커밋 `d882b91054d52c7a63a18a21f7db67bbbed82b63`, 수정 없음 |

코드는 Yoo 코드의 수정 없는 복사본을 `Lee/code/avedore/` 에 두었다 (파일별 sha256 은 해당 폴더 README).

## 원래 결과 (비교 기준)

같은 파이프라인에서 FCSRL + PPO-Lag (`ppolag_fcsrl`) 를 seed 5개로 직접 다시 돌려 기준으로 쓴다.
Yoo/e06 의 같은 방법 값은 비교 참고용이다 (검증 43일 x 구성원 5개, seed 2개, 선택 모델 기준:
공기량 38,527, 하루 비용 13.41, 한도 15 만족 2/2, 마지막 모델 평균 비용 16.12).

참고로 원본 코드 그대로의 FCSRL + TD3-Lag (Yoo/e02, 평가 구간 60일, seed 5개) 는 공기량 32,745,
NH4 위반 4.84%, DO 위반 0.36%, 하루 비용 14.92 이고 한도 만족 에피소드는 83.5% 다.
이 값은 TD3 계열이고 평가 구간이 달라서 직접 비교하지 않는다.

## 확장: feasibility 벌점

FCSRL 논문은 결론에서 feasibility 점수를 정책 학습의 보조 신호로 쓰는 것을 향후 과제로 남겼다.
원본 구현의 feasibility 는 표현 학습에만 쓰이고 정책 손실에는 들어가지 않는다. 행동을 한 뒤 도달한
다음 상태의 feasibility 예측 F(z_t+1) 을 표준화해 정책 이점에서 뺀다 (구현은
`Lee/code/avedore/feasibility_policy.py`).

    adjusted_advantage = advantage - weight x rms(advantage) x standardize(F(z_t+1))

- 현재 상태의 함수는 행동과 무관해 정책 경사의 기댓값이 0 이므로 다음 상태를 쓴다
- 벌점의 RMS 를 기본 이점 RMS 의 정확히 weight 배로 맞춘다 (`relative`). Lee/e05 에서 벌점을
  표준화한 F 에 weight 를 곱하고 1 + lambda 로 나누는 방식은 lambda 가 커질수록 기본 이점이
  함께 커져서 실제 크기가 명목 가중치의 1.5% 에서 18% 로 줄었다. 이를 고친 것이다
- 라그랑주 승수, 비용 제약, 표현 학습 손실은 그대로다. 안전을 보장하지 않는 실용적 벌점이다

## 실행 구성

| 항목 | 값 |
| --- | --- |
| 방법 | ppolag (PPO-Lag, 참고), ppolag_fcsrl (원래 FCSRL), ppolag_fcsrl_fpen 가중치 0.1, 0.3, 1.0 |
| seed | 0, 1, 2, 3, 4 (방법마다 5개, 총 25개 실행) |
| 학습량 | 각 404,096 전이 (Yoo/e05, e06 과 같다), 검증 5회 |
| 평가 | 검증 43일 x 대리 모델 구성원 5개 = 215 에피소드. 한도 15 를 만족하는 모델 중 보상 최대를 고르고(Yoo 규칙), 마지막 모델도 함께 보고 |
| test 구간 | 이 실험에서 쓰지 않는다. 최종 선택 뒤 한 번만 평가한다 |
| 나머지 | PPO 설정, 비용 한도 15, PID, 학습률, 대리 모델이 모두 Yoo/e05 와 같다 |

가중치 0.1, 0.3, 1.0 은 결과를 보기 전에 정했고 바꾸지 않는다. 세 값을 모두 보고하며 좋은 쪽만
고르지 않는다.

## 가설과 판정 기준 (결과를 보기 전에 정함)

- H18 (주 가설). 어떤 가중치에서든 feasibility 벌점을 더한 FCSRL 의 하루 평균 비용(선택 모델 기준,
  seed 5개 평균)이 원래 FCSRL 보다 낮다. 단 공기량이 원래 FCSRL 보다 3% 넘게 늘면 개선이 아니라
  에너지와 안전의 교환으로 적는다.
- H19. 가중치가 클수록 비용이 더 줄고 공기량은 늘어난다.
- H20. 마지막 모델(선택하지 않은 모델)에서도 같은 방향이다. Yoo/e06 에서 선택 모델은 한도를
  만족했지만 마지막 모델은 만족하지 못했으므로, 선택 효과가 아닌지 확인한다.

차이가 seed 간 표준편차보다 작으면 개선이라 단정하지 않는다. p 값은 seed 5개의 탐색적 Welch 검정이며
다중 비교 보정이 없다. 결과가 반대여도 그대로 기록한다.

## 실행

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u Lee/code/avedore/prep.py --raw <데이터>/aved_raw.csv --out <데이터>/aved_5min.pkl
python -u Lee/experiments/e06/run_av.py --data <데이터>/aved_5min.pkl --model <데이터>/surrogate.json \
  --upstream <원본 FCSRL 경로> --out <결과>/e06_runs --parallel 6 \
  --jobs ppolag_fcsrl:0 ppolag:0 ppolag_fcsrl_fpen@0.1:0 ppolag_fcsrl_fpen@0.3:0 ppolag_fcsrl_fpen@1.0:0 ...
python -u Lee/experiments/e06/report_av.py --runs <결과>/e06_runs --out Lee/experiments/e06
```

예상 시간은 6개 병렬로 약 1시간 30분이다. 시드 0에서 4까지 방법별 5개씩 25개 실행을 모두 돌린다.

## 미리 밝히는 한계

- 대리 모델 위의 결과이고 실제 공정이 아니다. Yoo 가 확인했듯 대리 모델은 NH4 위반을 실제보다 낮게
  예측하므로(하루 비용 예측 26.8, 실측 71.0) 절대 수치는 낙관적일 수 있다. 방법 간 비교만 의미가 있다.
- 검증 구간에서 모델을 고르고 같은 구간을 보고하므로 선택 편향이 있다. 마지막 모델 값을 함께 보고하고
  test 는 쓰지 않는다.
- 학습 404,096 전이는 수렴을 보장하지 않는다. seed 5개의 탐색적 비교다.
- 비용은 0, 1, 2 값이라 feasibility 의 "계속 안전할 확률" 해석(명제 4.1)이 엄밀히 성립하지 않는다 (Yoo/e05 와 같은 한계).
- Lee/e05 (BSM1) 의 벌점 크기 문제를 이 실험에서 고쳤으므로 두 실험의 가중치는 같은 뜻이 아니다.

## 실행 결과

실행 후 [results.md](results.md) 에 기록한다.

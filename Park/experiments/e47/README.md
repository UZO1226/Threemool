# Park/e47: FCSRL 기전 제거 대조, 학습 표현과 feasibility 손실 (Kaggle DARROW)

상태: **사전 등록 (2026-11-19).** 가설과 판정 규칙은 학습과 검증 결과를 보기 전에 정했다. 고를 설정은 없다. 대표는 바꾸지 않는다.

## 목적

팀 중간점검 (Yoo/analysis/TEAM_RESEARCH_REVIEW_20261117.md) 은 Park 에 대해 "현재 개선의 상당 부분은 FCSRL 보다 모델 예측 제어기의 성과로 보인다" (6.2 절) 고 했고, 다음 단계로 "살아남은 후보 하나의 기전 제거 대조" (8 절 4번, 예: feasibility 손실 제거 또는 FCSRL 표현 대 raw 입력) 와 "FCSRL 표현이 raw 입력보다 보정에 유용한가" (7.2 절 질문 4) 를 요구했다.

Park/e37 에서 같은 학습 설정과 같은 차단의 PPO-Lag (표현 없음) 와 비교하면 FCSRL 은 공기량이 같고 (-213) 비용이 0.244 낮았다 (95% 신뢰구간 -0.479 ~ -0.009, 판정 기준 미달). 이 작은 차이가 학습된 표현에서 오는지, 표현 학습 가운데 FCSRL 의 핵심인 feasibility 예측에서 오는지, 아니면 정책 입력에 64 차원을 더 붙인 것 자체 (무작위 특징) 에서 오는지 가린다.

## 방법

- 두 제거 대조를 `Park/code/fcsrl_ablation.py` 로 만든다. 망 구조, 입력 차원, 정책과 가치 학습, 학습 설정은 FCSRL 과 같고 표현 학습만 바꾼다. 같은 seed 면 초기값도 같다
  - R (무작위 고정 표현, `fcsrl_rand`): 인코더를 학습하지 않는다. 정책은 초기값 그대로인 인코더의 z 를 입력에 붙인다
  - D (feasibility 손실 없음, `fcsrl_dyn`): 원본 `train_encoder` 의 두 손실 가운데 동역학 일관성만 쓴다 (계수 0.5 그대로, feasibility 계수 0)
- 학습: 각각 seed 0 ~ 19 를 대표와 같은 명령으로 학습한다 (`train_e47.py --method <방법> --min-brake-target 3.5 --updates 16`). `train_e47.py` 는 두 이름을 등록한 뒤 `train_darrow.py` 의 main 을 그대로 부른다
- 평가: 구성원이 겹치지 않는 4 x 16 묶음 5개 (seed 0 ~ 3, 4 ~ 7, 8 ~ 11, 12 ~ 15, 16 ~ 19) 에 대표 차단 (e19 차단 + e28 원판 회복 계획) 을 붙인다. FCSRL 대표는 Park/e35 orig, PPO-Lag 는 Park/e37 의 같은 묶음 값을 쓴다
- 재현 검사: `train_e47.py --method fcsrl` 로 FCSRL seed 0 을 다시 학습해 학습 끝 평가가 `dw_u16_runs/A_s0` 기록과 같은지 본다 (진입점이 기존 학습을 바꾸지 않는지)

## 평가

검증 40일 x 구성원 5개 = 200 에피소드, 결정적. test 는 쓰지 않는다.

## 가설 (사전 등록)

Park/e37 에서 공기량은 차이가 없고 비용만 달랐으므로, 비용을 주 지표로 하고 공기량은 조건으로 둔다.

- H1 (학습된 표현이 필요한가). 대표의 하루 비용이 같은 묶음의 R 보다 낮은 묶음이 5개 중 4개 이상이고, 평균도 낮다. 단, 대표 평균 공기량이 R 보다 2% 넘게 많으면 판정하지 않는다
- H2 (feasibility 손실이 필요한가). 같은 규칙으로 대표 대 D

참고 (판정 없음): D 대 R (동역학 일관성만의 몫), R 대 PPO-Lag (무작위 특징을 붙인 것만의 몫), 대표 대 PPO-Lag (e37 재게시), 각 짝 차이의 95% 신뢰구간, DO 위반 차이, 차단 개입률.

## RESULT.md 처리 (미리 정함)

대표 방법은 바꾸지 않는다.

- H1, H2 지지: "같은 설정의 비용 이득은 학습된 표현, 그중 feasibility 예측 손실에서 온다" 고 적는다 (FCSRL 고유 기여의 근거)
- H1 지지, H2 기각: "학습된 표현이 기여하지만 feasibility 손실 없이도 같다. 기여는 동역학 일관성 표현의 몫" 이라고 적는다
- H1 기각: "FCSRL 학습 표현의 기여는 확인되지 않는다. e37 의 작은 비용 차이는 무작위 특징으로도 설명될 수 있다" 고 한계에 적는다. 이 경우 H2 는 참고로만 적는다

## 미리 밝히는 한계

- 묶음 5개는 판정 표본이 작다 (4/5 기준). e37 의 차이 (0.244) 가 작아 검정력이 낮다. 기각이 "기여 없음" 의 증명은 아니다
- 학습 설정은 FCSRL 로 골랐다 (Park/e12, e20 ~ e22)
- R 은 인코더를 학습하지 않을 뿐 feasibility 목표 계산 (학습 자료의 F 추정) 은 그대로 돈다. 정책 학습에는 쓰이지 않는다
- 검증 구간은 e10 부터 반복해서 썼다. e10 ~ e46 과 같은 대리 모델 한계

## 다시 돌리기

```
for m in fcsrl_rand:R:dw_rand16_runs fcsrl_dyn:D:dw_dyn16_runs; do IFS=: read M P DIR <<< "$m"; for s in $(seq 0 19); do OMP_NUM_THREADS=2 python train_e47.py --method $M --seed $s --data <darrow>/darrow_15min.pkl --surrogate <darrow>/darrow_surrogate_gbm.pkl --min-brake-target 3.5 --updates 16 --out <결과>/$DIR/${P}_s$s; done; done
OMP_NUM_THREADS=2 python train_e47.py --method fcsrl --seed 0 --data .. --surrogate .. --min-brake-target 3.5 --updates 16 --out <결과>/dw_chk_runs/A_s0
for i in 0 1 2 3; do OMP_NUM_THREADS=2 python run_e47.py run --part $i --nparts 4 --data .. --surrogate .. --runs <결과> --out <결과>/e47; done
python run_e47.py report --out <결과>/e47 --runs <결과> --e35 <결과>/e35 --e37 <결과>/e37 --md results.md
```

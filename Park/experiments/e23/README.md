# Park/e23: 새 seed 로 e22 앙상블 재현 (Kaggle DARROW)

상태: **사전 등록 (2026-11-12).** 가설과 판정 규칙은 학습과 검증 결과를 보기 전에 정했다.

## 목적

목표는 공기량과 하루 비용을 함께 낮추는 것이다. Park/e22 의 4 x 16 앙상블 (seed 0 ~ 3) 은 e19 H1 보다 둘 다 낮았지만, seed 0 ~ 4 의 다른 4개 조합에서는 판정이 0 ~ 2/5 였다. 또 e19 H1 기준 정책 (seed 0 ~ 4) 과 앙상블 구성원이 같은 seed 라서 서로 독립이 아니다. 이번에는 한 번도 쓰지 않은 seed 5 ~ 9 로 같은 방법을 처음부터 다시 해서, e22 결과가 seed 운인지 본다.

## 방법 (새로 고르는 설정 없음)

- 기준: seed 5 ~ 9 를 e12 A 설정 (학습 중 차단 target 3.5, 업데이트 66번) 으로 학습한 정책 하나씩 + e19 H1 차단
- 앙상블: seed 5 ~ 8 을 같은 설정으로 업데이트 16번씩 학습해 행동 평균 + e19 H1 차단 (e22 H1 과 같은 방법, 구성원은 새 seed 중 앞 4개)
- seed 9 의 업데이트 16번 정책은 참고 조합에만 쓴다
- 앙상블 합계 업데이트 64번으로 정책 하나 (66번) 보다 학습 예산이 작다

## 조건과 평가

검증 40일 x 구성원 5개 = 200 에피소드를 결정적으로 돈다. test 는 쓰지 않는다. 학습 구간에서 고르는 단계는 없다.

## 가설 (사전 등록)

- H1 (주 가설). 앙상블 (seed 5 ~ 8) 이 새 seed 5 ~ 9 정책 하나 + e19 차단 중 3개 이상보다 공기량과 하루 비용이 둘 다 낮고, 그 평균보다도 둘 다 낮다
- H2. 앙상블 (seed 5 ~ 8) 이 e19 H1 (seed 0 ~ 4) 평균 (80,748, 4.98) 보다 공기량과 하루 비용이 둘 다 낮다

새 seed 4개 조합 5개는 보고만 한다. 가설이 기각되는 결과도 그대로 기록한다.

## RESULT.md 처리 (미리 정함)

대표 결과 (e22 H1) 는 바꾸지 않는다. H1 결과 (지지 또는 기각) 와 새 seed 값들을 RESULT.md 해석에 재현 결과로 적는다.

## 미리 밝히는 한계

- seed 5개 묶음 하나를 더 볼 뿐이라, 앙상블 효과의 크기를 정확히 재지는 못한다
- e10 ~ e22 와 같은 대리 모델 한계

## 실행 명령

```bash
export OMP_NUM_THREADS=1
cd Park/code
for s in 5 6 7 8 9; do python train_darrow.py --method fcsrl --seed $s --data <Park/data/darrow>/darrow_15min.pkl --surrogate <Park/data/darrow>/darrow_surrogate_gbm.pkl --out <결과>/dw_new66_runs/A_s$s --min-brake-target 3.5; done
for s in 5 6 7 8 9; do python train_darrow.py --method fcsrl --seed $s --data <Park/data/darrow>/darrow_15min.pkl --surrogate <Park/data/darrow>/darrow_surrogate_gbm.pkl --out <결과>/dw_new16_runs/A_s$s --min-brake-target 3.5 --updates 16; done
cd ../experiments/e23
python run_e23.py run --data <Park/data/darrow>/darrow_15min.pkl --surrogate <Park/data/darrow>/darrow_surrogate_gbm.pkl --runs <결과> --out <결과>/e23
python run_e23.py report --out <결과>/e23 --e19 <결과>/e19 --md results.md
```

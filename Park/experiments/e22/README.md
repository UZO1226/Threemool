# Park/e22: 같은 학습 예산에서 구성원을 줄이고 더 길게 학습한 정책 앙상블 (Kaggle DARROW)

상태: **사전 등록 (2026-11-12).** 가설과 판정 규칙은 검증 결과를 보기 전에 정했다. 설정은 학습 구간만으로 골랐다.

## 목적

목표는 공기량과 하루 비용을 함께 낮추는 것이다. 비교 기준은 대표 결과인 Park/e19 H1 (검증 seed 평균 공기량 80,748, 비용 4.98) 이다.

## e21 의 실패 원인

- 업데이트 12번 x 5 앙상블은 비용을 크게 낮췄지만 (4.80) 공기량이 seed 3개보다 많았다.
- 공기량은 학습을 덜 할수록 많았다 (업데이트 10번 81,798, 12번 80,209, 66번 78,933). 정책이 공기를 아끼는 법은 학습 뒷부분에서 배운다.
- 그래서 같은 예산에서 구성원 수를 줄이고 하나하나를 더 길게 학습한다.

## 방법

seed 0 부터 k 개 정책을 각각 업데이트 u 번 학습해 (`train_darrow.py --updates u`, e12 A 와 같은 학습 중 차단 target 3.5) 행동을 평균하고, e19 H1 차단을 건다. k x u <= 66 이라 PROTOCOL v1 의 학습 예산 안이다. 구성원 seed 는 고르지 않고 0 부터 차례로 쓴다.

## 설정 고르기 (학습 구간만)

학습 구간 40일(4일 간격) x 구성원 5개. 정책은 e12 A 를 `--save-every 2` 로 다시 학습한 중간 정책 (업데이트 u 번 학습과 같음, e21 에서 확인) 을 썼다. 규칙은 결과를 보기 전에 정했다: "seed 0 ~ k-1 앙상블 중 평균 비용과 평균 공기량이 e19 H1 seed 평균보다 낮고, 둘 다 이긴 seed 가 가장 많은 것 (3 이상, 같으면 공기량이 적은 것). 없으면 버린다". 다른 seed 조합은 참고로만 봤다.

| 방법 | 공기량 | 하루 비용 | e19 H1 보다 둘 다 낮은 seed | 참고: 모든 seed 조합 평균 (공기량, 비용, 이긴 seed) |
| --- | --- | --- | --- | --- |
| e19 H1 seed 평균 | 68,224 | 2.375 | | |
| **4 x 16, seed 0 ~ 3 (규칙으로 고름)** | **65,836** | **2.065** | 5/5 | 65,979, 2.077, 4.6 (조합 5개) |
| 3 x 22, seed 0 ~ 2 | 65,958 | 2.055 | 4/5 | 66,148, 2.102, 3.6 (조합 10개) |
| 2 x 32, seed 0 ~ 1 | 66,757 | 1.970 | 3/5 | 67,683, 2.100, 2.8 (조합 10개) |
| 참고: 5 x 12 (e21 H1) | 66,636 | 2.030 | 4/5 | |

## 조건과 평가

seed 0 ~ 4 를 `--updates 16` 으로, seed 0 ~ 2 를 `--updates 22` 로 새로 학습한다 (seed 4 의 업데이트 16번 정책은 참고 조합에만 쓴다). 학습한 정책이 다시 학습한 e12 A 의 16번째, 22번째 정책과 같은지 확인해 결과에 적는다. 검증 40일 x 구성원 5개 = 200 에피소드를 결정적으로 돈다. test 는 쓰지 않는다.

## 가설 (사전 등록)

앙상블 값과 e19 H1 의 seed 5개를 비교한다. 앙상블이 seed 5개 중 3개 이상보다 공기량과 하루 비용이 둘 다 낮고, seed 평균보다도 둘 다 낮으면 지지로 적는다.

- H1 (주 가설). 4 x 16, seed 0 ~ 3
- H2. 3 x 22, seed 0 ~ 2

4 x 16 의 다른 seed 조합 4개는 보고만 한다 (seed 선택에 결과가 얼마나 달라지는지 보기 위해).

## RESULT.md 처리 (미리 정함)

H1 이 지지되면 학습 예산이 PROTOCOL v1 안이므로 대표 결과를 e22 H1 으로 바꾸고, 값이 하나라는 점과 다른 seed 조합 결과를 함께 적는다. H2 만 지지되면 대표 결과는 바꾸지 않는다.

## 미리 밝히는 한계

- 앙상블은 제어기 하나라 seed 사이 편차 대신 seed 조합 사이 차이로만 안정성을 본다
- 학습 구간에서 고른 구성이 검증 구간(겨울) 에서도 가장 좋을지는 모른다 (e21 에서 학습 구간 순서가 검증에서 바뀌었다)
- e10 ~ e21 과 같은 대리 모델 한계

## 실행 명령

```bash
export OMP_NUM_THREADS=1
cd Park/code
for s in 0 1 2 3 4; do python train_darrow.py --method fcsrl --seed $s --data <Park/data/darrow>/darrow_15min.pkl --surrogate <Park/data/darrow>/darrow_surrogate_gbm.pkl --out <결과>/dw_u16_runs/A_s$s --min-brake-target 3.5 --updates 16; done
for s in 0 1 2; do python train_darrow.py --method fcsrl --seed $s --data <Park/data/darrow>/darrow_15min.pkl --surrogate <Park/data/darrow>/darrow_surrogate_gbm.pkl --out <결과>/dw_u22_runs/A_s$s --min-brake-target 3.5 --updates 22; done
cd ../experiments/e22
python run_e22.py run --data <Park/data/darrow>/darrow_15min.pkl --surrogate <Park/data/darrow>/darrow_surrogate_gbm.pkl --runs <결과> --out <결과>/e22
python run_e22.py report --out <결과>/e22 --e19 <결과>/e19 --md results.md
```

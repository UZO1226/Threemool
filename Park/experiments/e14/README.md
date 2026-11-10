# Park/e14: DO 가 넘칠 때 공기량을 내리기도 하는 양방향 차단 (Kaggle DARROW)

상태: **사전 등록 (2026-11-11).** 가설과 판정 규칙은 학습 결과를 보기 전에 정했다.

## 목적

목표는 공기량(전기) 과 하루 비용을 함께 낮추는 것이다. 지금 가장 좋은 Park/e12 A (학습 중 최소 개입 차단 target 3.5) 는 공기량 84,031, 하루 비용 5.51 이고, 비용은 NH4 위반 3.43% 와 DO 위반 2.37% 로 이뤄진다. Park/e13 에서 차단 의존을 줄이는 방향은 공기량이 늘어 목표와 맞지 않았다. 이번에는 다시 성능을 노린다.

지금 차단은 공기량을 올리기만 한다. 정책이 공기를 너무 넣어 구간 DO 가 상한 2 를 넘으려 해도 막지 못한다. 공기량을 내리면 공기량과 DO 위반이 함께 준다. 그래서 NH4 를 지키는 데 필요한 최소 공기량 아래로는 내리지 않는 조건으로, DO 가 넘칠 것 같을 때 공기량을 내리는 양방향 차단을 만든다. Park/e11 의 실패 원인(NH4 와 DO 가 충돌할 때 NH4 여유를 버림) 을 피하려고, 충돌하면 언제나 NH4 를 우선한다.

## 공통 평가 조건

`Compare/PROTOCOL.md` v1 을 따른다 (데이터, 분할, 대리 모델, 관측, 행동, 보상, NH4 와 DO 비용, 하루 한도 12, 학습 예산, seed, 검증 200 에피소드). NH4 위반률과 DO 위반률을 따로 적고, 한도 만족은 에피소드 비율과 seed 평균 비용 <= 12 여부로 함께 본다.

## 양방향 차단 (`Park/code/min_brake.py` 의 `trim=True`)

예측 방법은 e10 과 같다 (환경이 쓰는 구성원을 뺀 4개 평균, 1스텝, 외부 입력 지속 예측, 격자 250 간격).

1. e10 처럼 예측 NH4 <= target 을 만족하는 가장 작은 공기량 r 을 구한다
2. 정책 공기량 < r 이면 r 로 올린다 (e10 과 같음)
3. 정책 공기량 >= r 이고 정책 공기량에서 예측 구간 DO 최댓값 > 2 이면, r 이상 정책 공기량 이하의 격자 값 중 예측 DO <= 2 인 가장 큰 공기량으로 내린다. 없으면 r 로 내린다

r 이상으로만 내리므로 예측 NH4 <= target 은 그대로 지킨다. e12 A seed 0 정책으로 검증 8일을 돈 시험에서 e10 차단과 비교해 DO 위반이 0.97% 에서 0.39% 로 줄었다 (판정에 쓰지 않음).

## 조건

데이터, 환경, 학습 설정은 Park/e12 A 와 같다 (FCSRL + PPO-Lag, 환경 8개 x 760 스텝 x 66 갱신, seed 0 ~ 4, 학습 비용 한도 12, 개입 비용 없음). 학습 중 처리 방식도 같다 (PPO 확률비는 정책이 뽑은 행동, 보상, 비용, 표현 학습은 실제로 넣은 행동).

| 조건 | 내용 | 명령 인자 |
| --- | --- | --- |
| A (주 조건) | 양방향 차단 target 3.5 로 학습 | `--min-brake-target 3.5 --brake-trim` |
| B | 양방향 차단 target 3.0 으로 학습 (더 엄격) | `--min-brake-target 3.0 --brake-trim` |
| E | 학습 없음. e12 A 정책에 차단만 양방향으로 바꿈 | 없음 |

`--brake-trim` 을 주지 않으면 e12 와 같다.

## 평가

검증 40일 x 구성원 5개 = 200 에피소드, 결정적 행동, 마지막 정책. A, B, E (e12 A 정책) 각각을 차단 없이, 한 방향 최소 개입 target 3.0, 3.5, 양방향 target 3.0, 3.5, 4.0 으로 평가한다. test 는 쓰지 않는다. 각 조건의 학습에 쓴 설정(A 는 양방향 3.5, B 는 양방향 3.0, E 는 양방향 3.5) 이 대표값이다.

## 가설 (사전 등록)

비교 기준은 같은 seed 의 e12 A 정책 + 한 방향 target 3.5 (e12 의 대표값, seed 평균 84,031, 5.51) 다. seed 5개 중 3개 이상에서 성립하고 seed 평균도 성립하면 지지로 적는다. 성립은 공기량과 하루 비용이 둘 다 기준보다 낮은 것이다.

- H1 (주 가설). A 의 대표값이 공기량과 하루 비용 모두 e12 A 보다 낮다
- H2. B 의 대표값이 공기량과 하루 비용 모두 e12 A 보다 낮다
- H3. E (e12 A 정책 + 양방향 target 3.5) 가 공기량과 하루 비용 모두 e12 A 보다 낮다

NH4 위반률, DO 위반률, 내린 스텝 비율, 차단 없이 평가한 비용, 학습 마지막 라그랑주 승수는 보고만 한다. 가설이 기각되는 결과도 그대로 기록한다.

## 미리 밝히는 한계

- e10, e12 와 같은 대리 모델 한계 (예측 정확도 낮음, 차단 모델과 환경 모델이 같은 방법의 구성원, 그 시각의 유입 성분을 안다고 둠)
- 차단이 올린 공기 때문에 DO 가 넘는 경우(NH4 와 충돌) 는 줄이지 않는다
- 대리 모델의 DO 는 해당 구간 공기량에 대해서만 비감소로 강제돼 있다. 공기량을 내리면 DO 가 내려간다고 기대하지만 보장은 아니다
- 학습 중 차단은 다시 정책의 차단 의존(e12, e13) 을 만들 수 있다
- seed 5개, 401,280 전이, 마지막 정책의 탐색적 실험이다

## 실행 명령

```bash
export OMP_NUM_THREADS=1 WANDB_MODE=disabled
D="--data <Park/data/darrow>/darrow_15min.pkl --surrogate <Park/data/darrow>/darrow_surrogate_gbm.pkl"
cd Park/code
for s in 0 1 2 3 4; do
  python train_darrow.py --method fcsrl --seed $s $D --out <결과>/dw_trim_runs/A_s$s --min-brake-target 3.5 --brake-trim
  python train_darrow.py --method fcsrl --seed $s $D --out <결과>/dw_trim_runs/B_s$s --min-brake-target 3.0 --brake-trim
done
cd ../experiments/e14
for c in A B E; do for s in 0 1 2 3 4; do python run_e14.py one --cond $c --seed $s $D --runs <결과>/dw_trim_runs --e12-runs <결과>/dw_minbrake_runs --out <결과>/e14; done; done
python run_e14.py report --out <결과>/e14 --runs <결과>/dw_trim_runs --e12 <결과>/e12 --md results.md
```

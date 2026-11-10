# Park/e13: 차단 개입에 학습 비용을 줘서 차단 의존 줄이기 (Kaggle DARROW)

상태: **사전 등록 (2026-11-10).** 가설과 판정 규칙은 학습 결과를 보기 전에 정했다.

## 목적

Park/e12 의 정책은 학습 중 최소 개입 차단을 켜서 target 3.5 차단과 함께 쓸 때 공기량 84,031, 하루 비용 5.51 로 e08 차단 기준(91,279, 6.62) 보다 둘 다 낮았다. 하지만 학습 중 비용이 한도보다 낮아 라그랑주 승수가 0 이 되었고, 정책은 차단에 기댔다. 차단을 끄면 하루 비용 34.44, 학습보다 느슨한 target 4.0 이면 16.85 다. 차단 모델이 틀리거나 차단을 쓸 수 없을 때 크게 무너진다는 뜻이다.

이 실험은 안정성을 본다. 차단이 개입한 스텝마다 학습 비용에 작은 값을 더한다. 개입이 많으면 학습 비용이 한도 12 를 넘어 라그랑주 승수가 살아나고, 정책은 차단이 올려 줄 공기를 스스로 넣도록 배운다. 목표는 차단에 덜 기대면서도 e12 처럼 공기량과 비용을 함께 낮게 유지하는 것이다.

## 공통 평가 조건

`Compare/PROTOCOL.md` v1 을 따른다 (데이터, 분할, 대리 모델, 관측, 행동, 보상, NH4 와 DO 비용, 하루 한도 12, 학습 예산, seed, 검증 200 에피소드). 개입 비용은 확장 기법의 학습 설정이며, 평가의 비용 정의와 한도 12 는 v1 그대로다. NH4 위반률과 DO 위반률을 따로 적고, 한도 만족은 에피소드 비율과 seed 평균 비용 <= 12 여부로 함께 본다.

## 조건

데이터, 환경, 학습 설정, 학습 중 차단은 Park/e12 조건 A 와 같다 (FCSRL + PPO-Lag, 환경 8개 x 760 스텝 x 66 갱신, seed 0 ~ 4, 최소 개입 1스텝, 다른 구성원, target 3.5, 학습 비용 한도 12). 다른 점은 학습 비용 하나다.

학습 비용 = 실제 비용(NH4 > 4 이면 1, 구간 DO 최댓값 > 2 이면 1) + w x 1[차단 개입]

학습 비용은 라그랑주 승수 갱신, 비용 가치 함수, FCSRL 실행 가능성 표현에 모두 들어간다. e12 A 의 학습 마지막 개입 비율은 약 30 ~ 60% (하루 약 30 ~ 55 스텝), 실제 학습 비용은 2 ~ 4 였다. w 는 그 값에서 학습 비용이 한도 12 를 넘도록 골랐다.

| 조건 | 명령 인자 | w | 한도 12 에서 실제 비용 3 일 때 하루 개입 허용 |
| --- | --- | --- | --- |
| A | `--min-brake-target 3.5 --intervention-cost 0.5` | 0.5 | 약 18 스텝 |
| B | `--min-brake-target 3.5 --intervention-cost 1.0` | 1.0 | 약 9 스텝 |

`--intervention-cost` 를 주지 않으면 e12 와 같다. 바꾼 뒤 seed 0 의 1 갱신 시험이 e12 A 기록과 같은지 확인했다.

## 평가

검증 40일 x 구성원 5개 = 200 에피소드, 결정적 행동, 마지막 정책. 각 정책을 차단 없이, 그리고 최소 개입 1스텝 (다른 구성원) target 2.0, 2.5, 3.0, 3.5, 4.0 으로 평가한다 (e12 와 같음). test 는 쓰지 않는다.

## 가설 (사전 등록)

조건 A 와 B 에 각각 같은 가설을 판정한다 (A 가 주 조건). seed 5개 중 3개 이상에서 성립하고 seed 평균도 성립하면 지지로 적는다. e12 와 비교할 때는 같은 seed 의 e12 조건 A 정책과 짝지어 비교하고, 평균은 e12 A 의 seed 평균과 비교한다.

- H1 (주 가설, 차단 의존 감소). 차단 없이 평가한 하루 비용이 같은 seed 의 e12 A 보다 낮다 (e12 A 평균 34.44)
- H2 (느슨한 차단에서의 안정성). target 4.0 으로 평가한 하루 비용이 같은 seed 의 e12 A 보다 낮다 (e12 A 평균 16.85)
- H3 (목표 유지). target 3.5 에서 공기량 < 91,279 이고 동시에 하루 비용 < 6.62 다 (e08 차단 기준, e12 의 H2 와 같은 판정)

target 3.5 에서 e12 A (84,031, 5.51) 와의 비교, 개입 비율, 학습 마지막 라그랑주 승수와 실제 학습 비용은 보고만 한다. 가설이 기각되는 결과도 그대로 기록한다.

## 미리 밝히는 한계

- e10, e12 와 같은 대리 모델 한계 (예측 정확도 낮음, 차단 모델과 환경 모델이 같은 방법의 구성원, 그 시각의 유입 성분을 안다고 둠)
- 개입 비용은 개입 여부만 세고 올린 공기량의 크기는 보지 않는다. w 는 두 값만 본다
- 개입 비용 때문에 정책이 공기를 더 쓰게 되어 H3 이 깨질 수 있다
- seed 5개, 401,280 전이, 마지막 정책의 탐색적 실험이다

## 실행 명령

```bash
export OMP_NUM_THREADS=1 WANDB_MODE=disabled
cd Park/code
for s in 0 1 2 3 4; do
  python train_darrow.py --method fcsrl --seed $s --data ../data/darrow/darrow_15min.pkl --surrogate ../data/darrow/darrow_surrogate_gbm.pkl --out <결과>/dw_intcost_runs/A_s$s --min-brake-target 3.5 --intervention-cost 0.5
  python train_darrow.py --method fcsrl --seed $s --data ../data/darrow/darrow_15min.pkl --surrogate ../data/darrow/darrow_surrogate_gbm.pkl --out <결과>/dw_intcost_runs/B_s$s --min-brake-target 3.5 --intervention-cost 1.0
done
cd ../experiments/e13
for c in A B; do for s in 0 1 2 3 4; do python run_e13.py one --cond $c --seed $s --data ../../data/darrow/darrow_15min.pkl --surrogate ../../data/darrow/darrow_surrogate_gbm.pkl --runs <결과>/dw_intcost_runs --out <결과>/e13; done; done
python run_e13.py report --out <결과>/e13 --runs <결과>/dw_intcost_runs --e12 <결과>/e12 --md results.md
```

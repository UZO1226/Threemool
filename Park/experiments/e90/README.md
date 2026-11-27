# Park/e90: 대표 대 공식 기준선 PPO-Lag, 공기와 비용을 날짜 블록으로 (학습 구간) (Kaggle DARROW)

상태: **사전 등록 (2026-11-26).** 가설과 판정 규칙은 이 실험의 결과를 보기 전에 정했다. 대표는 바꾸지 않는다.

## 목적

Park/e89 는 검증 40일에서 대표와 공식 기준선 PPO-Lag 의 공기와 비용을 날짜 블록으로 판정한다. Park/e80, e83 에서 검증 40일과 학습 구간 마지막 40일의 결과가 달랐으므로, 목표의 핵심 비교도 두 구간에서 모두 본다. 대표와 PPO-Lag 는 둘 다 학습 구간으로 학습했으므로 학습한 날의 이점은 양쪽에 같다.

## 방법

- Park/e89 와 같다. 날만 학습 구간 마지막 40일 (Park/e43, e60, e80, e83 과 같음)
- 대표 스텝 기록은 Park/e83 (묶음 5개, Park/e43 값과 5/5 같음). PPO-Lag seed 0 ~ 4 는 새로 돈다

## 평가

학습 구간 마지막 40일 x 구성원 5개, 결정적. test 는 쓰지 않는다.

## 가설 (사전 등록)

방향은 공식 표에서 정했다.

- H1. (대표 빼기 PPO-Lag) 하루 비용의 블록 95% 구간이 0 아래다
- H2. (대표 빼기 PPO-Lag) 하루 공기량의 블록 95% 구간이 0 아래다

## RESULT.md 처리 (미리 정함)

- 둘 다 지지: "학습 구간 마지막 40일에서도 대표는 PPO-Lag 보다 공기와 비용을 함께 줄인다" 고 적고, Park/e89 와 함께 두 구간 결과로 적는다
- 하나만 지지, 둘 다 아님: Park/e89 와 같은 방식으로 적는다

## 미리 밝히는 한계

- Park/e89 와 같다. 학습 구간은 두 방법 모두 학습한 날이다

## 다시 돌리기

```
for i in 0 1 2 3 4; do OMP_NUM_THREADS=2 python run_e90.py run --part $i --nparts 5 --data <darrow>/darrow_15min.pkl --surrogate <darrow>/darrow_surrogate_gbm.pkl --runs <runs> --out <결과>/e90; done
python run_e90.py report --out <결과>/e90 --e83 <결과>/e83 --md results.md
```

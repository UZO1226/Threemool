# Park/e51: 새 seed 블록 확인과 차단 개입률 (Kaggle DARROW)

상태: **사전 등록 (2026-11-20).** 가설과 판정 규칙은 새 seed 학습과 검증 결과를 보기 전에 정했다. 고를 설정은 없다. 대표는 바꾸지 않는다.

## 목적

팀 중간점검 (Yoo/analysis/TEAM_RESEARCH_REVIEW_20261117.md) 8절 5번은 "새 학습 묶음에서 고정 후보 확인. 새 seed 블록을 독립 보고. 앙상블은 구성원 비중복 묶음 단위. 실패하면 후보 미확증" 을 요구했다. 대표의 같은 공기량 비교 (Park/e43) 와 FCSRL 기전 대조 (Park/e47) 는 모두 seed 0 ~ 19 에서 나왔다. Park/e47 은 FCSRL 표현 전체를 쓴 정책의 차단 개입률 (24%) 이 무작위 고정 표현 (46%) 보다 5개 묶음 모두 낮다는 것을 결과를 본 뒤 찾았다. 이 실험은 한 번도 쓰지 않은 seed 20 ~ 35 로 세 결과를 다시 확인한다.

## 방법

- 학습: FCSRL 과 무작위 고정 표현 (`fcsrl_rand`, Park/e47 의 R) 을 seed 20 ~ 35 로 대표와 같은 명령으로 학습한다 (`train_e47.py --method <방법> --min-brake-target 3.5 --updates 16`)
- 평가: 구성원이 겹치지 않는 4 x 16 묶음 4개 (seed 20 ~ 23, 24 ~ 27, 28 ~ 31, 32 ~ 35) 에 대표 차단을 붙인다
- 같은 공기량 기준: Park/e43 이 검증에서 돈 ABAC 곡선 (설정값을 바꿔 공기량을 바꿈, 고른 설정 포함) 과 차단만 곡선 (여유를 바꿈) 을 각 묶음의 공기량에서 선형 보간한 비용. 새로 돌리거나 고르지 않는다

## 평가

검증 40일 x 구성원 5개 = 200 에피소드, 결정적. test 는 쓰지 않는다.

## 가설 (사전 등록)

- H1 (새 seed 에서 공식 기준선 대비). FCSRL 묶음 4개 모두 공기량과 하루 비용이 공식 기준선 (기본 PPO-Lag, 90,121, 10.79) 보다 낮다
- H2 (새 seed 에서 같은 공기량 기준 대비). FCSRL 묶음 4개 모두 하루 비용이 그 묶음 공기량에서 보간한 ABAC 와 차단만의 비용보다 낮다. 묶음 공기량이 곡선 범위 밖이면 판정하지 않는다
- H3 (차단 개입률). FCSRL 묶음 4개 모두 차단 개입률이 같은 seed 의 무작위 표현 묶음보다 낮다

참고 (판정 없음): FCSRL 대 무작위 표현의 비용, 공기량 차이와 95% 신뢰구간, 기존 seed 0 ~ 19 의 값.

## RESULT.md 처리 (미리 정함)

대표 방법은 바꾸지 않는다.

- H1, H2 지지: "대표 방법은 한 번도 쓰지 않은 seed 블록에서도 공식 기준선과 같은 공기량의 ABAC, 차단만보다 낫다" 고 적는다. 기각된 쪽은 "새 seed 에서 확증되지 않음" 으로 적는다
- H3 지지: "FCSRL 표현이 차단 개입을 줄인다" 를 사후 관찰에서 사전 등록 확인 결과로 올린다. 기각: e47 의 개입률 관찰은 seed 0 ~ 19 에만 맞았다고 적는다

## 미리 밝히는 한계

- 묶음 4개는 판정 표본이 작다. 4/4 를 요구하므로 묶음 하나의 실패로 기각된다
- 같은 공기량 기준은 보간이라 곡선 점 사이의 비선형을 무시한다. 기준 곡선은 대표 차단과 같은 검증 날짜에서 나왔다
- 검증 구간은 e10 부터 반복해서 썼다. 새 seed 는 새 날짜가 아니다

## 다시 돌리기

```
for m in fcsrl:A:dw_s20_runs fcsrl_rand:R:dw_rand16_runs; do IFS=: read M P DIR <<< "$m"; for s in $(seq 20 35); do OMP_NUM_THREADS=2 python ../e47/train_e47.py --method $M --seed $s --data <darrow>/darrow_15min.pkl --surrogate <darrow>/darrow_surrogate_gbm.pkl --min-brake-target 3.5 --updates 16 --out <결과>/$DIR/${P}_s$s; done; done
for i in 0 1 2 3; do OMP_NUM_THREADS=2 python run_e51.py run --part $i --nparts 4 --data .. --surrogate .. --runs <결과> --out <결과>/e51; done
python run_e51.py report --out <결과>/e51 --e43 <결과>/e43 --md results.md
```

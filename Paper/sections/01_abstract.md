<!-- owner: none -->
## Abstract

Using Tilburg wastewater records, we trained FCSRL aeration policies on past data at two time points and compared four configurations over the next 40 days. Over a common air-use range, adding risk-based action selection to an output-corrected actor reduced ammonium exceedance duration and severity without increasing oxygen exceedance duration in 5/5 and 4/5 seeds. The combined policy improved on basic FCSRL in 5/5 and 2/5 seeds. At fixed water quality targets, it saved air in only 1/5 and 2/5 seeds. Errors in replaying recorded aeration limit these findings to the surrogate environments.
<!-- exp: Yoo/e139 decision.csv, quality_pairs.csv and replay.csv, commit e3d6499 -->

## 요약

하수처리장의 폭기량을 줄이면서 수질을 유지할 수 있는지 알아보기 위해 Tilburg 처리장 운전 기록으로 안전 강화학습 기법 FCSRL을 평가하였다. 학습 중 정책 출력을 보정하는 방법과 운전 시점에 수질 위험이 낮은 행동을 선택하는 방법을 각각 또는 함께 적용했다. 두 시점에서 과거 자료로 정책과 대리 모델을 다시 학습하고 바로 다음 40일을 평가했다. 공기량이 겹치는 범위에서 출력 보정 정책에 위험 선택을 더하자 NH4 위반 시간과 초과 적분이 줄고 DO 위반 시간이 늘지 않은 시드는 각각 5개 중 5개와 4개였다. 결합 정책을 기본 FCSRL과 비교하면 각각 5개와 2개였고, 같은 수질 목표를 지키면서 공기까지 절약한 경우는 1개와 2개였다. 위험 선택의 수질 이득은 출력 보정 정책에서 두 기간에 걸쳐 관찰되었다. 결합 정책의 공기 절감은 일관되지 않았으며, 기록 운전 재생에서 나타난 수질 예측 오차는 현장 적용의 한계로 남는다.
<!-- exp: Yoo/e139 decision.csv, quality_pairs.csv and replay.csv, commit e3d6499 -->

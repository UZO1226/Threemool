<!-- owner: none -->
## Abstract

Using Tilburg wastewater records and learned surrogates, we evaluated FCSRL aeration policies with two execution-time modules. On 15 policies, a predictive shield cut daily cost over 188 training days from 11.5 to 12.7 to about 1.7 while using more air, and beat risk-based action selection on 40 validation days. With the shield on, learned policies differed little from an ammonium feedback controller, and FCSRL did not reduce shield intervention relative to PPO-Lag at equal air. Under a surrogate fitted to earlier data, the shield no longer improved the feedback controller.
<!-- exp: Yoo/e155 results.md, results commit 7080520 -->

## 요약

Tilburg 하수처리장 운전 기록과 이를 학습한 대리 모델로 안전 강화학습 기법 FCSRL의 폭기 정책과 실행 직전에 명령을 고치는 두 모듈을 평가하였다. 같은 정책 15개에 예측 차단을 붙이자 학습 188일의 하루 비용이 11.5~12.7에서 약 1.7로 낮아졌고 공기는 더 사용했다. 검증 40일에서는 차단이 위험 기반 행동 선택보다 비용과 공기가 모두 낮았다. 차단을 켜면 학습 정책과 암모늄 피드백 제어기의 차이는 작았고, FCSRL이 같은 공기량에서 PPO-Lag보다 차단 개입을 줄인다는 근거는 없었다. 과거 자료로 다시 학습한 시간순 비교에서 결합 정책이 기본 FCSRL의 수질을 지키며 공기를 절약한 시드는 두 기간 각각 1개와 2개였다. 앞선 기간 자료로 적합한 대리 모델에서는 피드백 제어기에 대한 차단 효과가 사라져, 결과는 대리 환경 안에서만 유효하다.
<!-- exp: Yoo/e155 results.md, results commit 7080520; Yoo/e139 quality_pairs.csv, commit e3d6499 -->

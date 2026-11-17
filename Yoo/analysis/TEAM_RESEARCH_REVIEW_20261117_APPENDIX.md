# 연구 검토 부록: 실험 색인과 확인 자료

[본문](TEAM_RESEARCH_REVIEW_20261117.md)의 근거 탐색 경로와 고정 버전을 기록한다. 이 색인의 존재는 모든 파일 또는 모든 원시 로그를 줄 단위로 감사했다는 뜻이 아니다. 전체 실험 경로와 제목을 목록화하고 종합 결과로 흐름을 확인한 뒤, 본문에서 인용한 주요 결과와 코드를 상세 검토했다.

## 1. 전체 실험 README 색인

결과 파일 존재는 완료 판정이 아니다. 계획, 중단, 일부 결과 여부는 해당 본문과 manifest로 판단한다. 같은 번호도 작성자가 다르면 다른 실험이다.

### Lee (72개 README)

| 실험 | 제목 | 함께 존재하는 결과 문서 |
| --- | --- | --- |
| [Lee/e01](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e01/README.md) | Lee/e01: 안전 비용을 평균 농도에서 순간 위반 지시값으로 바꾼 비교 | results.md |
| [Lee/e02](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e02/README.md) | Lee/e02: 같은 비교를 4배 긴 예산으로 다시 수행 | results.md |
| [Lee/e03](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e03/README.md) | Lee/e03: 제약이 엄격할수록 FCSRL 의 이점이 커지는가 | results.md |
| [Lee/e04](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e04/README.md) | Lee/e04: 수질 제약에 DO 를 함께 넣기 (수행계획서 정합) | results.md |
| [Lee/e05](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e05/README.md) | Lee/e05: feasibility 점수를 정책 학습의 보조 신호로 사용 | results.md |
| [Lee/e06](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e06/README.md) | Lee/e06: Kaggle 폐수처리 데이터(Avedøre)에서 feasibility 점수를 정책 학습에 쓰는 확장 | results.md |
| [Lee/e07](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e07/README.md) | Lee/e07: Kaggle 대회 데이터(DARROW)에서 폭기 제어 환경 구축과 원래 FCSRL 기준선 | results.md |
| [Lee/e08](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e08/README.md) | Lee/e08: 라그랑주 승수의 반응 속도를 높이는 확장 (DARROW 환경) | results.md |
| [Lee/e09](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e09/README.md) | Lee/e09: 실현 가능성 점수를 이용한 이점 벌점 (DARROW 환경) | README와 JSON 등 개별 확인 |
| [Lee/e14](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e14/README.md) | Lee/e14: DARROW environment audit and direct state prediction | results.md |
| [Lee/e15](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e15/README.md) | Lee/e15: Risk-stratified FCSRL training after environment audit | results.md |
| [Lee/e16](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e16/README.md) | Lee/e16: Safety calibration and recursive action-response audit | results.md |
| [Lee/e17](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e17/README.md) | Lee/e17: Contractive monotone autoregressive environment | results.md |
| [Lee/e18](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e18/README.md) | Lee/e18: Probabilistic residual innovations for safety frequencies | results.md |
| [Lee/e19](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e19/README.md) | Lee/e19: Temperature trend and predictive-variance continuation | results.md |
| [Lee/e20](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e20/README.md) | Lee/e20: Coupled contractive monotone state feedback | results.md |
| [Lee/e21](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e21/README.md) | Lee/e21: Forecast horizon and logged-action support diagnostic | results.md |
| [Lee/e22](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e22/README.md) | Lee/e22: Multi-step latent state model with causal history | results.md |
| [Lee/e23](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e23/README.md) | Lee/e23: Full-day latent-model training and convergence control | results.md |
| [Lee/e24](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e24/README.md) | Lee/e24: Repair dead NO3 initialization and repeat full-day fitting | results.md |
| [Lee/e25](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e25/README.md) | Lee/e25: Initial-condition anchoring as a persistence control | results.md |
| [Lee/e26](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e26/README.md) | Lee/e26: Long-correlated NH4 innovations and an expanded scale search | results.md |
| [Lee/e27](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e27/README.md) | Lee/e27: Test whether enforced contraction erases NH4 risk memory | results.md |
| [Lee/e28](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e28/README.md) | Lee/e28: Restore within-training seasonal coverage without touching test | results.md |
| [Lee/e29](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e29/README.md) | Lee/e29: Continue optimization past the selected training boundary | results.md |
| [Lee/e30](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e30/README.md) | Lee/e30: Tail-weighted environment fitting, unchanged evaluation | results.md |
| [Lee/e31](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e31/README.md) | Lee/e31: Joint one-step and full-day training | results.md |
| [Lee/e32](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e32/README.md) | Lee/e32: Persistent episode context in state forcing | results.md |
| [Lee/e33](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e33/README.md) | Lee/e33: Kinetic temperature scaling of NH4 retention and removal terms | results.md |
| [Lee/e34](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e34/README.md) | Lee/e34: Joint recursive and teacher-forced fitting with persistent context | results.md |
| [Lee/e35](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e35/README.md) | Lee/e35: Pre-episode innovation assimilation | results.md |
| [Lee/e36](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e36/README.md) | Lee/e36: Align shared bias calibration with the proper selection score | results.md |
| [Lee/e37](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e37/README.md) | Lee/e37: Center pre-episode innovations before adapting the frozen model | results.md |
| [Lee/e38](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e38/README.md) | Lee/e38: Conditional quantile transitions with a calibrated residual copula | results.md |
| [Lee/e39](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e39/README.md) | Lee/e39: Saturating implicit Monod removal for NH4 | results.md |
| [Lee/e40](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e40/README.md) | Lee/e40: Preserve declared monotone bounds after quantile leaf updates | results.md |
| [Lee/e41](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e41/README.md) | Lee/e41: Longer causal operating history | results.md |
| [Lee/e42](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e42/README.md) | Lee/e42: Freeze the validated quantile environment and train paired baselines | README와 JSON 등 개별 확인 |
| [Lee/e43](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e43/README.md) | Lee/e43: Constraint-wise feasibility supervision | README와 JSON 등 개별 확인 |
| [Lee/e44](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e44/README.md) | Lee/e44: Air-only matching and paired policy analysis | README와 JSON 등 개별 확인 |
| [Lee/e45](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e45/README.md) | Severity-aware auxiliary supervision | README와 JSON 등 개별 확인 |
| [Lee/e46](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e46/README.md) | Colder-day exposure with unchanged FCSRL | README와 JSON 등 개별 확인 |
| [Lee/e47](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e47/README.md) | Equal-air PPO diagnostic before another extension | README와 JSON 등 개별 확인 |
| [Lee/e48](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e48/README.md) | Finite-day feasibility supervision | README와 JSON 등 개별 확인 |
| [Lee/e49](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e49/README.md) | Lee/e49: Separate next-step event learnability from recursive drift | results.md |
| [Lee/e50](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e50/README.md) | Lee/e50: Quantile training exposed to accumulated state error | results.md |
| [Lee/e51](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e51/README.md) | Lee/e51: Calibrate PIT against the actual blended transition | results.md, support_results.md |
| [Lee/e52](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e52/README.md) | Lee/e52: Retain calibration tail magnitudes outside quantile support | results.md |
| [Lee/e53](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e53/README.md) | Lee/e53: Training-only probe of recent observations for FCSRL | results.md |
| [Lee/e54](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e54/README.md) | Lee/e54: Causal previous-observation FCSRL and repeated-current control | results.md |
| [Lee/e55](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e55/README.md) | Lee/e55: Learned one-step cost correction of an unchanged FCSRL policy | results.md |
| [Lee/e56](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e56/README.md) | Lee/e56: Four-step cost-rate correction with a one-step regression control | results.md |
| [Lee/e57](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e57/README.md) | Lee/e57: Fixed executed-air smoothing of the unchanged FCSRL policy | results.md |
| [Lee/e58](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e58/README.md) | Lee/e58: Conditional upper DO tail with the original quantile backbone fixed | results.md |
| [Lee/e59](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e59/README.md) | e59: Observed-state versus recursive DO prediction diagnosis | results.md |
| [Lee/e60](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e60/README.md) | e60: Fixed feedback horizon and temporal-noise diagnosis | results.md |
| [Lee/e61](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e61/README.md) | e61: Episode-start-conditioned whole-day PIT sampling | results.md |
| [Lee/e62](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e62/README.md) | e62: Calibration-only decomposition of context noise channels | results.md |
| [Lee/e63](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e63/README.md) | e63: Risk minimization inside the existing hard action trust region | results.md |
| [Lee/e64](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e64/README.md) | e64: Authenticate e63 as the new self-improvement baseline | README와 JSON 등 개별 확인 |
| [Lee/e65](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e65/README.md) | e65: Refine the risk-only action grid | results.md |
| [Lee/e66](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e66/README.md) | e66: Current-exogenous conditional innovation blocks | results.md |
| [Lee/e67](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e67/README.md) | e67: A larger fixed risk-only action radius | results.md |
| [Lee/e68](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e68/README.md) | e68: Fresh policy-seed baseline preparation | results.md |
| [Lee/e69](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e69/README.md) | e69: Training-state air-budget projection | results.md |
| [Lee/e70](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e70/README.md) | e70: Diagnose transport of the air constraint | results.md |
| [Lee/e71](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e71/README.md) | e71: Gate projection on expanded-search risk improvement | fresh_results.md, results.md |
| [Lee/e72](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e72/README.md) | e72: Diagnose component-risk trades after failed e71 replication | results.md |
| [Lee/e73](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e73/README.md) | e73: Componentwise predicted-rate guard for budget projection | results.md |
| [Lee/e74](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e74/README.md) | e74: Extend the authenticated matched reference directly | results.md |
| [Lee/e75](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e75/README.md) | e75: Validate a continuous next-DO predictor before shielding anchored actions | results.md |
| [Lee/e76](https://github.com/UZO1226/Threemool/blob/e077ba3b4684bac96ecf8faf0383d54531676be2/Lee/experiments/e76/README.md) | e76: Persistence plus learned DO change, with unchanged admission conditions | README와 JSON 등 개별 확인 |

### Park (33개 README)

| 실험 | 제목 | 함께 존재하는 결과 문서 |
| --- | --- | --- |
| [Park/e01](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e01/README.md) | Park/e01: FCSRL feasibility 점수로 실행 시점에 위험 행동을 막는 안전 차단 | results.md |
| [Park/e02](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e02/README.md) | Park/e02: 더 먼 미래를 보는 feasibility 점수 | results.md |
| [Park/e03](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e03/README.md) | Park/e03: F 조기 경보의 학습 표본 밖 확인과 단순 예측기 비교 | results.md |
| [Park/e04](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e04/README.md) | Park/e04: 단순 위반 예측기로 거는 안전 차단 | results.md |
| [Park/e05](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e05/README.md) | Park/e05: 위험에 비례해 미리 폭기를 올리는 차단 | results.md |
| [Park/e06](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e06/README.md) | Park/e06: 우기에서 차단 순서가 유지되는가 | results.md |
| [Park/e07](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e07/README.md) | Park/e07: Kaggle Avedøre 데이터에서 실행 시점 안전 차단 | results.md |
| [Park/e08](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e08/README.md) | Park/e08: 팀 공식 데이터(Kaggle DARROW) 에서 실행 시점 안전 차단 | results.md |
| [Park/e09](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e09/README.md) | Park/e09: 학습 중에도 NH4 규칙 차단을 켠 정책 (Kaggle DARROW) | results.md |
| [Park/e10](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e10/README.md) | Park/e10: 필요한 만큼만 공기량을 올리는 최소 개입 차단 (Kaggle DARROW) | results.md |
| [Park/e11](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e11/README.md) | Park/e11: DO 상한도 함께 보는 최소 개입 차단 (Kaggle DARROW) | results.md |
| [Park/e12](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e12/README.md) | Park/e12: 학습 중에도 최소 개입 차단을 켠 정책 (Kaggle DARROW) | results.md |
| [Park/e13](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e13/README.md) | Park/e13: 차단 개입에 학습 비용을 줘서 차단 의존 줄이기 (Kaggle DARROW) | results.md |
| [Park/e14](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e14/README.md) | Park/e14: DO 가 넘칠 때 공기량을 내리기도 하는 양방향 차단 (Kaggle DARROW) | results.md |
| [Park/e15](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e15/README.md) | Park/e15: 예측 불확실성에 맞춘 NH4 여유 (Kaggle DARROW) | results.md |
| [Park/e16](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e16/README.md) | Park/e16: DO 여유와 헛수고 방지를 더한 차단 (Kaggle DARROW) | results.md |
| [Park/e17](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e17/README.md) | Park/e17: 회복 가능성 검사를 더한 차단 (Kaggle DARROW) | results.md |
| [Park/e18](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e18/README.md) | Park/e18: 회복 기준과 DO 여유를 조인 차단 (Kaggle DARROW) | results.md |
| [Park/e19](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e19/README.md) | Park/e19: NH4 여유와 DO 가 부딪칠 때 여유를 줄이는 차단 (Kaggle DARROW) | results.md |
| [Park/e20](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e20/README.md) | Park/e20: 정책 앙상블 + e19 차단 (Kaggle DARROW) | results.md |
| [Park/e21](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e21/README.md) | Park/e21: 같은 학습 예산의 작은 정책 앙상블 (Kaggle DARROW) | results.md |
| [Park/e22](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e22/README.md) | Park/e22: 같은 학습 예산에서 구성원을 줄이고 더 길게 학습한 정책 앙상블 (Kaggle DARROW) | results.md |
| [Park/e23](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e23/README.md) | Park/e23: 새 seed 로 e22 앙상블 재현 (Kaggle DARROW) | results.md |
| [Park/e24](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e24/README.md) | Park/e24: 정책 학습률을 올려 덜 학습된 정책 보완 (Kaggle DARROW) | results.md |
| [Park/e25](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e25/README.md) | Park/e25: 한 학습의 중간 정책을 평균하는 체크포인트 앙상블 (Kaggle DARROW) | results.md |
| [Park/e26](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e26/README.md) | Park/e26: 4 x 16 앙상블 구성원의 학습률 올리기 (Kaggle DARROW) | README와 JSON 등 개별 확인 |
| [Park/e27](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e27/README.md) | Park/e27: 정책 앙상블의 의견 차이에 맞춘 NH4 여유 (Kaggle DARROW) | README와 JSON 등 개별 확인 |
| [Park/e28](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e28/README.md) | Park/e28: 막을 수 없는 NH4 위반 구간의 회복 계획 (Kaggle DARROW) | results.md |
| [Park/e29](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e29/README.md) | Park/e29: 학습률이 다른 구성원을 섞은 4 x 16 앙상블 (Kaggle DARROW) | results.md |
| [Park/e30](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e30/README.md) | Park/e30: 회복 계획의 적용 범위 넓히기 (Kaggle DARROW) | README와 JSON 등 개별 확인 |
| [Park/e31](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e31/README.md) | Park/e31: 최근 예측 오차로 보정한 차단 (Kaggle DARROW) | results.md |
| [Park/e32](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e32/README.md) | Park/e32: 새 seed 로 회복 계획 재현 (Kaggle DARROW) | results.md |
| [Park/e33](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/experiments/e33/README.md) | Park/e33: 남은 비용 진단과 상태에 따른 정책 공기량 조절 (Kaggle DARROW) | results.md |

### Yoo (65개 README)

| 실험 | 제목 | 함께 존재하는 결과 문서 |
| --- | --- | --- |
| [Yoo/e01](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e01/README.md) | Yoo/e01 FCSRL 원본 재현 | README와 JSON 등 개별 확인 |
| [Yoo/e02](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e02/README.md) | Yoo/e02 FCSRL 의 폭기 제어 적용 | results.md |
| [Yoo/e03](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e03/README.md) | Yoo/e03 PPO 비교군, UCB 확장과 승수 상한 점검 | results.md |
| [Yoo/e04](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e04/README.md) | Yoo/e04 학습 전 점검과 오류 분리 진단 | results.md |
| [Yoo/e05](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e05/README.md) | Yoo/e05 PPO 기반 FCSRL 적용 | results.md |
| [Yoo/e06](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e06/README.md) | Yoo/e06 FCSRL + PPO에 하루 비용 CVaR 항을 추가하는 확장 후보 | results.md |
| [Yoo/e07](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e07/README.md) | Yoo/e07 기존 구조를 유지하기 위한 원인 진단 | README와 JSON 등 개별 확인 |
| [Yoo/e08](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e08/README.md) | Yoo/e08 학습한 표현의 중간 고정 대조 | results.md |
| [Yoo/e09](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e09/README.md) | Yoo/e09 조건 차이와 대리 모델 연속 예측 진단 | results.md |
| [Yoo/e10](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e10/README.md) | Yoo/e10 위반 사건 진단과 과거 상태 입력 대조 | results.md |
| [Yoo/e11](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e11/README.md) | Yoo/e11 누락한 유입 및 유출 반응조 입력 대조 | results.md |
| [Yoo/e12](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e12/README.md) | Yoo/e12 위반 누락과 연속 예측 목적의 수정 | results.md |
| [Yoo/e13](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e13/README.md) | Yoo/e13 한 스텝 정확도 및 거짓경보 제한 대조 | results.md |
| [Yoo/e14](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e14/README.md) | Yoo/e14 하루 예측과 조건별 거짓경보 제한 | results.md |
| [Yoo/e15](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e15/README.md) | Yoo/e15 초기 오차 기반 편향 보정의 선행 검사 | results.md |
| [Yoo/e16](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e16/README.md) | Yoo/e16: BSM1을 이용한 별도 검증 환경의 연결 검사 | results.md |
| [Yoo/e17](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e17/README.md) | Yoo/e17: BSM1 평균 수질 제약의 짧은 학습 시험 | results.md |
| [Yoo/e18](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e18/README.md) | Yoo/e18: BSM1 DO 설정값 제어와 같은 출발점 시험 | results.md |
| [Yoo/e19](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e19/README.md) | Yoo/e19: BSM1 3개 조 DO 제어에서 PPO-Lag 와 FCSRL + PPO-Lag 본 실험 | results.md |
| [Yoo/e20](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e20/README.md) | Yoo/e20: 7일 연속 에피소드 학습으로 하루 reset 착취 제거 | results.md |
| [Yoo/e21](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e21/README.md) | Yoo/e21: 설정값 결정 주기 30분과 빠른 lambda 반응 | results.md |
| [Yoo/e22](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e22/README.md) | Yoo/e22: e21 비교 기준 정확도 검증 | results.md |
| [Yoo/e23](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e23/README.md) | Yoo/e23: 수행계획서의 Kaggle 데이터로 본 실험 복귀 | results.md |
| [Yoo/e24](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e24/README.md) | e24 Kaggle FCSRL 회복 지속시간 보조 학습 | results.md |
| [Yoo/e25](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e25/README.md) | e25 Kaggle FCSRL 표현 반영 시 정책과 가치 보정 | results.md |
| [Yoo/e26](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e26/README.md) | e26 e14 하루 길이 MSE의 외부 시간 구간 확인 | results.md |
| [Yoo/e27](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e27/README.md) | e27 거짓양성의 구분 능력 및 시간 전이 진단 | results.md |
| [Yoo/e28](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e28/README.md) | e28 동일 목표 시각의 예측 길이와 시작 상태 대조 | results.md |
| [Yoo/e29](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e29/README.md) | e29 Kaggle FCSRL 출력 보정의 Actor와 Critic 기여 분리 | results.md |
| [Yoo/e30](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e30/README.md) | e30: 사용자 지정 DARROW 데이터에서 FCSRL과 출력 보정 재검증 | results.md |
| [Yoo/e31](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e31/README.md) | e31: DARROW 설계 점검 (공기 배분 정보, 계절 이동, 위반 예측 가능성)과 계절 혼합 분할 시험 | results.md |
| [Yoo/e32](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e32/README.md) | e32: DARROW에서 PPO와 FCSRL의 차이 진단 | results.md |
| [Yoo/e33](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e33/README.md) | e33: NH4와 DO의 평균 제약을 분리하는 DARROW 적용 시험 | results.md |
| [Yoo/e34](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e34/README.md) | e34: 행동, 목적함수와 평가 지표 감사 | results.md |
| [Yoo/e35](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e35/README.md) | e35: 원출처 확인과 DO 직접 폭기 구간 오류 수정 | results.md |
| [Yoo/e36](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e36/README.md) | e36: 기존 정책의 단순 기준 대비 효율과 환경 민감도 | results.md |
| [Yoo/e37](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e37/README.md) | e37: DARROW 공기량 변화 구간의 모델 반응 대조 | results.md |
| [Yoo/e38](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e38/README.md) | e38: DARROW NH4 전이에 직전1시간 수질 이력 추가 | results.md |
| [Yoo/e39](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e39/README.md) | e39: 학습과 분리한 날짜로 DARROW NH4 위험 판정 보정 | results.md |
| [Yoo/e40](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e40/README.md) | e40: DARROW 연속 NH4 점수 판정 대조 | results.md |
| [Yoo/e41](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e41/README.md) | e41: 자기 예측 상태를 학습에 추가하는 NH4 전이 대조 | results.md |
| [Yoo/e42](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e42/README.md) | e42: 정상/위반 조건부 NH4 예측기 분리 | results.md |
| [Yoo/e43](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e43/README.md) | e43: 상태 피드백과 전체 제어 구조 감사 | README와 JSON 등 개별 확인 |
| [Yoo/e44](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e44/README.md) | e44: 전체 설계 검토와 과거 실행 정보 대조 | results.md |
| [Yoo/e45](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e45/README.md) | e45 기존 정책의 동일 조건 평가 | results.md |
| [Yoo/e46](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e46/README.md) | e46 시작 수온 조건별 NH4 제약 | results.md |
| [Yoo/e50](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e50/README.md) | e50: 공통 평가 조건 v1.1 기준 대리 모델에서 Yoo 세 조건 재측정 | results.md |
| [Yoo/e51](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e51/README.md) | e51: 공통 기준 대리 모델의 DO 구간 대응 오류 영향 점검 | results.md |
| [Yoo/e52](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e52/README.md) | e52: 대리 모델 구성원 불일치로 NH4 제약을 비관적으로 세는 학습 | results.md, results.md, results.md, results.md |
| [Yoo/e53](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e53/README.md) | e53: e52 의 대조, 불확실성 없이 NH4 문턱만 고정 여유로 낮춘 학습 | results.md |
| [Yoo/e54](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e54/README.md) | e54: 날짜 묶음을 빼고 학습한 앙상블의 불일치로 NH4 제약에 여유를 두는 학습 | results.md |
| [Yoo/e55](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e55/README.md) | e55: 저온 지원 범위와 NH4 반복 예측 오류 검산 | results.md |
| [Yoo/e56](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e56/README.md) | e56: 고정 모방 prior 사용 여부 대조 | results.md |
| [Yoo/e57](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e57/README.md) | e57: 새 보조망 이전의 고정 표현 진단 | results.md |
| [Yoo/e58](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e58/README.md) | e58: 연속 NH4 초과 크기 보조 표현 학습 | README와 JSON 등 개별 확인 |
| [Yoo/e59](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e59/README.md) | e59: Park 모델 예측 차단을 Yoo 정책에 적용해 정책의 기여를 분리 | results.md |
| [Yoo/e60](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e60/README.md) | e60: 학습 중 최소 개입 차단을 켠 Yoo PPO-Lag / FCSRL | results.md |
| [Yoo/e61](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e61/README.md) | e61: 학습 비용 한도를 묶이는 2.0 으로 낮춘 학습 중 차단 | results.md |
| [Yoo/e62](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e62/README.md) | e62: NH4 바닥 분해와 구간 배분 진단 | results.md |
| [Yoo/e63](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e63/README.md) | e63: 총 공기와 구간 배분을 함께 정하는 정책 | results.md |
| [Yoo/e64](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e64/README.md) | e64: 실행 상한과 연속 날짜 재평가 | results.md |
| [Yoo/e65](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e65/README.md) | e65: e63 배분 정책의 seed 5 ~ 9 재현 | results.md |
| [Yoo/e66](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e66/README.md) | e66: test 41일 최종 평가 | results.md |
| [Yoo/e67](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e67/README.md) | e67: 학습된 배분 대 고정 배분 규칙, 막을 수 없는 위반 진단 | results.md |
| [Yoo/e68](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e68/README.md) | e68: 문헌 기준 제약에서 원본 FCSRL 대 같은 망 PPO-Lag, ABAC | results.md |

## 2. 주요 코드 직접 확인 목록

| 코드 | 확인한 쟁점 |
| --- | --- |
| [Lee/experiments/e63/trust_policy.py](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Lee/experiments/e63/trust_policy.py) | 위험 합, 후보 반경, 벌점 제거, 동점 처리 |
| [Lee/experiments/e63/audit_trust.py](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Lee/experiments/e63/audit_trust.py) | 독립 sklearn 후보 결정과 최종 행동 감사 경로 |
| [Lee/experiments/e44/report.py](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Lee/experiments/e44/report.py) | 5개 정책 seed 짝 bootstrap과 사전 조건 |
| [Park/code/min_brake.py](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/code/min_brake.py) | 후보 rollout, 추가 정보, 회복 계획 비용식 |
| [Park/code/darrow_env.py](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/code/darrow_env.py) | 환경 최종 비용의 float 지시값 합 |
| [Park/code/train_darrow.py](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Park/code/train_darrow.py) | 정책 표본과 차단 실행 행동의 구분 |
| [Yoo/experiments/e68/train68.py](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e68/train68.py) | 원 에이전트 상속과 변경, 표본 행동 평가, 말단 처리 |
| [Yoo/experiments/e68/env68.py](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e68/env68.py) | 누적 NH4 예산, 비용, 실제 실행 구간 공기 합 |
| [Yoo/experiments/e68/baselines.py](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e68/baselines.py) | 축약 ABAC PI와 격자 선택 |
| [Yoo/experiments/e68/protocol.json](https://github.com/UZO1226/Threemool/blob/9fb396ddb58bc65ec2e64b1f4a6ab097a8816830/Yoo/experiments/e68/protocol.json) | 평가 방식, 한도와 holdout 정의의 문서 일치 |

## 3. 문헌 원문과 코드 보관

본문의 문헌 표가 읽은 절과 코드 함수를 지정한다. 아래 해시는 보관 파일 식별용이며, 논문 또는 소프트웨어를 재현 실행했다는 증명이 아니다.

| 자료 | SHA256 |
| --- | --- |
| bsm1.pdf | `68a12464fc9d195f0b0f266a9366d05de10712871c9e0dd6252edad901c27b65` |
| mbpo.pdf | `89d12b6142a285e541f6315a8911db469bea33514f3b80bdbe6e52439197b149` |
| ppg.pdf | `a5d5e56983c74f3007eda154103b4a008e9359aacb745fea5f8ba7f7bde4bd09` |
| rliable.pdf | `a6a5abb49e58a9e4a0be8556c8351ce8c3b52e70216bffc9fa80eac0823ec5ed` |
| safe_control_gym.pdf | `bcfbb42c3ed7d1d4d414835df71edd1bbb83ea530b582736685afa1ea0a4f72a` |
| safety_filter.pdf | `f1604802bf91f0eb672ca04b7772eee536dfccebd4a42072456130ba74fe8f52` |
| abac2026.xml | `f6211276210ba5e8d0b2f5aa507c062b72a00996de6c3157410a0c3c46e3a6e5` |
| fcsrl.pdf | `800f1f403d3bdfe62286c123736679fed4b76b3ddef766d1f40aefedad80e3b2` |
| unreal.pdf | `325d47327e008eb8606ece1017e1812607402cada7ce79b5f0d490ef2c816d96` |

외부 코드 고정 commit은 본문 5.2절에 있다. 원문 PDF와 XML을 Git에 복제하지 않았다. 다운로드한 텍스트는 원문 확인을 위한 추출물이며 도표의 모든 숫자를 수작업 검산한 것은 아니다.

## 4. 직접 재현한 작은 수식 확인

환경: `/home/jio/fcsrl_src/.venv-skl191/bin/python`, NumPy 1.26.4.

```python
import numpy as np
nh4 = np.array([False, True, False, True])
do = np.array([False, False, True, True])
print(nh4 + do)  # [False, True, True, True], dtype bool
print(nh4.astype(int) + do.astype(int))  # [0, 1, 1, 2]
```

Park 회복 계획의 식을 확인한 것이며 기존 모든 rollout에서 행동이 얼마나 달라지는지 측정하지 않았다. 최종 환경 비용은 float 변환 후 더하므로 별도 문제다. Park/e28 반영 커밋 `391039a`에도 동일 불리언 식이 있는 것을 확인했다. 원 사전 커밋 `2a0f542` 객체는 로컬에서 조회되지 않았으며 그 객체까지 확인했다고 주장하지 않는다.

## 5. 점수 산술 검산

| 대상 | 여섯 점수 | 합계 | 평균 |
| --- | --- | ---: | ---: |
| Lee | 5, 6.5, 7, 6, 5, 8 | 37.5 | 6.25 |
| Park | 5.5, 7, 5.5, 6.5, 7, 7.5 | 39.0 | 6.50 |
| Yoo | 5, 6.5, 6, 6.5, 5.5, 7.5 | 37.0 | 6.17 |

점수는 근거를 참고한 검토자의 정성 판단이며 통계적 추정량, 공식 교수 성적 또는 게재 확률이 아니다.

## 6. 서버에만 있는 e58 최종 결과

`/home/jio/fcsrl_src/results/e58/main/results.md`를 직접 읽었다. baseline/severity/detached의 5 seed 결과와 미채택 판정을 본문에 반영했다. 저장소 e58 README의 과거 실행 중 문장보다 이 완료 결과를 우선했다.

결과 파일 SHA256: `b16fb6e99dd90edf06cfb0d1e308788456db972cec861555f465f5eed16ed724`.

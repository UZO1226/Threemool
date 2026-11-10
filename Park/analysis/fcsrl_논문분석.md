# FCSRL 논문 분석 노트

## 1. 서지 정보

| 항목 | 내용 |
| --- | --- |
| 제목 | Feasibility Consistent Representation Learning for Safe Reinforcement Learning |
| 저자 | Zhepeng Cen, Yihang Yao, Zuxin Liu, Ding Zhao (Carnegie Mellon University) |
| 발표 | ICML 2024, PMLR 235 |
| 프로젝트 | https://sites.google.com/view/FCSRL |
| 코드 | https://github.com/czp16/FCSRL |

## 2. 한 줄 요약

안전 RL 에서 비용(제약) 추정이 어려운 문제를, 표현 학습에 "앞으로 위반이 한 번이라도 일어날지"를 나타내는 feasibility score 예측을 보조 과제로 넣어 해결한다. 기존 model free 안전 RL 알고리즘(PPO-Lag, TD3-Lag)에 그대로 붙일 수 있다.

## 3. 문제 정의

- 안전 RL 은 CMDP 로 정의한다. 누적 reward 를 최대화하되 누적 비용 기댓값이 한도 ε 이하가 되어야 한다 (식 1). 보통 Lagrangian 승수 λ 를 두고 primal dual 로 푼다 (식 2).
- 이 논문은 비용이 0 또는 1 인 binary cost 를 가정한다.
- 저자가 꼽는 핵심 어려움은 비용 추정이다.
  1. 원시 상태가 복잡해 한 스텝 비용 예측부터 어렵다.
  2. 비용 신호가 드물어(sparse) 비용 가치함수가 매끄럽지 않고 추정 잡음이 크다.
- 비용을 과대추정하면 정책이 지나치게 보수적이 되고, 과소추정하면 위반이 늘어 λ 가 커지면서 reward 가 떨어진다.

## 4. 방법

인코더 g 가 상태 s 를 표현 z 로 바꾸고, 정책과 가치함수는 이 표현을 입력으로 받는다. 표현 학습 손실은 두 항의 합이다.

### 4.1 전이 동역학 일관성 (dynamics loss)

- 전이 모델 h 가 (z_t, a_t) 로 다음 표현 z_{t+1} 을 예측한다.
- 목표값은 학습하지 않는 target encoder g^(m) (g 의 EMA) 가 만든 z^(m)_{t+1} 이다. 표현 붕괴를 막기 위한 장치다.
- 거리 함수는 SimSiam 방식의 음의 cosine 유사도이고, 투영망 p1, p2 (2층 MLP) 와 stop gradient 를 쓴다 (부록 B.1, 식 26).
- 길이 K 의 부분 궤적으로 확장하면 K 스텝을 연속으로 예측해 손실을 더한다.
- 목적은 동역학을 정확히 예측하는 model based RL 이 아니라, 상태 구조를 표현 공간에 담는 것이다.

### 4.2 Feasibility 일관성 (feasibility loss)

- **Feasibility score 정의 (식 4):** F(s, a) = E[ max_t γ^t c(s_t, a_t) ]. 미래 궤적에서 가장 큰 할인 비용이다. 누적합을 쓰는 비용 가치함수 V_c 와 달리 최댓값을 쓴다.
- **Bellman 식 (식 5):** F = (1 − γ) c + γ max{ c, E[F(s', a')] }. 기존 HJ reachability 연구와 달리 다음 행동에 대해 max 가 아니라 기댓값을 쓴다.
- **명제 4.1:** 비용이 binary 이고 γ → 1 이면 1 − F(s, a) 는 이후 궤적이 한 번도 위반하지 않을 확률이다. 즉 F 는 안전 정도의 지표다.
- **명제 4.3:** 한 궤적 위에서 F 의 추정값은 V_c 의 추정값보다 시간적으로 더 매끄럽다 (인접 스텝 간 변화량 L 이 작다). 그래서 표현 학습의 지도 신호로 쓰기 쉽다.
- **손실 (식 7, 8):** feasibility head f 가 z 로 F 를 예측하고, 목표는 bootstrap 추정 F^(m)(s_t) = max{ c_t, γ f(z^(m)_{t+1}) } 이다. 같은 상태라도 행동에 따라 F 가 분포를 이루므로, 출력 공간을 63개 구간으로 나눈 분포 회귀(discrete regression)로 예측하고 KL divergence 로 학습한다.

### 4.3 전체 목적과 학습 절차

- 표현 손실: L = E[ L_dyn + λ_fea L_fea ] (식 9), λ_fea = 2.0.
- 표현 갱신이 actor, critic 입력에 잡음을 주지 않도록 정책과 가치함수는 target encoder 의 표현 z^(m) 을 입력으로 받는다.
- 알고리즘 1 순서: 경험 수집, 길이 K 부분 궤적 샘플, 표현 손실로 g, h, f 갱신, g^(m) EMA 갱신, z^(m) 으로 critic 과 정책 갱신, Lagrangian 승수 갱신.
- On policy (PPO-Lag) 에서는 bootstrap 대신 Monte Carlo 추정을 쓴다. V_c(s_i) = Σ_{t≥i} γ^t c_t, F(s_i) = max_{i≤t≤T} γ^t c_t.
- 벡터 상태 과제에서는 SALE 처럼 원시 상태와 표현을 이어 붙여(concat) 정책과 가치함수에 넣는다.

### 4.4 Value consistency 와의 차이

기존 표현 학습은 표현으로 가치함수를 예측하게 하는 value consistency 를 쓰기도 한다. 안전 RL 에서는 비용이 드물어 V_c 가 거칠고 예측이 어렵다. FCSRL 은 더 매끄러운 F 를 예측 대상으로 바꾼 것이 핵심 차이다.

## 5. 실험

### 5.1 설정

- 과제: Safety Gymnasium 벡터 상태 6개 (PointGoal1, PointButton1, PointPush1, PointGoal2, CarGoal1, CarButton1), 이미지 3개 (PointButton1Vision, PointGoal2Vision, CarGoal1Vision).
- 기반 알고리즘: PPO-Lag (ε = 25), TD3-Lag (ε = 10). Lagrangian 승수는 PID 로 갱신한다 [Kp, Ki, Kd] = [0.02, 0.005, 0.01].
- 학습량 2M 환경 스텝, seed 5개.
- 비교 표현 학습: Raw state, Forward raw, Forward latent, Inverse, TCL, SALE, VC (value consistent). 이미지 과제는 CURL, SPR, VC.
- 지표: 정규화 reward NR = (R − R_l)/(R_h − R_l) (R_l 은 무작위 정책, R_h 는 제약 없는 PPO), 정규화 비용 NC = C/ε.

### 5.2 주요 하이퍼파라미터 (표 6)

| 항목 | 값 |
| --- | --- |
| actor, critic, 전이 h, 인코더 g 은닉층 | [256, 256] |
| 이미지 인코더 | 4층 CNN |
| 최적화, 학습률 | Adam, 3e-4 |
| 분포 회귀 구간 수 | 63 |
| γ | 0.99 |
| 예측 길이 K | 4 |
| 표현 차원 | 64 (벡터), 128 (이미지) |
| target soft update 계수 | 0.05 |
| λ_fea | 2.0 |

### 5.3 결과

- **벡터 상태 (그림 3):** 대부분 방법이 비용 한도 근처로 수렴하고, FCSRL 이 대부분 과제에서 가장 높은 reward 를 낸다. TD3-Lag 기반이 PPO-Lag 기반보다 전반적으로 낫다. 제약을 만족하기 어려운 과제(PointButton1, PointGoal2, CarButton1)에서 VC 대비 격차가 더 크다.
- **이미지 (그림 4):** CURL 은 지나치게 보수적이고, SPR 은 일부 과제에서 비용 한도를 넘는다. VC 와 FCSRL 이 reward 와 비용 모두 좋고 FCSRL 이 더 높다.
- **비용 한도 변화 (그림 5):** ε ∈ {5, 10, 20, 30, 40}. ε > 30 에서는 SALE, VC 와 비슷하지만 한도가 엄격할수록 FCSRL 의 이점이 커진다.
- **다른 안전 RL 과 비교 (표 4, TD3-Lag, ε = 10):** CPO, PPO-Lag, FOCOPS, CVPO 대비 6개 과제 모두에서 FCSRL 의 reward 가 가장 높고 비용도 한도 근처다.

| 과제 | FCSRL reward | FCSRL cost |
| --- | --- | --- |
| PointGoal1 | 24.4 ± 1.4 | 9.3 ± 5.8 |
| PointButton1 | 10.5 ± 2.3 | 10.1 ± 6.3 |
| PointPush1 | 11.8 ± 4.5 | 9.7 ± 8.1 |
| PointGoal2 | 13.5 ± 2.6 | 9.7 ± 7.8 |
| CarGoal1 | 27.6 ± 2.0 | 9.9 ± 6.4 |
| CarButton1 | 3.6 ± 1.3 | 10.4 ± 7.3 |

### 5.4 Ablation

- **구성 요소 (표 1, TD3-Lag, ε = 10):** feasibility loss 를 빼면 reward 가 크게 떨어진다 (PointGoal2 12.7 → 7.1). dynamics loss 를 빼는 영향은 작다 (12.7 → 10.3). 표현 z 만 입력하면 가장 크게 떨어진다 (5.9). 원시 상태에 reward 관련 정보가 남아 있기 때문으로 해석한다.
- **예측 길이 (표 2, PointPush1):** K 가 커질수록 대체로 좋아지지만 K ≥ 4 부터는 이득이 작다.
- **표현 품질 (부록 B.3, 표 3):** 고정된 궤적에서 표현으로 선형 회귀를 하면, 비용 가치 MSE 는 VC 와 비슷하고 (17.6 대 17.3) feasibility MSE 는 FCSRL 이 더 작다 (0.139 대 0.217).

## 6. 한계와 의문점

- binary cost 를 전제로 한다. 비용이 연속값이면 명제 4.1 의 확률 해석이 성립하지 않고, F 의 목표식에서 (1 − γ)c 항도 생략할 수 없다 (각주 1).
- 비용 표준편차가 크다 (표 4 에서 cost ± 6 ~ 8). 평균은 한도 근처여도 seed 별, 에피소드별 위반 편차가 크다.
- 제약은 기댓값 제약이다. 상태별(state-wise) 안전 보장은 저자도 후속 과제로 남겼다.
- 실험이 모두 시뮬레이션 로봇 과제이며 실제 공정이나 데이터 기반 대리 모델 환경은 다루지 않는다.
- 이득이 표현 학습 자체에서 오는지, 원시 상태와 concat 하는 구조에서 오는지는 SALE, VC 와의 비교로만 간접 확인한다.

## 7. 폐수처리 폭기 제어에 적용할 때 검토할 점

- **비용 정의:** NH4, DO 수질 제약을 binary 위반(한도 초과 여부)으로 둘지, 초과량 같은 연속값으로 둘지 정해야 한다. 연속값이면 feasibility score 의 의미와 목표식이 달라진다.
- **비용 희소성:** 위반이 드문 운전 구간일수록 논문이 주장하는 이점(매끄러운 F)이 나타날 가능성이 있다. 반대로 위반이 잦으면 F 가 대부분 1 에 붙어 정보가 적을 수 있다.
- **γ 와 에피소드 길이:** 명제 4.1 은 γ → 1 을 가정한다. 하루 단위 에피소드의 스텝 수와 γ = 0.99 의 유효 시야를 맞춰 봐야 한다.
- **기반 알고리즘:** 논문은 TD3-Lag 가 PPO-Lag 보다 좋았지만, 우리 환경에서 어느 쪽이 나은지는 별도 실험으로 확인해야 한다. On policy 에서는 Monte Carlo 목표를 쓴다는 점도 구현에서 확인한다.
- **평가 지표:** 논문은 reward 와 누적 비용만 본다. 우리 비교 지표(에너지 소비량, 수질 제약 위반률, reward)에 맞추려면 위반률과 에너지를 따로 집계해야 한다 (`Compare/PROTOCOL.md` 기준).
- **비교군:** 최소한 같은 기반 알고리즘의 raw state 입력, 표현 고정 대조를 두어야 표현 학습의 효과를 분리할 수 있다.

## 8. 참고 문헌 정보 (인용 시 `Paper/refs/park.md` 에 옮길 것)

Z. Cen, Y. Yao, Z. Liu, and D. Zhao, "Feasibility Consistent Representation Learning for Safe Reinforcement Learning," in *Proc. 41st Int. Conf. Machine Learning (ICML)*, PMLR vol. 235, Vienna, Austria, 2024.

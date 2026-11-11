# FCSRL 적용과 확장 범위를 유지하는 후속 연구 검토

조사일: 2026-11-11. 코드와 결과 기준: main b795a660ec1da54bbe5c540a5fd22bddfb99bd3d. 이번 작업은 문헌 및 설계 검토이며 새 학습이나 정책 평가를 실행하지 않았다.

## 1. 판단

다음 연구 질문은 **두 수질 제약을 각각 보존하는 feasibility 표현이 기존 FCSRL보다 안정적인 폭기 제어를 만드는가**로 좁히는 것이 적절하다. 공유 Encoder, dynamics consistency, feasibility consistency, EMA, PPO와 PID를 유지하면서 feasibility 감독 신호만 분리하는 후보다. 아직 성능 개선이나 신규성을 입증한 방법이 아니다.

단, 바로 본 학습을 시작할 근거는 부족하다. 먼저 기존 체크포인트의 학습 날짜와 개발 날짜를 같은 평가 방식으로 비교해 제약 학습 자체의 실패와 평가 조건 차이를 구분한다. 이 점검 한 번으로 후속 실험의 진행 또는 보류를 결정한다. 정보 추가, Actor 보정, 새 최적화기를 연속해서 누적하지 않는다.

## 2. 현재 확인된 문제

근거: [e44 결과](../experiments/e44/results.md), [기계 판독 결과](../experiments/e44/comparison.json), [완료 기록](../experiments/e44/completion.json).

| 관찰 | 해석할 수 있는 범위 |
| --- | --- |
| 12개 정책 모두 개발 평가 NH4 평균 비용이 한도 6 초과 | 현재 어떤 후보도 두 평균 한도 6/6을 만족하지 못함. 가장 가까운 6.010도 성공 아님 |
| 모든 정책의 DO 평균 비용은 6 미만 | NH4와 DO를 같은 정도의 미해결 문제로 취급하면 안 됨 |
| 직전 zone 명령을 추가한 FCSRL은 평균 세 지표가 개선되지만 seed별 NH4 변화 방향이 반대 | 입력 보완의 효과를 부정할 수도, 안정적인 개선으로 채택할 수도 없음 |
| 명령 정보가 있는 FCSRL에 Actor 보정을 더하면 NH4가 두 seed 모두 악화 | 독립적으로 관찰한 개선 효과를 단순히 더할 수 없음. 작동 기전은 미확정 |
| NH4 승수의 학습 중 최대가 0.645156, 상한은 20 | 과거 e03의 승수 상한 포화 설명을 현재에 재사용할 수 없음 |
| 학습은 무작위 날짜, 한 환경 모델, 확률적 행동. 개발 평가는 고정 40일, 모델 5개, 결정적 행동 | 마지막 학습 비용과 개발 비용 차이를 곧바로 일반화 실패라고 해석할 수 없음 |

공기량은 m³/23.75h이며 전력량 kWh가 아니다. 현재 학습 문제는 수질 제약을 둔 공기량 최소화이고, 사용자가 요구한 세 지표 동시 감소는 더 엄격한 채택 조건이다. 모든 지표의 동시 개선 가능성을 문헌만으로 보장할 수 없다.

## 3. 최근 문헌

아래 표는 확인한 arXiv 최초 공개일 기준 내림차순이다. 최초 공개일과 학회 발표일을 혼동하지 않는다. 최신 문헌이라는 이유만으로 직접 도입하지 않는다.

| 최초 공개 | 논문 및 확인한 상태 | 우리 연구에 참고할 점 | 도입 범위 판단 |
| --- | --- | --- | --- |
| 2026-11-07 | [Safe Offline Reinforcement Learning using Behavior Regularisation and Latent Feasibility Guidance, FOCAL](https://arxiv.org/abs/2611.10677). arXiv에 NeurIPS 2026 채택 표기 | 추정상 feasible한 상태에서는 보상을, infeasible한 상태에서는 제약 회복을 우선하는 업데이트 | 고정 데이터 기반 offline RL과 CVAE 정책이다. 우리 PPO를 그대로 대체하는 것은 이번 확장 범위에서 제외. 저자 구현은 확인하지 못함 |
| 2026-11-05 | [Accelerating Safe Reinforcement Learning with Massive Parallelism](https://arxiv.org/abs/2611.08273). arXiv에 NeurIPS 2026 채택 표기 | 짧은 수집 구간과 전체 에피소드 비용의 불일치를 분석 | 현재 수집은 760 step, 95 step 에피소드 8개이고 PID는 완전한 에피소드 비용을 집계한다. 논문의 짧은 구간 편향을 현재 원인으로 단정하거나 보정식을 추가하지 않음 |
| 2026-10-25 | [CORMA: Constraint-Routed Decision Mamba for Generalizable Multi-Constraint Offline Safe Reinforcement Learning](https://arxiv.org/abs/2610.34208). preprint, 특정 학회 채택 미확인 | 서로 다른 제약을 구분하는 전문가와 결합 구조로 다중 제약을 다룸 | 제약 정체성을 구분한다는 설계 단서만 참고. Mamba, router, 전문가 전체를 도입하지 않음. FCSRL feasibility head 분리를 검증한 논문은 아님 |
| 2026-06-12 | [CSPO: Constraint-Sensitive Policy Optimization for Safe Reinforcement Learning](https://arxiv.org/abs/2606.14415). [ICML 2026 공식 논문집](https://proceedings.mlr.press/v306/belouadah26a.html) 확인 | 비용 경계의 국소 민감도로 지연된 승수 갱신을 보완 | [저자 코드](https://github.com/serval-uni-lu/CSPO)의 PPO 기반 구현 확인. 현재의 실패가 제약 회복 지연인지 먼저 확인해야 함. 전체 도입은 FCSRL과 기존 최적화기의 결합 비교로 명시하며 우리의 독창적 기법으로 주장하지 않음 |
| 2026-01-28 | [Safety Generalization Under Distribution Shift in Safe Reinforcement Learning: A Diabetes Testbed](https://arxiv.org/abs/2601.21094). [ICML 2026 공식 논문집](https://proceedings.mlr.press/v306/kwon26c.html) 확인 | 학습 중 제약 만족과 새로운 조건에서의 제약 만족을 구분 | 동일 평가 조건에서 train/development 차이를 먼저 측정할 근거. 다른 공정의 shielding 성과나 보장을 우리 부정확한 대리 모델에 그대로 적용하지 않음 |

업데이트 보정과 관련해서는 [SafeMPO, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/1fa0c4e5a7e189729230d018b229abc7-Abstract-Conference.html)와 [Embedding Safety into RL: A New Take on Trust Region Methods, C-TRPO, ICML 2025](https://proceedings.mlr.press/v267/milosevic25a.html)도 검토했다. 전자는 점진적인 제약 회복, 후자는 비용을 반영한 trust region을 다룬다. 정책이 이전 출력에 가깝다는 조건만으로 안전한 갱신을 보장할 수 없다는 점이 우리 Actor 보정 검토와 연결된다. 두 알고리즘 전체를 이식하지 않고, 이 연결 자체도 우리 보정법의 성능 증명과 구분한다. [C-TRPO 저자 코드](https://github.com/milosen/ctrpo)는 공개되어 있다.

직접적인 적용 사례로 [Optimal distributed energy systems scheduling using a feasibility consistent safe deep reinforcement learning mechanism](https://doi.org/10.1016/j.meaene.2026.100110), Measurement: Energy 11, 2026년 9월호의 출판사 초록도 확인했다. 전체 본문 접근이 제한되어 세부 구현이나 우리 방법과의 차별성은 판단하지 않았다.

검색에 나온 [Efficient safe reinforcement learning for batch processes with multi-head critics](https://doi.org/10.1016/j.ces.2026.124608)는 2027년 1월호 배정이며 정확한 온라인 출판일은 확인하지 못했다. 따라서 2026년 최신 공개일 표에 넣지 않았다. 초록의 다중 critic은 FCSRL의 feasibility head 분리와 같은 방법이 아니다.

## 4. 원 FCSRL에서 보존할 부분

[원 논문](https://proceedings.mlr.press/v235/cen24b.html), [본문](https://arxiv.org/html/2405.11718v2), [원 구현](https://github.com/czp16/FCSRL)을 기준으로 한다. 사용한 원 구현 커밋은 d882b91054d52c7a63a18a21f7db67bbbed82b63이며 Apache 2.0이다. PPO 기반 FCSRL은 원 논문에 이미 있으므로 PPO 사용 자체를 확장 기여로 주장하지 않는다.

| 부분 | 다음 비교에서의 방침 |
| --- | --- |
| 공유 상태 Encoder와 행동을 받는 잠재 전이 모델 | 유지 |
| dynamics consistency 손실 | 유지 |
| feasibility의 할인된 max 재귀 목표와 분포 예측 손실 | 유지, 제약별 감독 신호만 후보로 분리 |
| EMA target Encoder와 raw state + embedding 입력 | 유지 |
| PPO, 두 비용 critic과 PID, 보상, 한도 6/6 | 현재 적용 기준으로 고정 |
| DARROW 분할, e35 환경, 총 공기량 행동과 기록된 zone 분배 | 고정 |
| Actor 보정과 직전 명령 입력 | 첫 분리 실험에서는 사용하지 않음. e44 padded FCSRL을 고정 기준으로 사용 |

현재 적용 기준 자체가 원 논문 그대로라는 뜻은 아니다. 환경 적용, 두 제약, 초기 정책, 정규화 등의 누적 차이는 이미 존재한다. 원본 재현 e01, DARROW 적용 기준, 이번 추가 변경을 별도로 보고한다. 비교 중 다른 변경을 더하면 같은 실험의 후속 튜닝으로 취급하지 않고 기준과 연구 질문을 다시 명시한다.

원 논문은 다단계 dynamics 학습을 설명하지만 설치된 원 PPO 구현의 `train_encoder`는 한 단계 전이를 사용한다. 현재 [ppo_fcsrl.py](../code/wwtp/ppo_fcsrl.py)는 이 원 메서드를 호출한다. 이것을 우리만의 변경이나 새로 발견한 수정 필요 버그로 단정하지 않으며 이번 후보에 rollout 길이 변경을 섞지 않는다.

## 5. 코드에서 발견한 두 후보

### 우선 후보: 제약별 feasibility 감독

[e44/train.py](../experiments/e44/train.py)의 `collect`는 NH4와 DO의 advantage, critic, PID 비용을 분리한다. 그러나 `feasibility_targets`에는 두 비용의 합인 `b['cost']`를 전달하고, 표현망은 하나의 feasibility head를 사용한다.

따라서 즉시 비용 `(NH4=1, DO=0)`과 `(NH4=0, DO=1)`은 동일한 합계 감독을 제공한다. 두 제약의 원인을 구분하는 표현 학습에는 부족할 가능성이 있다. 다만 raw state와 dynamics, 비용 critic은 구분 정보를 가질 수 있으므로 전체 정책에서 정보가 완전히 사라진다는 뜻은 아니다. 원래 합계 비용 문제에는 타당한 설계이며 코드 버그로 단정하지 않는다.

제안은 공유 표현을 유지하고 최종 feasibility head만 두 개로 만드는 것이다. 각 제약 k에 대해 기존과 같은 재귀를 사용한다.

```text
F_k(t) = max(c_k(t), gamma_f * (1 - terminated_t) * F_k(t+1))
L_encoder = 0.5 * L_dynamics + 0.5 * (L_feasibility_NH4 + L_feasibility_DO)
```

실제 구현 시 기존의 truncation bootstrap과 reset 경계를 그대로 보존한다. 원 손실 `0.5 * L_dynamics + 1.0 * L_feasibility` 대비 feasibility 총 가중치를 맞추고, bins, gamma_f, EMA tau도 고정한다. 가중치 평균이 gradient 크기까지 동일하게 만든다는 보장은 없으므로 gradient norm도 기록한다.

이는 문헌의 다중 제약 설계와 현재 코드에서 도출한 **우리의 검증 가설**이다. CORMA나 multi-head critic 논문이 이 정확한 변경의 효과를 입증한 것은 아니다. 즉시 사용할 완성된 FCSRL용 구현이나 신규성은 이번 조사에서 확인하지 못했다.

### 별도 후보: Actor 보정의 비용 확인

[e29/ablation.py](../experiments/e29/ablation.py)의 Actor 보정은 이전 출력과의 오차 감소를 수락 기준으로 삼는다. NH4와 DO의 정책 비용 증가를 직접 검사하지 않는다. 원래 teacher 정책도 제약을 위반하면 출력 보존만으로 안전하게 만들 수 없다.

후속 원인 진단에서 EMA와 보정 사이의 비용 악화가 확인되는 경우에만, 기존 보정의 수락 기준에 비용 검사를 넣는 별도 후보를 고려한다. 기존 cost advantage와 행동 확률비로 비용 변화의 surrogate를 계산하거나, 같은 상태와 환경 모델에서 짧은 대조 rollout을 사용한다. 현재 critic은 V(s)이므로 행동을 바꿔 같은 V(s)를 읽는 것을 행동별 비용 검사로 쓰면 안 된다.

이 후보는 SafeMPO 또는 C-TRPO를 구현했다는 뜻이 아니며 실제 안전 보장도 아니다. 비용 추정 오차와 모든 보정이 거절되는 경우를 기록해야 한다. 위 feasibility 분리와 동시에 넣지 않는다. e44의 최종 체크포인트만으로 과거 EMA 직전과 직후의 인과 효과를 복원할 수는 없다.

## 6. 실행 전 한 번의 판별과 후속 실험

1. **기존 정책 동일 조건 평가.** 날짜 목록을 결과 확인 전에 고정하고 train 날짜와 개발 val 날짜를 각각 모델 5개, 같은 초기화 방식, 같은 비용 및 공기 집계로 평가한다. 결정적 행동과 고정 난수의 확률적 행동을 별도로 측정한다. 가중치 갱신은 없다. 반복 사용한 val은 개발 자료이며 test 41일은 계속 닫아 둔다.
2. **원인에 따른 진행 판단.** train에서도 NH4를 못 지키면 제약 학습과 표현의 분리를 먼저 시험한다. train만 통과하면 분포 차이와 상태 지원 범위를 먼저 보고하고, 표현 head 분리를 일반화 해결책으로 주장하지 않는다. 확률적 행동과 결정적 행동의 차이가 크면 실행 방식의 효과를 우선 분리한다. 원인을 다중으로 발견해도 변경을 동시에 적용하지 않는다.
3. **표현 후보를 진행한다면 세 조건만 비교.** 기존 합계 head 하나, 같은 합계 목표를 배우는 head 두 개(용량 대조), NH4/DO별 head 두 개. 동일 seed, 학습량, 데이터, 초기 공통 가중치, 평가 조건을 맞춘다. 추가 head 초기 난수 때문에 Actor 초기값까지 달라지지 않도록 별도 난수 또는 초기 가중치 복사를 사용한다.
4. **탐색과 확증 분리.** 작은 동일 seed 탐색에서 기존의 세 지표, 두 한도, 초과 적분, 최악 날짜 기준을 유지한다. 통과 후보만 추가 seed와 계산 비용을 검증한다. rollout을 독립 학습 seed처럼 세어 유의성을 부풀리지 않는다. 개발 결과로 선택을 끝낸 뒤 별도 확증 계획을 정하며 지금 test를 열지 않는다.

공기 감소만으로 성공 처리하지 않는다. NH4 평균, DO 평균, NH4/DO 초과 적분과 최악 날짜, seed별 방향, 계산 시간, 모델 구성원별 차이를 함께 보고한다. 단기 예측 head의 정확도 개선만으로 제어 성능이 좋아졌다고 주장하지 않는다.

## 7. 주장 범위와 보류 항목

새 world model, Mamba/CVAE 정책, reward 재정의, 다른 데이터로의 전환은 이 후보에 포함하지 않는다. 제약별 feasibility가 실패하면 실패 결과를 유지하고 가설을 수정한다. 실패한 후보에 다른 모듈을 계속 붙여 처음의 FCSRL 확장이라고 부르지 않는다.

원 표현 학습의 역할과 추가 변경의 역할을 분리해 보여야 FCSRL 확장이라는 설명이 설득력을 갖는다. 구조를 유지한다는 사실만으로 학술적 신규성이나 개선 효과가 확보되지는 않는다.

현재 환경의 재귀 예측 오차와 기록 행동 밖의 반사실적 정확도는 여전히 제한이다. 같은 대리 환경에서 두 알고리즘을 비교하는 것은 가능하지만 실제 처리장의 절전이나 수질 안전을 입증하지 않는다. 이번 연구 결과는 검증한 범위에서 **DARROW 대리 환경 내부의 FCSRL 적용 및 제한적 확장 비교**로 기술한다.

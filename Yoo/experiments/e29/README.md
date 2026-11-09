# e29 Kaggle FCSRL 출력 보정의 Actor와 Critic 기여 분리

2026-11-09. e25 후속이며 결과를 보기 전에 아래 비교와 해석 기준을 고정한다. 원본 dynamics/feasibility 표현 학습, Kaggle Avedore 데이터, 기존 대리 환경을 유지한다. e26~e28 환경 후보와 결합하지 않는다.

## 질문과 선행 방법 비교

e25의 비용 안정화가 Actor 보정, 보상 및 비용 Critic 보정, 두 보정의 결합 중 어디에서 나왔는지 비교한다. 추가 최적화 자체의 영향도 별도 대조한다. e07의 표현 교체만으로 표현 자체가 개선됐거나 특정 head가 원인이라고 확정하지 않는다.

2026-11-09 논문 및 공식 구현 확인:

| 방법 | 이미 있는 아이디어 | e25와 구분할 점 |
| --- | --- | --- |
| [Phasic Policy Gradient, ICML 2021](https://proceedings.mlr.press/v139/cobbe21a.html), [공식 코드](https://github.com/openai/phasic-policy-gradient/blob/master/phasic_policy_gradient/ppg.py) | 정책과 보조 학습 단계를 나누고, 보조 학습의 정책 변화를 KL로 억제한다 | e25는 FCSRL 고정 encoder의 EMA 반영 직후, 표현을 더 바꾸지 않고 actor와 두 가치망을 직전 출력에 맞춘다. 보존 목적 자체를 새로운 아이디어라고 주장하지 않는다 |
| [ITER, ICLR 2021](https://openreview.net/pdf/ea444807010b334cd2b90645f1cfa31bd38f3ef7.pdf), [공식 코드](https://github.com/maximilianigl/rl-iter) | 정책과 가치 출력을 새로 초기화한 student에 증류해 비정상성의 영향을 줄인다 | e25는 모델 전체 재초기화나 교체 없이 기존 head를 수정하며, 안전 비용 가치도 대상으로 한다 |
| [원본 FCSRL, ICML 2024](https://arxiv.org/abs/2405.11718), [공식 코드](https://github.com/czp16/FCSRL/tree/d882b91054d52c7a63a18a21f7db67bbbed82b63) | dynamics와 feasibility 표현 학습, PPO 또는 TD3 기반 적용 | 원본 표현 손실과 EMA를 유지하며 출력 보정만 추가한다. PPO를 사용한 사실은 새 기여가 아니다 |

이 검토는 가까운 선행 방법과의 차이를 좁히는 작업이며 완전한 신규성 입증이나 문헌 전체 조사 완료가 아니다. 이번 대조는 PPG/ITER 전체 알고리즘과의 성능 비교도 아니다.

## 고정 비교

| 이름 | EMA 뒤 추가 작업 | 실행 |
| --- | --- | --- |
| fcsrl | 없음 | e25에 연결된 e24의 5개 재사용 |
| fcsrl_actor | actor 출력만 직전 teacher에 보정 | 신규 5개 |
| fcsrl_critic | 보상과 비용 critic 출력만 보정 | 신규 5개 |
| fcsrl_preserve | actor와 두 critic 출력 보정 | e25의 5개 재사용 |
| fcsrl_extra_ppo | 기존 rollout의 clipped PPO와 두 return 회귀 손실로 head 추가 학습 | 신규 5개 |

seed는 0~4로 고정한다. 신규 15개, 기존 10개다. 모두 202,048전이(22 rollout) 뒤 마지막 정책을 주 평가한다. 결과에 따라 seed를 추가하거나 가중치를 바꾸지 않는다. 새 seed 일반화나 유의성 확보를 위한 순차 표본 추가가 아니다.

### 보정 분리

e25와 같은 독립 RNG 25000으로 매 rollout 관측의 fit 2048개와 check 1024개를 고른다. 기존 PPO 갱신이 끝난 뒤 EMA 직전 출력을 저장하고 EMA 뒤 detach한 표현에서 head만 바꾼다. Actor-only는 actor KL만, Critic-only는 두 표준화 가치 MSE만 목적과 수용 점수에 넣는다. 해당하지 않는 head는 바꾸지 않는다.

선택한 항 각각이 초기 check 오차보다 1% 넘게 악화하지 않는 후보 중 합계가 최소인 것을 저장하며, 합계 20% 이상 개선해야 반영한다. full batch 50회, actor lr 3e-4, critic lr 1e-3, 합친 gradient norm 0.5, 매 보정 새 Adam, PPO Adam moment는 유지한다. fit에서만 가치 scale을 구하며 최소값은 1이다. check는 같은 rollout의 상관된 상태이고 독립 평가가 아니다.

두 항 구성의 비교에는 활성 손실 수, 합친 gradient clipping, 수용 규칙의 차이도 포함된다. 결과를 head별 독립 인과 기여율로 합산하지 않는다. 둘 다 보정하는 구현은 원 e25 메서드를 직접 호출하고 parity 검사 후 기존 결과를 재사용한다.

### 추가 계산 대조

`fcsrl_extra_ppo`는 동일 fit/check 관측과 50회 갱신 시도를 사용한다. encoder, prior, log_std, 정규화와 PID를 추가 단계에서 고정하며, 원 rollout의 자르기 전 행동/logp, 전체 rollout에서 정규화한 reward advantage 및 중심화한 cost advantage, 고정 return을 사용한다. λ도 같은 갱신의 값을 쓴다. 목적은 기존 clipped PPO 손실과 reward/cost return MSE이며 teacher 출력 보존 손실은 없다.

별도 Adam, 학습률과 합친 clipping은 보정군과 같다. 50개 후보마다 check에서 EMA 직후 정책과의 정확한 Gaussian KL을 구하고 0.02 이하인 마지막 후보를 선택한다. 조건을 만족한 후보가 없으면 head만 되돌린다. PPO 원래 수집 정책과의 KL은 별도 개념이며 이 추가 guard는 과도한 추가 행동 변화를 제한하기 위한 것이다. finite 검사 실패는 실행 오류로 처리한다.

이 대조는 같은 추가 optimizer 갱신 횟수 및 batch 크기의 대조다. 손실식과 수용 기준은 다르므로 정확히 같은 FLOP/벽시계 예산이나 모든 부가 장치의 완전한 단일 요인 분리가 아니다. 실제 추가 CPU 시간, 벽시계 시간, 최종 채택 단계와 전체 실행 시간을 보고한다. 기존 재사용 실행의 시간은 과거 서버 부하가 달라 참고값으로만 쓴다.

## 데이터, 평가 및 사전 해석 기준

e25와 동일한 Kaggle `jueyetao/wastewater-treatment-dataset`, Mendeley DOI 10.17632/xmbxhscgpr.1, CC BY 4.0. train330일, val43일, test60일. 기존 분할과 환경/MLP/GBM/prior/normalizer를 유지하고 입력 및 재사용 파일의 SHA256을 manifest에 보존한다. 새 test 평가는 하지 않는다. e02/e03에서 test를 사용한 과거 사실은 별도로 유지한다.

하루 287전이, NH4>4 및 DO>2, 하루 비용 한도15, 공기량 Nm³/일. 기존 e25의 PPOConfig, RepresentationConfig와 초기 정책을 그대로 쓴다. 기본 학습 수집 흐름과 PID 갱신은 바꾸지 않는다. 마지막 정책을 두 모델 각각 val43일 x 구성원5개=215에피소드로 평가하며 seed가 독립 집계 단위다.

사전에 고정한 주요 질문은 full 보정이 actor-only, critic-only, extra-PPO 각각보다 비용을 줄이는지다. 원시 공기량/비용과 seed별 값, 평균과 표본 SD를 모두 보고한다. 주 효율 비교는 각 대조 정책에 상수 행동 offset을 적용해 full 보정 공기량에 맞춘 뒤 full 빼기 대조의 paired seed 비용 차이를 구한다. 공기량 오차 0.1% 이내 여부와 paired t 95% CI를 보고한다. offset은 val를 사용한 사후 진단이며 배포 정책이 아니다.

세 대비 x 두 모델의 다중 비교에 Holm 보정한 양측 paired t p값도 함께 보고한다. 작은 5seed와 반복 사용 val에서의 탐색 결과이며 CI/유의성만으로 실제 안전성이나 신규성을 주장하지 않는다. 같은 공기량 대조가 실패하면 그 대비를 성공으로 분류하지 않는다. e25 대비 구성 요소의 차이는 원시 비용과 matched 비용의 방향, 두 모델의 일관성, 4/5 이상 seed 평균 한도 충족을 함께 본다. 차이가 불확실하면 원인 미확정으로 남기며 새 설정을 자동 채택하지 않는다.

보정 전후 KL, 보상/비용 출력 오차, 적용률과 실행 시간을 기전 지표로 별도 보고한다. 출력 오차 감소가 제어 개선을 뜻하지 않는다. 원 환경의 실제 비용 과소예측은 해결하지 않으며 이번 실험은 FPR 진단과 다른 작업이다.

## 실행과 재현

Python `/home/jio/fcsrl_src/.venv/bin/python`, 기존 torch cu130 유지. CPU 4작업, 각 thread1. 원 FCSRL 커밋 d882b91054d52c7a63a18a21f7db67bbbed82b63 고정, 원본 수정 없음, Apache-2.0 라이선스는 `Yoo/code/wwtp/FCSRL_LICENSE`.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/jio/fcsrl_src/.venv/bin/python -u Yoo/experiments/e29/run.py \
 --data /home/jio/사물/Yoo/data/avedore/aved_5min.pkl \
 --model /home/jio/fcsrl_src/results/e02/surrogate/surrogate.json \
 --gbm /home/jio/fcsrl_src/results/e03/surrogate_gbm/surrogate_gbm.pkl \
 --prepared /home/jio/fcsrl_src/results/e23/prepared \
 --reuse /home/jio/fcsrl_src/results/e25/pilot/runs \
 --out /home/jio/fcsrl_src/results/e29/pilot
```

`verify.py`가 학습 전 실제 갱신의 원본 보존, 보정 대상 분리, 난수열 보존, e25 parity와 저장 복원을 확인한다. `match_air.py`는 data/model/gbm/prepared와 `--root` pilot, `--out` matched_air.json을 받는다. `report.py`는 `--root`, `--matches`, `--out` 보고서 폴더를 받는다. 실행 커밋, 명령 및 입력 해시는 runtime manifest와 보관한 launch.json에 기록한다. 실행 commit `543bb8b`, 2026-11-09 17:20 KST경 신규15개 학습 착수. 완료 결과는 아래에 기록했다. `--wait-for-runs`로 종료된 정책의 matched-air 평가를 먼저 처리할 수 있다. 학습4개와 평가4개를 병행하며 각 작업 thread1을 유지한다. `plot.py --summary <summary.json> --out <comparison.png>`로 그림을 만든다.

## 완료 결과와 해석

2026-11-09 17:34:47 KST 신규 학습15개 정상 종료, 17:36:40 KST 같은 공기량 대조40건 완료. 총25개 마지막 정책을 비교했다. [전체 결과](results.md), [수치](summary.json), [같은 공기량 상세](matched_air.json), [그림](figures/comparison.png).

| 방법 | MLP 공기량 Nm³/일 | MLP 비용 | MLP 한도 만족 seed | GBM 비용 |
| --- | --- | --- | --- | --- |
| 기본 FCSRL | 30,627 ± 8,936 | 16.68 ± 4.95 | 2/5 | 9.64 ± 2.99 |
| Actor-only | 29,878 ± 3,487 | 13.50 ± 0.54 | 5/5 | 8.14 ± 0.57 |
| Critic-only | 30,482 ± 8,714 | 18.34 ± 7.73 | 2/5 | 10.63 ± 5.21 |
| Actor+Critic, e25 | 28,707 ± 2,485 | 13.50 ± 0.49 | 5/5 | 8.03 ± 0.41 |
| 추가 PPO | 37,395 ± 11,360 | 17.01 ± 6.32 | 4/5 | 11.74 ± 8.90 |

Actor-only도 e25와 같은 5/5 평균 비용 한도 충족과 작은 편차를 보였다. Critic-only는 가치 출력 오차를 줄였지만 그 안정성을 재현하지 못했다. 이는 Actor 출력 변화 억제가 중요한 단서임을 지지한다. 특정 head의 독립 인과 기여율을 입증하거나 Critic이 항상 불필요하다는 결론은 아니다.

full 보정이 Actor-only보다 같은 공기량에서 줄인 비용은 MLP -0.114, 95% CI [-0.374, 0.146], GBM -0.386, CI [-0.747, -0.026]이다. GBM의 보정 전 p=0.0410이지만 사전6검정 Holm 보정 후 p=0.2460이고 MLP는 p=0.8667이다. Critic 추가의 효율 이점은 두 모델에서 확증하지 못했다. 두 구성이 동등하다고 증명한 것도 아니다.

추가 PPO는 seed3에서 MLP 비용28.312, GBM27.628로 실패했다. 해당 하나의 seed를 제외하지 않고 모두 집계했다. full 대 추가 PPO의 같은 공기량 비용 차이는 MLP -5.740, CI [-18.796, 7.315], GBM -0.759, CI [-3.535, 2.017]로 불확실하다. 따라서 이 추가 학습 설정이 e25의 안정성을 재현하지 못했다고 말할 수 있지만, 추가 계산의 기여를 완전히 배제하거나 모든 추가 PPO 설정보다 우월하다고 주장할 수 없다.

신규15개 모두 22번 추가 단계가 수용됐고 각 50회 시도, 총1100회였다. Actor-only의 22갱신 정책 KL 합은 seed 평균0.01672→0.00067, 가치 오차 합0.3249→0.3249였다. Critic-only는 KL0.01646→0.01646, 가치 오차0.2976→0.0064였다. 의도한 출력에만 보정이 작동했다. 추가 CPU 시간은 Actor-only9.80초, Critic-only9.71초, 추가 PPO11.38초였다. 기존 full 실행에는 이 구간만의 CPU 기록이 없고 전체 시간은 다른 서버 부하에서 측정했으므로 Actor-only의 계산 절감을 확정하지 않는다.

이번 결과로 새 정책을 최종 채택하거나 e25의 기존 CI 실패 판정을 뒤집지 않는다. **다음 검증의 중심 후보는 Actor 출력 보정으로 좁힐 수 있고, 두 Critic까지 보정해야 한다는 주장은 보류한다.** 선행 PPG의 정책 보존과 가까워진 만큼, FCSRL의 EMA 시점 보정이라는 차이가 왜 필요한지를 기존 정책 보존 방식과 직접 비교하는 근거가 다음에 필요하다. 이는 다음 설계 제안이며 새 실험을 시작한 것은 아니다. Kaggle 환경 편향은 그대로 남아 있다.

원 입력과 공유 소스 해시 일치, 25개 설정 일치, 원본 e25 갱신 parity, head 분리, 난수열 보존 및 저장 복원은 `verification.json`, `reuse_verification.json`에 기록했다. `postcheck.json`은 완료된 25개 실행의 해시, prior와 정규화 불변, 학습량, 두 모델의 val215개, 보정 규칙 및 KL 한도를 확인한다. `analysis_audit.json`은 결과 집계와 통계의 별도 검산을 보관한다. 현재 학습과 평가는 모두 종료됐다.

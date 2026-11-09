# e25 Kaggle FCSRL 표현 반영 시 정책과 가치 보정

2026-11-09, 본 학습 전 고정한 설정이다. e01~e24 종합 검토 뒤 사용자가 승인한 첫 단계다. e14 환경 모델 후보는 이번 정책 비교에 함께 넣지 않는다.

## 근거와 가설

e07 및 e24 사후 진단에서 마지막 정책망을 유지하고 중간 표현망으로 바꾸면 행동과 제약 충족이 달라졌다. e24 기본 FCSRL seed 5개에서 MLP 평균 비용 16.68→13.47, 공기량 30,627→35,536 Nm³/일, 한도 만족 2/5→5/5였다. 공기량이 약 16% 증가한 사후 조합이며 GBM 및 같은 공기량 대조가 없으므로 성공 정책으로 채택하지 않았다.

이번 가설은 원본 FCSRL 표현 학습을 유지하면서 표현 반영 순간의 정책 및 가치 출력 변화를 줄이면 후반 정책 악화를 줄일 수 있다는 것이다. 기존 좋은 checkpoint를 사후 선택하거나 표현을 중간부터 고정하는 e08과 다르다. 출력 보존이 제어 성능 개선을 보장하지 않으므로 별도로 평가한다.

## 변경한 요소

`preserve.py`는 e23 PriorFCSRL을 상속한다. 원본 dynamics/feasibility 손실, PPO 갱신, target encoder EMA(tau 0.05), PID, 보상과 비용은 그대로다. PPO 갱신 후 EMA 직전의 정책 평균 및 두 가치망 출력을 teacher로 저장하고, EMA 직후 새 표현에서 이를 근사하도록 actor와 두 가치망을 보정한다. teacher는 실제 운전 행동이나 과거 최고 정책이 아니라 같은 갱신 안의 직전 판단이다.

- 매 rollout 9,184개 정규화 관측에서 독립 RNG seed 25000의 고정 방식으로 fit 2,048개와 check 1,024개를 뽑는다. 서로 겹치지 않지만 같은 rollout에서 나온 상관된 상태이며 독립 일반화 평가가 아니다. 환경 train 날짜만 사용한다.
- 목적은 정책 KL(같은 표준편차의 Gaussian) + 보상 가치 표준화 MSE + 비용 가치 표준화 MSE다. 가치 척도는 fit teacher의 표준편차, 최소 1로 고정한다. 항별 가중치는 모두 1이다.
- 보정용 Adam을 매번 새로 만들고 full batch 50회 갱신한다. actor lr 3e-4, 두 가치망 lr 1e-3, 합친 gradient norm clip 0.5. 이는 추가 계산량이며 같은 연산 예산 비교라고 주장하지 않는다.
- check에서 각 항이 보정 전 오차보다 1% 초과 악화하지 않는 후보 중 합계 오차가 최소인 것을 고른다. 합계가 20% 이상 줄어야 적용하며, 실패하면 EMA 후 보정 전 actor/가치망으로 되돌린다. encoder EMA는 되돌리지 않는다. 모든 단계와 거절 횟수를 기록한다.
- 보정은 encoder, feasibility head, prior, 정규화, log_std, PID를 바꾸지 않는다. 새 표현은 detach해서 보정한다. PPO Adam의 기존 moment 상태는 유지하며 별도 보정 Adam은 폐기한다. 이후 PPO와 별도 보정 갱신의 상호작용도 이 후보의 일부다.
- 원본 train_encoder는 그대로 호출한다. head 구조와 정책 입력 차원도 바꾸지 않는다. UCB, CVaR, 행동 벌점, e24 회복 head는 끈다.

원본 출처 https://github.com/czp16/FCSRL, 커밋 `d882b91054d52c7a63a18a21f7db67bbbed82b63`, Apache-2.0. 라이선스는 `Yoo/code/wwtp/FCSRL_LICENSE`. 이는 원 논문 환경의 재현이 아닌 FCSRL+PPO 폐수처리 적용의 확장 후보다.

## 사전 판정

마지막 checkpoint가 주 비교이며 중간 선택은 별도 보존한다. e24와 같은 탐색적 효과 기준을 결과 전에 고정한다.

1. MLP 및 GBM 각각 확장 평균 비용 <=15, seed 4/5 이상 개별 평균 비용 <=15.
2. 기본 FCSRL에 상수 행동 offset을 더한 사후 대조가 확장 공기량과 seed별 0.1% 이내로 일치. 두 모델 각각 같은 공기량 대조 대비 평균 비용 10% 이상 감소.
3. 각 모델의 paired seed 비용 차이(확장 빼기 대조군) t 95% CI 상한 <0.

전체 채택은 모두 통과해야 한다. 보정 전후 KL 감소 및 적용 비율은 별도 기전 지표이며 효과 기준 실패와 구분한다. 결과에 맞춘 seed/가중치/기준 변경은 하지 않는다. val에서 맞춘 offset은 배포 정책이나 최적 경계가 아닌 탐색 대조이며 반복 사용한 val의 결과를 독립 확증 검정으로 부르지 않는다.

## 데이터 및 고정 조건

수행계획서의 Kaggle `jueyetao/wastewater-treatment-dataset` Avedore. 원 출처 Mendeley DOI `10.17632/xmbxhscgpr.1`, CC BY 4.0. 기존 e23/e24와 동일한 원자료, 전처리, MLP/GBM, prior 및 정규화를 사용하고 해시를 manifest에 보관한다. BSM1은 사용하지 않는다.

- 데이터 `/home/jio/사물/Yoo/data/avedore/aved_5min.pkl`, SHA256 `3a01d5995e49d3b4f3fd42240a4579d1be991629ef70243435422f797d789315`.
- 원 CSV SHA256 `b57bc042a394d84da9fc9620aae5b90f029885dbafa14d3ce6ed437a0b005a2e`.
- MLP `/home/jio/fcsrl_src/results/e02/surrogate/surrogate.json`, SHA256 `d25dbe3ea1a8e33b17f0a2d0f11a5983fe6c4a6d81a1d0db851b530386658117`.
- GBM `/home/jio/fcsrl_src/results/e03/surrogate_gbm/surrogate_gbm.pkl`.
- train 330일(2022-09-15~2023-12-31), val 43일(2024년 1~2월), test 60일(2024년 3~6월). test는 평가 및 선택에 사용하지 않는다. prior는 기존 train 앞 264일로 학습한 것을 고정한다.
- 5분 간격 하루 287전이. 11차원 관측, 호기 공기량 0~8,000 Nm³/h. reward=-air/8000, cost=I(NH4>4)+I(DO>2), 하루 한도15. 같은 시점의 두 위반은 비용2. 하루 truncation bootstrap은 기존대로 유지.
- seed 0~4, 각 202,048전이(요청20만,22갱신), val 2회, 마지막 정책의 val 215에피소드(43일×member5개)를 MLP/GBM에서 평가. init log std -1.6, PID 상한20, value_scale1.
- 기존 e24의 PPO와 기본 FCSRL 10개를 재사용하고 보정 확장 5개를 새로 학습한다. 환경 전이 예산은 같고 추가 보정 계산량은 다르다. 재사용 파일 해시를 고정한다. 개인 실험이며 공동 PROTOCOL 채택 결과가 아니다.

## 사전 검사

`verification.json` 참조. 같은 초기 정책, 원시 행동 logp, 실제 FCSRL 및 EMA 갱신의 bit 단위 일치, RNG 보존, prior/정규화/std 고정, 저장 복원, 순차 및 배치 평가 일치를 확인했다. 표현 변화가 0인 입력에서는 보정도 정확히 0이다. 작은 초기 업데이트에서 보정이 기준을 못 만족하면 원래 출력을 보존하는 거절 경로도 확인했다.

학습된 e24 seed0 checkpoint와 train 첫 3일 관측만 사용한 보정 check에서 KL 0.000525→0.0000298, 합계 출력 오차 0.02449→0.000298로 줄었다. 이것은 상태 재생 검사이며 정책 제어 개선이나 val 일반화 결과가 아니다.

## 실행

Python 3.10.19, torch 2.14.1+cu130, numpy1.23.5, scipy1.15.3. 기존 venv 유지, CPU4작업, 작업당 thread1. 실행 코드 commit 및 소스/입력 해시는 `launch.json`과 runtime manifest에 기록한다. 전체 데이터 및 checkpoint는 Git에 넣지 않는다.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/jio/fcsrl_src/.venv/bin/python -u Yoo/experiments/e25/run.py \
 --data /home/jio/사물/Yoo/data/avedore/aved_5min.pkl \
 --model /home/jio/fcsrl_src/results/e02/surrogate/surrogate.json \
 --gbm /home/jio/fcsrl_src/results/e03/surrogate_gbm/surrogate_gbm.pkl \
 --prepared /home/jio/fcsrl_src/results/e23/prepared \
 --reuse /home/jio/fcsrl_src/results/e24/pilot/runs \
 --out /home/jio/fcsrl_src/results/e25/pilot
```

평가 `match_air.py`에는 위 data/model/gbm/prepared와 `--root /home/jio/fcsrl_src/results/e25/pilot --out /home/jio/fcsrl_src/results/e25/matched_air.json`을 준다. `report.py`에는 `--root` 같은 경로, `--matches` 해당 JSON, `--out Yoo/experiments/e25`. `postcheck.py`에는 `--root /home/jio/fcsrl_src/results/e25 --repo /home/jio/사물 --prepared-root /home/jio/fcsrl_src/results/e23/prepared --out /home/jio/fcsrl_src/results/e25/postcheck.json`. `plot.py`는 `--summary Yoo/experiments/e25/summary.json --out Yoo/experiments/e25/figures/comparison.png`.

## 한계

기존 실제 운전 재생의 비용은 실측70.95, MLP26.79, GBM28.39였다. 환경 모델의 과소예측은 이번 보정으로 해결되지 않는다. 같은 rollout의 check 오차 감소는 전체 상태 공간의 보존이나 안전성 보장이 아니다. 추가 보정은 이후 PPO optimizer moment와 상호작용할 수 있다. 제약 충족과 같은 공기량 성능을 따로 확인하며 실제 공정 절감 또는 안전성을 주장하지 않는다. e14 모델 후보의 확인과 결합은 이번 효과를 분리한 뒤 판단한다.

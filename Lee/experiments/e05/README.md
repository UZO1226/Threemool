# Lee/e05: feasibility 점수를 정책 학습의 보조 신호로 사용

## 목적

FCSRL 논문은 결론에서 feasibility 점수를 정책 학습의 보조 신호로 써서 모든 상황에서의
안전(state-wise safety)을 달성하는 것을 향후 과제로 남겼다. 원본 구현에서 feasibility
점수는 표현(encoder)을 훈련하는 데만 쓰이고 정책 손실에는 들어가지 않는다. 이 실험은
그 향후 과제를 직접 시도하는 확장이다.

동기는 우리 결과다. Lee/e02 에서 하루 평균 제약을 지켜도 순간 NH4 위반이 30% 에서
41% 였고, FCSRL + PPO 는 PPO-Lag 보다 위반이 많았다. FCSRL 이 이미 학습한 위험
예측(feasibility)을 정책이 직접 쓰게 하면 FCSRL 쪽 개선이 가능한지 본다.

## 방법

행동을 한 뒤 도달한 다음 상태 s_t+1 의 feasibility 예측 F(z_t+1) 을 표준화해 정책
이점에서 뺀다 (구현은 `Lee/code/bsm/feasibility_policy.py`).

    adjusted = base_advantage - weight * standardize(F(z_t+1)) / (1 + lambda)

- 현재 상태의 함수는 행동과 무관해서 정책 경사의 기댓값이 0 이 되므로, 선택한 행동이
  만든 결과인 다음 상태를 쓴다.
- 표준화로 weight 가 곧 보상 이점(크기 약 1) 대비 상대 가중치가 된다. 평균을 빼는 것은
  상태 의존 baseline 이라 경사의 기댓값을 바꾸지 않는다.
- (1 + lambda) 로 나누는 규칙은 Yoo 의 CVaR 확장(Yoo/e06)과 같다.
- 라그랑주 승수와 비용 제약, 표현 학습 손실은 그대로 둔다.
- 이 항은 안전을 보장하지 않는다. 학습된 추정값에 기반한 실용적 벌점이다.

실제 환경에서 짧게 확인한 결과(seed 999, 비교 seed 와 분리), 벌점 크기는 기본 이점
크기의 정확히 weight 배였다 (weight 0.3 에서 비율 0.30).

## 바꾼 것

| 항목 | Lee/e02 (binary, FCSRL + PPO) | Lee/e05 (binary, FCSRL + PPO + feasibility 벌점) |
| --- | --- | --- |
| 정책 이점 | 기본 이점 | 기본 이점에서 weight x 표준화한 F(z_t+1) 을 뺌 |
| weight | 해당 없음 | 0.1, 0.3 두 값 |

비용 모드(이진 NH4), 한도 72, seed(0, 1, 2), 40 갱신, 초기 상태, 관측 정규화,
신경망, 평가 방법이 모두 e02 와 같다. 비교 대상은 Lee/e02 의 같은 조건 결과
(FCSRL + PPO 와 PPO-Lag)이며 새로 돌리지 않는다. 새 실행은 2 weight x 3 seed = 6 개다.

weight 0.1 과 0.3 은 결과를 보기 전에 정했고 바꾸지 않는다. 두 값 모두 보고하며
좋은 쪽만 고르지 않는다.

## 가설 (사전 등록)

- H11. feasibility 벌점을 더하면 FCSRL + PPO 의 순간 NH4 위반이 e02 의 FCSRL + PPO
  (41.30%) 보다 줄어든다.
- H12. 벌점 가중치가 클수록(0.3) 위반이 더 줄고 폭기 에너지는 늘어난다.
- H13. 벌점을 더한 FCSRL 이 PPO-Lag(30.49%) 와 같거나 더 낮은 위반에 도달한다.

H11 이 맞아도 벌점이 정책을 더 보수적으로 만들어 에너지만 늘린 것일 수 있다. 그래서
위반과 에너지를 함께 보고, 에너지를 맞춘 비교는 하지 못하므로 교환 곡선 위에서의
이동인지 곡선 자체의 개선인지는 이 자료만으로 구분하지 못한다고 미리 적는다.

## 실행

```bash
export BSM1_ARCHIVE=.../BSM1.zip
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u Lee/experiments/e01/run_all.py --prepared <결과>/e01_prepared --out <결과>/e05_runs_w0.1 \
  --updates 40 --parallel 3 --modes binary --methods fcsrl_fpen --feasibility-weight 0.1 --upstream <원본 FCSRL 경로>
python -u Lee/experiments/e01/run_all.py --prepared <결과>/e01_prepared --out <결과>/e05_runs_w0.3 \
  --updates 40 --parallel 3 --modes binary --methods fcsrl_fpen --feasibility-weight 0.3 --upstream <원본 FCSRL 경로>
```

두 묶음을 동시에 돌린다. 준비 산출물과 학습 전 검사는 e01 의 것을 쓰고 run_all.py 가
매번 무결성을 다시 검사한다. 예상 시간은 약 1시간이다.

## 미리 밝히는 한계

- 40 갱신은 수렴이 아니다. seed 3개의 탐색적 비교이고 다중 비교 보정이 없다.
- 비교 대상 e02 는 같은 코드와 설정이지만 다른 시점에 실행한 결과다.
- weight 두 값만 시험했다. 최적 가중치를 찾는 실험이 아니다.
- feasibility 예측이 부정확하면 벌점이 잘못된 방향으로 정책을 밀 수 있다. 이 위험을
  측정하지 않았다.
- BSM1 물리 시뮬레이터이고 NH4 만 제약한다.

## 실행 결과

[results.md](results.md) 에 있다. 벌점의 실제 크기가 명목 가중치보다 작았고(기본 이점의 약 4%와 11%), 알려진
하루 reset 설계 결함 위의 결과라는 한계를 함께 적었다.

# e27 거짓양성의 구분 능력 및 시간 전이 진단

2026-11-09 실행 전 계획. e26의 하루 MSE 모델은 외부 구간에서 짧은 MSE보다 NH4 탐지율과 FPR이 모두 악화했다. 고정 모델의 점수 구분 능력과 보정 가능한 편향을 구분한다. Kaggle Avedore 및 FCSRL 적용과 확장 범위를 유지한다. 이번 작업은 환경 진단이며 정책 학습이나 환경 교체가 아니다.

## 고정 조건

- 모델: start_multi, short_mse, day_mse의 기존 seed0,1. 가중치 갱신 없음.
- 원본 모델 fit263일 뒤 purge1일과 내부 holdout66일을 그대로 사용한다. 첫22일 calibration, 1일 제외, 다음21일 평가와 첫44일 calibration, 1일 제외, 마지막21일 평가의 시간순 두 fold다. 이66일은 과거 실험에서 재사용한 개발 자료다.
- 외부 val43일은 전체 내부66일로 보정 설정을 고정한 뒤 평가한다. val 역시 반복 사용한 개발 자료이며 독립 최종시험이 아니다. test 날짜는 평가하거나 설정 선택에 쓰지 않는다.
- 첫1시간 실제 상태 뒤275전이 자율 재생. 실제 운전 행동/외생 경로를 사용한 관찰적 검증이며 새로운 행동의 인과 효과를 입증하지 않는다.
- 물리 기준 NH4>4, DO>2 유지. FPR은 실제 정상 중 위반으로 예측한 비율이다. 정책 비용이나 FDR, 구간 coverage와 다르다.

## 진단과 작은 대조

1. 고정 모델의 ROC AUC와 동일 recall50/80에서 FPR을 비교한다. 평가 정답으로 얻은 ROC 운영점은 구분 능력의 사후 진단이며 배포 성능이 아니다. 별도로 calibration에서 recall50/80 또는 FPR10 운영점을 고정해 뒤 구간에 적용한다. 이 분류 임계값은 물리 비용 기준을 바꾸지 않는다.
2. 예측 거리1~36,37~144,145~275전이, 운전 단계0/1/2, 기존 fit의 온도3분위별로 정상/위반의 오차와 FPR을 보고한다. 조건별 연관성이 인과 효과라는 주장은 하지 않는다.
3. NH4에만 보정 없음, 과거 전체 평균 편향, 예측 거리3구간별 평균 편향, 구간별 일간 EWMA 편향을 대조한다. DO는 그대로 둔다. EWMA 반감기는 사전 고정한 관측7일이며 검색하지 않는다. 결과 NH4는0미만만 자른다. 이는 후처리 예측 대조이며 보정값을 전이망 안에 되먹이는 새 환경이 아니다.
4. EWMA는 calibration 날짜를 차례로 읽어 초기화하고, 각 평가일의 하루 예측을 모두 저장한 뒤 그날의 보정 전 모델 오차로 다음 날 상태를 갱신한다. 하루 예측 중 미래 정답을 사용하지 않는다. 다른 행동을 택한 RL 가상 경로의 정답으로 실측값을 넣지 않는다.

## 판정 및 한계

각 모델의 보정 전 결과 대비, 내부 시간순 평가와 외부 val 모두에서 두 seed 각각 NH4 recall이 낮아지지 않고 FPR이 감소하며 하루 비용MAE가 악화하지 않는 보정만 후속 후보로 삼는다. 두 seed만으로 확증하지 않으며, 동일 날짜를 seed/방법에 함께 재표집하는2000회 paired bootstrap95%구간을 보고한다. 이는 날짜 변동의 탐색적 구간이며 시계열 독립/새 seed 일반화 보장이 아니다. 기존 절대 recall80/FPR10 미충족 사실과 환경 교체 기준은 유지한다.

단순 보정이 실패하면 더 복잡한 OMPB를 자동 실행하지 않는다. 분류 구분 능력, 정상과 위반의 편향 방향, 시간 전이 여부를 근거로 보정으로 풀 수 있는 문제인지 먼저 판단한다. OMPB/BC-ACI 참고 아이디어를 활용하지만 그 알고리즘의 재현이나 coverage 보장 실험이라고 부르지 않는다.

## 출처와 검증

데이터/모델 출처는 e14와 e26과 같다. 내부 예측은 e14 저장 배열을 재사용하고 외부는 e26과 동일한 모델/재생 함수로 다시 얻는다. 기존 score 완전일치, 데이터/소스/가중치/캐시 해시 불변을 검사한다. NH4/DO 미래 정답 교란과 보정의 당일 이후 정답 교란으로 미래 정답 비참조를 확인한다. 양의 편향 보정 방향, DO 불변, 동점이 있는 임계값 판정을 별도 검사한다.

기존 Python venv와 CPU1thread 사용. 패키지 설치 및 변경 없음. 출력 경로 `/home/jio/fcsrl_src/results/e27/`. 원시 예측과 상세 진단은 이 경로에 보관한다.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /home/jio/fcsrl_src/.venv/bin/python Yoo/experiments/e27/diagnose.py --verify
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /home/jio/fcsrl_src/.venv/bin/python Yoo/experiments/e27/diagnose.py \
 --out /home/jio/fcsrl_src/results/e27
```

참고: [OMPB](https://arxiv.org/abs/2609.31960), [BC-ACI](https://arxiv.org/abs/2604.13253), [분포 전이의 Neyman-Pearson 분류](https://arxiv.org/abs/2511.06641). 실행 결과와 실행 커밋은 완료 뒤 추가한다.

## 완료 결과

실행 코드 `adce5cd`, 2026-11-09 16:26 KST 완료. [전체 결과](results.md), [seed별 판정과 출처](summary.json), [그림](figures/tradeoff.png). 상세 원시 결과 `/home/jio/fcsrl_src/results/e27/diagnosis.json` 및6개 예측 배열. 내부 캐시 재사용과 외부 재생의 기존 score가 완전히 일치했다. 미래 정답 비참조 및 해시 불변 검사를 통과했고, 별도 정렬 기반 검산으로6개 모델의 recall80 FPR 및 FP 수를 확인했다.

day_mse의 외부 후반145~275전이 FPR은33.28%, 전체 오탐 중83.93%가 이 구간에 있었다. 정상 상태에서는 NH4 +1.414, 위반 상태에서는 -1.075의 평균 편향이 함께 있었다. 같은 recall50에서 외부 FPR은 short_mse10.69%, day_mse18.17%로, 임계값 위치만의 문제가 아닌 구분 능력 저하가 확인됐다. 세 모델 모두 사후 전역 임계값으로도 recall80/FPR10을 달성하지 못했다. 예측 거리와 시간대는 함께 변하므로 누적 오차만을 원인으로 확정하지 않는다.

과거 편향 보정9개 후보는 모두 탐지율 유지 기준에서 실패했다. day_mse의 일간 EWMA는 FPR17.75%→6.82%로 낮췄지만 recall49.25%→29.20%로 낮췄다. OMPB, 환경 교체 및 e25 결합은 실행하지 않는다. 평균 편향 제거보다 예측 후반의 정상/위반 구분 능력 저하를 다음 분석 대상으로 남긴다.

표와 그림 재생성:

```bash
/home/jio/fcsrl_src/.venv/bin/python Yoo/experiments/e27/report.py \
 --source /home/jio/fcsrl_src/results/e27/diagnosis.json --out Yoo/experiments/e27
```

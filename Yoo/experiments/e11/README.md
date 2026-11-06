# Yoo/e11 누락한 유입 및 유출 반응조 입력 대조

**2026-11-06 19:06:54 KST, 4개 학습과 내부 및 외부 검증 완료.** 경로 입력이 비용 MAE를 줄였지만 위반 recall이 낮아져 기존 환경 모델 교체와 RL 재학습은 하지 않았다.

## 배경

e10의 이력 추가는 내부 holdout 비용 MAE를 낮췄지만 NH4 위반 recall과 시작 F1, DO 장기 오차를 개선하지 못했다. 원자료를 재점검하니 유입 및 유출 반응조 신호가 전처리에서 빠져 있었다. [데이터 설명 논문 Table 1](https://arxiv.org/html/2407.05959v1)은 `INLET.Q`를 처리장 전체 유량, `PROCESSPHASE.INLET TANK`와 `OUTLET TANK`를 물이 들어가고 나가는 반응조 번호로 정의한다. 이는 반응조의 호기 단계와 별도 신호다.

e10 입력 진단에서 학습 263일의 신호 유효 비율은 99.996%, 내부 holdout 66일은 100%였다. 같은 호기 단계에서도 유입 반응조별 평균 NH4 변화가 달랐다. 다른 조건의 차이가 있어 인과 효과로 단정하지 않는다. SS도 원자료에 있으나 결측이 많아 이번 후보에는 넣지 않는다.

## 사전에 고정한 별도 대조

이 실험은 이력 시험 결과를 본 뒤 정한 탐색적 후속이다. 내부 holdout을 재사용한다는 사실을 명시하며 새 독립 검증이라 주장하지 않는다. 학습과 내부 판정에는 바깥 val과 test를 사용하지 않는다. FCSRL + PPO, 기존 환경 모델, 보상과 비용 정의는 바꾸지 않는다. 대리 환경 입력의 타당성 점검이지 새로운 RL 확장이 아니다.

- e10과 같은 학습 263일, 제외 1일, 내부 holdout 66일, seed 0과 1을 쓴다. 새 모델을 초기화해 두 군 모두 60 epochs 학습한다. 기존 모델은 가져오지 않는다.
- `without_routing`: 추가 입력 8개를 모두 0으로 둔다. `with_routing`: inlet 0, 1, 2와 outlet 0, 1, 2의 비율 및 두 신호의 관측 여부를 넣는다. 입력 비율은 각 5분 구간의 품질 0인 원자료만 집계한다. 미래 보간 없이 결측은 값 0과 관측 여부 0으로 표현한다.
- 각 DO 및 NH4 모델은 입력 18, ReLU 64 x 2다. 두 군의 파라미터 개수 및 초기 가중치, batch 순서와 optimizer 설정이 같다. e10의 `Model`을 재사용하지만 과거 상태 8개 자리에 운전 신호 8개를 넣는다. 과거 농도를 추가한 군과 직접적인 성능 우열을 주장하지 않는다.
- Adam lr 0.001, weight decay 0.00001, gradient clip 1, batch 1024, 마지막 epoch를 사용한다. 입력과 출력 정규화는 학습 263일만 사용한다. 목표는 변화량 표준편차로 정규화한 다음 농도 MSE다.
- 실제 공기량과 유입 및 유출 경로를 고정 재생한다. 각 날짜 01:00 상태에서 시작해 이후 275전이를 평가한다. 1, 12, 275스텝별 실측 재시작 결과를 구분한다. 비용은 NH4 > 4 및 DO > 2의 합이며 기존 287전이 정책 비용 한도와 직접 비교하지 않는다.
- e10과 같은 진행 기준을 쓴다. 두 seed의 비용 MAE 개선과 평균 5% 감소, NH4 onset F1 및 최장 관측 위반 길이 MAE 악화 없음, DO MAE 및 한 스텝 NH4 MAE 악화 5% 이내다. 유의성 검정이 아니다.

유입 및 유출 경로도 기존 운전자가 정한 제어 신호다. 이를 고정 재생하는 시험이 다른 폭기 정책에서도 경로가 그대로 유지된다는 사실이나 실제 정책 안전성을 입증하지 않는다. NH4 부하 측정치가 새로 생기는 것도 아니다. 성능이 좋아도 새 환경으로 모든 정책을 같은 조건에서 재검증해야 한다.

## 재현

CPU 1 thread, 기존 venv를 쓴다. 원자료 및 처리 자료 SHA256, 실행 소스와 커밋, 시간 및 학습 로그는 결과 JSON에 남긴다. 저장 후 출력 일치, 실제 학습 모델의 미래 실측 비참조, 같은 초기 가중치 및 batch 순서를 검사한다. 출력 경로가 비어 있지 않으면 실행을 거부한다.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /home/jio/fcsrl_src/.venv/bin/python -u Yoo/experiments/e11/routing_trial.py \
 --raw /home/jio/사물/Yoo/data/avedore/aved_raw.csv \
 --data /home/jio/사물/Yoo/data/avedore/aved_5min.pkl \
 --out /home/jio/fcsrl_src/results/e11/routing_trial
```

데이터 출처와 기존 전처리는 [e02](../e02/README.md), 입력 누락 근거는 [e10](../e10/README.md)을 따른다. 모델과 예측 배열은 서버 결과 경로에 보관한다. 팀 공식 비교는 아직 합의하지 않은 개인 탐색이다.

## 고정 모델의 외부 val 확인

내부 기준을 통과하여 저장한 모델 4개를 바꾸지 않고 기존 val 43일에서 한 번 평가한다. 새 학습이나 설정 탐색은 하지 않는다. 275전이, 1 및 12 및 275스텝 재시작과 같은 지표 및 진행 기준을 사용한다. 외부 결과를 본 뒤 모델을 재선택하지 않는다. 기존 연구에서 이 val을 관찰한 이력이 있으므로 완전히 독립적인 일반화 근거라고 주장하지 않는다. test는 사용하지 않는다.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /home/jio/fcsrl_src/.venv/bin/python Yoo/experiments/e11/validate.py \
 --raw /home/jio/사물/Yoo/data/avedore/aved_raw.csv \
 --data /home/jio/사물/Yoo/data/avedore/aved_5min.pkl \
 --trial /home/jio/fcsrl_src/results/e11/routing_trial \
 --out /home/jio/fcsrl_src/results/e11/outer_val.json
```

## 최종 결과와 판정

학습 실행 커밋 `0a97165790bb199efd7266a9d723e994e619c296`, 고정 외부 검증 커밋 `1b41a7400314b282c5cd8c500cf517de29adcf08`의 전체 SHA는 [외부 요약](outer_summary.json)에 있다. 모델 구조나 학습 설정을 결과에 맞춰 바꾸지 않았다. 두 군의 모델 크기는 10,882개 파라미터로 같다.

| 구간 | 경로 입력 | 날짜별 비용 MAE | NH4 위반 스텝 recall | NH4 시작 F1 | 최장 관측 위반 구간 MAE (분) |
| --- | --- | --- | --- | --- | --- |
| 내부 66일 | 없음 | 57.780 | 39.29% | 0.103 | 207.84 |
| 내부 66일 | 있음 | 42.424 | 30.12% | 0.135 | 125.49 |
| 기존 val 43일 | 없음 | 58.872 | 36.54% | 0.050 | 150.87 |
| 기존 val 43일 | 있음 | 48.140 | 19.41% | 0.081 | 100.99 |

seed 2개 평균이며 전체 275전이 연속 예측이다. 개별 seed 및 단기 예측은 [결과 표](results.md)에 있다. 경로 입력 추가는 내부 비용 MAE를 26.58%, 기존 val 비용 MAE를 18.23% 줄였다. 시작 F1과 최장 위반 구간 길이 오차도 두 구간의 두 seed에서 개선했다. val에서 유입 및 유출 신호는 100% 유효했다.

그러나 위반 스텝 recall은 기존 val에서 두 seed 모두 낮아졌다. 후보의 평균 예측 비용도 36.535로 실측 66.535보다 작다. 잘못 예측한 위반이 줄면서 실제 위반도 놓치는 절충이 생겼다. **사전 진행 기준은 통과했지만 기존 환경 모델을 교체하지 않는다.** 이 기준에는 위반 스텝 recall 유지 조건이 없었다는 한계도 드러났다. 사후에 기준을 바꿔 통과 여부를 다시 쓰지 않고, 통과와 최종 미채택 판정을 모두 남긴다.

현재 결론은 경로 신호 누락이 예측에 영향을 주는 구체적인 문제라는 것이다. 이것이 모든 장기 오차의 원인이거나 경로 입력 추가만으로 제약 예측이 해결된다는 결론은 아니다. 다음 보정은 평균 오차와 함께 위반 누락률을 줄여야 하며, 임계값 주변의 예측 편향과 예측 분포를 점검할 필요가 있다. 평균 오차만으로 후보를 고르지 않는다. FCSRL + PPO의 핵심을 바꾸거나 이 결과를 RL 확장 성공으로 보고하지 않는다.

## 실행 중 수정과 검사

최초 학습 4개와 평가 24개를 모두 마친 뒤 JSON 저장에서 numpy boolean 직렬화 오류가 났다. 학습 실패나 비유한 값은 아니었다. `4926f3c`에서 결과 판정을 Python bool로 바꾸고 `--finalize-only`로 저장을 완료했다. 기존 모델 4개의 SHA256과 예측 배열 24개에서 다시 계산한 지표가 저장된 진행 기록과 모두 일치함을 확인했으며 재학습하거나 결과를 선택하지 않았다. 원래 실행 커밋과 후처리 커밋을 둘 다 기록했다.

모든 loss와 gradient 유한, 같은 초기 가중치와 샘플 순서, 저장 및 복원 출력 일치, 실제 학습망의 미래 실측값 비참조 검사를 통과했다. 외부 확인은 저장된 모델의 SHA256이 그대로임을 검사했다. 원래 데이터와 기존 환경용 모델은 덮어쓰지 않았다. 결과는 `/home/jio/fcsrl_src/results/e11`에 보관한다.

표와 그림 재생성:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 /home/jio/fcsrl_src/.venv/bin/python Yoo/experiments/e11/summarize.py \
 --trial /home/jio/fcsrl_src/results/e11/routing_trial \
 --outer /home/jio/fcsrl_src/results/e11/outer_val.json \
 --out Yoo/experiments/e11
```

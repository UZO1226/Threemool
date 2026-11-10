# e43: 상태 피드백과 전체 제어 구조 감사

2026-11-10. e42 이후 승인된 전체 구조 검토다. 새 정책이나 대리 모델 학습 대신 기존 코드의 피드백, 관측 누락, 날짜 경계를 검사한다. DARROW, 총 공기량 행동, FCSRL의 dynamics/feasibility 표현과 공기/NH4/DO 동시 개선 목표를 유지한다.

## 실행 전 고정 범위

- e30 데이터의 train188일만 진단한다. val40/test41은 평가하지 않는다.
- e35 DO 대응 수정 모델을 고정한다. e38~e42 미채택 모델을 제어 환경에 넣지 않는다.
- 첫 train 날짜에서 기존 평가 함수를 사용해 95회 정책 호출과 상태 전달을 검사한다. 미래 실측 수질을999로 바꿔도 경로가 완전히 같은지 확인한다. 정책은 학습 정책이 아닌 고정 진단 함수다.
- 모든 train 날짜의 t=1,48,94에서 같은 정책 관측과 같은 총 공기량을 유지하고, 현재 배분/직전 구간 유량/누락 유입 성분을 각각 다른 train 날짜의 값으로 교체한다. 이는 모델의 입력 의존성을 확인하는 합성 검사이며 현실에서 가능한 개입이나 제어 개선의 근거가 아니다.
- GAE와 feasibility의 날짜 경계에서 다음 날 항은 끊고 마지막 상태 가치 bootstrap은 남는지 수치 검사한다.
- 기존 관측11개에 직전 실행 구간 유량5개를 추가한 16차원 관측 함수를 작성하고 인과적 읽기 범위를 검사한다. 기존 환경/학습기에는 연결하지 않는다.

가설은 정책에 이미 피드백이 있지만 물리 대리 환경의 자기 예측과 관측 누락이 별개로 존재한다는 것이다. 원인을 하나로 확정하거나 공기 절감 성능을 측정하는 실험이 아니다. 결과와 다음 설계는 실행 후 아래에 기록한다.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE \
/home/jio/fcsrl_src/.venv/bin/python Yoo/experiments/e43/audit.py \
  --prepared /home/jio/fcsrl_src/results/e30/prepared \
  --corrected /home/jio/fcsrl_src/results/e35/zone_fix/corrected_surrogate.pkl \
  --out /home/jio/fcsrl_src/results/e43/main
```

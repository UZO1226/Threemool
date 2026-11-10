# e36: 기존 정책의 단순 기준 대비 효율과 환경 민감도

## 질문과 사전 결정

e01~e35 전체 검토에 따라 추가 NH4 증상 진단과 신규 RL 학습을 보류했다. 기존 Actor 보정이 단순 공기량 조정보다 유리한지, DO 구간 연결 오류 수정에도 판단이 유지되는지 확인한다. FCSRL의 dynamics/feasibility 표현과 정책은 변경하지 않는다.

Kaggle DARROW e30 준비 파일의 train188일에서 BC 공기 보정 범위를 정한다. 유량의 q10~q90 폭 절반을 500m³/h 단위로 올림하고 최대4500으로 제한한다. 범위 안에서 500 간격으로 고정한 모든 후보를 평가한다. 행동은 기존 범위0~9000에 clip한다. BC는 실제 미래 수질 대신 자기 궤적의 상태를 입력받는다.

수정 환경의 train에서 NH4/DO 평균 비용이 각각6 이하인 후보 중 최소 공기량을 대표 기준으로 고른다. 없으면 두 비용 중 큰 값이 가장 작은 후보를 고르고 미충족이라고 표시한다. 동률이면 공기량으로 정한다. protocol.json과 selection.json의 SHA256을 검증 평가 전에 잠근다. val 결과로 범위를 넓히거나 대표 기준을 교체하지 않는다.

val40일 x 모델5구성원에서 모든 기준을 공개한다. 같은 NH4 평균 비용을 끼우는 인접 공기 보정 후보 중 양끝과 정책 DO비용이6 이하인 구간에만 선형 보간한다. 모든 교차 구간을 보고하며 유리한 교차점만 선택하지 않는다. 이는 곡선의 기술적 비교이며 보간된 실제 제어기를 실행한 결과가 아니다. 범위 밖 외삽은 하지 않는다. 초과 농도 적분과 나쁜 날짜도 확인한다.

## 평가 대상과 해석

- e33 PPO, FCSRL, Actor 보정 최종 정책 각seed0/1, 총6개. 각각401280전이 학습 체크포인트를 고정한다.
- e34 기존 환경 결과를 재사용한다. 새 환경은 e35 corrected_surrogate.pkl이다. 잘못된 기존 대응은 민감도 참고이며 채택 대상이 아니다.
- 초기 BC와 고정 유량 보정 기준, 기록 행동의 모델 재생, 실제 관측 수질을 구분한다.
- 15분 x95전이=23.75시간. 공기량은 m³, 전력(kWh)은 미측정이다. NH4>4, max(DO)>2의 표본 수와 평균 한도6/6은 역사적 연구 설정이며 현장 법적 기준으로 주장하지 않는다. 농도 단위는 원출처mg/L에 맞춘다.
- 통계 단위는 날짜다. 모델 구성원은 날짜 안에서 평균한다. 7일 이동블록 bootstrap2000회는 개발 자료의 탐색적 불확실성 표시이며, seed2개로 재현 성공을 확정하지 않는다.
- 단순 기준 대비 이점이 없으면 보정 강도/seed를 자동 확대하지 않는다. 환경 변경에 제약 충족이나 순위가 크게 달라지면 알고리즘 우열 주장을 보류한다. 양쪽 모델에서 좋아도 실제 공정의 인과적 절감 증거는 아니다.
- 관측 가능한 정보, 기록된 구간 배분, 일별 reset과 외생 입력은 기존 환경 그대로다. test41일은 사용하지 않는다.

## 실행

기존 Python3.10, torch cu130 환경에서 CPU1스레드를 사용한다. 학습이나 모델 재학습은 없다. 준비/체크포인트/환경 SHA256 및 정확한 실행 커밋은 manifest.json에 기록한다. 원본 데이터 출처와 전처리는 [e30](../e30/README.md), 환경 수정 근거는 [e35](../e35/README.md)를 따른다.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE \
/home/jio/fcsrl_src/.venv/bin/python Yoo/experiments/e36/run.py \
  --prepared /home/jio/fcsrl_src/results/e30/prepared \
  --runs /home/jio/fcsrl_src/results/e33/main \
  --corrected /home/jio/fcsrl_src/results/e35/zone_fix/corrected_surrogate.pkl \
  --out /home/jio/fcsrl_src/results/e36/main
```

결과는 실행 완료 후 추가한다. 전체 로그와 날짜별 지표는 위 외부 결과 폴더에 보관한다.

# Yoo/e22: e21 비교 기준 정확도 검증

## 목적과 고정 범위

e21의 에너지 절감률은 DO 설정값 0.25 간격의 고정 운전 결과를 선형 보간한 값이다. 절감률이 약 1%이고 CI 상한이 0 근처라 보간 오차가 결론에 영향을 줄 수 있다. 또한 e21 정책은 30분마다 설정값을 갱신하지만 기존 부하 비례 규칙은 5분마다 갱신한다.

e21의 마지막 정책 2방법 x seed 0~4와 규칙 파라미터를 고정한다. 신경망 재학습, 제약, 행동 차원, FCSRL 손실 변경은 없다. 기존 e21 결과와 판정은 보존한다. 이 실험은 사후 비교 검증이며 확장 성능의 새로운 사전 검정으로 포장하지 않는다.

## 검증 절차

1. 기존 고정 DO 2.0과 5분 부하 비례 규칙을 건기 및 우기에서 다시 실행한다. 에너지, 유량 가중 NH4, 초과 시간 비율의 저장값 대비 오차가 각각 1e-7 이내여야 진행한다.
2. 같은 규칙을 30분마다 결정하도록 감싼다. 계수 a=2, b=1.5와 학습 자료에서 정한 평균 부하는 그대로 둔다. 평가 자료로 계수를 다시 고르지 않는다.
3. 정책 10개와 규칙 2개의 평균 NH4 각각에 대응하는 고정 DO를 Brent 근 찾기로 구한다. 각 후보 DO마다 14일 연속 운전 후 후반 7일을 집계한다. 기존 경계 안에서만 탐색하며 외삽하지 않는다.
4. DO 탐색 허용 오차는 1e-6 mg/L, 최종 NH4 오차는 1e-4 mg N/L 이하다. 직접 계산한 고정 운전 에너지 대비 절감률과 기존 보간값의 차이를 보고한다.
5. seed 5개의 t 95% CI, 두 방법 Welch 검정, 30분 규칙과의 차이를 함께 보고한다. p값이 비유의라는 사실로 동등성을 주장하지 않는다.

수질을 맞추기 위해 평가 NH4를 사용하는 것은 비교용 고정 경계를 계산하기 위한 것이다. 이 설정값을 배포 정책이나 학습 후보로 채택하지 않는다. 우기는 e16 이후 반복 확인한 보조 평가이며 독립 최종 test가 아니다. 고정 고폭기 운전의 위반률을 물리적 최솟값이라고 단정하지 않는다.

## 환경과 실행

환경은 e21과 같은 BSM1, 3/4/5번 조 각각 1분 DO PI, 공통 설정값 0.5~3.0 mg/L다. 기존 e19 준비 상태와 공식 IWA BSM1 유입을 그대로 쓴다. 지표는 폭기 kWh/일, 유량 가중 NH4 mg N/L, NH4 > 4 mg N/L 시간 비율이다. 새로운 reward나 비용 한도는 없다. 원본 FCSRL 커밋 d882b91054d52c7a63a18a21f7db67bbbed82b63의 표현 학습을 유지한 e21 정책의 저장 지표를 읽는다. 원본 재현 결과와 구분되는 BSM1 적용 평가다.

```bash
BSM1_ARCHIVE=/home/jio/fcsrl_src/results/e16/BSM1.zip \
PYTHONPATH=/home/jio/fcsrl_src/bsm-deps-e16:/home/jio/사물/Yoo/code/wwtp \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/jio/fcsrl_src/.venv/bin/python Yoo/experiments/e22/audit.py \
  --prepared /home/jio/fcsrl_src/results/e19/prepared \
  --runs /home/jio/fcsrl_src/results/e21/all \
  --out /home/jio/fcsrl_src/results/e22 --workers 4
```

Python 3.10.19, numpy 1.23.5, scipy 1.15.3, bsm2-python 0.0.16, numba 0.63.1, CPU 4개 프로세스. 기존 venv와 torch는 변경하지 않는다. 실행 커밋, 입력 파일과 실행 소스의 SHA256은 실행 시작 시 생성되는 manifest.json에 기록한다. 대용량 원본, 준비 상태와 e21 모델은 `/home/jio/fcsrl_src/results/`에 있다. 재평가 로그는 `/home/jio/fcsrl_src/results/e22/run.log`다.

## 상태

비교 검증 실행 준비 완료. 결과는 results.md, summary.json, matches.json에 기록한다.

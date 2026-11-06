# Yoo/e16: BSM1을 이용한 별도 검증 환경의 연결 검사

## 목적과 범위

Avedore 데이터로 만든 대리 모델의 장기 예측 오차와 제어 기법의 효과를 분리할 수 있는지 검사한다. 기존 e01 원본 재현, e02부터 e15까지의 실제 데이터 적용과 진단은 보존한다. FCSRL 표현 학습, PPO 적용, e06 CVaR 확장 코드를 수정하지 않는다. PPO 전환이나 시뮬레이터 추가 자체를 새 확장이라고 부르지 않는다.

이번에는 공정 시뮬레이터 안정화, 제어 입력 반응, 단위, reset과 기존 agent의 추론 연결까지만 확인한다. RL 정책 학습이나 성능 우열 검증은 아니다. 개인 탐색 단계이며 팀 공통 PROTOCOL과 채택 결정은 변경하지 않는다.

## 출처와 고정 버전

- Python 구현: [FAU-EVT/bsm2-python](https://github.com/FAU-EVT/bsm2-python), PyPI `bsm2-python==0.0.16`, BSD-3-Clause. 사용 고지는 [BSM_LICENSE](BSM_LICENSE).
- 검토한 upstream 커밋: `73caa8bde193c0dd8a8803dfcd67b546501c950e`. 실제 실행은 PyPI 배포본이며 주요 실행 파일 SHA256은 summary.json에 기록한다.
- 원본 정의: [IWA BSM1 기술 보고서](https://iwa-mia.org/wp-content/uploads/2018/01/BSM_TG_Tech_Report_no_1_BSM1_General_Description.pdf).
- 유입 데이터: [IWA Task Group 배포 저장소](https://github.com/wwtmodels/Benchmark-Simulation-Models), `BSM1_v1`의 [BSM1.zip](https://github.com/wwtmodels/Benchmark-Simulation-Models/releases/download/BSM1_v1/BSM1.zip).
- ZIP SHA256: `f1b7683596512f2e432bb616d20fdc85ffdef3006e27a170b562789d30a050aa`. 원본은 서버에 보관하고 Git에는 넣지 않는다.
- 배포본 `BSM1/influent_files/convert.m`과 같은 열 변환으로 ASCII 자료를 ASM1 입력에 연결한다. 온도는 15 C, dummy 5개는 0이다. 유입 유량 단위는 m3/day다. 최종 15분은 마지막 관측값을 유지하며, 원본 데이터 수정이나 보간은 하지 않는다.
- IWA/Lund University의 자료와 구현 개발 기여를 인정한다. 원본 이용 안내는 ZIP 안 `Documents/BSM1_agreement_LU.pdf`에 있다. 외부 연락은 하지 않았다.

## 환경과 검사 설계

BSM2-Python 패키지 안의 BSM1OL을 사용한다. 5개 ASM1 반응조와 침전조로 구성되며, 에너지 설비까지 포함하는 BSM2 전체 공정으로 확대하지 않는다.

| 항목 | 설정 |
| --- | --- |
| 행동 | 5번 반응조 KLa, [-1, 1]을 [0, 240] day^-1로 변환 |
| 나머지 폭기 | KLa1=0, KLa2=0, KLa3=240, KLa4=240 |
| 내부 순환, 반송, 잉여 슬러지 | 원본 고정값 유지 |
| 제어 간격 | 5분, 내부 적분 간격 1분 |
| 비교 행동 | KLa5=0, 기본값 84, 240 |
| 안정화 | 원본 상수 유입 200일, 15분 간격. upstream MATLAB 기준값과 rtol=atol=1e-5로 비교 |
| 사전 조건화 | 안정화 후 건기 14일, 각 날씨 첫 7일을 기본 폭기로 진행 |
| 평가 | 같은 초기 상태에서 각 고정 행동으로 날씨 파일 후반 7일 연속 진행 |
| 관측 | 방류 NH4, 5번 조 DO/NO3/NH4/TSS, 현재 유입 유량/NH4/온도, 시간 sin/cos, 직전 KLa |
| 보상 | 해당 5분의 전체 폭기 에너지 음수, 기본 폭기 5분 에너지로 나눔 |
| 비용 | 내부 적분 시점 중 방류 NH4 > 4 mg N/L인 비율. 에피소드 비용/T가 위반 시간 비율 |
| DO | 5번 조 평균 및 2 mg/L 초과 시간 보고. 2 mg/L는 기본 제어 목표이며 위반 비용에 넣지 않음 |
| 에너지 | 원본의 폭기 전력 kW를 시간 적분한 kWh/day. 혼합 에너지도 별도 기록 |
| 수질 집계 | NH4 시간 평균, 유량 가중 평균, 위반 시간 비율, 관측된 상향 교차 수 |
| 적분 민감도 | 같은 초기 상태에서 건기 기본값 제어를 30초 적분으로 반복 |
| 기존 agent | PPO, FCSRL+PPO, FCSRL+PPO+CVaR 각각 seed 0으로 초기화, 12전이 추론. 최적화 갱신 0회 |

원본 보고서의 NH4 기준은 유량 가중 평균과 위반 시간 등을 함께 평가한다. 이 연결 시험의 시간 비율 비용만으로 전체 BSM 수질 기준을 만족했다고 말하지 않는다. COD, 총질소, TSS, BOD의 제약을 충족했다는 주장도 하지 않는다. 전체 공정 전력이나 실제 측정 전력을 보고한 것이 아니다.

관측은 잡음과 지연이 없는 이상적 측정이다. actuator 지연도 아직 없다. 반응조의 숨은 생물학적 상태 전체를 관측하지 않으므로 완전 관측 MDP라는 보장은 없다. 마지막 7일에만 시험 행동을 적용하므로 14일 전체를 후보 제어기로 운전하는 공식 BSM 비교 절차와도 구분한다.

5% 위반 시간 예산은 agent 생성 API의 척도를 점검하기 위한 예시일 뿐, 본 학습의 확정 제약이나 교수자 요구 조건이 아니다. 실제 채택할 예산은 기준 제어와 달성 가능성을 확인한 뒤 별도 고정해야 한다.

## 실행

기존 `/home/jio/fcsrl_src/.venv`의 torch와 패키지는 변경하지 않았다. 필요한 네 패키지만 별도 경로에 설치하고 해당 실행에서 PYTHONPATH로 추가한다.

```bash
uv pip install --python /home/jio/fcsrl_src/.venv/bin/python \
  --target /home/jio/fcsrl_src/bsm-deps-e16 --no-deps \
  -r Yoo/experiments/e16/requirements-extra.txt

curl -Lsf https://github.com/wwtmodels/Benchmark-Simulation-Models/releases/download/BSM1_v1/BSM1.zip \
  -o /home/jio/fcsrl_src/results/e16/BSM1.zip

BSM1_ARCHIVE=/home/jio/fcsrl_src/results/e16/BSM1.zip \
PYTHONPATH=/home/jio/fcsrl_src/bsm-deps-e16 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/home/jio/fcsrl_src/.venv/bin/python -u Yoo/experiments/e16/preflight.py \
  --out /home/jio/fcsrl_src/results/e16/preflight_v2
```

출력 경로는 없어야 한다. 재실행은 새 경로를 지정한다. Numba 첫 컴파일 시간은 별도 비용이며 반복 학습 처리량에 그대로 넣지 않는다. raw 상태, 시간별 출력과 전체 로그는 `/home/jio/fcsrl_src/results/e16/`에 보관한다.

## 사전 검사에서 발견하고 해결한 문제

1. 별도 경로의 Numba 0.60.0은 Splitter JIT에서 예외 직렬화에 실패했다. 기존 numpy 1.23.5를 지원하는 Numba 0.63.1, llvmlite 0.46.0으로 별도 경로만 변경해 구동했다.
2. PyPI 및 검토한 upstream의 raininfluent.csv에는 `30.044.50` 같은 잘못된 숫자가 있고, 유량에 천 단위 구분으로 보이는 점이 섞여 있다. 단순히 점을 지우거나 보간하지 않았다. IWA 원본 ASCII와 원본 변환식을 사용하도록 변경했다. 건기와 우기 모두 같은 출처로 고정했다.
3. 초기 반응조 상태를 그대로 쓰면 미생물 안정화 전의 수질을 평가하게 된다. 모든 비교에서 충분한 안정화와 같은 사전 운전 상태를 사용한다.
4. 기존 Avedore의 DO 상한 2와 하루 비용 한도 15를 이 환경으로 복사하지 않았다. 물리 단위와 목적이 다르다.

최초 실행 커밋 `031eed5`는 우기 CSV 파싱에서 중단됐다. 원본 데이터 연결 수정 후 실행 커밋은 `a8ed1be`이며 최종 결과의 전체 커밋은 summary.json에 기록한다.


## 결과와 후속 작업

**2026-11-06 연결 검사 완료.** [전체 결과](results.md), [기계 판독 요약](summary.json). 6개 고정 제어 시나리오와 적분 민감도 검사, 기존 agent 3종 추론을 완료했다. 최적화 갱신은 없으며, 본 RL 학습은 아직 시작하지 않았다.

최대 폭기에서 NH4 유량 가중 평균은 건기 2.530, 우기 3.144 mg N/L로 4 미만이다. 따라서 적어도 이 두 시나리오의 평균 NH4 기준을 만족하는 고정 제어가 존재한다. 그때도 초과 시간은 16.67%, 26.08%다. 임의의 5% 시간 예산을 본 학습에 옮기지 않는다.

후속 비교에서 고정할 내용:

1. 원본 BSM의 평균 NH4 4 mg N/L 기준과 일치하는 비용 및 단위를 정하고, 순간 초과 시간과 위험 꼬리를 함께 보고한다. 현재 wrapper의 시간 비율 cost는 연결 진단용이다. 평균 기준으로 바꾸려면 collector, Lagrange 예산, CVaR 척도, FCSRL feasibility 분포의 범위를 함께 점검해야 하며 기존 한도 15를 복사하지 않는다.
2. 고정 기본 폭기 외에 DO 목표 2 mg/L의 피드백 제어 기준을 추가한다. 같은 초기 상태와 유입 조건에서 비교하고, 후보 제어기를 전체 운전 구간에 적용한 뒤 정해진 후반 구간을 평가한다.
3. PPO와 FCSRL+PPO, CVaR 항을 붙인 각 대조를 같은 수집량, seed, 관측과 제약으로 비교한다. 원본 FCSRL 표현 손실은 유지한다. 먼저 짧은 최적화 연결 시험으로 비용 스케일, clipping, lambda, 유한 경사를 확인한다.
4. 건기와 우기는 이미 사전 검사에 사용했으므로 완전히 미관찰한 최종 시험으로 부르지 않는다. 폭풍 시나리오의 정책 성능은 아직 평가하지 않았다. 데이터 분할과 모델 선택 규칙을 정책 학습 전에 고정한다.
5. 실제 역전파와 평가가 포함된 측정 후 본 실행 예산과 종료 예상 시간을 산정한다. 현재 환경 속도에서 40만 전이당 약 25분이라는 값은 단일 프로세스 환경 계산만의 외삽이다.

이 작업은 실제 데이터 적용을 대체하지 않는 보조 검증이다. 기존 환경에서의 결과와 새 환경의 kWh 수치를 한 표의 동일 조건 비교처럼 섞지 않는다. 개선이 없으면 확장 후보의 음성 결과로 남긴다.

표와 그림 재생성:

```bash
/home/jio/fcsrl_src/.venv/bin/python Yoo/experiments/e16/report_preflight.py \
  --summary /home/jio/fcsrl_src/results/e16/preflight_v2/summary.json \
  --out Yoo/experiments/e16
```

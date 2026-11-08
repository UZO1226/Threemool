# Lee/e01: 안전 비용을 평균 농도에서 순간 위반 지시값으로 바꾼 비교

## 목적

과제 4단계(기법 적용과 확장)에 해당한다. FCSRL 논문의 전제 하나를 BSM1 폭기 제어에
맞춰 바꾸고 원래 설정의 결과와 비교한다. 바꾸는 것은 안전 비용의 정의 한 가지다.

FCSRL 의 feasibility 점수는 `F = E[max_t gamma^t c_t]` 이고, 논문 명제 4.1 은 비용이
0 또는 1 일 때 `1 - F` 가 앞으로 계속 안전할 확률이 된다고 말한다. 즉 이 방법의
이론적 근거는 이진 비용을 전제로 한다.

Yoo/e17 은 같은 BSM1 환경에서 비용을 연속값(정규화한 NH4 부하)으로 두고 하루 평균
NH4 <= 4 mg N/L 를 제약으로 걸었다. 그 결과 평균 기준은 지켰지만 FCSRL + PPO 의
순간 NH4 초과 시간은 건기 약 40%, 우기 약 47% 였다. 연속 비용에서는 feasibility 가
확률로 해석되지 않고, 제약도 순간 위반을 직접 다루지 않는다.

그래서 이 실험은 비용을 논문의 전제대로 이진값으로 바꾼다.

## 가설 (사전 등록)

- H1. 이진 비용에서 FCSRL + PPO 는 PPO-Lag 보다 순간 위반 시간이 짧다. 같은 에너지
  수준에서 비교한다.
- H2. FCSRL 과 PPO 의 차이는 평균 비용 설정(Yoo/e17 방식)보다 이진 비용 설정에서 더
  크다. 즉 FCSRL 의 이점은 비용이 논문 전제에 맞을 때 더 잘 드러난다.
- H3 (거의 정의상 성립 예상). 이진 비용 설정은 평균 비용 설정보다 순간 위반 시간을
  줄인다. 이것은 제약을 그렇게 걸었기 때문이므로 기법의 성과로 세지 않는다.

가설이 기각되는 결과도 그대로 기록한다.

## 바꾼 것과 바꾸지 않은 것

| 구분 | mean (기준, Yoo/e17 방식) | binary (Lee 의 변경) |
| --- | --- | --- |
| 한 스텝 비용 | 정규화한 NH4 부하 (연속값) | 5분 구간에 NH4 > 4 인 시점이 있으면 1, 없으면 0 |
| 하루 비용 | 유량 가중 평균 NH4 / 4 x 288 | 위반한 제어 스텝의 개수 |
| 하루 한도 | 288 (평균 NH4 <= 4 mg N/L) | 72 스텝 (하루 288 스텝의 25%) |
| PID 계수 | 기본값 x 15/288 | 기본값 x 15/72 |
| feasibility 해석 | 명제 4.1 적용 안 됨 | `1 - F` 가 계속 안전할 확률 |

나머지는 모두 같다. 환경(BSM1, 5분마다 KLa5 0~240), 관측 11차원, 보상(폭기 에너지의
음수), 하루 에피소드 경계, 초기 상태와 관측 정규화 통계, 신경망 구조, 학습량, seed,
평가 방법, PPO 와 FCSRL 구현이 같다. 원본 FCSRL 과 Yoo 의 코드는 수정하지 않는다
(`Lee/code/bsm/README.md` 의 출처와 sha256 표, `verify.py` 가 일치를 검사한다).

PID 계수 환산은 Yoo/e17 과 같은 규칙이다. 한도가 달라도 같은 상대 제약 오차에 같은
승수 응답을 주기 위한 단위 변환이며, 결과를 본 뒤 바꾸지 않는다.

## 이진 비용 한도 72 스텝의 근거

학습 전에 측정한 기준 제어 결과다 (건기, 평가 구간 7~14일, RL 결과를 보기 전에 고정).

| 기준 제어 | 폭기 kWh/day | 평균 NH4 | 순간 위반 시간 | 하루 위반 스텝 |
| --- | --- | --- | --- | --- |
| 기본 폭기 KLa5=84 | 3,341.4 | 4.831 | 63.24% | 182.1 |
| 최대 폭기 KLa5=240 | 4,265.6 | 2.308 | 15.82% | 45.6 |
| DO PI (목표 2 mg/L) | 3,717.8 | 2.567 | 17.11% | 49.3 |

최대 폭기에서도 건기 위반 시간이 15.8% 이므로 그보다 낮은 한도는 물리적으로 불가능
하다. 한도 72 스텝(25%)은 최대 폭기의 하한 45.6 과 기본 폭기 182.1 사이에 있어서
에너지와 안전 사이에 실제 교환 관계가 생기고, Yoo/e17 의 FCSRL + PPO 가 보인 약 40%
보다 뚜렷하게 엄격하다. 우기의 물리적 하한은 66.3 스텝(23.0%)이라 한도 72 에 거의
붙어 있다. 학습과 제약은 건기 조건에서만 하므로 우기는 분포 밖 점검으로만 본다.

## 사전 고정 조건

| 항목 | 값 |
| --- | --- |
| 비용 모드 | mean, binary |
| 방법 | PPO-Lag, FCSRL + PPO |
| seed | 0, 1, 2 |
| 실행 수 | 2 x 2 x 3 = 12 |
| 학습량 | 각 57,600 전이 (5,760 전이 x 10 갱신) |
| 수집 | 4환경 x 1,440 전이, 갱신마다 완결된 하루 20개 |
| 학습 데이터 | 건기 0~6일의 하루 시작 상태. 날 선택 RNG 는 seed 가 같으면 모든 조건에서 같다 |
| 평가 | 학습 전과 학습 후 정책을 같은 초기 상태에서 14일 연속 운전하고 후반 7일 집계. 건기와 우기 |
| 선택 | 마지막 정책만 보고한다. 중간 체크포인트나 설정을 결과를 보고 고르지 않는다 |
| PPO | actor 3e-4, critic 1e-3, gamma .99, GAE .95, clip .2, 10 epoch, batch 512, KL 조기 종료 .02, gradient clip .5, 초기 log_std -1.5, lambda 상한 20 |
| FCSRL | Yoo/e05 의 원본 encoder 학습 메서드, 동역학 및 feasibility 손실, EMA .05, feasibility discount .9, symlog 63구간 [-2,4] |
| 미사용 | CVaR 확장, Avedore 대리 모델, 폭풍 시나리오, 중간 체크포인트 선택 |

## 측정 지표 (두 모드에서 똑같이 측정)

- 폭기 에너지 kWh/day
- 유량 가중 평균 NH4 (mg N/L)
- 순간 위반 시간: 제어 스텝 기준 비율과 하루 위반 스텝 수. 1분 표본 기준 비율도 함께
  적어 Yoo/e17 의 수치와 비교할 수 있게 한다
- 하루 평균 NH4 의 상위 20% CVaR, 하루 위반 스텝의 상위 20% CVaR
- 각 모드의 자기 한도 만족 여부

주 비교는 각 비용 모드 안에서 FCSRL 과 PPO 의 차이다. 두 모드는 서로 다른 제약 문제를
풀므로 모드끼리 우열을 직접 가리지 않는다. 또 방법마다 초기 정책이 다르므로(FCSRL 의
정책망은 입력에 z 가 붙어 구조가 다르다) 자기 초기 정책 대비 변화량을 함께 적는다.
이는 Yoo/e17 이 남긴 한계를 반영한 것이다.

## 실행 환경

Windows 11, CPU 전용, conda 환경 `threemool` (Python 3.11.15). 주요 버전은 torch
2.15.0+cpu, numpy 1.26.4, scipy 1.17.1, numba 0.68.1, pandas 2.2.3,
bsm2-python 0.0.16 이다. 실행마다 한 스레드를 쓰고 6개를 병렬로 돌린다.

```bash
conda create -y -n threemool python=3.11
conda activate threemool
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "numpy==1.26.4" "pandas==2.2.3" scipy tqdm pyyaml gymnasium matplotlib cloudpickle wandb
pip install --no-deps bsm2-python==0.0.16 numba llvmlite control==0.10.2
git clone https://github.com/czp16/FCSRL.git
cd FCSRL && git checkout d882b91054d52c7a63a18a21f7db67bbbed82b63 && pip install -e .
curl -Lsf https://github.com/wwtmodels/Benchmark-Simulation-Models/releases/download/BSM1_v1/BSM1.zip -o BSM1.zip
```

원본 FCSRL 은 수정하지 않는다. Windows 에서는 safety_gymnasium 설치가 pygame 빌드
실패로 불가능한데, 이 실험은 Safety Gymnasium 과제를 쓰지 않으므로
`Lee/code/bsm/fcsrl_compat.py` 가 사용하지 않는 이름 하나만 자리표시자로 등록한다.

BSM1.zip 의 sha256 은 `f1b7683596512f2e432bb616d20fdc85ffdef3006e27a170b562789d30a050aa`
이며 Yoo/e16 이 기록한 값과 같다. 정상상태는 upstream 의 MATLAB 기준값과 최대 절대
오차 8.7e-08 로 일치했다.

## 실행 명령

```bash
export BSM1_ARCHIVE=.../BSM1.zip
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u Lee/experiments/e01/prepare.py --out <결과>/e01_prepared
python -u Lee/experiments/e01/verify.py --prepared <결과>/e01_prepared --out Lee/experiments/e01/verification.json
python -u Lee/experiments/e01/run_all.py --prepared <결과>/e01_prepared --out <결과>/e01_runs \
  --updates 10 --parallel 6 --upstream <원본 FCSRL 경로>
python -u Lee/experiments/e01/report.py --runs <결과>/e01_runs --prepared <결과>/e01_prepared \
  --out Lee/experiments/e01
```

검사가 통과한 뒤에만 비교 학습을 실행한다. 원시 전이, 체크포인트, 전체 로그는 저장소
밖(`IoT_project2/results/`)에 두고 작은 요약만 Git 에 기록한다 (BRANCH_RULES 6절).

## 미리 밝히는 한계

- 10 갱신은 수렴한 학습이 아니다. seed 3개의 탐색적 비교이며 기법의 우월성을 증명하는
  본 실행이 아니다.
- 방법 간 초기 정책이 완전히 같지 않다. 구조가 다르기 때문이며, 변화량을 함께 본다.
- 평균 기준 만족이 모든 시점의 안전을 뜻하지 않는다. 이진 비용의 한도 역시 하루 평균에
  대한 기대 제약이라 모든 날의 만족을 보장하지 않는다.
- BSM1 은 물리 시뮬레이터이고 실제 공정이 아니다. NH4 하나만 제약하며 전체 수질
  안전성을 다루지 않는다.
- 결과를 본 뒤 한도, 가중치, 학습량을 바꾸지 않는다. 바꾸면 새 실험 ID 로 기록한다.

## 실행 결과

실행 후 [results.md](results.md) 와 [summary.json](summary.json) 에 기록한다.

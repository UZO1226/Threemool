# Park/code

박성현이 직접 작성하는 BSM1 폭기 제어와 FCSRL 확장 코드다. 외부 코드와 참고한 팀원 코드는 [SOURCE.md](SOURCE.md) 에 적는다.

## 원칙

- 원본 FCSRL 과 `bsm2-python` 은 설치해서 그대로 쓰고 고치지 않는다.
- 환경 연결, 학습, 안전 차단은 직접 작성한다.
- 팀 결과와 비교할 수 있도록, 직접 만든 환경이 아래 기준값을 재현하는지 먼저 확인한다. 재현하지 못하면 학습으로 넘어가지 않는다.

## 파일 구성

| 파일 | 하는 일 |
| --- | --- |
| `env.py` | BSM1 연결. 5분마다 5번 조 KLa(0 ~ 240) 를 받고, 관측 11개, 보상(폭기 에너지의 음수), 이진 비용을 돌려준다 |
| `baselines.py` | 기준 제어. 고정 KLa, DO PI |
| `evaluate.py` | 14일 연속 운전, 후반 7일 집계. 에너지, 평균 NH4, 순간 위반, CVaR, 차단 개입 비율 |
| `prepare.py` | 정상상태, 평가 시작 상태, 학습용 하루 시작 상태 7개, 관측 정규화 자료를 만든다 |
| `check_env.py` | 기준 제어 3개가 아래 기준값을 재현하는지 검사한다 |
| `agent.py` | PPO-Lag 와 FCSRL + PPO-Lag. FCSRL 손실은 원본의 `train_encoder` 를 불러 쓴다 |
| `train.py` | 학습과 학습 전후 평가 |
| `shield.py` | 안전 차단 (Park/e01 의 확장). F 차단과 대조군 두 개 |
| `avedore_env.py` | Kaggle Avedøre 폭기 환경, 여러 에피소드를 한꺼번에 굴리는 배열 버전 (Park/e07). Yoo 의 대리 모델 가중치를 읽는다 |
| `check_avedore.py` | Avedøre 환경이 Yoo/e02 의 실제 운전 재생 지표 12개를 재현하는지 검사한다 |
| `train_avedore.py` | Avedøre 에서 PPO-Lag, FCSRL + PPO-Lag 학습 (Yoo/e05, e06 과 같은 설정) |
| `darrow_env.py` | 팀 공식 데이터 Kaggle DARROW 폭기 환경, 배열 버전 (Park/e08). Lee 의 방법으로 만든 단조 제약 GBM 대리 모델을 읽는다 |
| `check_darrow.py` | DARROW 환경이 같은 대리 모델의 Lee 코드 결과 15개를 재현하는지 검사한다 |
| `train_darrow.py` | DARROW 에서 PPO-Lag, FCSRL + PPO-Lag 학습 (Lee/e07 과 같은 설정). `--brake-theta` 를 주면 학습 중에도 NH4 규칙 차단을 켠다 (Park/e09). `--min-brake-target` 는 최소 개입 차단을 학습 중에 켜고, `--cost-limit` 은 학습 비용 한도를 바꾼다 (Park/e12). `--intervention-cost` 는 차단이 개입한 스텝마다 학습 비용을 더한다 (Park/e13). `--brake-trim` 은 양방향 차단을 쓴다 (Park/e14). `--adaptive-margin` 은 불확실성 여유 표를 쓴다 (Park/e15). `--save-every k` 는 k 업데이트마다 중간 정책을 저장한다 (Park/e21). `--actor-lr`, `--epochs` 는 정책 학습률과 업데이트마다 자료를 도는 횟수를 바꾼다 (Park/e24) |
| `min_brake.py` | DARROW 최소 개입 차단. 대리 모델로 앞을 내다보고 예측 NH4 를 기준 아래로 유지하는 데 필요한 만큼만 공기량을 올린다 (Park/e10). `do_cap=True` 이면 DO 상한도 함께 본다 (Park/e11). `trim=True` 이면 DO 가 넘칠 때 NH4 에 필요한 공기량까지 내리기도 한다 (Park/e14). `margins` 는 고정 target 대신 예측 불확실성에 맞춘 여유를 쓴다 (Park/e15). `do_limit`, `futile` 은 DO 여유와 헛수고 방지 (Park/e16). `recover=(H, lim)` 은 나중에 최대 공기로 NH4 를 되돌릴 수 있는지 보고, 안 되면 미리 공기를 넣는다 (Park/e17). `conflict=m` 은 NH4 여유와 DO 가 부딪칠 때 여유를 m 으로 줄인다 (Park/e19) |

2026-11-09 확인: `prepare.py` 의 MATLAB 정상상태 최대 절대 오차 8.68e-08, `check_env.py` 기준값 12개 모두 통과 (`../experiments/e01/check_env.json`). `train.py --updates 1` 과 차단기 3종의 연결 시험도 끝까지 돌았다. 연결 시험 수치는 결과가 아니다.

## 실행 순서

```bash
cd Park/code
export BSM1_ARCHIVE=<BSM1.zip 경로>
export WANDB_MODE=disabled
python prepare.py --out <결과>/prepared                                  # 1. 시작 상태 만들기
python check_env.py --prepared <결과>/prepared --out ../experiments/e01/check_env.json   # 2. 통과해야 다음으로
python train.py --method fcsrl --seed 0 --prepared <결과>/prepared --out <결과>/runs/fcsrl_s0  # 3. 학습
```

`<결과>` 는 저장소 밖 폴더로 둔다 (상태 파일과 체크포인트는 Git 에 넣지 않는다).

## 환경 검증 기준값

출처는 `Lee/experiments/e01/results.md` (기준 제어, 학습 전, 건기, 14일 운전의 후반 7일) 다. 같은 평가 구간 정의로 재현해야 한다.

| 기준 제어 | 폭기 kWh/day | 평균 NH4 (mg N/L) | 순간 위반 % | 하루 위반 스텝 |
| --- | --- | --- | --- | --- |
| 고정 KLa5 = 84 | 3,341.4 | 4.831 | 63.24 | 182.1 |
| 고정 KLa5 = 240 | 4,265.6 | 2.308 | 15.82 | 45.6 |
| DO PI (목표 2 mg/L) | 3,717.8 | 2.567 | 17.11 | 49.3 |

Yoo/e16 의 같은 이름 수치(예: 최대 폭기 NH4 2.530)는 평가 구간이 달라 위 값과 다르다. 비교 기준은 Lee/e01 표로 한다.

## 실행 환경

확인에 쓴 환경 (2026-11-09, macOS, CPU): Python 3.13.5, torch 2.15.0, numpy 2.5.4, scipy 1.18.1, numba 0.68.1, bsm2-python 0.0.16. Lee/e01 (Python 3.11, numpy 1.26.4) 과 버전이 달라도 기준값은 같게 나왔다.

설치 명령:

```bash
python3 -m venv ~/threemool-venv && source ~/threemool-venv/bin/activate
pip install torch numpy scipy gymnasium tqdm matplotlib pandas wandb cloudpickle pyyaml
pip install --no-deps bsm2-python==0.0.16 numba llvmlite control==0.10.2
git clone https://github.com/czp16/FCSRL.git ~/FCSRL
cd ~/FCSRL && git checkout d882b91054d52c7a63a18a21f7db67bbbed82b63 && pip install --no-deps -e .
curl -Lf https://github.com/wwtmodels/Benchmark-Simulation-Models/releases/download/BSM1_v1/BSM1.zip -o ~/BSM1.zip
shasum -a 256 ~/BSM1.zip   # f1b7683596512f2e432bb616d20fdc85ffdef3006e27a170b562789d30a050aa 이어야 한다
```

`safety_gymnasium` 은 설치하지 않아도 된다. `agent.py` 가 원본 fcsrl 을 가져올 때 필요한 이름 하나만 빈 자리로 등록한다.

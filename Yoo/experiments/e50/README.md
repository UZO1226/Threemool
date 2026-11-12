# e50: 공통 평가 조건 v1.1 기준 대리 모델에서 Yoo 세 조건 재측정

2026-11-12 시작. test 구간 사용 없음.

## 목적

Yoo/e30 ~ e46 은 Yoo 서버의 scikit-learn 1.7.2 로 만든 대리 모델(검증 재생 27.37)을 썼다. 공통 평가 조건 v1.1 은 scikit-learn 1.9.1 로 만든 모델(검증 재생 28.77)을 기준으로 정했으므로, Yoo 결과는 그대로는 Lee, Park 결과와 같은 표에 놓을 수 없다. 기준 모델 환경에서 Yoo 의 세 조건을 다시 학습해 팀 비교에 쓸 수 있는 값을 만든다.

## 조건

| 항목 | 내용 |
| --- | --- |
| 환경 | PROTOCOL v1.1 그대로. 날짜순 188 / 40 / 41일, 단조 GBM 5구성원, 관측 11개, 총 공기량 행동, 비용 NH4 > 4 와 DO > 2 스텝 합, 한도 12 |
| 준비물 | e30 `prepare.py` 를 scikit-learn 1.9.1 환경에서 실행 (`/home/jio/fcsrl_src/results/e50/prepared`). 기록 운전 재생이 검산값과 같다: 검증 28.77, NH4 28.48%, DO 1.80%, 학습 12.17 (`replay_check.json`) |
| 방법 | PPO-Lag, FCSRL + PPO-Lag, FCSRL + Actor 보정 (Yoo/e29, e30 의 `fcsrl_actor`) |
| 학습 | e30 `train.py` 를 수정 없이 부른다. 기록 운전 모방 초기 정책 + 잔차 정책, 초기 log std −1.6, 8환경 x 760 x 66 = 401,280 전이 |
| seed, 평가 | seed 0 ~ 4, 마지막 정책, 검증 40일 x 구성원 5개 = 200 에피소드, 결정적 행동 |

Lee/e07, Park 과 다른 점은 초기 정책(기록 운전 모방)과 초기 log std 다. 환경, 예산, seed, 평가는 같다.

### 실행 환경

`/home/jio/fcsrl_src/.venv-skl191`: Python 3.11, scikit-learn 1.9.1, numpy 1.26.4, pandas 2.2.3, torch 2.14.1+cu130 (서버 표준 방식 `uv pip install torch --torch-backend=cu130`), gymnasium 0.28.1, 원본 FCSRL d882b91 (`pip install -e`, 수정 없음). safety-gymnasium 1.0.0 은 Python 3.11 에서 import 오류가 나서, 원본 FCSRL 의 `fcsrl.env.gym_utils` 가 import 하는 `VisionEnvConf` 하나만 담은 대체 모듈을 venv 에 두었다. DARROW 학습은 Safety Gymnasium 과제를 쓰지 않는다.

병렬 6개로 돌린다. 처음 실행에서 OpenMP 대기 스레드가 CPU 를 잡아 갱신 한 번에 100초 이상 걸려, `OMP_WAIT_POLICY=PASSIVE` 를 더해 다시 시작했다 (중단한 실행은 `main_aborted_spin`, 결과에 쓰지 않음).

## 판정 (결과를 보기 전에 정함)

- H1 Yoo 확장: FCSRL + Actor 보정이 FCSRL + PPO-Lag 보다 seed 짝 비교에서 비용이 낮은 seed 가 4/5 이상이고, 평균 비용 차이 < 0 이며, 평균 공기량 차이가 +2% 이하이면 "기준 모델에서도 Actor 보정 이점이 나타남" 으로 적는다
- Q2 모델 민감도: 세 방법의 평균 비용 순위가 e30 (1.7.2 모델: PPO-Lag 11.45 < Actor 보정 13.68 < FCSRL 16.58) 과 같은지 적는다. 다르면 Yoo 의 이전 순위 주장은 대리 모델 버전에 민감하다고 적는다
- 팀 비교: Lee/e07, Park/RESULT.md 의 같은 조건 수치와 나란히 적되, 초기 정책이 달라 방법 우열로 해석하지 않는다. NH4 와 DO 위반률은 따로 적는다

## 실행

```bash
cd Yoo/experiments/e30
PY=/home/jio/fcsrl_src/.venv-skl191/bin/python
PYTHONPATH=$PWD/../../code/wwtp $PY prepare.py --raw <원자료 폴더> --out /home/jio/fcsrl_src/results/e50/prepared
cd ../e50
PYTHONPATH=$PWD/../../code/wwtp:$PWD/../e30 $PY run.py --prepared /home/jio/fcsrl_src/results/e50/prepared --out /home/jio/fcsrl_src/results/e50/main --jobs 6
$PY report.py --root /home/jio/fcsrl_src/results/e50/main --out .
```

## 결과

<!-- RESULTS -->

# Yoo/e01 FCSRL 원본 재현

수행계획서의 "구현체 구동" 단계다. 공개 구현체로 논문 Table 4 (FCSRL + TD3 Lag) 결과를 재현한다. 폐수처리 환경 적용 결과와 섞지 않는다.

## 가설과 기준

- 기준: Cen et al., "Feasibility Consistent Representation Learning for Safe Reinforcement Learning", ICML 2024, Table 4
- 가설: 원본 코드와 원본 설정 그대로 돌리면 논문 값의 seed 편차 범위 안에 들어온다
- 바꾼 요소: 없음. 실행기(`run_repro.py`)가 원본 스크립트를 그대로 부르고, 저장 경로만 바꾼 설정 사본을 넘긴다

## 원본 구현체

| 항목 | 내용 |
| --- | --- |
| 저장소 | https://github.com/czp16/FCSRL |
| 커밋 | `d882b91054d52c7a63a18a21f7db67bbbed82b63` (2024-07-12) |
| 라이선스 | Apache 2.0 |
| 수정 사항 | 없음 |
| 이 서버의 위치 | `/home/jio/fcsrl_src/FCSRL` (저장소 밖) |

## 실행 조건

| 항목 | 값 |
| --- | --- |
| 스크립트 | `scripts/td3_repr_CMDP.py` (repr_type FCSRL) |
| 설정 | `hyper_params/TD3Repr_Lag.yaml` 원본 값 (25 epoch x 2500 gradient step, 2M 환경 스텝, 비용 한도 10) |
| 과제 | PointGoal1, PointButton1, PointPush1, PointGoal2, CarGoal1, CarButton1 (Safety Gymnasium v0) |
| seed | 100, 200, 300, 400, 500 (논문도 5개, 값은 공개되지 않음) |
| 평가 | epoch 마다 10 에피소드 |
| 장치 | RTX 3090 x2, 동시 3개 실행 |
| 패키지 | Python 3.10.19, torch 2.14.1+cu130, safety-gymnasium 1.0.0, gymnasium 0.28.1, mujoco 2.3.3, numpy 1.23.5 |
| 로그 | wandb offline 모드 (계정 없이 로컬 저장) |

환경 만들기 (원본 README 는 Python 3.8 conda 를 쓰지만 서버 표준 torch cu130 에 맞춰 3.10 으로 올렸다. pygame 2.1.0 휠이 3.10 까지만 있어서 3.11 이상은 안 된다):

```bash
cd /home/jio/fcsrl_src
git clone https://github.com/czp16/FCSRL.git && git -C FCSRL checkout d882b91
uv venv --python 3.10 .venv && source .venv/bin/activate
export UV_LINK_MODE=copy
uv pip install torch --torch-backend=cu130
uv pip install safety-gymnasium==1.0.0 numpy tqdm pyyaml wandb
uv pip install -e FCSRL
```

실행 (tmux 세션 안에서 돌리면 `tmux attach` 로 전체 진행 막대를 볼 수 있다):

```bash
python run_repro.py --fcsrl /home/jio/fcsrl_src/FCSRL --out /home/jio/fcsrl_src/results/e01 --jobs 3
```

중단해도 같은 명령으로 다시 돌리면 끝난 조합은 건너뛴다. 결과는 `--out` 아래 `logs/`, `model/`, `wandb/`, `test_by_epoch.csv` 에 생긴다.

## 결과

2026-11-04 19:32 에 시작해 2026-11-05 19:07 에 30회 모두 정상 종료했다 (약 23.5 시간, 동시 3개, 서버 CPU 를 다른 사용자와 나눠 씀).

값은 seed 5개의 평균 ± 표준편차(모표준편차)다. "마지막 5 epoch" 는 seed 마다 epoch 21 에서 25 의 시험 값을 평균한 뒤 seed 사이 통계를 냈다. 위의 맵 문제 때문에 epoch 하나의 시험 값은 사실상 에피소드 하나라서 마지막 5 epoch 평균을 대표값으로 쓴다.

| Task | 논문 reward | 논문 cost | 재현 reward (마지막 5 epoch) | 재현 cost (마지막 5 epoch) | 재현 reward (epoch 25) | 재현 cost (epoch 25) |
| --- | --- | --- | --- | --- | --- | --- |
| PointGoal1 | 24.4 ± 1.4 | 9.3 ± 5.8 | 23.8 ± 1.0 | 7.5 ± 4.6 | 23.8 ± 3.4 | 8.2 ± 16.4 |
| PointButton1 | 10.5 ± 2.3 | 10.1 ± 6.3 | 9.5 ± 1.6 | 12.8 ± 7.7 | 10.7 ± 2.3 | 14.2 ± 17.9 |
| PointPush1 | 11.8 ± 4.5 | 9.7 ± 8.1 | 11.5 ± 4.2 | 8.3 ± 6.1 | 13.1 ± 5.2 | 15.6 ± 10.0 |
| PointGoal2 | 13.5 ± 2.6 | 9.7 ± 7.8 | 10.3 ± 2.3 | 12.3 ± 4.9 | 9.9 ± 6.6 | 34.0 ± 32.6 |
| CarGoal1 | 27.6 ± 2.0 | 9.9 ± 6.4 | 26.1 ± 4.7 | 11.2 ± 3.6 | 25.7 ± 3.4 | 4.0 ± 8.0 |
| CarButton1 | 3.6 ± 1.3 | 10.4 ± 7.3 | 2.5 ± 1.2 | 12.8 ± 8.7 | 0.6 ± 3.2 | 0.0 ± 0.0 |

해석:

- PointGoal1, PointPush1, CarGoal1 은 reward 와 cost 모두 논문 값의 seed 편차 안이다. 재현됐다고 본다.
- PointButton1, CarButton1 은 reward 가 논문보다 약간 낮고 cost 가 약간 높지만 편차 범위 안이다.
- PointGoal2 는 reward 가 논문보다 3.2 낮다 (10.3 대 13.5, 편차 2.3 과 2.6). 6개 중 유일하게 차이가 뚜렷하다. 원인은 확인하지 않았다. 위 맵 문제로 학습 환경 다양성이 줄어드는 영향이 장애물이 많은 Level 2 에서 더 클 수 있다는 정도만 추정한다.
- 평균 cost 는 대체로 한도 10 근처다. Lagrangian 제약 처리가 원본대로 동작한다.

파일:

- `test_by_epoch.csv`: 과제, seed, epoch 별 시험 reward 와 cost (이 폴더, 실행기가 만든 원본의 사본)
- 전체 로그 336 MB, 모델 1.1 GB, wandb 오프라인 기록 537 MB 는 `/home/jio/fcsrl_src/results/e01/` 에 있고 Git 에 넣지 않는다

## 확인할 점

- 논문은 2M 스텝 학습 뒤의 값을 보고하지만 마지막 epoch 값인지 마지막 몇 epoch 평균인지 적혀 있지 않다. 두 방식 모두 계산해 둔다.
- 원본 로그의 TD3 `test_reward ±` 표준편차가 0 으로 찍힌다. 원인을 확인해 보니 `SubprocVectorEnv` 의 작업 프로세스들이 seed 없이 같은 난수 상태를 물려받아, 병렬 환경이 모두 같은 맵으로 초기화된다 (시험 환경 4개로 확인, 두 번 reset 해도 모두 동일). 그래서 epoch 마다 시험 10 에피소드가 사실상 같은 에피소드 하나이고 학습 환경 8개도 같은 맵을 본다. 또 환경 seed 가 `--seed` 와 연결되지 않아서 같은 seed 로 다시 돌려도 맵 순서가 같다는 보장이 없다. 재현 실험이라 고치지 않았다. 폐수처리 적용 단계에서는 작업 프로세스마다 다른 seed 를 주도록 고친다.
- seed 간 편차는 `test_by_epoch.csv` 로 따로 계산한다.
- 논문 Table 1 과 Table 4 의 FCSRL 값이 PointGoal2, PointButton1 에서 조금 다르다. Table 4 를 기준으로 삼는다.

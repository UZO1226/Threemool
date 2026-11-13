# e56: 고정 모방 prior 사용 여부 대조

**2026-11-13 09:40 KST 본 학습 시작, 10:02 GPU1 전환 실행 시작.** prior-off PPO/FCSRL 각 seed5개, 총10개다. 먼저 돌던 3개는 보존하고 대기 7개만 CPU1 수집/FCSRL GPU1 갱신으로 넘겼다. 최종 결과는 아직 없으며 현재 전체 상태는 `main_hybrid/status.json`에서 확인한다. [전환 검증과 장치별 해석 제한](HYBRID.md). [고정 조건](protocol.json).

## 질문과 단일 변경

기록 운전에서 학습한 고정 prior를 정책 출력에 계속 더하는 방식이 PPO/FCSRL의 공기량과 수질 개선을 제한하는지 확인한다. 이 prior는 행동을 강제로 묶는 제한이나 모방 벌점이 아니다. 원래 정책 평균은 `prior(s) + pi(s)`이며 `pi` 출력은 자유롭게 학습한다. 이번 대조는 **초기화뿐 아니라 학습 전체의 가산 prior 사용 여부**를 바꾼다.

원 prior를 같은 절차로 로드하고 마지막 층의 가중치와 편향만 0으로 만든다. 다른 가중치, 난수 상태, 정규화, 초기 log std -1.6, 학습률과 PID를 유지한다. 초기 residual head가 0이므로 prior-off의 초기 평균 행동은 정규화값 0, 총 공기 4,500 m³/h다. 이 시작 정책 변화도 비교의 일부이며 제어 변수에서 제거할 수 있는 혼입으로 취급하지 않는다.

기존 loader로도 복원되도록 영 출력 prior 자체를 checkpoint에 저장한다. FCSRL Encoder, dynamics/feasibility 손실, EMA, PPO와 비용 정의는 유지한다. Actor 보정, 조건별 제약, 불확실성 여유, grey-box 모델은 추가하지 않는다.

## 왜 e50 조건인가

문제 제기의 출발점인 Yoo와 팀 PPO 차이를 같은 환경에서 다루고, 기존 PPO/FCSRL 각 seed5개의 기준선을 재사용하기 위해 e50 조건을 고정했다. Yoo/e54의 수정 환경이나 6/6 한도로 옮기지 않는다. 팀 구현과 차이가 prior뿐이라는 가정도 하지 않는다. 이 실험은 Yoo 코드 안에서 prior의 영향만 비교한다.

| 항목 | 고정 조건 |
| --- | --- |
| 데이터 | Kaggle DARROW, e50 준비물, train188일/개발 val40일. test41일 행 추출/평가 없음 |
| 대리 모델 | e50 공통 v1.1 기준 GBM, scikit-learn 1.9.1 |
| 관측/행동 | 원 11차원, 총 공기 유량 하나. 기존 날짜별 구간 배분 |
| 비용 | NH4>4와 max(DO)>2 전이 수 합, 평균 한도12. NH4와 DO는 따로도 공개 |
| 학습 | PPO-Lag와 FCSRL+PPO-Lag, seed0~4, 401,280전이(66갱신), 마지막 정책 |
| 비교군 | e50 prior-on10개 고정 결과. 신규 prior-off10개 |
| 평가 | 결정적 행동, train/val 각각 날짜 x 모델5구성원, 95전이. 구성원은 독립 날짜가 아님 |
| 단위 | 공기 m³/23.75h, 초과 적분 mg h/L. 실제 전력 절감이 아님 |

학습 본체는 e30 `collect`, 기존 에이전트와 갱신을 재사용한다. `load_selected`는 train/val 행만 추출한다. 정책 평가는 e45의 검산된 상세 지표 함수를 사용하되 그 함수의 6/6 충족 항목을 제거하고 e50 합산 한도12로 계산한다. 기준선 상세 지표는 [e50 사후 검산](../e52/audit/README.md)의 정확 재현 결과를 사용한다.

## 결과 전에 고정한 판단

각 방법에서 같은 seed의 prior-off minus prior-on 공기량, NH4, DO를 우선 비교한다. 평균만 아니라 모든 seed의 원값을 공개한다. 다음을 모두 충족할 때만 전체 개선의 탐색 후보로 표시한다.

- 공기량, NH4, DO의 세 평균이 모두 감소한다.
- 최소 4/5 seed에서 세 지표가 모두 악화하지 않는다.
- NH4/DO 초과 적분과 최악 날짜 비용의 seed 평균이 증가하지 않는다.
- 합산 평균 비용12를 만족하는 seed 수가 감소하지 않는다.

수치 동등 허용오차는 공기0.001m³, 비용1e-9, 적분1e-8이다. 후보의 제약 한도는 완화하지 않는다. 반복 사용한 개발 val의 seed5 탐색이며 이 표식만으로 통계적 확증이나 공식 채택을 주장하지 않는다. 일부 지표만 좋아지면 부분 개선/교환관계로 설명한다. 결과 뒤 seed 제외, 학습량 연장 또는 prior/탐색 폭 탐색은 하지 않는다.

## 사전 검사

아래 검사를 모두 통과했다. 실행 코드 `7033994`, [검사 요약](preflight_checks.json). PPO와 FCSRL의 prior-on 첫 2갱신은 e50과 모든 로그 항목이 정확히 같았다. prior-off 두 조건도 2갱신과 기존 loader 복원 검사를 통과했다.

- e50 원 입력의 SHA256과 공유 학습 본체를 당시 커밋과 대조한다.
- 같은 seed에서 prior 이외 초기 가중치, 정규화, log std와 난수 상태가 정확히 같아야 한다.
- prior-on PPO/FCSRL seed0의 첫 2갱신 전체 로그가 각각 e50과 정확히 같아야 기준선을 재사용한다.
- prior-off도 2갱신과 checkpoint 복원을 확인한다. 학습 출력이 prior 없이 바뀔 수 있어야 한다.
- 보고서가 동일 결과 또는 공기량 악화/비용 개선을 전체 개선으로 표시하지 않는지 인공 결과로 검사한다.

## 실행과 기록

기존 환경 `/home/jio/fcsrl_src/.venv-skl191`을 사용하고 PyTorch를 변경하지 않는다. 서버의 다른 작업을 고려해 기존 작업을 포함하여 동시 학습 3개만 실행한다. **정정:** 원 실행은 `DeviceConfig`가 PyTorch 스레드를 4로 다시 설정했다. 신규 backend는 CPU1을 명시적으로 검사하고 FCSRL 갱신에만 GPU1을 사용한다. 학습 중 현재 갱신과 잔여 시간은 각 `progress.json`과 tqdm 로그에 저장하고, 10개가 끝나면 자동으로 표를 만든다.

```bash
cd /home/jio/threemool-prior-ablation
export OMP_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/jio/fcsrl_src/.venv-skl191/bin/python Yoo/experiments/e56/preflight.py --out <새 사전검사 경로>
/home/jio/fcsrl_src/.venv-skl191/bin/python -u Yoo/experiments/e56/run.py --preflight <검사 경로>/checks.json --out <새 본 실행 경로> --jobs 3
```

사전 검사: `/home/jio/fcsrl_src/results/e56/preflight/`.
원 실행: `/home/jio/fcsrl_src/results/e56/main/`, 커밋 `a173e3b`. **현재 실행:** `/home/jio/fcsrl_src/results/e56/main_hybrid/`, 커밋 `670ccd3`, tmux 세션 `yoo-e56-hybrid`. 기존 3개는 원 경로를 가리킨다. 기존 배차는 추가 작업을 시작하지 않으며 원 학습이 모두 끝나면 종료한다. 10:03 기준 종료 추정은 11:10~11:30 KST이며 서버 부하에 따라 달라진다.
위 명령은 원 CPU 실행의 재현용이다. 현재 이관 실행의 조건과 명령은 [HYBRID.md](HYBRID.md)를 따른다. 실행 커밋과 입력/소스 해시, 정확한 시작/종료 시각은 각 manifest에 남긴다. `status.json`이 전체 진행, 각 `runs/<method>/seed_<n>/progress.json`이 개별 진행이다. 완료 후 같은 루트에 `summary.json`과 `results.md`를 자동 생성한다.

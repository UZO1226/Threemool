# e58: 연속 NH4 초과 크기 보조 표현 학습

**본 학습 전 구현 및 검사 단계.** 사용자는 팀과 겹치는 실험을 피하되 새 아이디어는 실행하도록 지시했다. Lee/Park 문서 대조와 원 논문 방법 검토를 끝냈다. [중복 검토](../../analysis/TEAM_EXPERIMENT_REVIEW_20261113.md), [논문과 구현 대응](LITERATURE.md), [고정 조건](protocol.json).

## 질문과 변경 범위

기존 FCSRL이 배우는 위험 신호에 위반의 크기를 추가하면 공기량, NH4와 DO를 함께 개선할 수 있는지 시험한다. 현재 관측 정보가 완전히 사라졌다고 가정하지 않는다. 원본 FCSRL의 dynamics/feasibility 손실, Encoder, EMA, 원 관측과 잠재값을 함께 받는 Actor/Critic, PPO/PID와 고정 모방 prior를 유지한다.

온라인 Encoder의 상태-행동 잠재값 zsa 뒤에 64→64 Tanh→1 head만 추가한다. 목표는 현재 실행 행동 뒤 4개 전이에서 `max(NH4-4,0)`을 평균내고4로 나눈 값이다. 한 시간의 농도 초과 크기를 학습하는 회귀이며, e24의 한 시간 평균 이진 위반 비용과 다르다. 원래 비용을 연속값으로 바꾸지는 않는다. 따라서 연속 초과 적분을 정책이 직접 제약하게 됐다고 주장하지 않는다.

UNREAL의 보조 예측과 희소 사건 균형 학습을 참고해 유효한 양수/0 목표의 평균 MSE를 같은 비중으로 합친다. 한 그룹만 있으면 그 그룹만 쓴다. PPO 표본/비용에는 균형 가중치를 적용하지 않는다. 시간4, 단위 척도4, 가중치1은 결과 전 고정한다. 원 손실은 `.5*dynamics + feasibility`이며 여기에 보조 손실을 더한다. 추론에는 head 출력이 필요 없다.

## 대조와 데이터

| 조건 | 학습 |
| --- | --- |
| baseline | 기존 FCSRL |
| severity | 기존 FCSRL + 연속 NH4 초과 보조 손실 |
| detached | 같은 head와 목표를 학습하되 zsa를 detach, 보조 경사가 Encoder로 전달되지 않음 |

세 조건 모두 seed0~4, 401,280전이(66갱신), 마지막 checkpoint. e50 기준 대리 모델(scikit-learn1.9.1), 관측11, 총 공기 행동0~9000 m³/h, 비용 `I(NH4>4)+I(maxDO>2)`, 합산 평균 한도12. prior-on과 초기logstd-1.6을 유지한다. 비교의 목적은 추가 지도 신호이며 e56 prior-off는 쓰지 않는다.

Kaggle DARROW train188일, 개발 val40일. 각 평가 날짜마다 모델5구성원을 사용한다. test41일 행 추출/평가 없음. 보조 목표는 학습 날짜 대리 rollout의 실제 상태 배열에서 계산하며 정규화 잘림 후 관측으로 복구하지 않는다. 4전이 창은 하루 reset과 buffer 끝을 넘지 않고 마지막 종료 전이를 포함하는 완전한 창은 허용한다. 하루95개 중92개 창이 유효하다.

기존 e50 CPU4 정책을 이번 기준선으로 재사용하지 않고15개 모두 CPU1 수집과 물리 GPU1 갱신으로 새 학습한다. 최대 동시3개, GPU 갱신은 기존 잠금으로 직렬화한다. 기존 PyTorch cu130은 변경하지 않는다. 매 갱신 optimizer/PID/RNG까지 저장한다. detached는 추가 head 계산 대조지만 정확한 벽시계 예산 대조는 아니므로 시간도 보고한다.

## 판단과 사전 검사

severity를 baseline과 detached 각각에 대응 비교한다. 세 주 지표의 평균이 모두 감소하고4/5seed 이상에서 세 지표가 모두 악화하지 않아야 한다. NH4/DO 초과 적분과 최악 날짜의 평균, 한도 충족 seed 수가 악화하지 않아야 동시 개선 후보로 표시한다. 판정과 무관하게 모든 수치 및 paired seed 차이의 탐색적 bootstrap95% 구간을 공개한다. 결과를 보고 seed, 가중치, 학습량 또는 목표를 바꾸지 않는다.

사전 검사는 초기 공유 가중치/RNG 일치, 원 수집 buffer 정확 일치, 경계 목표와 균형 손실, detached의 한 갱신 뒤 기본 정책 일치, 보조 경사의 Encoder 전달, 유한 GPU 갱신, 저장/복원, 재개 뒤 다음 갱신 일치다. 첫512개 표본에서 원 표현 손실 대비 보조 경사의 cosine과 norm 비율을 매 갱신 기록한다. 진단값으로 가중치를 바꾸거나 안전 보장을 주장하지 않는다.

원 FCSRL `d882b91`, Apache-2.0. e56 CPU/GPU 실행 기능과 e24 보조 손실 구조를 재사용했다. 이 결과는 반복 사용한 개발 자료와 같은 불완전한 대리 환경에서의 탐색이며 실제 폐수처리장 성능을 입증하지 않는다.

## 실행

```bash
export OMP_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
/home/jio/fcsrl_src/.venv-skl191/bin/python Yoo/experiments/e58/check.py --out <새 검사 경로>
/home/jio/fcsrl_src/.venv-skl191/bin/python -u Yoo/experiments/e58/run.py --check <검사 경로>/checks.json --out <새 본 실행 경로> --jobs 3
```

본 실행의 `manifest.json`에 코드 커밋, 입력/소스 SHA256, 장치 및 명령을 남긴다. `status.json`은 전체 상태, 각 `progress.json`과 tqdm 로그는 갱신과 잔여 추정, `resume.pt`는 재개 상태다. 모두 완료되면 `results.md`와 `summary.json`을 자동 집계한다.

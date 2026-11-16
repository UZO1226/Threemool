# e56 실행 장치 전환

2026-11-13. 사용자가 비어 있는 GPU1을 활용한 실행을 요청해 CPU 수집과 신경망 갱신을 분리했다. 데이터, 비용, FCSRL 손실과 학습량을 변경하는 새 기법 실험이 아니다.

**완료 상태:** 2026-11-13 11:03:41 KST에 10개 모두 종료하고 결과를 집계했다. 현재 실행 중인 e56 작업은 없다. [최종 비교](results.md). 아래는 당시 전환 절차의 기록이다.

## 변경 이유와 스레드 정정

기존 e56 코드가 `torch.set_num_threads(1)` 뒤에 원본 `DeviceConfig.select_device(-1)`을 호출하면서 PyTorch 스레드가 4로 다시 설정됐다. 따라서 이전 README의 작업당 CPU1스레드 설명은 정확하지 않았다. 환경 변수만으로 그 설정을 고정할 수 없었다.

신규 backend는 원본 장치 선택 함수를 반복 호출하지 않고 장치 값을 지정한 뒤 CPU 스레드를 실제로 1로 고정하고 검사한다. 기존 e50 및 먼저 실행된 e56 작업은 CPU4 조건으로 기록한다.

## 실행 구조

- 환경 예측과 rollout: 기존 sklearn GBM과 CPU1의 정책 복사본. 실제 관측 재주입이나 모델 변경 없음.
- PPO 갱신: CPU1. 작은 망은 GPU 이득이 확인되지 않았다.
- FCSRL 갱신: 물리 GPU1만 사용한다. `CUDA_VISIBLE_DEVICES=1`로 격리하므로 프로세스 내부 이름은 `cuda:0`이다. GPU0 작업에 접근하지 않는다.
- 수집 전 learner의 모든 정책, 가치, Encoder, feasibility 모듈과 log std를 CPU 복사본에 반영한다. 원래 관측 정규화와 prior는 고정한다.
- 미니배치 순서는 CPU RNG로 뽑아 해당 장치로 옮긴다. 수집 난수열 소비 순서를 보존하며 손실, EMA, 갱신 횟수와 KL 종료 조건은 그대로다.
- GPU 갱신은 우리 작업끼리 파일 잠금으로 직렬화한다. GPU 대기 시간을 포함한 갱신 시간과 수집 시간을 각각 로그에 남긴다.
- 매 갱신 뒤 모델, optimizer, PID, CPU/NumPy/환경 샘플러 난수 상태를 원자적으로 `resume.pt`에 저장한다. 이 경로에서는 GPU 난수를 쓰는 dropout/샘플링이 없으며 GPU 갱신 중 미니배치 난수도 CPU에서 생성한다.
- 최종 checkpoint는 CPU 복사본으로 저장해 기존 loader로도 읽힌다. 최종 train/val 평가도 CPU에서 수행한다.

## 검증

실행 코드 `634d18b`. [검사 수치](hybrid_checks.json). 원 출력은 `/home/jio/fcsrl_src/results/e56/hybrid_preflight/`다.

| 검사 | PPO | FCSRL |
| --- | --- | --- |
| CPU 수집 복사본의 초기 가중치 일치 | 정확 일치 | 정확 일치 |
| 같은 buffer와 미니배치 CPU RNG 소비 | 정확 일치 | 정확 일치 |
| CPU1 대비 갱신 후 평균 행동 최대 차이 | 0 | 7.45e-8 |
| CPU1 대비 가치 최대 차이 | 0 | 3.81e-6 |
| optimizer/PID/RNG 복원 뒤 다음 갱신 | 정확 일치 | 정확 일치 |
| 동일 CPU1 기준 갱신 시간 | 1.56초 | 10.21초 |
| 새 backend 갱신 시간 | 1.07초 | 4.67초 |
| CPU1 수집 시간 | 21.88초 | 20.49초 |

실제 초기 rollout 하나와 고정 buffer에서 검사했다. 단발 측정이라 전체 학습 속도를 확정하지 않으며, 수집 시간 때문에 전체 학습이 GPU 갱신 속도만큼 빨라지는 것은 아니다. CPU/GPU 전체 학습 궤적의 비트 단위 동일성을 증명한 것도 아니다. 별도 실제 실행기 검사도 1갱신(6,080전이), checkpoint 저장/복원, 최종 val200/train940 에피소드 평가까지 통과했다. test는 사용하지 않았다.

## 기존 실행 보존과 전환

09:40에 시작된 PPO seed0/1, FCSRL seed0은 매 갱신 checkpoint가 없으므로 중단하거나 재시작하지 않는다. 이 세 작업은 완료시켜 기존 결과를 보존한다. 기존 배차 프로세스만 일시 정지해 나머지 7개가 원 backend로 추가 시작되지 않게 했다.

새 실행기는 기존 실행을 포함해 동시 작업 3개를 넘지 않는다. 빈 자리가 생기면 FCSRL seed1~4는 GPU1 갱신, PPO seed2~4는 CPU1로 시작한다. 이미 돌던 세 작업이 모두 종료된 뒤 기존 배차 프로세스만 종료하고 원 `main/status.json`에 후속 경로를 남긴다. 기존 metrics, 로그와 checkpoint는 수정하지 않는다.

- 원 경로: `/home/jio/fcsrl_src/results/e56/main/`
- 후속 경로: `/home/jio/fcsrl_src/results/e56/main_hybrid/`
- 전환 시점 원 배차 상태: `/home/jio/fcsrl_src/results/e56/migration_dispatch_pause.json`
- 후속의 기존 3개 `runs` 경로는 원 결과를 가리키는 심볼릭 링크다. 나머지 7개만 새로 만든다.

후속 `status.json`의 `legacy_running`, `running`, `pending`, `completed`로 전체 상태를 확인한다. 원 `main`의 상태가 `superseded`가 되면 후속 경로를 기준으로 확인한다.

## 결과 해석 제한

비교 기준 e50은 CPU4이고, 이번 후보에는 원 CPU4 작업 3개와 새 backend 작업 7개가 섞인다. 수식과 CPU 난수 순서를 유지하고 작은 수치 차이를 검사했지만 장치에 따른 전체 학습 차이는 완전히 분리하지 못했다. 자동 결과 표에 이 이력을 표시하고 `prior_only_causal_confirmation=false`를 기록한다. 탐색 선별이 좋아도 prior만의 확정된 인과 효과나 공식 채택으로 주장하지 않으며, 채택 전 같은 backend의 확인이 필요하다.

속도 개선을 새 FCSRL 알고리즘의 성능 개선으로 세지 않는다. 환경, 비용 또는 seed를 결과에 따라 바꾸지 않는다.

## 재현

```bash
cd /home/jio/threemool-hybrid-gpu
export OMP_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 /home/jio/fcsrl_src/.venv-skl191/bin/python Yoo/experiments/e56/check_hybrid.py --out <새 검사 경로>
```

`run_hybrid.py`는 이번 원 배차 상태를 이관하는 실행기이므로 이미 이관된 작업에 다시 호출하지 않는다. 개별 신규 실행은 `train_hybrid.py`를 사용하고, 같은 manifest와 설정의 중단 작업은 `--resume`으로 마지막 완료 갱신부터 복원한다. PyTorch나 시스템 CUDA를 설치/변경하지 않았다.

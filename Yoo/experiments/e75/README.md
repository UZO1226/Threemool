# e75: 현재 대리 환경의 개선 여지 진단

2026-11-21 15:19 KST 본 실행 시작. 16개 조건 중8개 병렬 실행,8개 대기. 사전 검사와2일 전체 파이프라인 스모크를 통과했다. 새 RL/환경 학습이 아니라 e74의 고정 환경에서 ABAC를 개선하는 행동열이 존재하는지 검사한다. [실행 전 근거와 설계](REFERENCES_AND_DESIGN.md).

## 제안 검토

환경 신뢰도는 큰 병목이지만 환경 하나만 문제라고 확정하지 않는다. 누적 비용이 순간 NH4 초과를 놓치는 설계 문제도 있었고, 13차원 관측은 전이가 쓰는 모든 변수를 포함하지 않는다. 원 FCSRL은 벡터 관측에서도 효과를 시험한 방법이므로 저차원이라는 이유만으로 주제를 폐기할 수 없다.

Park/e57,e58은 이미 짧은 MPC를 평가했다. 이번에는 e74의 비용과 모델에서 정확한 환경 구성원과 미래 외생 입력을 주고 하루 전체 행동열을 탐색한다. 기존 결과와 같은 순위표에 섞지 않는다. 이 정보는 실제 제어기에 허용되는 정보보다 많다.

**발견한 실행 가능해는 최적 성능 상한이 아니다.** 개선을 찾으면 대리 환경 안의 여지가 있다는 근거다. 못 찾으면 탐색기/예산/후보 제한 때문일 수도 있어 환경이나 FCSRL이 쓸모없다고 결론 내리지 않는다. 실제 공기의 인과 효과나 환경 신뢰도를 검증하는 실험도 아니다.

## 비교

- 기존 e74 선택 ABAC (sp3.5, kp0, ki1e6), 재조정 없음.
- dev/val 각 날짜순 등간격12일 x 모델5개, holdout/test 평가 없음.
- 미래 입력을 아는95스텝 행동열, 공기 최소/비용 최소 두 목적.
- CEM과 같은 후보 수 random shooting, 탐색 seed0,1. 32후보 x 8세대, 2세대 성과도 보존. 모두16개 계산이며 정책 학습 seed16개가 아니다.
- 각 날짜/구성원에서 공기, 비용, NH4 초과 적분, 순간 NH4 위반 수, 누적 위반 수, DO 위반 수가 ABAC보다 나쁘지 않은 후보만 채택. 비악화 자체는 baseline fallback의 설계 성질이다.
- 최종 행동열을 다른 구성원에 그대로 옮겨 민감도 확인. 독립 모델/실제 환경 검증은 아님.

## 실행

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PYTHON=/home/jio/fcsrl_src/.venv-skl191/bin/python
$PYTHON Yoo/experiments/e75/verify75.py --out /home/jio/fcsrl_src/results/e75/preflight.json
$PYTHON Yoo/experiments/e75/run75.py --out /home/jio/fcsrl_src/results/e75/main --jobs 8
```

원문과 공식 CEM 코드 확인본은 `/home/jio/fcsrl_src/results/e75/literature/`에 있다. 기존 cu130 torch와 sklearn 환경을 변경하지 않았다. CPU 작업이다.

상태는 `/home/jio/fcsrl_src/results/e75/main/status.json`, 조건별 진행은 `jobs/*/progress.json`, 전체 결과는 `results.md`, 원 궤적은 `jobs/*/*.npz`에 저장한다. 실행 코드, 자료, 기준선과 설계의 SHA256은 manifest에 남긴다. 실행 중에는 Python과 설계 파일을 수정하지 않는다.

## 사전 검사 결과

[검사 기록](preflight.json). 기존 ABAC의 dev 공기47,852.722489/비용4.184932와 val 공기55,256.098745/비용8.495 및 모든 기존 지표가 일치했다. ABAC 행동열 상태 재생, 후보/episode 배열 순서, OR 비용 경계, 실행 불가능 후보 탈락을 확인했다. 2일 x 5구성원 x 32후보 x95스텝 검사는5.20초 걸렸다. 이 검사는 새 제어 성과가 아니다.

본 실행 커밋: `74361c6`. 실행기는 PID3158185로 시작했다. PID는 시작 기록이며 현재 상태는 status.json과 실제 프로세스로 확인해야 한다. 첫2일 스모크는 `/home/jio/fcsrl_src/results/e75/smoke/`에 보존하며 본 성과 표에 합치지 않는다.

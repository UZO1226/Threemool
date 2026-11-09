# 코드 출처

`Park/code/` 의 코드는 박성현이 직접 작성한다. 팀원 코드를 파일째 복사하지 않는다.

## 그대로 설치해서 쓰는 외부 코드 (수정하지 않음)

| 이름 | 버전 | 라이선스 | 용도 |
| --- | --- | --- | --- |
| 원본 FCSRL (`https://github.com/czp16/FCSRL`) | 커밋 `d882b91054d52c7a63a18a21f7db67bbbed82b63` | Apache-2.0 | 표현 학습 손실과 신경망 |
| `bsm2-python` | 0.0.16 | 패키지 라이선스를 따른다 | BSM1 물리 시뮬레이터 |
| BSM1 유입 자료 (`BSM1.zip`, IWA 배포본) | sha256 `f1b7683596512f2e432bb616d20fdc85ffdef3006e27a170b562789d30a050aa` | `Yoo/experiments/e16/BSM_LICENSE` 참고 | 건기, 우기 유입 |

원본 FCSRL 의 커밋과 BSM1 자료의 체크섬은 Yoo/e16, Yoo/e17, Lee/e01 과 같게 맞춰 결과를 비교할 수 있게 한다.

## 참고만 한 팀원 코드

코드는 복사하지 않고, 같은 조건으로 비교하기 위해 정의와 기준값을 맞춘다. 참고한 내용은 아래 표에 적는다.

| 참고한 곳 | 기준 커밋 | 맞춘 내용 |
| --- | --- | --- |
| `Yoo/experiments/e16/bsm_env.py` | `1f7a14a` | BSM1 연결 방식 (5분마다 5번 조 KLa 제어, 1분 내부 적분), 관측 구성 |
| `Lee/code/bsm/bsm_binary.py` | `1f7a14a` | 이진 비용 정의 (5분 안에 NH4 > 4 mg N/L 인 시점이 있으면 1), 하루 한도 72 스텝 |
| `Lee/experiments/e01/results.md` | `1f7a14a` | 환경 검증 기준값 (`Park/code/README.md` 참고) |
| `Yoo/code/wwtp/env.py`, `surrogate.py`, `evaluate.py` | `db0a448` | Avedøre 환경 규칙(관측, 행동, 비용, 보상), 대리 모델 추론 방식, 평가 지표 정의 |
| `Yoo/experiments/e02/surrogate.json` | `db0a448` | 대리 모델 가중치. 복사하지 않고 그 경로에서 읽는다 (sha256 d25dbe3e...) |
| `Yoo/experiments/e02/historical.json` | `db0a448` | Avedøre 환경 검증 기준값 |
| `Yoo/code/wwtp/prep.py` | `db0a448` | 데이터 전처리. 고치지 않고 그대로 실행한다 |

## 수정 이력

| 날짜 | 파일 | 내용 | 실험 |
| --- | --- | --- | --- |
| 2026-11-08 | 전체 | 직접 작성으로 방향을 정하고 구조를 잡음 | Park/e01 준비 |
| 2026-11-09 | `avedore_env.py`, `check_avedore.py`, `train_avedore.py`, `agent.py` | Avedøre 환경과 학습 추가, 관측 정규화 표준편차 하한을 인자로 받게 함 | Park/e07 |

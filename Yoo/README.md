# Yoo (유지오)

유지오 의 개인 작업 공간이다. `yoo/<주제>` 브랜치에서만 수정한다 (RULES/BRANCH_RULES.md 4절).

| 위치 | 내용 |
| --- | --- |
| `analysis/` | 논문 분석 노트 |
| `code/` | 구현 코드 (FCSRL 재현, 환경, 확장 기법) |
| `experiments/<실험ID>/README.md` | 실험 기록. 실험 ID 는 e01, e02 처럼 이 폴더 안에서 겹치지 않게 붙인다 |
| `RESULT.md` | 결과 비교용 요약 |

## 실행 환경

- 서버 .32 (i9-10900K, RTX 3090 x2), Python 3.10.19 venv `/home/jio/fcsrl_src/.venv`
- torch 2.14.1+cu130, safety-gymnasium 1.0.0, gymnasium 0.28.1, mujoco 2.3.3, numpy 1.23.5, pandas, scikit-learn
- 원본 FCSRL: `/home/jio/fcsrl_src/FCSRL` (커밋 d882b91, 수정 없음, `pip install -e` 로 설치)
- 설치 명령은 `experiments/e01/README.md` 에 있다

## 현재 상태

| 실험 | 내용 | 상태 |
| --- | --- | --- |
| e01 | FCSRL 원본 재현 (Safety Gymnasium 6개 과제 x seed 5개) | 완료. PointGoal2 를 뺀 5개 과제가 논문 편차 안 |
| e02 | FCSRL 을 Avedøre 폭기조 데이터 기반 폭기 제어에 적용, TD3 Lag 와 TD3 와 실제 운전 비교 | 완료. 대리 모델 위에서 공기량 약 45% 절감, 비용 실제 운전 22.9 에서 15 안팎. FCSRL 과 TD3 Lag 차이는 seed 편차 안. `experiments/e02/README.md` |

다음 할 일: `Compare/PROTOCOL.md` 합의 (e02 의 정의를 후보로 제안), 합의 뒤 `RESULT.md` 작성

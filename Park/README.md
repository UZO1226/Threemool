# Park (박성현)

박성현 의 개인 작업 공간이다. `park/<주제>` 브랜치에서만 수정한다 (RULES/BRANCH_RULES.md 4절).

| 위치 | 내용 |
| --- | --- |
| `analysis/` | 논문 분석 노트 |
| `code/` | 구현 코드 (FCSRL 재현, 환경, 확장 기법) |
| `experiments/<실험ID>/README.md` | 실험 기록. 실험 ID 는 e01, e02 처럼 이 폴더 안에서 겹치지 않게 붙인다 |
| `RESULT.md` | 결과 비교용 요약 |

## 실행 환경

<!-- Python 버전, 주요 패키지 버전, GPU, 설치 명령 -->

## 현재 상태

BSM1 환경에서 FCSRL 확장을 준비하고 있다. 코드는 `code/` 에 직접 작성한다. 구성과 순서는 `code/README.md`, 외부 코드와 참고한 곳은 `code/SOURCE.md` 에 있다.

| 실험 | 내용 | 상태 |
| --- | --- | --- |
| e01 | feasibility 점수로 실행 시점에 위험 행동을 막는 안전 차단 | 2026-11-09 완료. 위반은 줄었지만 F 가 NH4 규칙보다 낫다는 주 가설은 지지 안 됨. [실험 기록](experiments/e01/README.md) |
| e02 | 더 먼 미래를 보는 F (feasibility discount 0.97, 0.99) | 준비 중 |

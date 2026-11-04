# Threemool 브랜치 및 협업 규칙

## 1. 목적과 진행 단계

유지오, 이하경, 박성현이 FCSRL 구현체를 재현하고 폐수처리 공정의 폭기 제어에 적용, 확장하는 연구를 진행하면서 각자의 실험과 공동 논문을 한 저장소에서 충돌 없이 관리하기 위한 규칙이다.

작업은 다음 순서로 진행한다.

1. **개인 실험.** 각자 자기 이름 폴더(`Yoo/`, `Lee/`, `Park/`)에서 논문 분석, 구현체 구동, 확장 기법 실험을 진행한다. 다른 사람 폴더는 건드리지 않는다.
2. **결과 비교와 채택.** `Compare/PROTOCOL.md` 에 합의한 같은 조건으로 측정한 결과를 `Compare/RESULTS.md` 에서 비교하고, 논문에 쓸 실험을 세 명이 함께 정한다.
3. **논문 분담 작성.** 채택한 실험을 근거로 논문 절을 나눠 맡는다. 각자 맡은 `Paper/sections/` 파일을 작성해 `main` 에 merge 하면 `Paper/build/paper.md` 가 자동으로 합쳐진다.
4. **제출과 발표.** 자동 생성된 `paper.hwpx` 를 한글에서 마무리해 제출하고 `Slides/` 에서 발표 자료를 만든다.

비교 대상은 PPO/SAC, FCSRL, 확장 기법이며 주요 지표는 에너지 소비량, 수질 제약 위반률, reward 다. 개선 효과는 실험으로 확인한 범위에서만 기술한다.

## 2. 저장소 구조

```
Threemool/
├── README.md
├── CLAUDE.md, AGENTS.md     AI 도구가 자동으로 읽는 지침 (RULES 를 가리킨다)
├── RULES/
│   ├── BRANCH_RULES.md      이 문서
│   └── PAPER_RULES.md       논문 작성과 통합 규칙
├── Yoo/  Lee/  Park/        개인 작업 공간 (분석, 코드, 실험, RESULT.md)
├── Compare/
│   ├── PROTOCOL.md          공통 평가 조건
│   └── RESULTS.md           세 결과 비교, 채택 결정, 논문 절 배정
├── Paper/
│   ├── sections/            절 파일 (원본, 담당자만 수정)
│   ├── refs/                각자의 참고문헌 (yoo.md, lee.md, park.md)
│   ├── figures/             그림 (식별자_이름.png)
│   ├── template/            학회 투고 양식 (HWP)
│   └── build/               통합본 paper.md, 투고 양식 paper.hwpx (자동 생성, 수정 금지)
├── Docs/                    수행 계획서 등 과제 자료
├── Slides/                  발표 자료
├── scripts/                 build_paper.py, build_hwpx.py, check_scope.py
└── .github/workflows/       PR 검사, 논문 자동 통합
```

개인 폴더 안의 구성은 `Yoo/README.md` 를 기본으로 하되 각자 필요에 맞게 바꿔도 된다. 단 `RESULT.md` 는 비교에 쓰므로 양식을 유지한다.

## 3. 브랜치

공유 브랜치는 `main` 하나다. 모든 작업은 최신 `main` 에서 만든 작업 브랜치에서 하고 PR 로 `main` 에 합친다.

| 팀원 | 식별자 | GitHub | 개인 폴더 | 브랜치 예시 |
| --- | --- | --- | --- | --- |
| 유지오 | `yoo` | `UZO1226` | `Yoo/` | `yoo/fcsrl-baseline` |
| 이하경 | `lee` | `dlgkrudha` | `Lee/` | `lee/do-constraint` |
| 박성현 | `park` | `qkrtjdgus1123` | `Park/` | `park/survey-analysis` |

| 브랜치 | 용도 |
| --- | --- |
| `main` | 통합 상태. 직접 push 하지 않는다 (논문 자동 통합 봇만 예외) |
| `<식별자>/<주제>` | 개인 작업. 자기 폴더와 자기에게 배정된 논문 파일만 수정한다 |
| `team/<주제>` | 공동 파일 수정. RULES, Compare, Docs, Slides, scripts, .github, Paper/template, 논문 절 배정 변경, 절 파일 추가 |

- 주제는 영문 소문자, 숫자, 하이픈으로 쓴다. 예: `yoo/paper-introduction`
- 작업 하나마다 브랜치를 새로 만들고, merge 뒤 삭제한다. 개인 장기 브랜치를 두지 않는다.
- PR 전에 최신 `main` 을 자기 브랜치에 merge 해서 확인한다.
- `main` 에는 force push 하지 않는다. 자기 작업 브랜치는 자유롭게 정리해도 된다.

## 4. 수정 범위 (자동 검사)

PR 을 열면 GitHub Actions 가 `scripts/check_scope.py` 로 브랜치 이름과 바뀐 파일을 대조한다. 범위를 벗어나면 검사가 실패하고, 실패한 PR 은 merge 하지 않는다.

| 브랜치 | 수정할 수 있는 경로 |
| --- | --- |
| `yoo/*` | `Yoo/` 전체, 첫 줄이 `<!-- owner: yoo -->` 인 `Paper/sections/` 파일, `Paper/refs/yoo.md`, `Paper/figures/yoo_*` |
| `lee/*`, `park/*` | 같은 방식으로 자기 식별자에 해당하는 경로 |
| `team/*` | `Paper/build/` 를 제외한 모든 경로 |
| 모든 브랜치 | `Paper/build/` 는 수정할 수 없다 (자동 생성) |

다른 사람 폴더의 코드나 결과가 필요하면 그 사람에게 요청하거나, 자기 폴더로 복사하고 출처 경로와 커밋을 적는다.

PR 전에 로컬에서 확인할 수 있다.

```
git fetch origin
python3 scripts/check_scope.py "$(git branch --show-current)"
```

## 5. PR 과 merge

| PR 종류 | 검토 | merge |
| --- | --- | --- |
| 개인 폴더만 바뀐 PR | 선택 | 검사 통과 후 작성자가 merge |
| 논문 절(`Paper/sections/`)이 바뀐 PR | 다른 팀원 한 명 이상 승인 | 승인한 팀원 |
| `team/*` PR | 다른 팀원 한 명 이상 승인 | 승인한 팀원 |
| 평가 조건, 채택 결정, 절 배정, 규칙 변경 | 세 명 모두 승인 | 마지막으로 승인한 팀원 |

- squash merge 만 쓴다.
- 논문 통합 시험이 실패한 PR 은 merge 하지 않는다.
- PR 본문은 `.github/pull_request_template.md` 양식으로 자동으로 채워진다.

GitHub 에서 다음을 자동으로 강제한다 (규칙 세트 "main 보호").

- `main` 에는 PR 로만 반영된다. 직접 push, 강제 push, 브랜치 삭제가 막혀 있다.
- PR 검사(`check`)가 통과해야 merge 할 수 있다.
- `.github/CODEOWNERS` 에 적힌 공동 경로를 바꾸는 PR 은 작성자가 아닌 팀원 한 명 이상이 승인해야 merge 할 수 있다. 승인 뒤 새 커밋을 올리면 승인이 취소된다.
- 개인 폴더만 바꾼 PR 은 승인 없이 작성자가 merge 할 수 있다.
- 세 명 모두 승인해야 하는 경우(위 표 마지막 줄)는 GitHub 가 강제하지 않으므로 팀 규칙으로 지킨다.
- 예외는 논문 자동 통합 봇 하나다. 봇은 저장소 배포 키(Deploy key)로 `Paper/build/` 만 커밋한다.
- 승인 뒤 내용이 바뀌면 다시 검토한다.
- PR 본문에는 목적, 바뀐 범위, 실행한 확인과 결과를 적는다. 학습을 돌리지 않았다면 그 사실을 적는다. 실험이면 실험 ID 와 결과 위치를, 논문 수치가 바뀌면 근거 실험을 적는다.

## 6. 개인 실험

### 실험 기록

실험 ID 는 개인 폴더 안에서 `e01`, `e02` 처럼 겹치지 않게 붙이고, 다른 문서에서는 `Yoo/e03` 처럼 폴더 이름과 함께 부른다. 실험 기록은 `<개인 폴더>/experiments/<실험ID>/README.md` 에 두고 다음을 적는다.

- 가설과 기준 대비 바꾼 요소
- 실제 실행한 코드 커밋
- 데이터 출처와 버전, 전처리, 분할
- 환경 정의, 보상, NH4 와 DO 제약, 지표의 단위와 집계 방식 (`Compare/PROTOCOL.md` 와 다른 점)
- 실행 명령, 설정 파일, seed, 학습량, 평가 횟수, 패키지와 장치 환경
- 결과와 seed 간 편차, 로그와 그림의 위치, 재생성 방법
- 실패하거나 개선되지 않은 결과, 해석과 한계

### 원본 구현체

FCSRL 원본 코드(`https://github.com/czp16/FCSRL`)를 가져올 때는 사용한 원본 커밋, 라이선스, 원본 대비 수정 사항을 기록한다. 원본 재현 결과와 폐수처리 환경 적용 결과를 구분한다.

### 대용량 파일

데이터 원본, 체크포인트, 전체 로그는 Git 에 넣지 않는다 (`.gitignore` 참고). 팀이 접근할 수 있는 별도 저장소에 두고 위치와 체크섬을 실험 기록에 적는다. 작은 설정 파일, 결과 요약, 재현 스크립트는 Git 으로 관리한다.

## 7. 결과 비교와 채택

1. 개인 실험을 본격적으로 돌리기 전에 `Compare/PROTOCOL.md` 를 `team/protocol` PR 로 합의한다. 조건이 다른 결과끼리는 우열을 비교하지 않는다.
2. 각자 `RESULT.md` 에 그 조건으로 측정한 값을 적는다.
3. `team/compare-v1` PR 에서 `Compare/RESULTS.md` 에 세 결과를 모아 비교하고, 채택 실험과 이유를 적는다.
4. 채택한 실험의 실행 커밋에 `selected-v1` 태그를 붙인다. 채택하지 않은 결과도 지우지 않는다. 비교 실험이나 한계 분석에 쓸 수 있다.
5. 같은 PR 또는 이어지는 `team/paper-assign` PR 에서 논문 절 담당을 정하고, `Paper/sections/` 각 파일 첫 줄의 owner 를 바꾼다.

## 8. 논문 작성

`RULES/PAPER_RULES.md` 를 따른다. 요약하면 다음과 같다.

- 각자 owner 로 배정된 절 파일만 개인 브랜치에서 고친다.
- PR 이 `main` 에 merge 되면 GitHub Actions 가 `Paper/build/paper.md` 와 투고 양식에 채운 `Paper/build/paper.hwpx` 를 다시 만들어 커밋한다.
- 그림, 표, 수식, 인용 번호는 통합할 때 자동으로 매겨진다.

## 9. 태그

| 태그 | 의미 |
| --- | --- |
| `baseline-fcsrl-v1` | 원본 FCSRL 구동과 재현을 확인한 상태 |
| `selected-v1` | 논문에 쓰기로 채택한 실험의 실행 커밋 |
| `paper-v1` | 논문 초안 통합본과 근거 실험이 연결된 상태 |
| `submission-v1` | 최종 제출 상태 |

태그는 새 커밋으로 옮기지 않는다. 기준에 오류가 있으면 고친 뒤 `v2` 로 새 태그를 만든다.

## 10. 문장부호

코드 주석, 문서 본문, 커밋 메시지, 화면 문구에는 가운뎃점, 줄표, 이중 하이픈을 사용하지 않는다. 쉼표, 마침표, 괄호로 표현한다. 명령행 옵션, 코드 연산자, Markdown 표 구분선 등 실행이나 문서 형식에 필요한 문법은 그대로 사용한다.

## 11. 처음 설정

설정 담당자 한 명이 순서대로 진행한다.

1. 이 구조를 `main` 에 올린다. 초기 구조 반영은 직접 push 금지의 예외다.
2. 세 명의 저장소 쓰기 권한을 확인한다 (Settings > Collaborators).
3. 봇용 배포 키를 만든다. 쓰기 권한이 있는 Deploy key 로 등록하고, 개인 키는 Actions 비밀값 `PAPER_BOT_KEY` 로 저장한 뒤 로컬 사본은 지운다.
4. 규칙 세트 "main 보호"를 만든다 (Settings > Rules > Rulesets). 대상은 기본 브랜치, 규칙은 PR 필수(코드 소유자 승인, 새 커밋 시 승인 취소, squash 만 허용), 상태 검사 `check` 필수, 강제 push 와 삭제 금지, 우회 허용은 Deploy key 만이다.
5. 규칙을 잠시 풀어야 할 때는 저장소 관리자(유지오)가 규칙 세트를 끄고, 작업 뒤 바로 다시 켠다. 끈 이유는 팀에 알린다.

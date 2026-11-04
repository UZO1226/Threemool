# Threemool 작업 지침

이 저장소에서 작업하기 전에 아래 문서를 읽고 따른다.

- `RULES/BRANCH_RULES.md`: 개인 폴더, 브랜치, PR, 실험 기록 규칙
- `RULES/PAPER_RULES.md`: 논문 작성과 자동 통합 규칙 (`Paper/` 작업 시 필수, 9절이 AI 작업 지침)

핵심:

- 현재 브랜치 이름의 앞부분(`yoo/`, `lee/`, `park/`, `team/`)이 고칠 수 있는 경로를 정한다. 범위 밖 파일은 고치지 않고 사용자에게 알린다.
- `Paper/build/` 는 자동 생성 폴더라 직접 고치지 않는다.
- `main` 에 직접 push 하지 않는다.
- 작업 뒤 `python3 scripts/check_scope.py "$(git branch --show-current)"` 와 `python3 scripts/build_paper.py --check` 를 실행한다.
- 문서, 주석, 커밋 메시지에 가운뎃점, 줄표, 이중 하이픈을 쓰지 않는다.

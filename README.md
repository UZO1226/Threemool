# Threemool

사물인터넷 Project 2 (7조). FCSRL(ICML 2024) 구현체를 재현하고 폐수처리 공정의 폭기 제어에 적용, 확장하는 Safe RL 연구.

| 폴더 | 내용 |
| --- | --- |
| `Yoo/`, `Lee/`, `Park/` | 각자의 분석, 코드, 실험 |
| `Compare/` | 공통 평가 조건, 결과 비교, 채택 결정 |
| `Paper/` | 논문 절 파일과 자동 생성물(`Paper/build/paper.md`, 투고 양식 `paper.hwpx`) |
| `RULES/` | 협업 규칙 (`BRANCH_RULES.md`, `PAPER_RULES.md`) |
| `Docs/`, `Slides/` | 과제 자료, 발표 자료 |

작업 방법은 `RULES/BRANCH_RULES.md` 를 먼저 읽는다.

```
git switch main && git pull
git switch -c yoo/fcsrl-baseline      # 자기 식별자/주제
# Yoo/ 안에서 작업
python3 scripts/check_scope.py "$(git branch --show-current)"
git push -u origin yoo/fcsrl-baseline # 이후 GitHub 에서 main 대상 PR
```

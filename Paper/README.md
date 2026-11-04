# Paper

규칙 전체는 `RULES/PAPER_RULES.md` 에 있다.

| 위치 | 내용 | 수정 |
| --- | --- | --- |
| `sections/` | 절 파일 (원본) | 첫 줄 owner 가 자기 식별자인 파일만 |
| `refs/<식별자>.md` | 각자 추가한 참고문헌 | 자기 파일만 |
| `figures/<식별자>_*.png` | 그림 | 자기 식별자로 시작하는 파일만 |
| `template/` | 학회 투고 양식 원본(HWP)과 변환본(`template.hwpx`) | team/ 브랜치 |
| `build/paper.md` | 통합본 | 수정 금지, main 에 merge 되면 자동 생성 |
| `build/paper.hwpx` | 투고 양식에 채운 논문 | 수정 금지, 자동 생성. 한글에서 열어 마무리 |

로컬 확인: `python3 scripts/build_paper.py --check` (HWPX 까지 만들어 보는 방법은 PAPER_RULES 7절)

# Threemool 논문 작성 및 통합 규칙

`Paper/` 를 다루는 모든 작업은 사람이든 AI 도구든 이 문서를 따른다. 브랜치와 PR 규칙은 `RULES/BRANCH_RULES.md` 를 따른다.

## 1. 원칙

- 원본은 `Paper/sections/` 의 절 파일이다. 사람은 원본만 고친다.
- `Paper/build/` 의 두 파일은 자동 생성물이다. 직접 고치지 않는다. 고칠 내용이 있으면 해당 절 파일을 고친다.
  - `paper.md`: `scripts/build_paper.py` 가 절 파일을 합치고 번호를 매긴 통합본
  - `paper.hwpx`: `scripts/build_hwpx.py` 가 `paper.md` 를 투고 양식(`Paper/template/template.hwpx`)에 채운 파일. 한글에서 열어 HWP 로 저장할 수 있다
- PR 이 `main` 에 merge 되면 GitHub Actions(논문 통합)가 두 파일을 다시 만들어 `main` 에 커밋한다.
- 자동 생성 HWPX 에도 사람이 마무리할 부분이 남는다 (8절).
- 논문의 수치와 결과는 `Compare/RESULTS.md` 에서 채택한 실험과 각 개인 폴더의 실험 기록에 근거한다.

## 2. 투고 양식 요약

`Paper/template/1. 논문 투고 양식(20.4.24 개정)-접수용(국문).hwp` 에서 확인한 요구사항이다. 원문과 다르면 원문을 따른다.

| 항목 | 요구사항 |
| --- | --- |
| 제목 | 국문 제목, 영문 제목 |
| 저자 | 국문, 영문 이름과 소속, 책임저자 연락처. 양식에 "(신규논문 시 삭제)" 표시가 있으므로 신규 접수본에서는 넣지 않는다 |
| Abstract | 영문, 500자 내외 |
| 요약 | 국문, 500자 내외 |
| Key words | 영문 키워드 5개 이상 |
| 본문 구성 | Ⅰ. 서론, Ⅱ. 본론, Ⅲ. 결론, References |
| 번호 체계 | 절 `1.`, 항 `가.`, 그 아래 `(1)` |
| 인용 | 인용문 오른쪽 대괄호 안에 번호. 예: `[1]` |
| 수식 | 오른쪽에 `(1)` 형태 번호 |
| 그림 | 그림 안 글자는 영문. 캡션은 `Fig. 1. 영문제목` 과 `그림 1. 국문제목` 두 줄. 국문 캡션 끝에 마침표를 찍지 않는다 |
| 표 | 표 안 내용은 영문. 표 위에 `Table 1. 영문제목` 과 `표 1. 국문제목` |
| References | 영문, 본문 인용 순서. 저널명, 도서명, 학회명은 이탤릭 |
| 저자 약력 | 양식 끝 BIOGRAPHY (영문). 신규 접수본에서는 넣지 않는다 |

참고문헌 유형별 형식 (양식 예시 기준):

- 저널: `저자, "제목," *저널명*, vol. 권, no. 호, pp. 쪽, 월. 연도.`
- 학회: `저자, "제목," in *Proc. of 학회명*, 연도, pp. 쪽.`
- 도서: `저자, *도서명*, 판. 출판지: 출판사, 연도.`
- 학위논문: `저자, "제목," 학위, 학교, 연도.`
- 웹사이트: `저자, "제목," URL`

## 3. 파일과 절 대응

| 파일 | 통합본 위치 | 쓸 내용 |
| --- | --- | --- |
| `00_title.md` | 제목, Key words | 국문 제목, 영문 제목, 영문 키워드 |
| `01_abstract.md` | Abstract, 요약 | 영문 초록, 국문 요약. 다른 절을 다 쓴 뒤 마지막에 쓴다 |
| `10_introduction.md` | Ⅰ. 서론 | 배경(공정 산업 RL, 폐수처리 폭기 에너지), 해결할 문제, 기여 |
| `20_related_work.md` | Ⅱ. 1. | 공정 산업 RL 서베이, Safe RL, FCSRL |
| `21_problem.md` | Ⅱ. 2. | 폭기 제어 문제 정의, 상태, 행동, 보상, NH4 와 DO 제약 |
| `22_method.md` | Ⅱ. 3. | FCSRL 적용 방식과 확장 기법 |
| `23_experiment.md` | Ⅱ. 4. | 데이터, 환경, 비교 방법, 지표, 학습 설정 |
| `24_results.md` | Ⅱ. 5. | 결과 비교와 분석, 한계 |
| `30_conclusion.md` | Ⅲ. 결론 | 요약, 한계, 향후 연구 |

- 파일 이름 앞 숫자가 위치를 정한다. 첫 숫자 0 은 앞부분, 1 은 Ⅰ, 2 는 Ⅱ(본론), 3 은 Ⅲ 이다. 같은 장 안에서는 두 번째 숫자 순서로 합친다.
- 절을 추가, 삭제하거나 순서를 바꾸는 일은 `team/` 브랜치에서 파일을 추가하거나 이름을 바꿔서 한다.

## 4. 분담과 수정 범위

- 모든 절 파일의 첫 줄은 `<!-- owner: 식별자 -->` 다. 처음에는 `none` 이고, 결과 채택 뒤 `Compare/RESULTS.md` 의 배정표와 함께 `team/` PR 로 정한다.
- 자기 식별자가 owner 인 파일만 자기 브랜치(`yoo/...` 등)에서 고친다. owner 줄은 개인 브랜치에서 바꾸지 않는다.
- 다른 절에 고칠 점이 있으면 직접 고치지 않고 PR 코멘트나 이슈로 담당자에게 요청한다.
- 참고문헌은 `Paper/refs/<식별자>.md` 에, 그림은 `Paper/figures/<식별자>_<이름>.png` 로 추가한다.
- 한 PR 에는 한 절만 담는다.

## 5. 작성 문법

통합 스크립트가 아래 문법을 읽어 번호를 매긴다. 번호는 직접 쓰지 않는다.

### 제목

본론 파일(`2x_*.md`):

```
# 관련 연구            → 1. 관련 연구
## Safe RL             → 가. Safe RL
### 제약 처리 방식      → (1) 제약 처리 방식
```

서론, 결론 파일(`1x_*.md`, `3x_*.md`)은 첫 `#` 이 장 제목(Ⅰ. 서론)이 되고, `##` 부터 `1.`, `###` 은 `가.`, `####` 은 `(1)` 이 된다.

`00_title.md` 는 `국문 제목:`, `영문 제목:`, `Key words:` 세 줄을 채운다. 키워드는 쉼표로 구분한다. `01_abstract.md` 는 `## Abstract` 와 `## 요약` 아래에 쓴다.

### 인용

```
FCSRL 은 표현 학습으로 안전 제약을 다룬다 [@cen2024fcsrl].
여러 문헌은 [@a2024; @b2023] 처럼 쓴다.
```

통합본에서는 처음 인용된 순서대로 `[1]`, `[1, 2]` 로 바뀌고 References 가 자동으로 만들어진다. 문헌은 `Paper/refs/<식별자>.md` 에 한 줄에 하나씩 적는다.

```
[cen2024fcsrl] 저자, "Feasibility Consistent Representation Learning for Safe Reinforcement Learning," in *Proc. of the 41st International Conference on Machine Learning (ICML)*, 2024, pp. 쪽.
```

- key 는 첫 저자 성, 연도, 핵심 단어를 영문 소문자로 붙인다. 예: `cen2024fcsrl`
- 같은 문헌을 두 사람이 쓰면 key 와 문헌 정보를 똑같이 맞춘다. 내용이 다르면 통합이 실패한다. 추가하기 전에 다른 사람 refs 파일에서 key 를 찾아본다.
- 인용되지 않은 문헌은 References 에 들어가지 않는다.

### 그림

```
![Energy consumption by method | 방법별 에너지 소비량](../figures/yoo_energy.png){#fig:energy}
```

`|` 앞은 영문 캡션, 뒤는 국문 캡션이다. 본문에서 `@fig:energy` 라고 쓰면 `그림 1` 로 바뀐다. 그림 파일은 PNG 또는 JPG 만 쓴다 (HWPX 에 넣기 위해). 그림은 단 폭(약 73mm)에 맞춰 들어가므로 글자를 충분히 크게 만든다. 그림 안의 글자(축 이름, 범례)는 영문으로 만든다. 가능하면 그림을 만든 스크립트 경로를 실험 근거 주석에 적는다.

### 표

```
Table: Comparison of control methods | 제어 방법별 비교 {#tab:main}

| Method | Energy (kWh) | Violation rate (%) | Reward |
| --- | --- | --- | --- |
| PPO | 0.0 | 0.0 | 0.0 |
```

캡션 줄과 표 사이에 빈 줄을 둔다. 본문에서 `@tab:main` 은 `표 1` 로 바뀐다. 표 안은 영문으로 쓴다.

### 수식

```
$$ r_t = -E_t - \lambda c_t $$ {#eq:reward}
```

여러 줄 수식은 마지막 줄을 `$$ {#eq:이름}` 으로 끝낸다. 본문에서 `@eq:reward` 는 `식 (1)` 로 바뀐다. HWPX 에는 수식이 글자(예: `r_t = -E_t - λ c_t (1)`)로 들어가므로 제출 전에 한글 수식 편집기로 다시 입력한다.

### 실험 근거 주석

결과 수치가 나오는 문단, 표, 그림 바로 아래에 근거를 적는다. 화면에는 보이지 않고 통합본에 남는다.

```
<!-- exp: Yoo/e03, commit abc1234, figure script: Yoo/code/plot_energy.py -->
```

## 6. 내용 규칙

- 결과 수치는 실험 기록에서 옮기고 근거 주석을 단다. 근거 없이 수치를 고치지 않는다.
- 원본 FCSRL 의 방법, 이 팀이 폐수처리 환경에 적용한 부분, 새로 확장한 기법을 구분해서 쓴다.
- 채택하지 않은 팀원의 결과를 쓸 때도 출처 실험을 밝힌다.
- 실험으로 확인한 범위만 주장한다. 실데이터 관측 결과와 시뮬레이션 평가 결과를 구분한다.
- 문체는 평서문 "~다" 로 쓴다. 영문 약어는 처음 나올 때 풀어 쓴다. 예: 안전 강화학습(Safe Reinforcement Learning, Safe RL)
- 문장부호는 BRANCH_RULES 10절을 따른다.

### 용어집

아래 표기로 통일한다. 새 용어는 처음 쓰는 사람이 PR 본문에 적고, 다음 `team/` PR 에서 이 표에 추가한다.

| 국문 | 영문 | 약어 |
| --- | --- | --- |
| 강화학습 | Reinforcement Learning | RL |
| 안전 강화학습 | Safe Reinforcement Learning | Safe RL |
| 제약 마르코프 결정 과정 | Constrained Markov Decision Process | CMDP |
| 폐수처리 | wastewater treatment | |
| 폭기 | aeration | |
| 용존산소 | Dissolved Oxygen | DO |
| 암모니아성 질소 | ammonium nitrogen | NH4-N |
| 에너지 소비량 | energy consumption | |
| 제약 위반률 | constraint violation rate | |

## 7. 통합과 검사

| 시점 | 자동으로 하는 일 |
| --- | --- |
| PR 을 열거나 갱신할 때 | 수정 범위 검사, 통합 시험 (`paper.md`, `paper.hwpx` 를 임시로 만들어 본다) |
| `main` 에 merge 된 뒤 | `Paper/build/paper.md`, `Paper/build/paper.hwpx` 생성과 커밋 |

통합 시험의 오류(통합 실패):

- 절 파일 첫 줄에 owner 가 없음
- 정의되지 않은 그림, 표, 수식 참조
- `Paper/refs` 에 없는 인용 key, 같은 key 의 문헌 정보 불일치
- 없는 그림 파일, `영문 | 국문` 형식이 아닌 캡션
- 제목 단계가 맞지 않음

경고(통합은 됨): 비어 있는 절, 초록 길이, 키워드 5개 미만.

로컬에서 확인하는 방법:

```
pip install lxml                                          # 처음 한 번
python3 scripts/build_paper.py --check
python3 scripts/build_paper.py --out paper.preview.md     # 미리보기, 커밋하지 않는다
python3 scripts/build_hwpx.py --md paper.preview.md --out paper.preview.hwpx
```

### 투고 양식 파일

`Paper/template/template.hwpx` 는 원본 HWP 양식을 리눅스 변환기(rhwp)로 HWPX 로 바꾼 파일이다. 변환 과정에서 일부 글자 서식(안내 문구 색 등)이 합쳐졌다. 한컴오피스가 있는 팀원이 원본 HWP 를 열어 "다른 이름으로 저장, HWPX" 로 저장한 파일로 바꾸면 더 정확하다. 바꿀 때는 `team/` 브랜치에서 같은 이름으로 덮어쓰고, 양식의 안내 문구("Ⅱ. 본론", "1. 각 절 제목", "Fig. 1." 등)는 지우지 않는다. 스크립트가 이 문구로 복제할 문단을 찾는다.

## 8. 제출

1. 통합본과 근거 실험이 맞는지 세 명이 확인하고 `paper-v1` 태그를 붙인다.
2. `Paper/build/paper.hwpx` 를 한글에서 열어 마무리한다.
   - 수식을 한글 수식 편집기로 다시 입력한다
   - 그림 크기, 표 열 너비, 쪽 나눔을 확인한다 (표 열은 같은 폭으로 들어간다)
   - 그림과 표 안 글자가 영문인지 확인한다
   - 저자 정보와 약력이 빠져 있는지 확인한다 (접수본 기준. 게재 확정 뒤에는 다시 넣는다)
3. 한글에서 HWP 로 저장해 제출하고, 제출한 파일은 `team/submission` PR 로 `Paper/submission/` 에 저장한 뒤 `submission-v1` 태그를 붙인다.

## 9. AI 작업 지침

AI 도구가 이 저장소에서 논문 작업을 할 때 따른다.

1. 작업 전에 `git branch --show-current` 로 브랜치를 확인하고, 이 문서 4절과 BRANCH_RULES 4절에 따라 고칠 수 있는 파일을 정한다. 범위 밖 수정이 필요하면 하지 않고 사용자에게 알린다.
2. 할 수 있는 일: 사용자가 맡은 절의 초안 작성과 수정, 5절 문법 맞추기, 통합 시험 실행과 오류 수정, 용어와 문체 점검, 실험 기록의 수치를 근거 주석과 함께 옮기기.
3. 하지 않는 일: 실험 기록에 없는 수치나 결과 서술, 원문에서 확인하지 않은 문헌 정보 작성, 존재하지 않는 문헌 인용, `Paper/build/` 직접 수정, owner 줄 변경, `main` 에 직접 push.
4. 다른 절과 맞지 않는 부분(용어, 수치, 기호)이나 규칙 위반을 발견하면 고치지 않고 목록으로 보고한다.
5. 작업이 끝나면 `scripts/check_scope.py` 와 `build_paper.py --check` 를 실행하고 결과를 사용자에게 알린다.
6. HWPX 를 직접 편집하지 않는다. 양식 처리 방식을 바꿔야 하면 `scripts/build_hwpx.py` 를 `team/` 브랜치에서 고친다. HWP, HWPX 파일을 읽거나 검사할 때는 hwpx 스킬(https://github.com/jkf87/hwpx-skill)을 써도 되지만, 그 코드를 이 저장소에 복사하지 않는다 (라이선스 표시가 없다).

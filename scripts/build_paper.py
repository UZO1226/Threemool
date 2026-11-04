#!/usr/bin/env python3
"""Paper/sections 의 절 파일을 합쳐 Paper/build/paper.md 를 만든다.

사용법:
  python3 scripts/build_paper.py             Paper/build/paper.md 생성 (main 에서는 GitHub Actions 가 실행)
  python3 scripts/build_paper.py --check     파일을 쓰지 않고 검사만
  python3 scripts/build_paper.py --out 경로   다른 위치에 생성 (로컬 미리보기)

작성 문법과 규칙: RULES/PAPER_RULES.md
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "Paper"
SECTIONS = PAPER / "sections"
REFS = PAPER / "refs"
DEFAULT_OUT = PAPER / "build" / "paper.md"

ROMAN = ["Ⅰ", "Ⅱ", "Ⅲ", "Ⅳ", "Ⅴ", "Ⅵ"]
HANGUL = "가나다라마바사아자차카타파하"
# 여러 파일이 모이는 장. 키는 파일 이름 첫 숫자, 값은 장 제목
GROUP_TITLES = {"2": "본론"}

FILE_RE = re.compile(r"^(\d)\d_[A-Za-z0-9_\-]+\.md$")
OWNER_RE = re.compile(r"^<!--\s*owner:\s*([a-z]+)\s*-->\s*$")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
FIG_RE = re.compile(r"^!\[(.*)\]\((.+?)\)\{#fig:([\w\-]+)\}\s*$")
TAB_RE = re.compile(r"^Table:\s*(.*?)\s*\{#tab:([\w\-]+)\}\s*$")
EQ_END_RE = re.compile(r"^(.*?)\$\$\s*\{#eq:([\w\-]+)\}\s*$")
CITE_RE = re.compile(r"\[(@[\w\-.]+(?:\s*;\s*@[\w\-.]+)*)\]")
XREF_RE = re.compile(r"@(fig|tab|eq):([\w\-]+)")
REF_LINE_RE = re.compile(r"^\[([\w\-.]+)\]\s+(.+?)\s*$")
FIELD_RE = re.compile(r"^(국문 제목|영문 제목|Key words)\s*:\s*(.*?)\s*$")
XREF_WORD = {"fig": "그림 {}", "tab": "표 {}", "eq": "식 ({})"}


class Report:
    def __init__(self):
        self.errors, self.warnings, self.notes = [], [], []

    def error(self, msg):
        self.errors.append(msg)

    def warn(self, msg):
        self.warnings.append(msg)


def read_file(path, rep):
    """owner 줄을 떼고 (owner, 본문 줄 목록)을 돌려준다."""
    lines = path.read_text(encoding="utf-8").splitlines()
    rel = path.relative_to(ROOT)
    first = next((i for i, l in enumerate(lines) if l.strip()), None)
    if first is None or not OWNER_RE.match(lines[first].strip()):
        rep.error(f"{rel}: 첫 줄에 <!-- owner: 식별자 --> 가 없다")
        return None, lines
    owner = OWNER_RE.match(lines[first].strip()).group(1)
    return owner, lines[first + 1:]


def is_empty(lines):
    """제목과 주석을 빼면 내용이 없는지."""
    text = "\n".join(l for l in lines if not HEADING_RE.match(l))
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    return not text.strip()


def build_front(rep):
    """00_title.md, 01_abstract.md 로 제목, 초록, 키워드 부분을 만든다."""
    out = []
    fields = {}
    title = SECTIONS / "00_title.md"
    if title.exists():
        owner, lines = read_file(title, rep)
        for l in lines:
            m = FIELD_RE.match(l.strip())
            if m:
                fields[m.group(1)] = m.group(2)
    else:
        rep.error("Paper/sections/00_title.md 가 없다")
    out += [f"# {fields.get('국문 제목') or '(국문 제목 미작성)'}", ""]
    out += [f"## {fields.get('영문 제목') or '(영문 제목 미작성)'}", ""]

    abstract = SECTIONS / "01_abstract.md"
    parts = {}
    if abstract.exists():
        owner, lines = read_file(abstract, rep)
        cur = None
        for l in lines:
            m = HEADING_RE.match(l)
            if m:
                cur = m.group(2).replace(" ", "")
                parts[cur] = []
            elif cur:
                parts[cur].append(l)
    else:
        rep.error("Paper/sections/01_abstract.md 가 없다")
    for key, label in (("Abstract", "Abstract"), ("요약", "요  약")):
        body = re.sub(r"<!--.*?-->", "", "\n".join(parts.get(key, [])), flags=re.S).strip()
        if not body:
            rep.warn(f"{label}: 미작성")
            body = f"({label} 미작성)"
        else:
            n = len(body)
            rep.notes.append(f"{label}: {n}자 (양식 기준 500자 내외)")
            if n > 650:
                rep.warn(f"{label}: {n}자로 500자 내외 기준보다 길다")
        out += [f"**{label}**", "", body, ""]

    kw = [k.strip() for k in fields.get("Key words", "").split(",") if k.strip()]
    if len(kw) < 5:
        rep.warn(f"Key words: {len(kw)}개 (양식 기준 영문 5개 이상)")
    out += [f"**Key words** : {', '.join(kw) if kw else '(미작성)'}", ""]
    return out


def build_body(rep):
    """1x, 2x, 3x 파일을 장 단위로 합치고 제목 번호를 붙인다."""
    files = sorted(p for p in SECTIONS.glob("*.md") if not p.name.startswith("0"))
    groups = {}
    for p in files:
        m = FILE_RE.match(p.name)
        if not m:
            rep.error(f"{p.relative_to(ROOT)}: 파일 이름은 두 자리 숫자_영문이름.md 형식이어야 한다")
            continue
        groups.setdefault(m.group(1), []).append(p)

    out = []
    for gi, key in enumerate(sorted(groups)):
        roman = ROMAN[gi]
        named = key in GROUP_TITLES
        if named:
            out += [f"## {roman}. {GROUP_TITLES[key]}", ""]
        part_done = named
        c = [0, 0, 0]  # 1. / 가. / (1)
        for path in groups[key]:
            rel = path.relative_to(ROOT)
            owner, lines = read_file(path, rep)
            if owner is not None:
                state = "미작성" if is_empty(lines) else "작성"
                rep.notes.append(f"{rel.name}: owner={owner}, {state}")
                if state == "미작성":
                    rep.warn(f"{rel.name}: 내용이 없다")
            in_code = False
            for l in lines:
                if l.strip().startswith("```"):
                    in_code = not in_code
                m = None if in_code else HEADING_RE.match(l)
                if not m:
                    out.append(l)
                    continue
                level, text = len(m.group(1)), m.group(2)
                if not part_done:
                    if level != 1:
                        rep.error(f"{rel}: 장 제목(# 제목)이 먼저 와야 한다")
                    out.append(f"## {roman}. {text}")
                    part_done = True
                    continue
                depth = level if named else level - 1
                if depth < 1 or depth > 3:
                    rep.error(f"{rel}: 제목 '{l.strip()}' 의 단계가 맞지 않다 (PAPER_RULES 5절)")
                    out.append(l)
                    continue
                c[depth - 1] += 1
                for i in range(depth, 3):
                    c[i] = 0
                if depth == 1:
                    num = f"{c[0]}."
                elif depth == 2:
                    num = f"{HANGUL[(c[1] - 1) % len(HANGUL)]}."
                else:
                    num = f"({c[2]})"
                out.append(f"{'#' * (depth + 2)} {num} {text}")
            out.append("")
    return out


def load_refs(rep):
    refs, src = {}, {}
    for path in sorted(REFS.glob("*.md")):
        rel = path.relative_to(ROOT)
        text = re.sub(r"<!--.*?-->", "", path.read_text(encoding="utf-8"), flags=re.S)
        for i, l in enumerate(text.splitlines(), 1):
            if not l.strip() or l.lstrip().startswith("#"):
                continue
            m = REF_LINE_RE.match(l.strip())
            if not m:
                rep.error(f"{rel}: '{l.strip()[:40]}' 는 [key] 문헌정보 형식이 아니다")
                continue
            k, v = m.groups()
            if k in refs and refs[k] != v:
                rep.error(f"참고문헌 key '{k}' 가 {src[k]} 와 {rel} 에서 내용이 다르다. 하나로 맞춘다")
            refs.setdefault(k, v)
            src.setdefault(k, rel)
    return refs


def resolve(lines, refs, rep):
    """그림, 표, 수식 번호와 인용 번호를 매긴다."""
    labels, count = {}, {"fig": 0, "tab": 0, "eq": 0}

    def define(kind, name):
        if (kind, name) in labels:
            rep.error(f"{kind}:{name} 라벨이 두 번 정의되었다")
            return
        count[kind] += 1
        labels[(kind, name)] = count[kind]

    in_code = False
    for l in lines:
        if l.strip().startswith("```"):
            in_code = not in_code
        if in_code:
            continue
        if m := FIG_RE.match(l):
            define("fig", m.group(3))
        elif m := TAB_RE.match(l):
            define("tab", m.group(2))
        elif m := EQ_END_RE.match(l):
            define("eq", m.group(2))

    cite_no, out, in_code = {}, [], False

    def caption(raw, where):
        if "|" not in raw:
            rep.error(f"{where}: 캡션은 '영문 | 국문' 형식이어야 한다")
            return raw.strip(), raw.strip()
        en, ko = (s.strip() for s in raw.split("|", 1))
        return en.rstrip("."), ko.rstrip(".")

    def xref(m):
        n = labels.get((m.group(1), m.group(2)))
        if n is None:
            rep.error(f"참조 @{m.group(1)}:{m.group(2)} 의 정의가 없다")
            return f"??{m.group(0)}"
        return XREF_WORD[m.group(1)].format(n)

    def cite(m):
        nums = []
        for key in (k.strip().lstrip("@") for k in m.group(1).split(";")):
            if key not in refs:
                rep.error(f"인용 [@{key}] 가 Paper/refs 에 없다")
                nums.append(f"?{key}")
                continue
            cite_no.setdefault(key, len(cite_no) + 1)
            nums.append(str(cite_no[key]))
        return f"[{', '.join(nums)}]"

    for l in lines:
        if l.strip().startswith("```"):
            in_code = not in_code
        if in_code:
            out.append(l)
            continue
        if m := FIG_RE.match(l):
            en, ko = caption(m.group(1), f"그림 {m.group(3)}")
            n = labels[("fig", m.group(3))]
            img = (SECTIONS / m.group(2)).resolve()
            if not img.exists():
                rep.error(f"그림 파일이 없다: {m.group(2)}")
            out += [f"![{en}]({m.group(2)})", "", f"Fig. {n}. {en}", "", f"그림 {n}. {ko}"]
            continue
        if m := TAB_RE.match(l):
            en, ko = caption(m.group(1), f"표 {m.group(2)}")
            n = labels[("tab", m.group(2))]
            out += [f"Table {n}. {en}", "", f"표 {n}. {ko}"]
            continue
        if m := EQ_END_RE.match(l):
            n = labels[("eq", m.group(2))]
            l = f"{m.group(1).rstrip()} \\tag{{{n}}} $$"
        l = XREF_RE.sub(xref, l)
        l = CITE_RE.sub(cite, l)
        out.append(l)

    unused = sorted(set(refs) - set(cite_no))
    if unused:
        rep.notes.append(f"인용되지 않아 빠진 참고문헌: {', '.join(unused)}")
    out += ["## References", ""]
    for key, n in sorted(cite_no.items(), key=lambda kv: kv[1]):
        out += [f"[{n}] {refs[key]}", ""]
    rep.notes.append(f"그림 {count['fig']}개, 표 {count['tab']}개, 수식 {count['eq']}개, 참고문헌 {len(cite_no)}개")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="파일을 쓰지 않고 검사만 한다")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="결과 파일 경로")
    args = ap.parse_args()

    rep = Report()
    lines = build_front(rep) + build_body(rep)
    lines = resolve(lines, load_refs(rep), rep)
    header = ["<!-- 자동 생성 파일이다. 직접 수정하지 않는다. 원본은 Paper/sections, 규칙은 RULES/PAPER_RULES.md -->", ""]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(header + lines)).rstrip() + "\n"

    for n in rep.notes:
        print(f"[정보] {n}")
    for w in rep.warnings:
        print(f"[경고] {w}")
    for e in rep.errors:
        print(f"[오류] {e}")
    if rep.errors:
        print(f"\n오류 {len(rep.errors)}개. 통합본을 만들지 않았다.")
        return 1
    if args.check:
        print("\n검사 통과 (파일은 쓰지 않음)")
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"\n생성: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Paper/build/paper.md 를 투고 양식(HWPX)에 채워 Paper/build/paper.hwpx 를 만든다.

사용법:
  python3 scripts/build_hwpx.py                  기본 경로 사용 (main 에서는 GitHub Actions 가 실행)
  python3 scripts/build_hwpx.py --md 입력 --out 출력

양식은 Paper/template/template.hwpx 다. 양식의 안내 문구("Ⅱ. 본론", "1. 각 절 제목",
"Fig. 1." 등)로 복제할 문단을 찾으므로 양식을 바꿀 때 안내 문구를 지우지 않는다.
필요 패키지: lxml. 규칙: RULES/PAPER_RULES.md
"""
import argparse
import copy
import re
import struct
import sys
import zipfile
from pathlib import Path

from lxml import etree

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "Paper" / "template" / "template.hwpx"
MD = ROOT / "Paper" / "build" / "paper.md"
OUT = ROOT / "Paper" / "build" / "paper.hwpx"

NS = {
    "hp": "http://www.hancom.co.kr/hwpml/2011/paragraph",
    "hh": "http://www.hancom.co.kr/hwpml/2011/head",
    "hc": "http://www.hancom.co.kr/hwpml/2011/core",
    "opf": "http://www.idpf.org/2007/opf/",
}
HP, HH, HC, OPF = ("{%s}" % NS[k] for k in ("hp", "hh", "hc", "opf"))

FIG_WIDTH = 20781    # 양식 그림 폭 (HWPUNIT, 약 73mm)
FIG_MAX_H = 22000    # 그림 최대 높이
TABLE_WIDTH = 22482  # 양식 표 폭
ROW_H = 1482         # 표 행 높이 (내용에 맞게 한글에서 늘어난다)
PX = 75              # 96 dpi 기준 픽셀당 HWPUNIT
PART_RE = re.compile(r"^[ⅠⅡⅢⅣⅤⅥ]\.")

GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε", "theta": "θ",
    "lambda": "λ", "mu": "μ", "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "φ",
    "psi": "ψ", "omega": "ω", "Delta": "Δ", "Sigma": "Σ", "Pi": "Π", "Omega": "Ω",
    "le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "neq": "≠", "times": "×", "cdot": "⋅",
    "in": "∈", "sum": "Σ", "infty": "∞", "approx": "≈", "pm": "±", "to": "→", "max": "max",
    "min": "min", "log": "log", "exp": "exp", "mathbb": "", "mathrm": "", "text": "",
    "mathcal": "", "left": "", "right": "", "quad": " ", ",": " ",
}


def fail(msg):
    print(f"[오류] {msg}")
    sys.exit(1)


# ---------------------------------------------------------------- markdown

def latex_plain(s):
    """간단한 LaTeX 를 읽을 수 있는 글자로 바꾼다. 최종본 수식은 한글 수식 편집기로 다시 입력한다."""
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\([A-Za-z]+|,)", lambda m: GREEK.get(m.group(1), m.group(1)), s)
    s = re.sub(r"[{}]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def inline(text):
    """**굵게**, *기울임*, `코드`, $수식$ 을 (글자, 굵게, 기울임) 조각으로 나눈다."""
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\$([^$]+)\$", lambda m: latex_plain(m.group(1)), text)
    parts = []
    for tok in re.split(r"(\*\*[^*]+\*\*|\*[^*\s][^*]*\*)", text):
        if not tok:
            continue
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            parts.append((tok[2:-2], True, False))
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            parts.append((tok[1:-1], False, True))
        else:
            parts.append((tok, False, False))
    return parts


def blank_if_placeholder(s):
    return "" if re.fullmatch(r"\(.*미작성\)", s.strip()) else s.strip()


def parse_md(path):
    text = re.sub(r"<!--.*?-->", "", path.read_text(encoding="utf-8"), flags=re.S)
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if l.startswith("## ") and PART_RE.match(l[3:])), len(lines))

    front = {"title_ko": "", "title_en": "", "abstract": [], "yoyak": [], "keywords": ""}
    mode = None
    for l in lines[:start]:
        s = l.strip()
        if s.startswith("# "):
            front["title_ko"] = blank_if_placeholder(s[2:])
        elif s.startswith("## "):
            front["title_en"] = blank_if_placeholder(s[3:])
        elif s == "**Abstract**":
            mode = "abstract"
        elif s.replace(" ", "") == "**요약**":
            mode = "yoyak"
        elif s.startswith("**Key words**"):
            front["keywords"] = blank_if_placeholder(s.split(":", 1)[1] if ":" in s else "")
            mode = None
        elif s and mode:
            front[mode].append(s)
    for k in ("abstract", "yoyak"):
        front[k] = blank_if_placeholder(" ".join(front[k]))

    blocks, para, in_refs = [], [], False
    body_dir = ROOT / "Paper" / "build"  # 그림 경로는 통합본 위치 기준이다

    def flush():
        if para:
            blocks.append(("para", " ".join(para)))
            para.clear()

    i = start
    while i < len(lines):
        s = lines[i].strip()
        i += 1
        if not s:
            flush()
            continue
        if s.startswith("```"):
            flush()
            while i < len(lines) and not lines[i].strip().startswith("```"):
                blocks.append(("code", lines[i]))
                i += 1
            i += 1
            continue
        if s.startswith("$$"):
            flush()
            buf = [s]
            while not (len(buf[-1]) > 2 and buf[-1].rstrip().endswith("$$")) and i < len(lines):
                buf.append(lines[i].strip())
                i += 1
            expr = " ".join(buf).strip()[2:-2]
            m = re.search(r"\\tag\{([^}]*)\}", expr)
            blocks.append(("eq", (latex_plain(re.sub(r"\\tag\{[^}]*\}", "", expr)), m.group(1) if m else "")))
            continue
        if s.startswith("|"):
            flush()
            rows = [s]
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows
                     if not re.fullmatch(r"\|?[\s:\-|]+\|?", r)]
            blocks.append(("table", cells))
            continue
        m = re.match(r"^(#{2,5})\s+(.*)$", s)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2)
            if level == 2:
                in_refs = title == "References"
                blocks.append(("part", title))
            else:
                blocks.append((f"h{level - 2}", title))
            continue
        if in_refs and (m := re.match(r"^\[(\d+)\]\s+(.*)$", s)):
            flush()
            blocks.append(("ref", s))
            continue
        if not para:
            if m := re.match(r"^!\[(.*?)\]\((.+?)\)$", s):
                blocks.append(("img", (body_dir / m.group(2)).resolve()))
                continue
            if m := re.match(r"^Fig\. (\d+)\. (.*)$", s):
                blocks.append(("fig_en", s))
                continue
            if m := re.match(r"^그림 (\d+)\. (.*)$", s):
                blocks.append(("fig_ko", (m.group(1), m.group(2))))
                continue
            if m := re.match(r"^Table (\d+)\. (.*)$", s):
                blocks.append(("tab_en", s))
                continue
            if m := re.match(r"^표 (\d+)\. (.*)$", s):
                blocks.append(("tab_ko", (m.group(1), m.group(2))))
                continue
        if m := re.match(r"^([-*+]|\d+\.)\s+(.*)$", s):
            flush()
            bullet = "• " if m.group(1) in "-*+" else m.group(1) + " "
            blocks.append(("para", bullet + m.group(2)))
            continue
        para.append(s)
    flush()
    return front, blocks


# ---------------------------------------------------------------- template

def text_of(p):
    return re.sub(r"\s+", " ", "".join(p.itertext())).strip()


class CharPrs:
    """기존 글자 모양을 바탕으로 굵게, 기울임, 검정 글자 변형을 만든다."""

    def __init__(self, header):
        self.props = header.find(f".//{HH}charProperties")
        self.byid = {c.get("id"): c for c in self.props}
        self.next = max(int(k) for k in self.byid) + 1
        self.cache = {}

    def get(self, base, bold=False, italic=False):
        src = self.byid[base]
        has_b, has_i = src.find(HH + "bold") is not None, src.find(HH + "italic") is not None
        bold, italic = bold or has_b, italic or has_i
        black = (src.get("textColor") or "#000000").upper() == "#000000"
        under = src.find(HH + "underline") is not None
        if bold == has_b and italic == has_i and black and not under:
            return base
        key = (base, bold, italic)
        if key not in self.cache:
            new = copy.deepcopy(src)
            new.set("id", str(self.next))
            new.set("textColor", "#000000")
            for tag in ("italic", "bold", "underline"):
                for e in new.findall(HH + tag):
                    new.remove(e)
            idx = list(new).index(new.find(HH + "offset")) + 1
            if bold:
                new.insert(idx, etree.Element(HH + "bold"))
            if italic:
                new.insert(idx, etree.Element(HH + "italic"))
            self.props.append(new)
            self.props.set("itemCnt", str(len(self.props)))
            self.byid[str(self.next)] = new
            self.cache[key] = str(self.next)
            self.next += 1
        return self.cache[key]


class Builder:
    def __init__(self, template):
        self.zin = zipfile.ZipFile(template)
        self.section = etree.fromstring(self.zin.read("Contents/section0.xml"))
        self.header = etree.fromstring(self.zin.read("Contents/header.xml"))
        self.hpf = etree.fromstring(self.zin.read("Contents/content.hpf"))
        self.chars = CharPrs(self.header)
        self.top = list(self.section)
        self.pid = max(int(p.get("id") or 0) for p in self.top) + 1
        self.oid = 1500000000
        self.images = []
        self.proto = self.find_protos()

    def find(self, pred, what, items=None):
        for p in items if items is not None else self.top:
            if pred(text_of(p), p):
                return p
        fail(f"양식에서 '{what}' 문단을 찾지 못했다. Paper/template/template.hwpx 의 안내 문구를 확인한다")

    def find_protos(self):
        has = lambda p, tag: p.find(f".//{HP}{tag}") is not None
        pr = {
            "part": self.find(lambda t, p: t.startswith("Ⅱ. 본론"), "Ⅱ. 본론"),
            "refs_head": self.find(lambda t, p: t == "References", "References"),
            "h1": self.find(lambda t, p: t.startswith("1. 각 절 제목"), "1. 각 절 제목"),
            "h2": self.find(lambda t, p: t.startswith("가. 각 항 제목"), "가. 각 항 제목"),
            "h3": self.find(lambda t, p: t.startswith("(1) 각 항 제목"), "(1) 각 항 제목"),
            "body": self.find(lambda t, p: t.startswith("여기에 내용을 입력하세요"), "본문"),
            "eq": self.find(lambda t, p: "수식" in t and "(1)" in t, "수식 (1)"),
            "pic": self.find(lambda t, p: has(p, "pic"), "그림"),
            "fig_en": self.find(lambda t, p: t.startswith("Fig. 1."), "Fig. 1."),
            "fig_ko": self.find(lambda t, p: t.startswith("그림") and "국문제목" in t, "그림 1. 국문제목"),
            "tab_en": self.find(lambda t, p: t.startswith("Table 1."), "Table 1."),
            "tab_ko": self.find(lambda t, p: t.startswith("표") and "국문제목" in t, "표 1. 국문제목"),
            "table": self.find(lambda t, p: has(p, "tbl") and "Parameters" in t, "예시 표"),
            "ref": self.find(lambda t, p: t.startswith("(예2) D. L. Goeckel"), "참고문헌 예시"),
            "start": self.find(lambda t, p: t.startswith("Ⅰ. 서론") or t.startswith("Ⅰ.서론"), "Ⅰ. 서론"),
        }
        idx = self.top.index(pr["h2"])
        prev = self.top[idx - 1]
        pr["blank"] = prev if not text_of(prev) else pr["body"]
        return pr

    # -- 문단 만들기

    def new_id(self):
        self.oid += 1
        return str(self.oid)

    @staticmethod
    def make_t(text):
        t = etree.Element(HP + "t")
        parts = text.split("\t")
        t.text = parts[0]
        for part in parts[1:]:
            tab = etree.SubElement(t, HP + "tab", width="4000", leader="0", type="1")
            tab.tail = part
        return t

    def set_text(self, p, text, rich=True):
        """문단의 글자를 바꾼다. 첫 run 의 글자 모양을 바탕으로 쓴다."""
        runs = p.findall(HP + "run")
        base = runs[0].get("charPrIDRef") if runs else "0"
        for r in runs:
            p.remove(r)
        for e in p.findall(HP + "linesegarray"):
            p.remove(e)
        pieces = inline(text) if rich else [(text, False, False)]
        if not pieces:
            pieces = [("", False, False)]
        for k, (s, b, it) in enumerate(pieces):
            run = etree.Element(HP + "run", charPrIDRef=self.chars.get(base, b, it))
            run.append(self.make_t(s))
            p.insert(k, run)
        return p

    def clone(self, key, text=None, rich=True):
        p = copy.deepcopy(self.proto[key])
        p.set("id", str(self.pid))
        self.pid += 1
        for e in p.iter(HP + "linesegarray"):
            e.getparent().remove(e)
        if text is not None:
            self.set_text(p, text, rich)
        return p

    def picture(self, path):
        data = path.read_bytes()
        w_px, h_px, ext = image_size(data, path)
        used = [int(m.group(1)) for it in self.hpf.iter(OPF + "item")
                if (m := re.fullmatch(r"image(\d+)", it.get("id", "")))]
        item_id = f"image{max(used, default=0) + 1}"
        href = f"BinData/{item_id}.{ext}"
        manifest = self.hpf.find(f".//{OPF}manifest")
        etree.SubElement(manifest, OPF + "item", id=item_id, href=href,
                         **{"media-type": "image/png" if ext == "png" else "image/jpeg", "isEmbeded": "1"})
        self.images.append((href, data))

        ow, oh = w_px * PX, h_px * PX
        w = FIG_WIDTH
        h = round(w * h_px / w_px)
        if h > FIG_MAX_H:
            h = FIG_MAX_H
            w = round(h * w_px / h_px)
        p = self.clone("pic")
        pic = p.find(f".//{HP}pic")
        pic.set("id", self.new_id())
        pic.find(HP + "orgSz").attrib.update(dict(width=str(ow), height=str(oh)))
        pic.find(HP + "curSz").attrib.update(dict(width=str(w), height=str(h)))
        rect = pic.find(HP + "imgRect")
        for name, (x, y) in zip(("pt0", "pt1", "pt2", "pt3"), ((0, 0), (w, 0), (w, h), (0, h))):
            rect.find(HC + name).attrib.update(dict(x=str(x), y=str(y)))
        pic.find(HP + "imgClip").attrib.update(dict(left="0", top="0", right=str(ow), bottom=str(oh)))
        pic.find(HP + "imgDim").attrib.update(dict(dimwidth=str(ow), dimheight=str(oh)))
        pic.find(HC + "img").set("binaryItemIDRef", item_id)
        pic.find(HP + "sz").attrib.update(dict(width=str(w), height=str(h)))
        return p

    def table(self, rows):
        p = self.clone("table")
        tbl = p.find(f".//{HP}tbl")
        tbl.set("id", self.new_id())
        trs = tbl.findall(HP + "tr")
        head_tc = copy.deepcopy(trs[0].find(HP + "tc"))
        body_tc = copy.deepcopy(trs[1].find(HP + "tc")) if len(trs) > 1 else copy.deepcopy(head_tc)
        for tr in trs:
            tbl.remove(tr)
        ncol = max(len(r) for r in rows)
        colw = TABLE_WIDTH // ncol
        for ri, row in enumerate(rows):
            tr = etree.SubElement(tbl, HP + "tr")
            for ci in range(ncol):
                tc = copy.deepcopy(head_tc if ri == 0 else body_tc)
                sub = tc.find(HP + "subList")
                cps = sub.findall(HP + "p")
                for extra in cps[1:]:
                    sub.remove(extra)
                self.set_text(cps[0], row[ci] if ci < len(row) else "")
                tc.find(HP + "cellAddr").attrib.update(dict(colAddr=str(ci), rowAddr=str(ri)))
                tc.find(HP + "cellSpan").attrib.update(dict(colSpan="1", rowSpan="1"))
                tc.find(HP + "cellSz").attrib.update(dict(width=str(colw), height=str(ROW_H)))
                tr.append(tc)
        tbl.attrib.update(dict(rowCnt=str(len(rows)), colCnt=str(ncol)))
        tbl.find(HP + "sz").attrib.update(dict(width=str(colw * ncol), height=str(ROW_H * len(rows))))
        return p

    # -- 전체 조립

    def fill_front(self, front):
        title_tbl = self.top[0].find(f".//{HP}tbl")
        if title_tbl is None:
            fail("양식 첫 문단에 제목 표가 없다")
        for p in title_tbl.iter(HP + "p"):
            t = text_of(p)
            if "논문제목" in t:
                self.set_text(p, front["title_ko"], rich=False)
            elif "Paper Title" in t:
                self.set_text(p, front["title_en"], rich=False)
            elif "저자이름" in t:
                self.set_text(p, "", rich=False)  # 신규 접수본은 저자 정보를 넣지 않는다
            elif t.startswith("Abstract"):
                self.set_text(p, "Abstract", rich=False)
            elif "Please put the abstract" in t:
                self.set_text(p, front["abstract"])
            elif t.replace(" ", "").startswith("요약"):
                self.set_text(p, "요  약", rich=False)
            elif "이곳에 한글 요약문" in t:
                self.set_text(p, front["yoyak"])
            elif t.startswith("Key words"):
                self.set_text(p, f"Key words : {front['keywords']}", rich=False)
        for top in self.top:
            if "Corresponding author" in text_of(top) and top.find(f".//{HP}tbl") is not None:
                for p in top.iter(HP + "p"):
                    t = text_of(p)
                    if "신규논문 시 삭제" in t or "Corresponding author" in t:
                        self.set_text(p, "", rich=False)

    def build_body(self, blocks):
        out = []

        def blank():
            if out and out[-1][0] != "blank":
                out.append(("blank", self.clone("blank", "", rich=False)))

        prev = None
        for kind, val in blocks:
            if kind == "part":
                blank()
                key = "refs_head" if val == "References" else "part"
                out.append((kind, self.clone(key, val, rich=False)))
                blank()
            elif kind in ("h1", "h2", "h3"):
                if prev != "part":
                    blank()
                out.append((kind, self.clone(kind, val, rich=False)))
            elif kind == "para":
                out.append((kind, self.clone("body", val)))
            elif kind == "code":
                out.append((kind, self.clone("body", val, rich=False)))
            elif kind == "eq":
                expr, num = val
                text = f"{expr}\t({num})" if num else expr
                out.append((kind, self.clone("eq", text, rich=False)))
            elif kind == "img":
                blank()
                out.append((kind, self.picture(val)))
            elif kind == "fig_en":
                out.append((kind, self.clone("fig_en", val, rich=False)))
            elif kind == "fig_ko":
                out.append((kind, self.clone("fig_ko", f"그림\t{val[0]}.\t{val[1]}", rich=False)))
                blank()
            elif kind == "tab_en":
                blank()
                out.append((kind, self.clone("tab_en", val, rich=False)))
            elif kind == "tab_ko":
                out.append((kind, self.clone("tab_ko", f"표\t{val[0]}.  {val[1]}", rich=False)))
            elif kind == "table":
                out.append((kind, self.table(val)))
                blank()
            elif kind == "ref":
                out.append((kind, self.clone("ref", val)))
            prev = kind
        return [p for _, p in out]

    def assemble(self, front, blocks):
        self.fill_front(front)
        body = self.build_body(blocks)
        start = self.top.index(self.proto["start"])
        for p in self.top[start:]:
            self.section.remove(p)
        for p in body:
            self.section.append(p)
        for e in list(self.section.iter(HP + "linesegarray")):
            e.getparent().remove(e)

    def save(self, out, preview):
        xml = lambda root: etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
        replaced = {
            "Contents/section0.xml": xml(self.section),
            "Contents/header.xml": xml(self.header),
            "Contents/content.hpf": xml(self.hpf),
            "Preview/PrvText.txt": preview.replace("\n", "\r\n").encode("utf-8"),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        with zipfile.ZipFile(tmp, "w") as z:
            for info in self.zin.infolist():
                data = replaced.get(info.filename, self.zin.read(info.filename))
                method = zipfile.ZIP_STORED if info.filename == "mimetype" else zipfile.ZIP_DEFLATED
                z.writestr(zipfile.ZipInfo(info.filename, info.date_time), data, compress_type=method)
            for href, data in self.images:
                z.writestr(zipfile.ZipInfo(href, (2020, 1, 1, 0, 0, 0)), data, compress_type=zipfile.ZIP_STORED)
        tmp.replace(out)


def image_size(data, path):
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack(">II", data[16:24])
        return w, h, "png"
    if data[:2] == b"\xff\xd8":
        i = 2
        while i < len(data) - 9:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            length = struct.unpack(">H", data[i + 2:i + 4])[0]
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                h, w = struct.unpack(">HH", data[i + 5:i + 9])
                return w, h, "jpg"
            i += 2 + length
    fail(f"그림은 PNG 또는 JPG 만 쓸 수 있다: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", type=Path, default=TEMPLATE)
    ap.add_argument("--md", type=Path, default=MD)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if not args.template.exists():
        fail(f"양식이 없다: {args.template}")
    if not args.md.exists():
        fail(f"통합본이 없다: {args.md}. 먼저 scripts/build_paper.py 를 실행한다")

    front, blocks = parse_md(args.md)
    b = Builder(args.template)
    b.assemble(front, blocks)
    preview = "\n".join([front["title_ko"], front["title_en"], front["abstract"]] +
                        [v for k, v in blocks if k in ("part", "h1", "para")])[:1000]
    b.save(args.out, preview)
    counts = {k: sum(1 for kind, _ in blocks if kind == k) for k in ("para", "img", "table", "eq", "ref")}
    print(f"생성: {args.out} (문단 {counts['para']}, 그림 {counts['img']}, 표 {counts['table']}, "
          f"수식 {counts['eq']}, 참고문헌 {counts['ref']})")
    if counts["eq"]:
        print("[안내] 수식은 글자로 들어간다. 제출 전에 한글 수식 편집기로 다시 입력한다")
    return 0


if __name__ == "__main__":
    sys.exit(main())

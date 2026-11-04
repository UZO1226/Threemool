#!/usr/bin/env python3
"""브랜치가 자기 범위의 파일만 바꿨는지 검사한다. 규칙: RULES/BRANCH_RULES.md 4절

사용법:
  python3 scripts/check_scope.py <브랜치 이름> [기준, 기본값 origin/main]
  예: python3 scripts/check_scope.py "$(git branch --show-current)"
"""
import re
import subprocess
import sys
from pathlib import Path

PEOPLE = {"yoo": "Yoo", "lee": "Lee", "park": "Park"}
OWNER_RE = re.compile(r"^<!--\s*owner:\s*([a-z]+)\s*-->")
SECTION_RE = re.compile(r"^Paper/sections/[^/]+\.md$")
ROOT = Path(__file__).resolve().parent.parent


def git(*args):
    return subprocess.run(["git", "-c", "core.quotepath=false", *args],
                          capture_output=True, text=True, cwd=ROOT)


def first_owner(text):
    for line in text.splitlines():
        if line.strip():
            m = OWNER_RE.match(line.strip())
            return m.group(1) if m else None
    return None


def owner_at(ref, path):
    r = git("show", f"{ref}:{path}")
    return first_owner(r.stdout) if r.returncode == 0 else None


def owner_now(path):
    p = ROOT / path
    return first_owner(p.read_text(encoding="utf-8")) if p.exists() else "deleted"


def check_person(who, path, base):
    if path.startswith(PEOPLE[who] + "/"):
        return None
    if path == f"Paper/refs/{who}.md" or path.startswith(f"Paper/figures/{who}_"):
        return None
    if SECTION_RE.match(path):
        before = owner_at(base, path)
        if before is None:
            return "새 절 파일 추가는 team/ 브랜치에서 한다"
        if before != who:
            return f"owner 가 {before} 인 절이다. 담당자에게 요청한다"
        if owner_now(path) not in (who, "deleted"):
            return "owner 줄은 개인 브랜치에서 바꿀 수 없다 (team/ 브랜치에서 배정 변경)"
        return None
    return f"{who}/ 브랜치의 수정 범위가 아니다. 공동 파일은 team/ 브랜치에서 수정한다"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    branch = sys.argv[1]
    base = sys.argv[2] if len(sys.argv) > 2 else "origin/main"
    r = git("diff", "--name-only", "--no-renames", f"{base}...HEAD")
    if r.returncode != 0:
        print(f"[오류] git diff 실패: {r.stderr.strip()}")
        return 2
    files = [f for f in r.stdout.splitlines() if f]
    prefix = branch.split("/", 1)[0] if "/" in branch else ""

    if prefix not in PEOPLE and prefix != "team":
        print(f"[오류] 브랜치 이름 '{branch}' 는 yoo/, lee/, park/, team/ 중 하나로 시작해야 한다")
        return 1

    problems = []
    for f in files:
        if f.startswith("Paper/build/"):
            problems.append((f, "자동 생성 폴더라 직접 수정할 수 없다"))
        elif prefix in PEOPLE:
            why = check_person(prefix, f, base)
            if why:
                problems.append((f, why))

    print(f"브랜치 {branch}, 기준 {base}, 바뀐 파일 {len(files)}개")
    if prefix == "team":
        print("team/ 브랜치: 다른 팀원 승인이 필요하다 (BRANCH_RULES 5절)")
    if any(SECTION_RE.match(f) for f in files):
        print("논문 절이 바뀌었다: 다른 팀원 한 명 이상 승인 후 merge 한다")
    for f, why in problems:
        print(f"[범위 위반] {f}: {why}")
    if problems:
        return 1
    print("범위 검사 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())

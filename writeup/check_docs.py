"""Check the Markdown pages before they go on GitHub.

Flags relative links and images that point at missing files, pandoc-only markup that GitHub shows as raw text
(including any ^ outside $…$ math and 'Table:' captions), images that are PDFs (GitHub won't show them inline), figures in docs/ without a visible caption line under them,
and leftover words. Prints path:line: message for each problem and exits 1 if there are any.

    python3 writeup/check_docs.py                 # README.md, docs/*.md, data/README.md
    python3 writeup/check_docs.py docs/build.md   # just this file
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = [
    "README.md",
    *sorted(str(p.relative_to(ROOT)) for p in (ROOT / "docs").glob("*.md")),
    "data/README.md",
]

LINK = re.compile(r"(!?)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HTML_SRC = re.compile(r"<img[^>]*\ssrc=\"([^\"]+)\"")
PANDOC = [
    (re.compile(r"~[^\s~]+~"), "pandoc subscript ~…~"),
    (re.compile(r"\[@"), "pandoc citation [@…]"),
    (re.compile(r"(?<![\w.])@[a-z]+\d{4}"), "pandoc citation @key"),
    (re.compile(r"\{width="), "pandoc image attribute {width=…}"),
]
MATH = re.compile(r"\$\$.*?\$\$|\$[^$]*\$")  # one-line math only; the pages have no multi-line $$ blocks
LEFTOVER = [r"\bTODO\b", r"Claude", r"LOGBOOK", r"/Users/", r"liam-ser"]


def check(rel):
    path = ROOT / rel
    problems = []
    lines = path.read_text(encoding="utf-8").splitlines()
    in_code = False
    for i, line in enumerate(lines, 1):
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        bare = re.sub(r"`[^`]*`", "", line)  # inline code is allowed to show anything
        targets = [(m.group(1) == "!", m.group(2)) for m in LINK.finditer(bare)]
        targets += [(True, m.group(1)) for m in HTML_SRC.finditer(bare)]
        for is_img, target in targets:
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            file_part = target.split("#")[0]
            if file_part and not (path.parent / file_part).exists():
                problems.append((i, f"missing file: {target}"))
            if is_img and file_part.endswith(".pdf"):
                problems.append((i, f"image is a PDF (GitHub won't show it): {target}"))
            if is_img and rel.startswith("docs/") and not bare.lstrip().startswith("<"):
                nxt = next((ln for ln in lines[i:] if ln.strip()), "")
                if not nxt.startswith("*Figure "):
                    problems.append(
                        (i, f"no '*Figure N.*' caption line under image: {target}")
                    )
        for pat, msg in PANDOC:
            if pat.search(bare):
                problems.append((i, msg))
        if "^" in MATH.sub("", bare):
            problems.append((i, "^ outside $…$ math (GitHub shows it raw)"))
        if line.startswith("Table:"):
            problems.append((i, "pandoc caption 'Table:' (plain text on GitHub; use a '*Table N.*' line)"))
        for word in LEFTOVER:
            if re.search(word, line) and "TODO (Liam)" not in line:
                problems.append((i, f"leftover word: {word.strip(chr(92) + 'b')}"))
    return [f"{rel}:{i}: {msg}" for i, msg in problems]


def main(argv):
    files = argv or DEFAULT
    out = [p for f in files for p in check(f)]
    for p in out:
        print(p)
    if out:
        print(f"{len(out)} problem(s)")
        return 1
    print("docs ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

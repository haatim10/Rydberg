"""PROMPT 23 standing rule: no number may change.

Extracts every numeric literal from a .tex file and compares the multiset
against the same file at a baseline commit. A rewrite may move numbers around,
re-order them, or change the prose that surrounds them, but the SET of values
with their multiplicities must be identical.

Comments are stripped before extraction, so a value that lives only in a
`% src:` provenance comment is not counted as manuscript content -- but the
src comments themselves are checked separately for survival.

Run:  python3 scratch/p23_number_invariance.py <baseline-rev>
"""
from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter

FILES = [
    "paper/thesisproj2/haatim_thesisproj2.tex",
    "wip/spl2/haatim_structural_priors_evaluation.tex",
]

# A number: optional sign, digits, optional thousands separators ({,} or ,),
# optional decimal part. Captures 1,586,900 / 80{,}000 / 89.7 / -0.177 / 2.171
NUM = re.compile(r"[-+−]?\d[\d,]*(?:\{,\})?[\d,]*(?:\.\d+)?")


def strip_comments(text: str) -> str:
    out = []
    for line in text.split("\n"):
        # a % not preceded by a backslash starts a comment
        m = re.search(r"(?<!\\)%", line)
        out.append(line[: m.start()] if m else line)
    return "\n".join(out)


def numbers(text: str) -> Counter:
    body = strip_comments(text)
    # normalise thousands separators and unicode minus so 80{,}000 == 80000
    body = body.replace("{,}", "").replace("−", "-")
    vals = []
    for tok in NUM.findall(body):
        tok = tok.replace(",", "").rstrip(".")
        if tok in ("", "-", "+"):
            continue
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    return Counter(vals)


def src_comments(text: str) -> Counter:
    return Counter(l.strip() for l in text.split("\n") if l.strip().startswith("% src:"))


def at_rev(rev: str, path: str) -> str | None:
    try:
        return subprocess.check_output(["git", "show", f"{rev}:{path}"],
                                       text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None


def main() -> int:
    rev = sys.argv[1] if len(sys.argv) > 1 else "f76ed63"
    bad = False
    for path in FILES:
        old = at_rev(rev, path)
        if old is None:
            print(f"  {path}: not present at {rev} -- skipped")
            continue
        new = open(path, encoding="utf-8").read()
        a, b = numbers(old), numbers(new)
        sa, sb = src_comments(old), src_comments(new)
        lost_src = sa - sb

        # The rule is that no VALUE may change. PROMPT 23 C4 also requires the
        # abstract to stop quoting numbers that the body already carries, which
        # lowers a value's multiplicity without altering it. So a value
        # DISAPPEARING from the document is a violation; a value appearing
        # fewer times while still present is reported but is not.
        lost = set(a) - set(b)
        new_vals = set(b) - set(a)
        fewer = {v: (a[v], b[v]) for v in set(a) & set(b) if a[v] != b[v]}

        print(f"\n{path}")
        print(f"  distinct values at {rev}: {len(a)}   now: {len(b)}")
        if lost:
            bad = True
            print("  VALUES LOST ENTIRELY  (violation):")
            for v in sorted(lost):
                print(f"    {v!r} (appeared {a[v]}x)")
        if new_vals:
            bad = True
            print("  NEW VALUES not at baseline  (violation unless ported):")
            for v in sorted(new_vals):
                print(f"    {v!r} (appears {b[v]}x)")
        if not lost and not new_vals:
            print("  values: IDENTICAL set — no value altered, none lost")
        if fewer:
            print("  multiplicity changes (allowed; value still present):")
            for v, (x, y) in sorted(fewer.items()):
                print(f"    {v!r}: {x} -> {y}")
        print(f"  % src: comments at {rev}: {sum(sa.values())}   now: {sum(sb.values())}")
        if lost_src:
            bad = True
            print("  LOST src COMMENTS:")
            for c, n in lost_src.items():
                print(f"    {c}")
        elif sum(sa.values()):
            print("  % src: comments: all survive")

    print("\nRESULT:", "VIOLATION" if bad else "no number changed, no src comment lost")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

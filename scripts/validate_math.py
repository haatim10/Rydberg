"""Validate every equation/inline-math fragment by intercepting the real calls.

Regex-scraping the source truncates implicitly-concatenated string literals,
so this instead monkeypatches the document builder and checks the exact
strings that would be rendered.
"""
from __future__ import annotations

import re
import sys

import matplotlib
matplotlib.use("Agg")

import make_report_pdf as M

FAILS: list[tuple[str, str, str]] = []

_orig_eq = M.Doc.eq
_orig_para = M.Doc.para
_orig_cap = M.Doc.caption
_orig_bul = M.Doc.bullets
_orig_tab = M.Doc.table

INLINE = re.compile(r"\$((?:[^$\\]|\\.)+?)\$", re.S)


def _check_inline(text, where):
    for m in INLINE.finditer(str(text)):
        frag = M.sanitize(m.group(1))
        if not frag.strip():
            continue
        err = M.check_math(frag)
        if err:
            FAILS.append((where, m.group(1)[:80], err))


def eq(self, latex, tag=None, size=12):
    err = M.check_math(M.sanitize(latex))
    if err:
        FAILS.append(("display", latex[:100], err))
    return _orig_eq(self, latex, tag, size)


def para(self, text, size=9.3, indent=0.0, color="black"):
    _check_inline(text, "para")
    return _orig_para(self, text, size, indent, color)


def caption(self, text):
    _check_inline(text, "caption")
    return _orig_cap(self, text)


def bullets(self, items, size=9.2):
    for it in items:
        _check_inline(it, "bullet")
    return _orig_bul(self, items, size)


def table(self, rows, widths, size=8.4, header=True):
    for row in rows:
        for cell in row:
            _check_inline(cell, "table")
    return _orig_tab(self, rows, widths, size, header)


M.Doc.eq = eq
M.Doc.para = para
M.Doc.caption = caption
M.Doc.bullets = bullets
M.Doc.table = table

import report_content  # noqa: E402

report_content.build()

if FAILS:
    print(f"\n{len(FAILS)} MATH FAILURES\n")
    for where, frag, err in FAILS:
        print(f"  [{where}] {frag}\n      {err}\n")
    sys.exit(1)
print("\nALL MATH PARSES")

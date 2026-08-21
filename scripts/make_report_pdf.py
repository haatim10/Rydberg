"""Build the Rydberg atomic-MIMO technical report as a vector PDF.

Rendered with matplotlib (mathtext) so every equation is real typeset maths
and every embedded figure stays vector where possible. No LaTeX needed.

Every formula in this document was read out of the implementation, not from
memory; scripts/deep_audit.py verifies them numerically.
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPO = Path(__file__).resolve().parent.parent
FIGS = REPO / "results/final_figures"
OUT = REPO / "results/Rydberg_Atomic_MIMO_Report.pdf"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9.5,
})

PAGE = (8.27, 11.69)          # A4 portrait
L, R = 0.085, 0.915           # text margins in figure coords
TOP, BOT = 0.945, 0.055


_SUB = [
    (r"\\bigl", r"\\left"), (r"\\bigr", r"\\right"),
    (r"\\Bigl", r"\\left"), (r"\\Bigr", r"\\right"),
    (r"\\biggl", r"\\left"), (r"\\biggr", r"\\right"),
    (r"\\textbf", r"\\mathbf"), (r"\\textit", r"\\mathit"),
    (r"\\text", r"\\mathrm"),
    (r"\\mathsf\s*\{([^}]*)\}", r"\\mathrm{\1}"),
    (r"\\mathsf\s+([A-Za-z])", r"\\mathrm{\1}"),
    # \le / \ge must not clobber \left / \geq etc.
    (r"\\le(?![a-zA-Z])", r"\\leq"),
    (r"\\ge(?![a-zA-Z])", r"\\geq"),
]


def sanitize(latex: str) -> str:
    """Map the LaTeX subset used here onto what matplotlib mathtext accepts."""
    out = latex
    for a, b in _SUB:
        out = re.sub(a, b, out)
    return out


_MATH = re.compile(r"\$((?:[^$\\\\]|\\\\.)+?)\$", re.S)


def _visual_len(text: str) -> int:
    """Approximate rendered width: LaTeX markup is much longer than it draws."""
    def strip(m):
        body = m.group(1)
        body = re.sub(r"\\\\[a-zA-Z]+", "x", body)
        body = re.sub(r"[{}^_\\\\]", "", body)
        return body
    return len(_MATH.sub(strip, text))


def wrap_width(text: str, base: int) -> int:
    """Widen the wrap column in proportion to how much of the text is markup."""
    vis = max(1, _visual_len(text))
    return int(min(base * (len(text) / vis), base * 4.5))


def check_math(latex: str) -> str | None:
    """Return an error string if mathtext cannot parse ``latex``."""
    from matplotlib import mathtext
    try:
        mathtext.MathTextParser("path").parse(f"${latex}$")
        return None
    except Exception as exc:  # noqa: BLE001
        return str(exc).strip().splitlines()[-1][:120]


class Doc:
    """Simple flowing-text page builder with a cursor."""

    def __init__(self, pdf):
        self.pdf = pdf
        self.fig = None
        self.y = 0.0
        self.page = 0

    def new_page(self, title=None):
        if self.fig is not None:
            self._footer()
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
        self.fig = plt.figure(figsize=PAGE)
        self.page += 1
        self.y = TOP
        if title:
            self.fig.text(L, self.y, title, fontsize=14, weight="bold",
                          va="top")
            self.y -= 0.030
            self.fig.add_artist(plt.Line2D([L, R], [self.y, self.y],
                                           color="0.65", lw=0.8))
            self.y -= 0.020

    def _footer(self):
        self.fig.text(0.5, 0.028, f"{self.page}", ha="center", fontsize=8,
                      color="0.45")
        self.fig.text(R, 0.028, "Rydberg atomic-MIMO — formulae & results",
                      ha="right", fontsize=7, color="0.6")

    def space(self, dy=0.014):
        self.y -= dy

    def need(self, dy, title=None):
        if self.y - dy < BOT:
            self.new_page(title)

    def h2(self, text):
        self.need(0.06)
        self.space(0.010)
        self.fig.text(L, self.y, text, fontsize=11.5, weight="bold", va="top")
        self.y -= 0.026

    def h3(self, text):
        self.need(0.05)
        self.space(0.006)
        self.fig.text(L, self.y, text, fontsize=10, weight="bold",
                      style="italic", va="top", color="0.15")
        self.y -= 0.021

    def para(self, text, size=9.3, indent=0.0, color="black"):
        text = sanitize(text)
        flat = " ".join(text.split())
        wrapped = textwrap.fill(flat, width=wrap_width(flat, int(104 - indent * 300)))
        n = wrapped.count("\n") + 1
        self.need(0.017 * n + 0.008)
        self.fig.text(L + indent, self.y, wrapped, fontsize=size, va="top",
                      linespacing=1.5, color=color)
        self.y -= 0.0165 * n + 0.006

    def eq(self, latex, tag=None, size=12):
        latex = sanitize(latex)
        self.need(0.055)
        self.space(0.006)
        self.fig.text(0.5, self.y, f"${latex}$", fontsize=size, va="top",
                      ha="center")
        if tag:
            self.fig.text(R, self.y - 0.004, tag, fontsize=8.5, va="top",
                          ha="right", color="0.4")
        self.y -= 0.046

    def bullets(self, items, size=9.2):
        for it in items:
            it = sanitize(it)
            flat = " ".join(it.split())
            wrapped = textwrap.fill(flat, width=wrap_width(flat, 98))
            n = wrapped.count("\n") + 1
            self.need(0.017 * n)
            self.fig.text(L + 0.016, self.y, "•", fontsize=size, va="top")
            self.fig.text(L + 0.032, self.y, wrapped, fontsize=size, va="top",
                          linespacing=1.5)
            self.y -= 0.0158 * n + 0.004
        self.space(0.006)

    def table(self, rows, widths, size=8.4, header=True):
        self.need(0.020 * len(rows) + 0.01)
        self.space(0.004)
        for i, row in enumerate(rows):
            x = L
            w = "bold" if (header and i == 0) else "normal"
            for cell, cw in zip(row, widths):
                self.fig.text(x, self.y, sanitize(str(cell)), fontsize=size, va="top",
                              weight=w)
                x += cw
            self.y -= 0.0175
            if header and i == 0:
                self.fig.add_artist(plt.Line2D([L, R], [self.y + 0.006] * 2,
                                               color="0.75", lw=0.6))
                self.y -= 0.004
        self.space(0.008)

    def image(self, path, height=0.30):
        p = Path(path)
        if not p.exists():
            self.para(f"[missing figure: {p.name}]", color="red")
            return
        img = mpimg.imread(p)
        h, w = img.shape[:2]
        width = height * (w / h) * (PAGE[1] / PAGE[0])
        width = min(width, R - L)
        height = width * (h / w) * (PAGE[0] / PAGE[1])
        self.need(height + 0.02)
        ax = self.fig.add_axes([L + (R - L - width) / 2, self.y - height,
                                width, height])
        ax.imshow(img)
        ax.axis("off")
        self.y -= height + 0.012

    def caption(self, text):
        text = sanitize(text)
        flat = " ".join(text.split())
        wrapped = textwrap.fill(flat, width=wrap_width(flat, 110))
        n = wrapped.count("\n") + 1
        self.need(0.016 * n)
        self.fig.text(0.5, self.y, wrapped, fontsize=8.2, va="top",
                      ha="center", color="0.3", linespacing=1.45)
        self.y -= 0.015 * n + 0.008

    def close(self):
        if self.fig is not None:
            self._footer()
            self.pdf.savefig(self.fig)
            plt.close(self.fig)

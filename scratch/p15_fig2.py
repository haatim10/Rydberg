"""Fig. 2, decluttered. Same data, same source, fewer visual conventions.

The previous version carried four different ways of encoding a series -- a
solid grey line with no marker for N=32, a dashed blue line with circles for
N=16, a dotted red line with triangles for N=64, plus a floating "N = 32" text
label competing with a two-entry legend -- and a dashed zero rule on top of a
full grid. This draws all three series the same way, puts all three in one
legend, and keeps a single light horizontal rule at the level that matters.

Data source is unchanged: reports/trackD_partB9_analysis.json (B1_collapse) for
N in {16, 64}, and the frozen eight-point A2 relation for N=32, which is the
same relation used for both cross-generator predictions and is not refitted.

Run:  PYTHONPATH=. python3 scratch/p15_fig2.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("paper/paper1/fig")
SRC = Path("reports/trackD_partB9_analysis.json")

# The A2 relation, as used for BOTH cross-generator predictions. Not refitted.
# Copied verbatim from scratch/paper1_figures.py:56.
A2_CURVE = [(0.119, 7.043), (0.212, 3.556), (0.285, 1.792), (0.356, 1.038),
            (0.408, 0.577), (0.460, 0.266), (0.507, 0.046), (0.546, -0.117)]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8880", "axes.linewidth": 0.7,
    "xtick.color": "#52514e", "ytick.color": "#52514e",
})

SERIES = {
    "16": ("#2a78d6", "o"),
    "32": ("#4a4a48", "s"),
    "64": ("#e34948", "^"),
}


def main():
    b1 = json.loads(SRC.read_text())["B1_collapse"]

    fig, ax = plt.subplots(figsize=(3.45, 2.35))

    # One light rule at the level the section is about, and a horizontal-only
    # grid: the vertical grid added ruling without adding information.
    ax.axhline(0, color="#b9b7b1", lw=0.7, zorder=0)
    ax.grid(axis="y", color="#d8d6d0", lw=0.4, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)

    for N in ("16", "32", "64"):
        col, mk = SERIES[N]
        if N == "32":
            px, py = zip(*A2_CURVE)
        else:
            px, py = zip(*sorted(b1["delta_by_N"][N]))
        ax.plot(px, py, "-", color=col, marker=mk, ms=3.6, mew=0, lw=1.3,
                zorder=3, label=rf"$N={N}$")

    ax.set_xlabel(r"$\rho = r_{\mathrm{eff}}/r_{\max}$")
    ax.set_ylabel(r"$\Delta_{\mathrm{HS}}$ (dB)")
    ax.set_xlim(0.10, 0.57)
    ax.legend(frameon=False, loc="upper right", handlelength=1.8,
              labelspacing=0.25, borderpad=0.2, handletextpad=0.5)

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig2_boundary_clean.{ext}", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / "fig2_boundary_clean.pdf")


if __name__ == "__main__":
    main()

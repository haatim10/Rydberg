"""PROMPT 19 E1 -- Fig. 5 for Paper 2: every bin, every seed.

The 18-of-18 result is the strongest single thing in the project and it has
been a table row. This makes it the figure.

Form
----
A dot-and-interval plot, not grouped bars. The job is polarity first (is the
cell above zero?) and magnitude second, with an interval on every point; bars
would spend ink on the distance from zero to the mark and make the CI the
subordinate element, which inverts the question the panel exists to answer.

Two panels, not one axis with two scales. Values run from +0.04 to +3.69 dB, so
on a single linear axis the two lowest bins collapse onto the zero line --
which is exactly the claim the figure is meant to show. The lower panel is the
same encoding on the two lowest bins alone, at an expanded scale. A second
y-axis on one panel would be the standard dual-axis mistake.

Identity is never colour alone: each seed carries its own marker shape as well
as its own hue, because IEEE prints in grayscale. The palette was run through
the validator on the white print surface -- all checks pass, with a contrast
WARN against the surface that the dark marker edges and the legend discharge.

Run:  PYTHONPATH=. python3 scratch/p19_fig5_allcells.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCORE = Path("reports/p16/p27_score.json")
OUT = Path("wip/spl2/fig")

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.0,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8880", "axes.linewidth": 0.7,
    "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "lines.linewidth": 1.4,
})
BLUE, AMBER, AQUA = "#2a78d6", "#eda100", "#1baf7a"
INK, MUTED = "#0b0b0b", "#52514e"
# Interval labels, rotated, to match Fig. 2 on the facing column -- two
# different bin-labelling conventions in adjacent figures is the kind of thing
# a reader notices and an author does not. Horizontal interval labels collide
# at 3.45 in, so they are rotated exactly as Fig. 2 rotates its own.
BINLAB = ["$[-10,-5)$", "$[-5,0)$", "$[0,5)$", "$[5,10)$", "$[10,15)$",
          "$[15,20)$"]
SEEDS = [("seed1", BLUE, "o"), ("seed2", AMBER, "s"), ("seed3", AQUA, "^")]


def draw(ax, d, bins, ylim, show_xlabel, dodge):
    for j, (tag, col, mk) in enumerate(SEEDS):
        s = d["per_seed"][tag]
        x = np.asarray(bins, dtype=float) + (j - 1) * dodge
        y = np.asarray([s["per_bin_db"][b] for b in bins])
        ci = np.asarray([s["per_bin_ci95_over_test_realisations"][b]
                         for b in bins])
        err = np.vstack([y - ci[:, 0], ci[:, 1] - y])
        ax.errorbar(x, y, yerr=err, fmt="none", ecolor=col, elinewidth=1.0,
                    capsize=1.8, capthick=1.0, zorder=2)
        ax.plot(x, y, mk, color=col, markersize=4.0, markeredgecolor="white",
                markeredgewidth=0.6, linestyle="none", zorder=3,
                label=f"seed {tag[-1]}")
    ax.axhline(0.0, color=INK, linewidth=0.8, zorder=1)
    ax.set_xticks(list(bins))
    # Six interval labels only fit rotated; two fit flat, and rotating them
    # would spend a fifth of the panel's height on two words.
    rot = len(list(bins)) > 2
    ax.set_xticklabels([BINLAB[b] for b in bins],
                       rotation=45 if rot else 0,
                       ha="right" if rot else "center",
                       rotation_mode="anchor" if rot else None, fontsize=6.8)
    ax.set_xlim(min(bins) - 0.5, max(bins) + 0.5)
    ax.set_ylim(*ylim)
    ax.set_ylabel("gain (dB)")
    if show_xlabel:
        ax.set_xlabel("SNR bin (dB)")


def main() -> int:
    d = json.loads(SCORE.read_text())
    vals = np.array([d["per_seed"][t]["per_bin_db"] for t, _, _ in SEEDS])
    assert (vals > 0).all(), "figure asserts 18 of 18 positive; data disagrees"

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(3.45, 3.30), height_ratios=[2.0, 1.0])

    draw(ax1, d, range(6), (-0.25, 4.15), show_xlabel=False, dodge=0.20)
    ax1.legend(loc="upper left", frameon=False, handletextpad=0.3,
               borderaxespad=0.2, labelspacing=0.25)

    # Same encoding, the two down-weighted bins only, at an expanded scale.
    draw(ax2, d, range(2), (-0.06, 0.52), show_xlabel=True, dodge=0.12)
    ax2.set_title("the two down-weighted bins, at "
                  "$\\approx\\!1/18$ and $1/9$ weight",
                  fontsize=7.0, color=MUTED, pad=3)

    fig.align_ylabels([ax1, ax2])
    fig.tight_layout(pad=0.3, h_pad=0.7)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig5_allcells.{ext}")
    plt.close(fig)
    print(f"  wrote {OUT / 'fig5_allcells.pdf'}")
    print(f"  18 cells, min {vals.min():+.4f} dB, max {vals.max():+.4f} dB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

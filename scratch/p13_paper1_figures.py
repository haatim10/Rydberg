"""PROMPT 13 -- Paper 1 figures, rebuilt on the bridging run.

Fig. 1 changes in two ways required by the brief and by PROMPT 12:

  * the curves are now the TD operator at the B3 principal cell (results/p12/
    B1.json), which the B1 decision rule made THE proposed estimator, instead
    of the old family-B3 curves;
  * the geometric CCRB is the PRIMARY reference and the unconstrained CRLB is
    demoted to an annotated secondary line, because a skimming reader takes a
    curve below a line labelled "CRLB" as an error however carefully the bias
    discussion explains it.

Both bounds are computed at exactly this operating point (N=32, K=3, P=30,
RSR 12 dB), so the overlay is like-for-like and no family is mixed.

Run:  PYTHONPATH=. python3 scratch/p13_paper1_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("paper/paper1/fig")
SNRS = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "lines.linewidth": 1.2, "lines.markersize": 3.8,
    "axes.grid": True, "grid.alpha": 0.28, "grid.linewidth": 0.4,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def key(s):
    return f"N32_P30_snr{'+' if s >= 0 else '-'}{abs(int(s))}"


def fig1():
    b1 = json.loads(Path("results/p12/B1.json").read_text())
    cc = json.loads(Path("results/track_b/constrained_crlb.json").read_text())
    em = np.array([p["em_gs_db"] for p in b1["points"]])
    hs = np.array([p["hs_gs_db"] for p in b1["points"]])
    ccrb = np.array([cc["constrained"]["b3"][key(s)] for s in SNRS])
    crlb = np.array([cc["unconstrained_rank1"]["b3"][key(s)] for s in SNRS])
    x = np.array(SNRS)

    fig, ax = plt.subplots(figsize=(3.45, 2.55))

    # PRIMARY reference: the geometric CCRB, solid and prominent.
    ax.plot(x, ccrb, "-", color="#1a1a1a", lw=1.5, zorder=4,
            label=r"geometric CCRB (primary reference)")
    # SECONDARY: the unconstrained bound, faint, dotted, annotated in place.
    ax.plot(x, crlb, ls=(0, (1, 2.2)), color="#9b9992", lw=1.0, zorder=2)

    ax.plot(x, em, "s--", color="#2a78d6", mew=0, zorder=5, label="EM-GS")
    ax.plot(x, hs, "o-", color="#e34948", mew=0, zorder=6,
            label="HS-GS (proposed)")

    ax.annotate("unconstrained CRLB\n(does not model the prior)",
                xy=(x[2], crlb[2]), xytext=(-2, 26), textcoords="offset points",
                fontsize=6.4, color="#52514e", ha="left",
                arrowprops=dict(arrowstyle="-", color="#9b9992", lw=0.6,
                                shrinkA=1, shrinkB=2))

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("NMSE (dB)")
    ax.set_xticks(x)
    ax.legend(frameon=False, loc="lower left", handlelength=2.0,
              labelspacing=0.28, borderpad=0.2)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_nmse_bridging.{ext}", facecolor="white")
    plt.close(fig)

    # Numbers the manuscript quotes, printed so they can be checked.
    d = em - hs
    print(f"  HS-GS over EM-GS (ratio-of-sums): {d.min():.3f}..{d.max():.3f} dB")
    print(f"  EM-GS minus unconstrained CRLB, SNR>=5: "
          f"max |gap| {np.max(np.abs(em[2:] - crlb[2:])):.3f} dB")
    print(f"  CCRB below unconstrained CRLB: "
          f"{(crlb - ccrb).min():.3f}..{(crlb - ccrb).max():.3f} dB")
    print(f"  HS-GS above CCRB at all six points: "
          f"{bool(np.all(hs > ccrb))}  "
          f"({(hs - ccrb).min():.3f}..{(hs - ccrb).max():.3f} dB)")
    print("  wrote", OUT / "fig1_nmse_bridging.pdf")


if __name__ == "__main__":
    fig1()

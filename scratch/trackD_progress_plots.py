"""Track D progress plots: arm 1a training curves + Part A classical headroom.

Palette: the dataviz reference categorical slots, validated with
scripts/validate_palette.js --mode light.

  2-series (curves), --pairs all : ALL CHECKS PASS
  5-series (A2), adjacent        : ALL CHECKS PASS, contrast WARN on aqua /
                                   yellow / magenta -> relief rule applied by
                                   DIRECT LABELLING every series on the plot.

The repo's own conventions (audit item 19) are kept: Agg, rcParams set once,
figsize (3.6,2.7) single / (7.0,2.7) two-panel, dpi 200, both .png and .pdf,
output under results/track_d/.

Run:  PYTHONPATH=. python3 scratch/trackD_progress_plots.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("results/track_d/progress")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
})

# Validated categorical slots (dataviz references/palette.md, light mode).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")
INK, MUTED = "#0b0b0b", "#52514e"


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {OUT/name}.png")


# ---------------------------------------------------------------------------
# 1. arm 1a training curves
# ---------------------------------------------------------------------------
rows = list(csv.DictReader(
    open("results/track_d/stage1/arm1a_full_warmstart/curves.csv")))
ep = [int(r["epoch"]) for r in rows]
tr = [float(r["train_loss_db"]) for r in rows]
va = [float(r["val_nmse_db"]) for r in rows]
al = [float(r["mean_alpha"]) for r in rows]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))

ax.plot(ep, tr, color=BLUE, lw=2, marker="o", ms=3.5, label="train")
ax.plot(ep, va, color=ORANGE, lw=2, marker="s", ms=3.5, label="validation")
# Direct labels at the last point (selective, not every point).
ax.annotate(f"train {tr[-1]:.2f}", (ep[-1], tr[-1]), textcoords="offset points",
            xytext=(4, -8), color=BLUE, fontsize=7, fontweight="bold")
ax.annotate(f"val {va[-1]:.2f}", (ep[-1], va[-1]), textcoords="offset points",
            xytext=(4, 4), color=ORANGE, fontsize=7, fontweight="bold")
ax.set_xlabel("epoch")
ax.set_ylabel("NMSE (dB)")
ax.set_title(f"arm 1a — training, {len(ep)}/50 epochs", color=INK)
ax.legend(loc="upper right", framealpha=0.9, edgecolor="none")
ax.set_xlim(-0.5, max(50, ep[-1] + 1))

ax2.plot(ep, al, color=AQUA, lw=2, marker="o", ms=3.5)
ax2.axhline(0.1192, color=MUTED, ls=":", lw=1)
ax2.annotate("init 0.119 (near-GS)", (0, 0.1192), textcoords="offset points",
             xytext=(4, -10), color=MUTED, fontsize=6.5)
ax2.annotate(f"{al[-1]:.3f}", (ep[-1], al[-1]), textcoords="offset points",
             xytext=(4, 2), color=AQUA, fontsize=7, fontweight="bold")
ax2.set_xlabel("epoch")
ax2.set_ylabel(r"mean gate $\alpha$")
ax2.set_title(r"gate drifts toward EM-GS ($\alpha\!\to\!1$)", color=INK)
ax2.set_xlim(-0.5, max(50, ep[-1] + 1))

fig.tight_layout()
_save(fig, "arm1a_training")

# ---------------------------------------------------------------------------
# 2. Part A: classical estimators vs the oracle-phase line
# ---------------------------------------------------------------------------
A = json.load(open("reports/trackD_partA.json"))["A2"]["rows"]
snr = [r["snr_db"] for r in A]

series = [
    ("GS (spectral)", "gs_spectral_db", BLUE, "o"),
    ("EM-GS (spectral)", "em_gs_spectral_db", ORANGE, "s"),
    ("linearised LS", "linearised_ls_db", MAGENTA, "^"),
    ("oracle phase", "oracle_phase_db", AQUA, "D"),
]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
for label, key, c, m in series:
    y = [r[key] for r in A]
    ls = "--" if key == "oracle_phase_db" else "-"
    ax.plot(snr, y, color=c, lw=2, marker=m, ms=3.5, ls=ls, label=label)
    # Relief for the contrast-WARN slots: label every series directly.
    ax.annotate(label.split(" (")[0], (snr[-1], y[-1]),
                textcoords="offset points", xytext=(4, -2), color=c,
                fontsize=6.5, fontweight="bold")
ax.set_xlabel("SNR (dB)")
ax.set_ylabel("channel NMSE (dB)")
ax.set_title("Part A — classical vs oracle-phase, N=32", color=INK)
ax.legend(loc="lower left", framealpha=0.9, edgecolor="none")
ax.set_xlim(-11, 27)

hd = [r["headroom_emgs_minus_oracle_db"] for r in A]
ax2.plot(snr, hd, color=YELLOW, lw=2, marker="o", ms=4)
ax2.axhline(3.0, color=MUTED, ls="--", lw=1)
ax2.annotate("go/no-go floor 3 dB", (-10, 3.0), textcoords="offset points",
             xytext=(2, 4), color=MUTED, fontsize=6.5)
ax2.axhline(2.0, color="#e34948", ls=":", lw=1)
ax2.annotate("STOP below 2 dB", (-10, 2.0), textcoords="offset points",
             xytext=(2, -10), color="#e34948", fontsize=6.5)
for x, y in zip(snr, hd):
    ax2.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                 xytext=(0, 5), ha="center", color=INK, fontsize=6)
ax2.set_xlabel("SNR (dB)")
ax2.set_ylabel("EM-GS − oracle (dB)")
ax2.set_title("headroom: 4.34–5.31 dB, all points pass", color=INK)
ax2.set_ylim(0, 6.5)
ax2.set_xlim(-11, 21)

fig.tight_layout()
_save(fig, "partA_headroom")
print("done")

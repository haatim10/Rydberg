"""Track D stage-2 figures (PROMPT 5 Part C).

Palette: validated dataviz categorical slots. Repo conventions kept.
Run:  PYTHONPATH=. python3 scratch/trackD_stage2_plots.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("results/track_d/stage2")
D2 = json.load(open("reports/trackD_stage2_results.json"))
D1 = json.load(open("reports/trackD_stage1_results.json"))

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
})
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#e34948")
INK, MUTED = "#0b0b0b", "#52514e"


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {OUT/name}.png")


# ---------------------------------------------------------------------------
# 1. THE headline: paired improvement vs data budget
# ---------------------------------------------------------------------------
pv1, pv2 = D1["test"]["paired_vs_em_gs"], D2["test"]["paired_vs_em_gs"]
n = [20_000, 40_000, 80_000]
full = [-pv1["arm1b_full_random"]["median_diff_db"],
        -pv2["B2_40k_25ep"]["median_diff_db"],
        -pv2["B3_80k_13ep"]["median_diff_db"]]
lo = [pv1["arm1b_full_random"]["boot_ci95_median"],
      pv2["B2_40k_25ep"]["boot_ci95_median"],
      pv2["B3_80k_13ep"]["boot_ci95_median"]]
err = np.array([[f - (-c[1]), (-c[0]) - f] for f, c in zip(full, lo)]).T
oracle = -pv2["oracle_phase"]["median_diff_db"]
fo = [-pv1["arm2_filteronly_warmstart"]["median_diff_db"], None,
      -pv2["B3_80k_13ep_filteronly"]["median_diff_db"]]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))
ax.axhspan(2.0, 5.0, color=AQUA, alpha=0.07)
ax.errorbar(n, full, yerr=err, color=BLUE, lw=2, marker="o", ms=6, capsize=3,
            label="full URformer")
ax.plot([n[0], n[2]], [fo[0], fo[2]], color=AQUA, lw=2, marker="D", ms=5,
        ls="--", label="filter-only (980 p)")
ax.axhline(oracle, color=MAGENTA, ls="--", lw=1.5)
ax.annotate(f"unstructured-LS oracle {oracle:.2f}", (21_000, oracle),
            textcoords="offset points", xytext=(0, 4), color=MAGENTA,
            fontsize=6.5, fontweight="bold")
ax.axhline(2.0, color=RED, ls="--", lw=1.5)
ax.annotate("bar 2 dB", (76_000, 2.0), textcoords="offset points",
            xytext=(0, -11), color=RED, fontsize=7, fontweight="bold",
            ha="right")
for x, y in zip(n, full):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 9),
                ha="center", color=BLUE, fontsize=7, fontweight="bold")
ax.set_xscale("log"); ax.set_xticks(n); ax.set_xticklabels(["20k", "40k", "80k"])
ax.set_xlabel("training samples (matched compute ~1M passes)")
ax.set_ylabel("median gain over EM-GS (dB)")
ax.set_title("bar CLEARED at 40k and 80k", color=INK)
ax.legend(loc="center left", framealpha=0.9, edgecolor="none")
ax.set_ylim(-0.3, 4.7)

# overfitting gap shrinks with data
gaps = [4.0,
        D2["runs"]["B2_40k_25ep"]["history"][-1]["val_nmse_db"]
        - D2["runs"]["B2_40k_25ep"]["history"][-1]["train_loss_db"],
        D2["runs"]["B3_80k_13ep"]["history"][-1]["val_nmse_db"]
        - D2["runs"]["B3_80k_13ep"]["history"][-1]["train_loss_db"]]
ax2.plot(n, gaps, color=ORANGE, lw=2, marker="s", ms=6)
for x, y in zip(n, gaps):
    ax2.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8),
                 ha="center", color=ORANGE, fontsize=7, fontweight="bold")
ax2.axhline(0, color=MUTED, lw=1)
ax2.set_xscale("log"); ax2.set_xticks(n); ax2.set_xticklabels(["20k", "40k", "80k"])
ax2.set_xlabel("training samples")
ax2.set_ylabel("final train−val gap (dB)")
ax2.set_title("overfitting collapses: 4.0 → 0.65 dB", color=INK)
ax2.set_ylim(-0.4, 4.6)
fig.tight_layout(); _save(fig, "stage2_data_scaling")

# ---------------------------------------------------------------------------
# 2. attribution at 80k vs 20k
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.6, 2.7))
labels = ["20k", "80k"]
filt = [-pv1["arm2_filteronly_warmstart"]["median_diff_db"],
        -pv2["B3_80k_13ep_filteronly"]["median_diff_db"]]
tot = [-pv1["arm1b_full_random"]["median_diff_db"],
       -pv2["B3_80k_13ep"]["median_diff_db"]]
tf = [t - f for t, f in zip(tot, filt)]
y = np.arange(2)
ax.barh(y, filt, color=AQUA, height=0.5, label="gated filter (980 p)")
ax.barh(y, tf, left=filt, color=BLUE, height=0.5, label="Transformer (+1.586M p)")
ax.axvline(2.0, color=RED, ls="--", lw=1.5)
ax.axvline(oracle, color=MAGENTA, ls="--", lw=1.5)
ax.annotate("bar", (2.0, 1.5), color=RED, fontsize=6.5, ha="center",
            fontweight="bold")
ax.annotate("oracle", (oracle, 1.5), color=MAGENTA, fontsize=6.5, ha="center",
            fontweight="bold")
for i in range(2):
    ax.annotate(f"{filt[i]:.2f}", (filt[i] / 2, y[i]), ha="center", va="center",
                color="white", fontsize=6.5, fontweight="bold")
    ax.annotate(f"+{tf[i]:.2f}", (filt[i] + tf[i] / 2, y[i]), ha="center",
                va="center", color="white", fontsize=7, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.set_ylabel("training samples")
ax.set_xlabel("median gain over EM-GS (dB)")
ax.set_title("Transformer share: 80% → 96%", color=INK)
ax.legend(loc="lower right", framealpha=0.9, edgecolor="none", fontsize=6)
ax.grid(axis="y", alpha=0); ax.set_xlim(0, 4.7)
fig.tight_layout(); _save(fig, "stage2_attribution")

# ---------------------------------------------------------------------------
# 3. training curves
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.6, 2.7))
runs = [("B2_40k_25ep", "40k / 25 ep", ORANGE),
        ("B3_80k_13ep", "80k / 13 ep", BLUE),
        ("B3_80k_13ep_filteronly", "80k filter-only", AQUA)]
for key, lab, c in runs:
    rows = list(csv.DictReader(open(OUT / key / "curves.csv")))
    fr = [(int(r["epoch"]) + 1) / len(rows) for r in rows]
    va = [float(r["val_nmse_db"]) for r in rows]
    ax.plot(fr, va, color=c, lw=1.8, label=lab)
    ch = D2["runs"][key]["chosen_epoch"]
    ax.plot([(ch + 1) / len(rows)], [va[ch]], marker="o", ms=6, color=c,
            mec="white", mew=1.2, zorder=5)
r1 = list(csv.DictReader(open("results/track_d/stage1/arm1b_full_random/curves.csv")))
ax.plot([(int(r["epoch"]) + 1) / len(r1) for r in r1],
        [float(r["val_nmse_db"]) for r in r1], color=MUTED, lw=1.5, ls=":",
        label="20k / 50 ep (stage 1)")
ax.set_xlabel("fraction of training budget")
ax.set_ylabel("validation NMSE (dB)")
ax.set_title("markers = one-SE selected epoch", color=INK)
ax.legend(loc="upper right", framealpha=0.9, edgecolor="none", fontsize=6.5)
fig.tight_layout(); _save(fig, "stage2_curves")
print("done")

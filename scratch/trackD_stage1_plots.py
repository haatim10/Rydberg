"""Track D stage-1 result figures (PROMPT 4 Part C).

Palette: validated dataviz categorical slots (validate_palette.js, light mode,
ALL CHECKS PASS). Contrast-WARN slots carry direct labels per the relief rule.
Repo conventions kept: Agg, rcParams once, dpi 200, .png + .pdf,
results/track_d/stage1/.

Run:  PYTHONPATH=. python3 scratch/trackD_stage1_plots.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("results/track_d/stage1")
D = json.load(open("reports/trackD_stage1_results.json"))

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

ARMS = {
    "arm1a_full_warmstart": ("arm 1a full, warm start", BLUE, "o"),
    "arm1b_full_random": ("arm 1b full, random", ORANGE, "s"),
    "arm2_filteronly_warmstart": ("arm 2 filter-only (980p)", AQUA, "D"),
}


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {OUT/name}.png")


# ---------------------------------------------------------------------------
# 1. training curves, all three arms
# ---------------------------------------------------------------------------
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
for arm, (lab, c, m) in ARMS.items():
    rows = list(csv.DictReader(open(OUT / arm / "curves.csv")))
    ep = [int(r["epoch"]) for r in rows]
    va = [float(r["val_nmse_db"]) for r in rows]
    al = [float(r["mean_alpha"]) for r in rows]
    best = D["arms"][arm]["best_epoch"]
    ax.plot(ep, va, color=c, lw=1.8, label=lab)
    ax.plot([best], [va[best]], marker=m, ms=7, color=c, mec="white", mew=1.2,
            zorder=5)
    ax.annotate(f"{va[best]:.2f}", (best, va[best]), textcoords="offset points",
                xytext=(3, 6), color=c, fontsize=6.5, fontweight="bold")
    ax2.plot(ep, al, color=c, lw=1.8, label=lab)
    ax2.annotate(f"{al[-1]:.3f}", (ep[-1], al[-1]), textcoords="offset points",
                 xytext=(3, -1), color=c, fontsize=6.5, fontweight="bold")

ax.set_xlabel("epoch"); ax.set_ylabel("validation NMSE (dB)")
ax.set_title("validation — markers = selected checkpoint", color=INK)
ax.legend(loc="upper right", framealpha=0.9, edgecolor="none")
ax2.axhline(0.1192, color=MUTED, ls=":", lw=1)
ax2.annotate("init 0.119", (0, 0.1192), textcoords="offset points",
             xytext=(2, 4), color=MUTED, fontsize=6.5)
ax2.set_xlabel("epoch"); ax2.set_ylabel(r"mean gate $\alpha$")
ax2.set_title(r"gate: filter-only leans hardest on $R(\kappa)$", color=INK)
fig.tight_layout(); _save(fig, "stage1_training_curves")

# ---------------------------------------------------------------------------
# 2. THE decision figure: paired difference vs EM-GS
# ---------------------------------------------------------------------------
pv = D["test"]["paired_vs_em_gs"]
per = D["test"]["per_trial_nmse"]
base = 10 * np.log10(np.array(per["em_gs_spectral"]))

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))
for arm, (lab, c, m) in ARMS.items():
    d = 10 * np.log10(np.array(per[arm])) - base
    xs = np.sort(d)
    ax.plot(xs, np.linspace(0, 100, len(xs)), color=c, lw=1.8, label=lab)
    med = pv[arm]["median_diff_db"]
    ax.plot([med], [50], marker=m, ms=7, color=c, mec="white", mew=1.2, zorder=5)
d = 10 * np.log10(np.array(per["oracle_phase"])) - base
ax.plot(np.sort(d), np.linspace(0, 100, len(d)), color=MAGENTA, lw=1.8, ls="--",
        label="oracle phase (floor)")
ax.axvline(0, color=MUTED, lw=1)
ax.axvline(-2.0, color=RED, ls="--", lw=1.5)
ax.annotate("bar: −2 dB", (-2.0, 8), color=RED, fontsize=7, fontweight="bold",
            rotation=90, textcoords="offset points", xytext=(-10, 0))
ax.set_xlabel("paired NMSE difference vs EM-GS (dB)")
ax.set_ylabel("percentile")
ax.set_title("paired difference — ECDF, 2000 trials", color=INK)
ax.legend(loc="lower right", framealpha=0.9, edgecolor="none", fontsize=6.5)
ax.set_xlim(-9, 6)

names = list(ARMS) + ["oracle_phase"]
labs = [ARMS[a][0] for a in ARMS] + ["oracle phase"]
cols = [ARMS[a][1] for a in ARMS] + [MAGENTA]
y = np.arange(len(names))[::-1]
for i, (n, c) in enumerate(zip(names, cols)):
    v = pv[n]
    lo, hi = v["boot_ci95_median"]
    ax2.plot([lo, hi], [y[i], y[i]], color=c, lw=3, solid_capstyle="round")
    ax2.plot([v["median_diff_db"]], [y[i]], marker="o", ms=6, color=c,
             mec="white", mew=1.2, zorder=5)
    ax2.annotate(f"{v['median_diff_db']:+.2f}", (v["median_diff_db"], y[i]),
                 textcoords="offset points", xytext=(0, 8), ha="center",
                 color=c, fontsize=7, fontweight="bold")
ax2.axvline(0, color=MUTED, lw=1)
ax2.axvline(-2.0, color=RED, ls="--", lw=1.5)
ax2.annotate("bar −2 dB", (-2.0, 3.35), color=RED, fontsize=6.5,
             fontweight="bold", ha="center")
ax2.set_yticks(y); ax2.set_yticklabels(labs, fontsize=7)
ax2.set_xlabel("median paired difference (dB), 95% bootstrap CI")
ax2.set_title("NOT MET — best arm is −1.43 dB", color=RED)
ax2.set_xlim(-5.2, 0.6); ax2.set_ylim(-0.6, 3.7)
ax2.grid(axis="y", alpha=0)
fig.tight_layout(); _save(fig, "stage1_paired_difference")

# ---------------------------------------------------------------------------
# 3. attribution: what the 1.59M parameters bought
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.6, 2.7))
gain_filter = -pv["arm2_filteronly_warmstart"]["median_diff_db"]
gain_full = -pv["arm1a_full_warmstart"]["median_diff_db"]
gain_tf = gain_full - gain_filter
head = -pv["oracle_phase"]["median_diff_db"]

ax.barh([0], [gain_filter], color=AQUA, height=0.55,
        label=f"gated filter, 980 p ({gain_filter:.2f} dB)")
ax.barh([0], [gain_tf], left=[gain_filter], color=BLUE, height=0.55,
        label=f"Transformer, +1.586M p ({gain_tf:.2f} dB)")
ax.barh([1], [head], color=MAGENTA, height=0.55,
        label=f"oracle-phase headroom ({head:.2f} dB)")
ax.axvline(2.0, color=RED, ls="--", lw=1.5)
ax.annotate("bar 2 dB", (2.0, 1.55), color=RED, fontsize=6.5,
            fontweight="bold", ha="center")
ax.annotate(f"{gain_filter:.2f}", (gain_filter / 2, 0), ha="center", va="center",
            color="white", fontsize=7, fontweight="bold")
ax.annotate(f"+{gain_tf:.2f}", (gain_filter + gain_tf / 2, 0), ha="center",
            va="center", color="white", fontsize=7, fontweight="bold")
ax.annotate(f"{head:.2f}", (head / 2, 1), ha="center", va="center",
            color="white", fontsize=7, fontweight="bold")
ax.set_yticks([0, 1])
ax.set_yticklabels(["URformer\ngain", "available\nheadroom"], fontsize=7)
ax.set_xlabel("median gain over EM-GS (dB)")
ax.set_title(f"{100*gain_tf/gain_full:.0f}% of the gain is the Transformer",
             color=INK)
ax.legend(loc="lower right", framealpha=0.9, edgecolor="none", fontsize=6)
ax.grid(axis="y", alpha=0)
ax.set_xlim(0, 4.6)
fig.tight_layout(); _save(fig, "stage1_attribution")
print("done")

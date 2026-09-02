"""PROMPT 9 Part A figures: the two normalization collapses.

Palette: validated dataviz slots (#2a78d6, #1baf7a, #eda100, #e34948, #7a4fd6
passed all six checks under the skill's validator in the PROMPT 8 pass).

Run:  PYTHONPATH=. python3 scratch/trackD_partA9_plots.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D = json.load(open("reports/trackD_partA9_normalization.json"))
OUT = Path("results/track_d/normalization")

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
    "legend.fontsize": 7.6, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8880", "axes.linewidth": 0.8,
    "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
})
BLUE, AQUA, AMBER, RED, PURPLE = "#2a78d6", "#1baf7a", "#eda100", "#e34948", "#7a4fd6"
INK, MUTED = "#0b0b0b", "#52514e"

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.5))

# ---- A2: Delta_HS re-indexed by effective rank ---------------------------
rows = sorted(D["A2"]["trackB_reindexed"], key=lambda r: r["r_eff_over_cap"])
x = np.array([r["r_eff_over_cap"] for r in rows])
y = np.array([r["delta_hs_db"] for r in rows])
lo = np.array([r["ci"][0] for r in rows])
hi = np.array([r["ci"][1] for r in rows])
xL = np.array([r["L_over_cap"] for r in rows])

ax.plot(xL, y, color=MUTED, lw=1.3, ls=(0, (4, 2)), marker="s", ms=4,
        label="indexed by $L/\\mathrm{cap}$ (old)")
ax.errorbar(x, y, yerr=[y - lo, hi - y], color=BLUE, lw=2.2, marker="o", ms=5,
            capsize=3, label="indexed by $r_{\\rm eff}/\\mathrm{cap}$ (new)")
ax.axhline(0, color=INK, lw=1.0)
zc = 0.518
ax.axvline(zc, color=RED, ls=":", lw=1.3)
ax.annotate(f"zero at {zc:.2f}", (zc, 5.4), color=RED, fontsize=7,
            fontweight="bold", ha="center")
for m, col, mk in (("clustered", AQUA, "D"), ("literal", RED, "X")):
    rc = D["A2"]["xiao_sv_r_eff"][m] / D["A2"]["cap_N32"]
    pred = float(np.interp(rc, x, y))
    ax.plot(rc, pred, mk, color=col, ms=10, markeredgecolor="white",
            markeredgewidth=1.0, zorder=5,
            label=f"Xiao SV {m}: predict {pred:+.2f} dB")
ax.set_xlabel("normalized rank")
ax.set_ylabel("$\\Delta_{\\rm HS}$  (dB)")
ax.legend(frameon=False, fontsize=6.9, loc="upper right")
ax.set_title("a. A2 — one curve across channel models", loc="left",
             fontweight="bold", fontsize=8.5, pad=12)
ax.annotate("Track B exp. C, adaptive rank, SNR 5 dB", xy=(0, 1.012),
            xycoords="axes fraction", fontsize=7.1, color=MUTED, va="bottom")

# ---- A3: the EM filter's value collapses onto kappa -----------------------
s = sorted(D["A3"]["snr_family"], key=lambda r: r["median_kappa"])
k = np.array([r["median_kappa"] for r in s])
d = np.array([r["median_diff_db"] for r in s])
p = sorted(D["A3"]["pilot_family"], key=lambda r: r["P"])
kp = np.array([r["median_kappa"] for r in p])
dp = np.array([r["median_diff_db"] for r in p])

c = float(np.median(d * k))
kk = np.logspace(np.log10(k.min()), np.log10(k.max()), 100)
ax2.plot(kk, c / kk, color=MUTED, lw=1.4, ls=(0, (4, 2)),
         label=f"$\\Delta \\approx {c:.1f}/\\kappa$")
ax2.plot(k, d, color=PURPLE, lw=2.2, marker="o", ms=5,
         label="SNR family ($P=20$)")
ax2.plot(kp, dp, "s", color=AMBER, ms=6, markeredgecolor="white",
         markeredgewidth=0.8, label="pilot family (SNR 5, $P$ 10–35)")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("median $\\kappa = 2Z|Y|/\\sigma^2$")
ax2.set_ylabel("GS $-$ EM-GS, paired median (dB)")
ax2.legend(frameon=False, fontsize=7.1, loc="lower left")
ax2.set_title("b. A3 — the EM filter's value is a function of $\\kappa$",
              loc="left", fontweight="bold", fontsize=8.5, pad=12)
ax2.annotate("$\\kappa$ spans 375×; the product varies ±15%", xy=(0, 1.012),
             xycoords="axes fraction", fontsize=7.1, color=MUTED, va="bottom")

OUT.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_normalization_collapse.{ext}")
print(f"wrote {OUT}/fig_normalization_collapse.png")
print(f"kappa law constant: median {c:.2f}, "
      f"range [{(d*k).min():.2f}, {(d*k).max():.2f}]")

"""Figures for SPL Paper 1 (effective-rank scaling of Hankel-structured CE).

IEEE single-column width is 3.5 in; these are drawn at 3.45 in and dropped in
at \\columnwidth with no scaling, so the 8 pt tick labels stay 8 pt on the page.

Design notes
------------
* Palette is the project's validated set, re-validated for this target against
  the **white** print surface (the earlier run used the #fcfcfb screen surface):
  ``#2a78d6,#1baf7a,#eda100,#e34948,#7a4fd6`` -> ALL CHECKS PASS. The contrast
  WARN on the aqua and amber slots is discharged by the visible labels every
  figure here carries.
* **IEEE prints in grayscale.** Every series therefore carries a second,
  non-colour encoding -- line style and marker shape -- so no comparison in
  these figures depends on hue.
* Fig. 1 puts both curves on ONE axis (dB improvement over EM-GS) rather than
  pairing an NMSE axis with a gain axis, which would be a dual-scale chart.
* The genie-aided LS curve is drawn as a light dotted reference and labelled
  in the figure itself as *not a bound*, because a reader who sees a curve
  above everything else will otherwise read it as one.

Run:  PYTHONPATH=. python3 scratch/paper1_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ANALYSIS = Path("reports/trackD_partB9_analysis.json")
PREDICT = Path("reports/trackD_step0_cui_prediction.json")
CELLS = Path("results/track_d/partB9")
OUT = Path("paper/spl1/fig")

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8880", "axes.linewidth": 0.7,
    "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "lines.linewidth": 1.4,
})
BLUE, AQUA, AMBER, RED, PURPLE = "#2a78d6", "#1baf7a", "#eda100", "#e34948", "#7a4fd6"
INK, MUTED, FAINT = "#0b0b0b", "#52514e", "#9b9992"
W, H = 3.45, 2.45

# The A2 relation, as used for BOTH out-of-model predictions. Not refitted.
A2_CURVE = [(0.119, 7.043), (0.212, 3.556), (0.285, 1.792), (0.356, 1.038),
            (0.408, 0.577), (0.460, 0.266), (0.507, 0.046), (0.546, -0.117)]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {OUT / name}.pdf")


def centers(bins):
    return [0.5 * (b["bin"][0] + b["bin"][1]) for b in bins]


def fig1_gain_vs_snr(a):
    """Improvement over EM-GS at the default configuration, per SNR bin."""
    c = a["cells"]["B3_default"]
    hs = c["delta_hs"]["bins"]
    # vs_oracle stores (method - genie); EM-GS minus genie IS the headroom.
    genie = c["vs_oracle"]["EM-GS"]["bins"]
    x = centers(hs)
    y = [b["median_diff_db"] for b in hs]
    lo = [b["median_diff_db"] - b["boot_ci95_median"][0] for b in hs]
    hi = [b["boot_ci95_median"][1] - b["median_diff_db"] for b in hs]
    g = [b["median_diff_db"] for b in genie]

    fig, ax = plt.subplots(figsize=(W, H))
    ax.plot(centers(genie), g, ls=(0, (1, 2)), color=FAINT, marker="",
            lw=1.2, zorder=2)
    ax.annotate("genie-aided LS reference\n(not a bound)",
                (centers(genie)[2], g[2]), xytext=(0, 7),
                textcoords="offset points", color=MUTED, fontsize=7,
                ha="center")
    ax.errorbar(x, y, yerr=[lo, hi], color=AQUA, marker="s", ms=4.0,
                lw=1.5, capsize=2.0, elinewidth=0.8, mew=0, zorder=3,
                label="adaptive-rank HS-EM-GS")
    ax.axhline(0, color=FAINT, lw=0.6, ls=(0, (4, 3)), zorder=0)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("improvement over EM-GS (dB)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="lower right", handletextpad=0.5)
    save(fig, "fig1_gain_vs_snr")


def fig2_boundary(a):
    """Delta_HS against r_eff/cap for N in {16, 32, 64}."""
    b1 = a["B1_collapse"]
    fig, ax = plt.subplots(figsize=(W, H))
    xr, yr = zip(*A2_CURVE)
    ax.plot(xr, yr, "-", color=FAINT, lw=1.1, zorder=1)
    ax.axhline(0, color=FAINT, lw=0.6, ls=(0, (4, 3)), zorder=0)
    style = {"16": (BLUE, "o", (0, (5, 2))), "64": (RED, "^", (0, (1, 1.6)))}
    for N in ("16", "64"):
        col, mk, ls = style[N]
        px, py = zip(*sorted(b1["delta_by_N"][N]))
        ax.plot(px, py, ls=ls, color=col, marker=mk, ms=4.2, mew=0, lw=1.4,
                zorder=3, label=f"$N={N}$")
    # Anchored on the sparse upper-left stretch of the reference curve; the
    # obvious spot mid-curve collides with the N=16 series.
    ax.annotate("$N=32$", (0.16, 5.506), xytext=(11, 6),
                textcoords="offset points", color=MUTED, fontsize=7.5)
    ax.set_xlabel(r"$r_{\mathrm{eff}}/r_{\max}$")
    ax.set_ylabel(r"$\Delta_{\mathrm{HS}}$ (dB)")
    ax.legend(frameon=False, loc="upper right", handletextpad=0.5)
    save(fig, "fig2_boundary_invariance")


def fig3_predictions(a):
    """The two out-of-model predictions against their measurements."""
    pred = json.loads(PREDICT.read_text())
    rows = [("Xiao SV, clustered", 0.331, 1.30, None,
             a["B6_xiao"]["B6_xiao_clustered"]["measured_db"],
             a["B6_xiao"]["B6_xiao_clustered"]["ci95"])]
    m = a.get("B8_cui_measured")          # present only once B8 has landed
    if m is not None:
        rows.append((r"$L=10$ configuration",
                     pred["r_eff_over_cap"], pred["PREDICTED_delta_hs_db"],
                     pred["prediction_interval_db"],
                     m["measured_db"], m["ci95"]))

    fig, ax = plt.subplots(figsize=(W, H))
    xr, yr = zip(*A2_CURVE)
    ax.plot(xr, yr, "-", color=FAINT, lw=1.1, zorder=1)
    # Bottom-left is the only region of this panel with no curve, no marker
    # and no legend in it.
    ax.annotate("relation, fitted on the\nULA model only", (0.113, 0.75),
                color=MUTED, fontsize=7, ha="left", va="center")
    ax.axhline(0, color=FAINT, lw=0.6, ls=(0, (4, 3)), zorder=0)

    # Predicted and measured are separated on x by a fixed offset, because
    # drawn at the same abscissa they overlap -- which is the result, but it
    # makes the figure unable to show WHICH is which.
    DX = 0.013
    for i, (name, x, p, pint, meas, ci) in enumerate(rows):
        col = (AMBER, PURPLE)[i]
        mk = ("D", "o")[i]
        if pint:
            ax.plot([x - DX, x - DX], pint, color=col, lw=3.0, alpha=0.28,
                    solid_capstyle="butt", zorder=2)
        ax.plot([x - DX], [p], marker=mk, ms=4.6, mfc="white", mec=col,
                mew=1.3, zorder=4)
        ax.errorbar([x + DX], [meas], yerr=[[meas - ci[0]], [ci[1] - meas]],
                    color=col, marker=mk, ms=4.6, mew=0, capsize=2.0,
                    elinewidth=0.9, lw=0, zorder=5)
        # Leader lines into the empty region right of the curve; the two
        # channels sit 0.07 apart in x and their labels collide if placed
        # directly above their own points.
        ax.annotate(name, (x, max(p, meas)), xytext=(0.50, 0.56 - 0.14 * i),
                    textcoords="axes fraction", color=INK, fontsize=7,
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=FAINT, lw=0.7,
                                    shrinkA=2, shrinkB=5))

    # One legend explaining the hollow/filled convention, hue-independent.
    h = [plt.Line2D([], [], ls="", marker="o", ms=4.6, mfc="white",
                    mec=INK, mew=1.3),
         plt.Line2D([], [], ls="", marker="o", ms=4.6, color=INK)]
    ax.legend(h, ["predicted", "measured, 95% CI"], frameon=False,
              loc="upper right", handletextpad=0.5, labelspacing=0.3)
    ax.set_xlabel(r"$r_{\mathrm{eff}}/r_{\max}$")
    ax.set_ylabel(r"$\Delta_{\mathrm{HS}}$ (dB)")
    ax.set_xlim(0.10, 0.58)
    save(fig, "fig3_out_of_model")


def fig4_pilots(a):
    """Delta_HS as the pilot count falls."""
    b7 = a["B7_P15_classical"]
    P = sorted(int(k) for k in b7["delta_hs_by_P"])
    y = [b7["delta_hs_by_P"][str(p)] for p in P]
    yh = [b7["delta_hs_high_snr_by_P"][str(p)] for p in P]
    fig, ax = plt.subplots(figsize=(W, H))
    ax.plot(P, y, ls="-", marker="s", ms=4.2, mew=0, color=AQUA,
            label="all SNR")
    ax.plot(P, yh, ls=(0, (5, 2)), marker="o", ms=4.0, mew=0, color=BLUE,
            label=r"SNR $\geq 5$ dB")
    ax.set_xlabel("pilots $P$")
    ax.set_ylabel(r"$\Delta_{\mathrm{HS}}$ (dB)")
    ax.legend(frameon=False, loc="upper right", handletextpad=0.5)
    save(fig, "fig4_pilots")


def main() -> int:
    a = json.loads(ANALYSIS.read_text())
    fig1_gain_vs_snr(a)
    fig2_boundary(a)
    fig3_predictions(a)
    fig4_pilots(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Figures for SPL Paper 2 (evaluation of structural priors in unrolling).

Same drawing conventions as scratch/paper1_figures.py: 3.45 in for
\\columnwidth, the project palette re-validated on the white print surface, and
a non-colour encoding on every series because IEEE prints in grayscale.

Form choices
------------
* Fig. 1 is a horizontal decomposition, not a pie or a grouped bar: the job is
  "how does one total split", and the total (available headroom) is itself a
  bar so the unclaimed part is visible rather than implied.
* Fig. 2 is grouped bars over SNR bins, because the comparison is between two
  training regimes at the same bins; a slope chart would imply the bins are a
  sequence being traversed.
* Fig. 3 pairs the CAUSE (gradient share) with the EFFECT (per-bin gain) as
  two stacked panels sharing one x, since the paper's argument is that the
  first produces the second.

Run:  PYTHONPATH=. python3 scratch/paper2_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STAGE5 = Path("reports/trackD_stage5_results.json")
EVAL = Path("reports/trackD_stage5_eval.json")
PILOTS = Path("reports/trackD_pilot_three_way.json")
OUT = Path("paper/spl2/fig")

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8880", "axes.linewidth": 0.7,
    "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "lines.linewidth": 1.4,
})
BLUE, AQUA, AMBER, RED, PURPLE = "#2a78d6", "#1baf7a", "#eda100", "#e34948", "#7a4fd6"
INK, MUTED, FAINT = "#0b0b0b", "#52514e", "#9b9992"
W, H = 3.45, 2.35
BINLAB = ["$[-10,-5)$", "$[-5,0)$", "$[0,5)$", "$[5,10)$", "$[10,15)$",
          "$[15,20)$"]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {OUT / name}.pdf")


def fig1_attribution():
    """Where URformer's gain over the classical estimator comes from."""
    # src: reports/trackD_stage2_report.md:107-110 (filter 0.147, +Transformer
    # -> 3.345 of 4.183 available) and trackD_stage3_report.md:98 (unrolling
    # worth 0.920 dB against the matched non-unrolled control X1).
    filt, total, avail, unroll = 0.147, 3.345, 4.183, 0.920
    fig, ax = plt.subplots(figsize=(W, 1.55))
    ax.barh([0], [avail], color="#e9e7e1", height=0.55, zorder=1)
    ax.barh([0], [filt], color=AMBER, height=0.55, zorder=3)
    ax.barh([0], [total - filt], left=filt, color=BLUE, height=0.55, zorder=3)
    # A 2 px surface gap between adjacent fills, per the mark spec.
    ax.plot([filt, filt], [-0.275, 0.275], color="white", lw=1.6, zorder=4)
    ax.plot([total, total], [-0.275, 0.275], color="white", lw=1.6, zorder=4)
    ax.annotate(f"learned filter\n{filt:.3f} dB", (0.0, -0.42),
                ha="left", va="top", fontsize=7, color=INK)
    ax.annotate(f"Transformer  {total - filt:.3f} dB",
                ((filt + total) / 2, 0.42), ha="center", va="bottom",
                fontsize=7, color=INK)
    ax.annotate(f"unclaimed\n{avail - total:.3f} dB",
                ((total + avail) / 2, -0.42), ha="center", va="top",
                fontsize=7, color=MUTED)
    ax.set_yticks([])
    ax.set_ylim(-1.15, 0.95)
    ax.set_xlim(0, avail * 1.02)
    ax.set_xlabel("improvement over EM-GS (dB)")
    ax.grid(axis="y", visible=False)
    save(fig, "fig1_attribution")


def fig2_confound():
    """The structural gain under mixed-SNR versus matched focused training."""
    # src: reports/trackD_stage4_report.md:74-78
    bins = ["$[5,10)$", "$[10,15)$", "$[15,20)$", "over $[5,20]$"]
    mixed = [0.398, 1.305, 2.226, 1.209]
    focused = [-0.076, 0.130, 0.232, 0.078]
    x = np.arange(len(bins))
    w = 0.36
    fig, ax = plt.subplots(figsize=(W, H))
    ax.bar(x - w / 2, mixed, w, color=RED, zorder=3,
           label="trained on $[-10,20]$ dB")
    ax.bar(x + w / 2, focused, w, color=BLUE, zorder=3,
           label="trained on $[5,20]$ dB")
    ax.axhline(0, color=FAINT, lw=0.6, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(bins, fontsize=7)
    ax.set_xlabel("SNR bin (dB)")
    ax.set_ylabel("Hankel gain $\\Delta_H$ (dB)")
    ax.legend(frameon=False, loc="upper left", handletextpad=0.5,
              labelspacing=0.3)
    ax.grid(axis="x", visible=False)
    save(fig, "fig2_confound")


def fig3_balanced():
    """Cause and effect: gradient share, then the gain it buys."""
    s = json.loads(STAGE5.read_text())["runs"]["C1_snr_balanced_P20"]
    unw = s["unweighted_shares"]["grad_share"]
    bal = s["realized_shares"]["grad_share"]
    e = json.loads(EVAL.read_text())["P13_balanced_vs_uniform_P20"]["contrast"]
    y = [b["median_diff_db"] for b in e["bins"]]
    lo = [b["median_diff_db"] - b["boot_ci95_median"][0] for b in e["bins"]]
    hi = [b["boot_ci95_median"][1] - b["median_diff_db"] for b in e["bins"]]

    x = np.arange(6)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 3.25), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 1],
                                              "hspace": 0.18})
    w = 0.36
    a1.bar(x - w / 2, unw, w, color=RED, zorder=3, label="per-sample NMSE")
    a1.bar(x + w / 2, bal, w, color=BLUE, zorder=3, label="SNR-balanced")
    a1.axhline(1 / 6, color=FAINT, lw=0.7, ls=(0, (3, 2)), zorder=2)
    # Placed in the gap between the 0-5 and 5-10 groups; at the right edge it
    # collides with the tallest balanced bar.
    a1.annotate("uniform", (2.55, 1 / 6), xytext=(0, 3),
                textcoords="offset points", fontsize=6.5, color=MUTED,
                ha="center")
    a1.set_ylabel("gradient share")
    a1.legend(frameon=False, loc="upper right", handletextpad=0.5,
              labelspacing=0.25, fontsize=7)
    a1.grid(axis="x", visible=False)

    a2.errorbar(x, y, yerr=[lo, hi], color=AQUA, marker="s", ms=4.0, lw=1.5,
                capsize=2.0, elinewidth=0.8, mew=0, zorder=3)
    a2.axhline(0, color=FAINT, lw=0.6, ls=(0, (4, 3)), zorder=2)
    a2.set_ylabel("gain from\nbalancing (dB)")
    a2.set_xticks(x)
    a2.set_xticklabels(BINLAB, fontsize=6.5, rotation=30, ha="right")
    a2.set_xlabel("SNR bin (dB)")
    a2.grid(axis="x", visible=False)
    save(fig, "fig3_balanced_loss")


def fig4_pilots():
    """Pilot efficiency against pilot-count generalization."""
    d = json.loads(PILOTS.read_text())
    cg = d["classical_and_generalization"]
    P = sorted(int(k) for k in cg)
    fig, ax = plt.subplots(figsize=(W, H))
    for key, col, ls, mk, lab in (
            ("HS-EM-GS", AQUA, (0, (5, 2)), "^", "HS-EM-GS"),
            ("unstructured-LS oracle", FAINT, (0, (1, 2)), "", "genie-aided LS"),
            ("URformer", RED, "-", "o", "trained at $P=20$")):
        ax.plot(P, [cg[str(p)][key] for p in P], ls=ls, color=col, marker=mk,
                ms=3.6, mew=0, lw=1.4, label=lab)
    mt = d["matched_trained"]
    key = {10: "C2_P10_matched", 20: "U1_P20_uniform", 35: "C3_P35_matched"}
    mp = sorted(int(k) for k in mt)
    ax.plot(mp, [mt[str(p)][key[p]] for p in mp], ls=(0, (2, 2)), color=AMBER,
            marker="D", ms=4.4, mew=0, lw=1.3, label="trained AT each $P$")
    ax.axvline(20, color=FAINT, lw=0.6, ls=(0, (1, 3)), zorder=0)
    ax.set_xlabel("pilots $P$")
    ax.set_ylabel("NMSE (dB)")
    ax.legend(frameon=False, loc="upper right", handletextpad=0.5,
              labelspacing=0.28, fontsize=7)
    save(fig, "fig4_pilots_three_way")


def main() -> int:
    fig1_attribution()
    fig2_confound()
    fig3_balanced()
    fig4_pilots()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

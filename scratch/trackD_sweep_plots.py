"""Paper-style figures: NMSE vs SNR, and NMSE vs number of pilots.

Two panels each. LEFT is the paired per-trial MEDIAN, this project's standing
primary statistic. RIGHT is RATIO-OF-SUMS, the energy-pooled aggregate the
literature usually plots, included so the curves are directly comparable to
the paper's Fig. 3 / Fig. 4. They are labelled, never mixed: PROMPT 5
established that reading a difference between these two as a difference
between conditions produces false findings.

Run:  PYTHONPATH=. python3 scratch/trackD_sweep_plots.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SRC = Path("results/track_d/sweeps")
OUT = Path("results/track_d/sweeps")

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.28, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
})
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#e34948")
INK, MUTED = "#0b0b0b", "#52514e"

# Classical arms dashed, learned arms solid, the oracle a dotted reference.
STYLE = {
    "GS":                      dict(c=MUTED,   ls="--", m="v", lw=1.6),
    "EM-GS":                   dict(c=BLUE,    ls="--", m="^", lw=1.6),
    "HS-EM-GS":                dict(c=AQUA,    ls="--", m="D", lw=1.6),
    "X1 EM-GS+former":         dict(c=YELLOW,  ls="-",  m="P", lw=1.8),
    "URformer":                dict(c=ORANGE,  ls="-",  m="o", lw=2.2),
    "HS-URformer":             dict(c=MAGENTA, ls="-",  m="s", lw=1.8),
    "G1 gated":                dict(c=RED,     ls="-",  m="x", lw=1.6),
    "G2 SNR-gated":            dict(c="#7a4fd6", ls="-", m="*", lw=2.2),
    "unstructured-LS oracle":  dict(c=INK,     ls=":",  m=None, lw=1.5),
}
ORDER = list(STYLE)


def load(mode):
    rows = []
    for f in sorted(SRC.glob(f"{mode}_*.json")):
        r = json.loads(f.read_text())
        if r["n"] < 100:                      # ignore smoke-test leftovers
            continue
        rows.append(r)
    if not rows:
        raise SystemExit(f"no {mode} sweep files in {SRC}")
    key = (lambda r: r["snr_db"]) if mode == "snr" else (lambda r: r["P"])
    return sorted(rows, key=key), key


def curves(rows, key):
    x = np.array([key(r) for r in rows], dtype=float)
    med, ros = {}, {}
    for m in ORDER:
        med[m] = np.array([10 * np.log10(np.median(np.array(r["num"][m])
                                                   / np.array(r["den"][m])))
                           for r in rows])
        ros[m] = np.array([10 * np.log10(np.sum(r["num"][m])
                                         / np.sum(r["den"][m])) for r in rows])
    return x, med, ros


def panel(ax, x, d, xlabel, title, xticks=None):
    for m in ORDER:
        s = STYLE[m]
        ax.plot(x, d[m], color=s["c"], ls=s["ls"], lw=s["lw"], marker=s["m"],
                ms=4.5, label=m, markerfacecolor="none" if s["m"] in
                ("o", "s", "D", "v", "^") else s["c"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("NMSE [dB]")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=8)
    if xticks is not None:
        ax.set_xticks(xticks)


def figure(mode, xlabel, fname, suptitle, note, xticks=None):
    rows, key = load(mode)
    x, med, ros = curves(rows, key)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.4))
    panel(a1, x, med, xlabel, "a. per-trial MEDIAN  (project primary)", xticks)
    panel(a2, x, ros, xlabel, "b. ratio-of-sums  (paper convention)", xticks)
    a1.legend(frameon=False, ncol=2, fontsize=6.2, loc="lower left")
    fig.suptitle(suptitle, fontsize=9, fontweight="bold", y=1.02)
    fig.text(0.5, -0.07, note, ha="center", fontsize=6.2, color=MUTED)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{fname}.{ext}")
    plt.close(fig)
    print(f"  wrote {OUT/fname}.png")
    return x, med, ros


n = ("K=3 users, M=32 atomic antennas, RSR=10 dB (repo convention; 5.23 dB in the "
     "paper's multi-user convention), L_k~U{3..7} paths, 400 trials per point.\n"
     "Classical arms dashed, learned arms solid. All learned arms trained at "
     "P=20 on 80k samples (URformer/HS-URformer/G1/G2) or on cached EM-GS (X1).")

x, med, ros = figure(
    "snr", "SNR [dB]", "fig_nmse_vs_snr",
    "NMSE vs SNR — all methods (P = 20 pilots)", n,
    xticks=[-10, -5, 0, 5, 10, 15, 20])

xp, medp, rosp = figure(
    "pilots", "Number of pilots  P", "fig_nmse_vs_pilots",
    f"NMSE vs pilots — all methods (SNR = 5 dB)",
    n + "\nPILOT SWEEP IS A GENERALIZATION TEST: every learned arm was trained "
        "at P=20 and is evaluated unchanged at other P. Nothing was retrained.",
    xticks=[10, 12, 15, 20, 25, 30, 35])

print("\n=== NMSE vs SNR, per-trial median [dB] ===")
hdr = "  " + "".join(f"{v:>7.1f}" for v in x)
print(f"  {'method':<24}" + hdr)
for m in ORDER:
    print(f"  {m:<24}" + "".join(f"{v:>7.2f}" for v in med[m]))
print("\n=== NMSE vs pilots at 5 dB, per-trial median [dB] ===")
print(f"  {'method':<24}" + "".join(f"{int(v):>7d}" for v in xp))
for m in ORDER:
    print(f"  {m:<24}" + "".join(f"{v:>7.2f}" for v in medp[m]))

json.dump({"snr": {"x": x.tolist(),
                   "median_db": {m: med[m].tolist() for m in ORDER},
                   "ratio_of_sums_db": {m: ros[m].tolist() for m in ORDER}},
           "pilots": {"x": xp.tolist(),
                      "median_db": {m: medp[m].tolist() for m in ORDER},
                      "ratio_of_sums_db": {m: rosp[m].tolist() for m in ORDER}}},
          open("reports/trackD_sweep_curves.json", "w"), indent=2)
print("\nwrote reports/trackD_sweep_curves.json")

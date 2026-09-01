"""Paper-style figures: NMSE vs SNR, and NMSE vs number of pilots.

Design notes (dataviz skill)
----------------------------
The first version put nine hues and a legend box on each of two panels, which
hit two anti-patterns at once: "eight categorical hues when the story is a
grouping", and a legend colliding with the data. This version instead:

* **one panel per figure** -- median (the project's primary statistic) is the
  headline; ratio-of-sums moves to its own supplementary figure rather than
  sitting beside a near-identical twin;
* **emphasis over enumeration** -- HS-URformer, G1 and G2 agree to within a few
  tenths of a dB, so they are drawn as one labelled band rather than three
  competing lines;
* **a frameless legend in the empty upper-right**, which discharges the
  validator's contrast WARN on the aqua and amber slots. Direct end-of-line
  labels were tried and abandoned: the curves converge into a ~6 dB band at
  the right edge, so the collision solver pushed labels off the axes;
* **classical arms dashed, learned arms solid**, so family is legible without
  relying on hue alone.

Palette validated with the skill's own script (light mode, surface #fcfcfb):
``#2a78d6,#1baf7a,#eda100,#e34948,#7a4fd6`` -> ALL CHECKS PASS. An earlier
attempt using orange #eb6834 beside amber #eda100 FAILED the normal-vision
floor at dE 13.7 (< 15) and was re-stepped to red #e34948.

Run:  PYTHONPATH=. python3 scratch/trackD_sweep_plots.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SRC = OUT = Path("results/track_d/sweeps")

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 10,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8880", "axes.linewidth": 0.8,
    "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
})
BLUE, AQUA, AMBER, RED, PURPLE = "#2a78d6", "#1baf7a", "#eda100", "#e34948", "#7a4fd6"
INK, MUTED, FAINT = "#0b0b0b", "#52514e", "#8a8880"

# The three structured URformer variants agree closely; they are one band.
BAND = ["HS-URformer", "G1 gated", "G2 SNR-gated"]
LINES = [
    # key,                    color,  dashed, width, label
    ("GS",                    FAINT,  True,  1.3, "GS"),
    ("EM-GS",                 BLUE,   True,  1.6, "EM-GS"),
    ("HS-EM-GS",              AQUA,   True,  1.8, "HS-EM-GS"),
    ("X1 EM-GS+former",       AMBER,  False, 1.6, "X1  EM-GS + 1 former"),
    ("URformer",              RED,    False, 2.2, "URformer"),
]
ORACLE = "unstructured-LS oracle"
ALL = [k for k, *_ in LINES] + BAND + [ORACLE]


def load(mode):
    rows = [json.loads(f.read_text()) for f in sorted(SRC.glob(f"{mode}_*.json"))]
    rows = [r for r in rows if r["n"] >= 100]
    if not rows:
        raise SystemExit(f"no {mode} sweep files in {SRC}")
    key = (lambda r: r["snr_db"]) if mode == "snr" else (lambda r: r["P"])
    rows.sort(key=key)
    x = np.array([key(r) for r in rows], dtype=float)
    med, ros = {}, {}
    for m in ALL:
        num = [np.array(r["num"][m]) for r in rows]
        den = [np.array(r["den"][m]) for r in rows]
        med[m] = np.array([10 * np.log10(np.median(a / b))
                           for a, b in zip(num, den)])
        ros[m] = np.array([10 * np.log10(a.sum() / b.sum())
                           for a, b in zip(num, den)])
    return x, med, ros


def draw(ax, x, d, xlabel, xticks):
    """One panel. Legend goes in the empty upper-right, never on the data.

    End-of-line direct labels were tried first and do not work here: all seven
    curves converge into a ~6 dB band at the right edge, so the collision
    solver stacks the labels straight off the bottom of the axes. The upper
    right is empty in both sweeps (every curve descends left to right), so a
    frameless legend there costs nothing and collides with nothing.
    """
    h, lab = [], []
    for k, c, dash, lw, name in LINES:
        ln, = ax.plot(x, d[k], color=c, lw=lw, zorder=3,
                      ls=(0, (5, 2.5)) if dash else "-")
        h.append(ln)
        lab.append(name)

    band = np.array([d[k] for k in BAND])
    ax.fill_between(x, band.min(0), band.max(0), color=PURPLE, alpha=0.20,
                    lw=0, zorder=4)
    ln, = ax.plot(x, band.mean(0), color=PURPLE, lw=2.4, zorder=5)
    h.append(ln)
    lab.append("structured URformer  (HS · G1 · G2)")

    ln, = ax.plot(x, d[ORACLE], color=INK, ls=(0, (1, 2)), lw=1.4, zorder=2)
    h.append(ln)
    lab.append("unstructured-LS oracle")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("NMSE  [dB]")
    ax.set_xticks(xticks)
    ax.set_xlim(x.min(), x.max())
    ax.legend(h, lab, frameon=False, loc="upper right", fontsize=7.9,
              handlelength=2.4, labelspacing=0.42, borderaxespad=0.2)


def figure(mode, xlabel, xticks, fname, title, sub, stat="median"):
    x, med, ros = load(mode)
    d = med if stat == "median" else ros
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    draw(ax, x, d, xlabel, xticks)
    ax.set_title(title, loc="left", fontweight="bold", pad=30)
    ax.annotate(sub, xy=(0, 1.008), xycoords="axes fraction", fontsize=7.4,
                color=MUTED, va="bottom", linespacing=1.5)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{fname}.{ext}")
    plt.close(fig)
    print(f"  wrote {OUT/fname}.png")
    return x, med, ros


COMMON = ("K=3, M=32 antennas, L_k~U{3..7} paths, 400 trials/point.\n"
          "Dashed = classical, solid = learned.  Per-trial median.")

x, med, ros = figure(
    "snr", "SNR  [dB]", [-10, -5, 0, 5, 10, 15, 20], "fig_nmse_vs_snr",
    "NMSE vs SNR   ·   P = 20 pilots",
    COMMON)

xp, medp, rosp = figure(
    "pilots", "Number of pilots  P", [10, 15, 20, 25, 30, 35],
    "fig_nmse_vs_pilots", "NMSE vs pilots   ·   SNR = 5 dB",
    COMMON + "  Learned arms trained at P=20, evaluated unchanged elsewhere.")

# Supplementary: the same two sweeps under the literature's energy-pooled
# aggregate. Kept separate so the headline figures are not two near-identical
# panels side by side.
figure("snr", "SNR  [dB]", [-10, -5, 0, 5, 10, 15, 20],
       "fig_nmse_vs_snr_ratio_of_sums",
       "NMSE vs SNR   ·   ratio-of-sums (literature convention)",
       COMMON.replace("Per-trial median.", "SUPPLEMENTARY: ratio-of-sums; "
                              "the project's primary statistic is the median."),
       stat="ros")
figure("pilots", "Number of pilots  P", [10, 15, 20, 25, 30, 35],
       "fig_nmse_vs_pilots_ratio_of_sums",
       "NMSE vs pilots   ·   ratio-of-sums (literature convention)",
       COMMON.replace("Per-trial median.", "SUPPLEMENTARY: ratio-of-sums; "
                              "the project's primary statistic is the median."),
       stat="ros")

hdr = lambda v: "".join(f"{t:>8.1f}" for t in v)
print("\n=== NMSE vs SNR, per-trial median [dB] ===")
print(f"  {'method':<24}{hdr(x)}")
for m in ALL:
    print(f"  {m:<24}" + "".join(f"{v:>8.2f}" for v in med[m]))
print("\n=== NMSE vs pilots at 5 dB, per-trial median [dB] ===")
print(f"  {'method':<24}{hdr(xp)}")
for m in ALL:
    print(f"  {m:<24}" + "".join(f"{v:>8.2f}" for v in medp[m]))

json.dump({"snr": {"x": x.tolist(),
                   "median_db": {m: med[m].tolist() for m in ALL},
                   "ratio_of_sums_db": {m: ros[m].tolist() for m in ALL}},
           "pilots": {"x": xp.tolist(),
                      "median_db": {m: medp[m].tolist() for m in ALL},
                      "ratio_of_sums_db": {m: rosp[m].tolist() for m in ALL}}},
          open("reports/trackD_sweep_curves.json", "w"), indent=2)
print("\nwrote reports/trackD_sweep_curves.json")

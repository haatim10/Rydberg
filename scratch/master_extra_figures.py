"""The two master-manuscript figures that did not already exist.

Both are drawn from ALREADY-STORED results. No simulation is run here.

  fig6  gain versus the RAW path-count coordinate L/r_max, from the Track B
        controlled path-count sweep -- the panel that motivates replacing L
        by effective rank in Fig. 7.
        SOURCE: trackB_hankel_emgs/results/experiment_C_path_count.csv

  fig9  K-invariance at fixed pilot adequacy P/2K = 3.33.
        SOURCE: reports/trackD_partB9_analysis.json, cells B2_K{2,3,4}_P{13,20,27}

Run:  PYTHONPATH=. python3 scratch/master_extra_figures.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("paper/master/fig")

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "lines.linewidth": 1.1, "lines.markersize": 3.6,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", facecolor="white")
    plt.close(fig)
    print("  wrote", name)


def fig6_raw_L():
    rows = list(csv.DictReader(
        open("trackB_hankel_emgs/results/experiment_C_path_count.csv")))
    L = np.array([int(r["L"]) for r in rows], float)
    g = np.array([float(r["gain_db"]) for r in rows])
    lo = np.array([float(r["gain_ci_lo"]) for r in rows])
    hi = np.array([float(r["gain_ci_hi"]) for r in rows])
    rmax = 16.0                       # ceil(32/2), the sweep's array size
    fig, ax = plt.subplots(figsize=(3.3, 1.72))
    ax.errorbar(L / rmax, g, yerr=[g - lo, hi - g], fmt="^-", color="C3",
                capsize=2, elinewidth=0.8)
    ax.axhline(0, color="0.5", lw=0.8, ls=":")
    ax.set_xlabel(r"$L/r_{\max}$  (raw path-count coordinate)")
    ax.set_ylabel(r"$\Delta_{\mathrm{HS}}$ (dB)")
    save(fig, "fig6_gain_vs_raw_L")


def fig9_k_invariance():
    a = json.loads(Path("reports/trackD_partB9_analysis.json").read_text())
    cells = {2: "B2_K2_P13", 3: "B2_K3_P20", 4: "B2_K4_P27"}
    K = sorted(cells)
    pooled, plo, phi, high, hlo, hhi = [], [], [], [], [], []
    for k in K:
        d = a["cells"][cells[k]]["delta_hs"]
        p = d["pooled_SAMPLING_DESIGN_DEPENDENT"]
        h = d["high_snr_ge5"]
        pooled.append(p["median_diff_db"]); plo.append(p["boot_ci95_median"][0])
        phi.append(p["boot_ci95_median"][1])
        high.append(h["median_diff_db"]); hlo.append(h["boot_ci95_median"][0])
        hhi.append(h["boot_ci95_median"][1])
    pooled, plo, phi = map(np.asarray, (pooled, plo, phi))
    high, hlo, hhi = map(np.asarray, (high, hlo, hhi))
    fig, ax = plt.subplots(figsize=(3.3, 1.72))
    ax.errorbar(np.array(K) - 0.05, high, yerr=[high - hlo, hhi - high],
                fmt="s-", color="C0", capsize=2, elinewidth=0.8,
                label=r"SNR $\geq 5$ dB")
    ax.errorbar(np.array(K) + 0.05, pooled, yerr=[pooled - plo, phi - pooled],
                fmt="o--", color="C3", capsize=2, elinewidth=0.8,
                label="pooled over all SNR")
    ax.set_xticks(K)
    ax.set_xlabel(r"users $K$  (pilot adequacy $P/2K=3.33$ held fixed)")
    ax.set_ylabel(r"$\Delta_{\mathrm{HS}}$ (dB)")
    ax.legend(frameon=False, loc="lower left", handlelength=1.8)
    save(fig, "fig9_k_invariance")


if __name__ == "__main__":
    fig6_raw_L()
    fig9_k_invariance()

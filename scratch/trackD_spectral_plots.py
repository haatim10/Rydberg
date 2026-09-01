"""PROMPT 8 Part C figure: Hankel cumulative spectral energy.

One figure, two panels sharing a y-axis:

  a. cumulative energy vs q, curves indexed by L_k, NOISELESS true channel
     (solid) and the converged EM-GS ESTIMATE at 5 dB (dashed) -- the
     distinction the deliverable asks to be made visible
  b. Xiao's Saleh-Valenzuela channel under both readings of its Table I,
     against two sparse references

Sequential ramp for L_k (an ORDERED variable -- the dataviz skill's rule is
one hue light->dark for magnitude, never a categorical rainbow). Panel b uses
validated categorical slots because its series are identities, not magnitudes.

Run:  PYTHONPATH=. python3 scratch/trackD_spectral_plots.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D = json.load(open("reports/trackD_spectral_diagnostics.json"))
OUT = Path("results/track_d/spectral")

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9.5, "axes.titlesize": 9.5,
    "legend.fontsize": 7.4, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.22, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8a8880", "axes.linewidth": 0.8,
    "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
})
BLUE, AQUA, AMBER, RED, PURPLE = "#2a78d6", "#1baf7a", "#eda100", "#e34948", "#7a4fd6"
INK, MUTED = "#0b0b0b", "#52514e"

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.6), sharey=True)

# ---- panel a: L sweep, noiseless vs estimate -----------------------------
c1 = D["C1_L_sweep_M32"]
Ls = sorted(int(k) for k in c1["noiseless"])
ramp = plt.cm.viridis(np.linspace(0.08, 0.92, len(Ls)))
for c, L in zip(ramp, Ls):
    y = c1["noiseless"][str(L)]["cum_energy_mean"]
    ax.plot(range(1, len(y) + 1), y, color=c, lw=1.9, label=f"L={L}")
    y2 = c1["emgs_5dB"][str(L)]["cum_energy_mean"]
    ax.plot(range(1, len(y2) + 1), y2, color=c, lw=1.3, ls=(0, (3, 2)))
ax.axvline(7, color=RED, ls=":", lw=1.3)
ax.annotate("r = 7", (7, 0.10), color=RED, fontsize=7.5, fontweight="bold",
            ha="right", rotation=90, va="bottom")
ax.axhline(0.99, color=MUTED, lw=0.8, ls=(0, (1, 3)))
ax.set_xlabel("retained components  q")
ax.set_ylabel("cumulative Hankel energy  $E(q)$")
ax.set_xlim(1, 16)
ax.set_ylim(0, 1.02)
ax.legend(frameon=False, ncol=2, fontsize=6.9, loc="lower right")
ax.set_title("a. sparse channel, $M{=}32$", loc="left", fontweight="bold",
             fontsize=8.5, pad=14)
ax.annotate("solid = noiseless   ·   dashed = EM-GS estimate (5 dB)",
            xy=(0, 1.012), xycoords="axes fraction", fontsize=7.2,
            color=MUTED, va="bottom")

# ---- panel b: Xiao's own channel -----------------------------------------
c4 = D["C4_xiao_SV_M32"]
for mode, col, lab in (("clustered", AQUA,
                        "Xiao SV, clustered rays ($\\pm5^\\circ$, Cui precedent)"),
                       ("literal", RED,
                        "Xiao SV, literal Table I (40 indep. DoAs)")):
    y = c4[mode]["cum_energy_mean"]
    ax2.plot(range(1, len(y) + 1), y, color=col, lw=2.2, label=lab)
for L, col, ls in ((5, BLUE, "-"), (16, MUTED, (0, (4, 2)))):
    y = D["C1_L_sweep_M32"]["noiseless"][str(L)]["cum_energy_mean"]
    ax2.plot(range(1, len(y) + 1), y, color=col, lw=1.4, ls=ls,
             label=f"sparse reference, $L={L}$")
ax2.axvline(7, color=RED, ls=":", lw=1.3)
ax2.axhline(0.99, color=MUTED, lw=0.8, ls=(0, (1, 3)))
ax2.annotate("99% energy", (15.7, 0.985), color=MUTED, fontsize=6.8,
             ha="right", va="top")
ax2.set_xlabel("retained components  q")
ax2.set_xlim(1, 16)
ax2.legend(frameon=False, loc="lower right", fontsize=6.9)
ax2.set_title("b. the paper's own channel model", loc="left",
              fontweight="bold", fontsize=8.5, pad=14)
ax2.annotate("is it approximately low rank?", xy=(0, 1.012),
             xycoords="axes fraction", fontsize=7.2, color=MUTED, va="bottom")

OUT.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"fig_hankel_spectrum.{ext}")
print(f"wrote {OUT}/fig_hankel_spectrum.png")

# ---- the table the deliverable asks for ----------------------------------
def row(d, name):
    # M=16 has cap 8, so tail@13 does not exist there; print "-" rather than
    # crashing or silently inventing a value.
    t = {int(k): v for k, v in d["tail_energy_median"].items()}
    cell = lambda r: f"{t[r]*100:8.2f}" if r in t else f"{'-':>8}"
    return (f"  {name:<34} {d['effective_rank_median']:8.2f} "
            f"{d['effective_rank_energy_median']:9.2f} {d['stable_rank_median']:8.2f} "
            f"{cell(3):>9} {cell(7)} {cell(13)}")


hdr = (f"  {'configuration':<34} {'erank':>8} {'erank_E':>9} {'srank':>8} "
       f"{'tail@3 %':>9} {'tail@7 %':>8} {'tail@13 %':>8}")
print("\n=== C1  L sweep, M=32 (cap 16) ===\n" + hdr)
for L in Ls:
    print(row(c1["noiseless"][str(L)], f"true channel, L={L}"))
print()
for L in Ls:
    print(row(c1["emgs_5dB"][str(L)], f"EM-GS estimate 5 dB, L={L}"))

print("\n=== C2  M sweep, L=5 noiseless ===\n" + hdr)
for N in sorted(int(k) for k in D["C2_M_sweep_L5"]):
    print(row(D["C2_M_sweep_L5"][str(N)],
              f"M={N} (cap {D['rank_cap'][str(N)]})"))

print("\n=== C3  SNR sweep, EM-GS estimate, L=5 M=32 ===\n" + hdr)
for s in sorted(float(k) for k in D["C3_SNR_sweep_estimate"]):
    print(row(D["C3_SNR_sweep_estimate"][str(s)], f"SNR = {s:+.0f} dB"))

print("\n=== C4  Xiao Saleh-Valenzuela, M=32 ===\n" + hdr)
for m in ("clustered", "literal"):
    print(row(c4[m], f"Xiao SV, {m}"))

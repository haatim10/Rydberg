"""PROMPT 9 figures: the r_eff/cap collapse, and the three-way pilot figure.

Design notes (dataviz skill)
----------------------------
Two figures, each answering one question, rather than one multi-panel grid:

* **fig_collapse_reff.png** -- the P12 test. The N=32 reference curve is the
  grey backdrop (context, not a competitor); the N=16 and N=64 measurements sit
  on it as markers. If the collapse holds, the markers land on the line, which
  is a shape the eye checks in one glance and a table cannot show.
* **fig_pilots_three_way.png** -- classical efficiency vs learned pilot-count
  GENERALIZATION vs learned pilot EFFICIENCY. The generalization curve is one
  model evaluated everywhere (solid); the efficiency points are separate models
  trained at each P (markers on a dashed connector, because the connector is an
  interpolation between three different networks and not a swept curve).

Palette (validated in `scratch/trackD_sweep_plots.py`, light mode, #fcfcfb):
``#2a78d6,#1baf7a,#eda100,#e34948,#7a4fd6``. Frameless legends, for the reason
recorded there: end-of-line labels collide once curves converge.

Run:  PYTHONPATH=. python3 scratch/trackD_partB9_plots.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ANALYSIS = Path("reports/trackD_partB9_analysis.json")
SWEEPS = Path("results/track_d/sweeps")
OUT = Path("results/track_d/partB9")

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

N32_REF = [(0.119, 7.043), (0.212, 3.556), (0.285, 1.792), (0.356, 1.038),
           (0.408, 0.577), (0.460, 0.266), (0.507, 0.046), (0.546, -0.117)]


def fig_collapse() -> None:
    a = json.loads(ANALYSIS.read_text())
    if "B1_collapse" not in a:
        print("no B1 cells; skipping the collapse figure")
        return
    by_n = a["B1_collapse"]["delta_by_N"]
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    x, y = zip(*N32_REF)
    ax.plot(x, y, "-", color=FAINT, lw=1.6, zorder=1)
    ax.axhline(0.0, color=FAINT, lw=0.8, ls=(0, (4, 3)), zorder=0)
    ax.annotate("N = 32 reference\n(Track B experiment C)", (x[3], y[3]),
                xytext=(6, 12), textcoords="offset points", color=MUTED,
                fontsize=8.2, ha="left")
    style = {"16": (BLUE, "o"), "64": (RED, "s")}
    for N, pts in sorted(by_n.items(), key=lambda kv: int(kv[0])):
        c, mk = style.get(N, (PURPLE, "^"))
        px, py = zip(*sorted(pts))
        ax.plot(px, py, mk, color=c, ms=6.5, mew=0, zorder=3,
                label=f"N = {N}  (measured)")
    ax.set_xlabel(r"$r_{\mathrm{eff}}\,/\,\mathrm{cap}$   "
                  r"(effective rank relative to the Hankel ceiling)")
    ax.set_ylabel(r"$\Delta_{\mathrm{HS}}$  (dB, EM-GS $-$ HS-EM-GS)")
    ax.set_title("The Hankel gain collapses onto effective rank, not aperture",
                 loc="left", pad=8)
    ax.legend(frameon=False, loc="upper right", handletextpad=0.5)
    f = OUT / "fig_collapse_reff.png"
    fig.savefig(f)
    fig.savefig(f.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {f}")


def med_db(d, key):
    n = np.asarray(d["num"][key])
    return float(10 * np.log10(np.median(n / np.asarray(d["den"][key]))))


def fig_pilots() -> None:
    rows = {}
    for f in sorted(SWEEPS.glob("pilots_snr+5.0_P*_n*.json")):
        d = json.loads(f.read_text())
        rows[d["P"]] = d
    mp = SWEEPS / "matched_pilot_points.json"
    if not rows or not mp.exists():
        print("missing pilot sweep or matched points; skipping")
        return
    matched = json.loads(mp.read_text())
    P = sorted(rows)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for key, c, ls, lw, lab in (
            ("EM-GS", BLUE, (0, (5, 2)), 1.6, "EM-GS  (classical)"),
            ("HS-EM-GS", AQUA, (0, (5, 2)), 1.8, "HS-EM-GS  (classical)"),
            ("unstructured-LS oracle", FAINT, (0, (1, 2)), 1.4,
             "unstructured-LS oracle")):
        ax.plot(P, [med_db(rows[p], key) for p in P], ls=ls, color=c, lw=lw,
                label=lab)
    ax.plot(P, [med_db(rows[p], "URformer") for p in P], "-", color=RED, lw=2.2,
            label="URformer, trained at P = 20  (generalization)")
    mk = sorted(int(k) for k in matched)
    my = [matched[str(p)]["median_db"][
        {10: "C2_P10_matched", 20: "U1_P20_uniform",
         35: "C3_P35_matched"}[p]] for p in mk]
    ax.plot(mk, my, ls=(0, (2, 2)), color=AMBER, lw=1.4, zorder=2)
    ax.plot(mk, my, "D", color=AMBER, ms=6, mew=0, zorder=3,
            label="URformer, trained AT each P  (efficiency)")
    ax.axvline(20, color=FAINT, lw=0.8, ls=(0, (1, 3)), zorder=0)
    ax.annotate("training P", (20, ax.get_ylim()[0]), xytext=(3, 6),
                textcoords="offset points", color=MUTED, fontsize=8.2)
    ax.set_xlabel("number of pilots  $P$")
    ax.set_ylabel("NMSE (dB), per-trial median")
    ax.set_title("Pilot efficiency and pilot-count generalization are different "
                 "curves\nSNR = 5 dB, 400 paired trials per point",
                 loc="left", pad=8)
    ax.legend(frameon=False, loc="upper right", fontsize=8.2,
              handletextpad=0.6, labelspacing=0.35)
    f = OUT / "fig_pilots_three_way.png"
    fig.savefig(f)
    fig.savefig(f.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {f}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if ANALYSIS.exists():
        fig_collapse()
    fig_pilots()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

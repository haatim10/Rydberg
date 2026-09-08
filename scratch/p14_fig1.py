"""PROMPT 14 C2 -- Fig. 1, two panels, one configuration.

(a) NMSE versus SNR at the headline cell with both bounds (unchanged from
    PROMPT 13, results/p12/B1.json).
(b) Aperture. Delta_HS versus N for both pilot counts, with EM-GS's absolute
    NMSE overlaid on a secondary axis, so that "the structured curve moves and
    the unstructured one does not" is legible in one glance. The sign change
    at N = 8 is marked and labelled as the predicted failure.

Both panels come from configuration A (n_cz = 4, T = 100, selection budget 25,
RSR 12 dB, Track B generator, 400 paired trials per point), which is the whole
point of the C1 sweep -- the previous aperture numbers were T = 50, budget 20.

Panel (b) plots BOTH pooling rules, because they disagree at N = 8 and the
letter says so: filled markers are the mean over the six SNR points of the
paired per-trial median (the primary statistic), open markers are the mean of
the ratio of summed errors (what a bound on mean-square error governs).

Run:  PYTHONPATH=. python3 scratch/p14_fig1.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path("paper/paper1/fig")
AP = Path("results/p14/aperture")
SNRS = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
NS = (8, 16, 32)
PS = (10, 30)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.4,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "lines.linewidth": 1.2, "lines.markersize": 3.8,
    "axes.grid": True, "grid.alpha": 0.28, "grid.linewidth": 0.4,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def key(s):
    return f"N32_P30_snr{'+' if s >= 0 else '-'}{abs(int(s))}"


def load_cell(N, P):
    """Six SNR points for one (N,P) cell, from whichever form is on disk.

    The N=32,P=30 cell is served from the bridging run: it is that cell at
    exactly this operator, which is what brought the sweep inside budget.
    """
    if (N, P) == (32, 30):
        rows = json.loads(Path("results/p12/B1.json").read_text())["points"]
        return sorted(rows, key=lambda r: r["snr_db"])
    cell = AP / f"N{N}_P{P}.json"
    if cell.exists():
        return sorted(json.loads(cell.read_text())["points"],
                      key=lambda r: r["snr_db"])
    rows = []
    for si, s in enumerate(SNRS):
        f = AP / f"pt_N{N}_P{P}_s{si}.json"
        if not f.exists():
            raise SystemExit(f"missing point: {f}")
        rows.append(json.loads(f.read_text()))
    return sorted(rows, key=lambda r: r["snr_db"])


def assemble():
    """Collect every cell and write one summary JSON the manuscript can cite."""
    out = {}
    for N in NS:
        for P in PS:
            rows = load_cell(N, P)
            out[f"N{N}_P{P}"] = {
                "points": rows,
                "mean_delta_median_db": float(
                    np.mean([r["delta_median_db"] for r in rows])),
                "mean_delta_ratio_of_sums_db": float(
                    np.mean([r["delta_ratio_of_sums_db"] for r in rows])),
                "mean_em_gs_db": float(np.mean([r["em_gs_db"] for r in rows])),
                "mean_active_frac": float(
                    np.mean([r["active_frac"] for r in rows])),
            }
    return out


def panel_a(ax):
    b1 = json.loads(Path("results/p12/B1.json").read_text())
    cc = json.loads(Path("results/track_b/constrained_crlb.json").read_text())
    em = np.array([p["em_gs_db"] for p in b1["points"]])
    hs = np.array([p["hs_gs_db"] for p in b1["points"]])
    ccrb = np.array([cc["constrained"]["b3"][key(s)] for s in SNRS])
    crlb = np.array([cc["unconstrained_rank1"]["b3"][key(s)] for s in SNRS])
    x = np.array(SNRS)

    ax.plot(x, ccrb, "-", color="#1a1a1a", lw=1.5, zorder=4,
            label="geometric CCRB (primary reference)")
    ax.plot(x, crlb, ls=(0, (1, 2.2)), color="#9b9992", lw=1.0, zorder=2)
    ax.plot(x, em, "s--", color="#2a78d6", mew=0, zorder=5, label="EM-GS")
    ax.plot(x, hs, "o-", color="#e34948", mew=0, zorder=6,
            label="HS-GS (proposed)")
    ax.annotate("unconstrained CRLB\n(does not model the prior)",
                xy=(x[2], crlb[2]), xytext=(-2, 26), textcoords="offset points",
                fontsize=6.2, color="#52514e", ha="left",
                arrowprops=dict(arrowstyle="-", color="#9b9992", lw=0.6,
                                shrinkA=1, shrinkB=2))
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("NMSE (dB)")
    ax.set_xticks(x)
    ax.legend(frameon=False, loc="lower left", handlelength=2.0,
              labelspacing=0.26, borderpad=0.15)
    ax.set_title(r"(a) $N=32$, $P=30$", fontsize=8, pad=3)
    return em, hs, ccrb, crlb


def panel_b(ax, S):
    x = np.arange(len(NS))
    cols = {10: "#e34948", 30: "#8c3fb5"}
    ax.axhline(0.0, color="#666", lw=0.7, ls="-", zorder=1)

    for P in PS:
        med = [S[f"N{N}_P{P}"]["mean_delta_median_db"] for N in NS]
        ros = [S[f"N{N}_P{P}"]["mean_delta_ratio_of_sums_db"] for N in NS]
        ax.plot(x, med, "o-", color=cols[P], mew=0, zorder=6,
                label=rf"$\Delta_{{\rm HS}}$, $P={P}$ (paired median)")
        ax.plot(x, ros, "^--", color=cols[P], mfc="white", mew=0.9, lw=0.9,
                zorder=5, label=rf"$\Delta_{{\rm HS}}$, $P={P}$ (ratio of sums)")

    # The predicted failure, marked.
    lo = min(min(S[f"N8_P{P}"]["mean_delta_median_db"] for P in PS),
             min(S[f"N8_P{P}"]["mean_delta_ratio_of_sums_db"] for P in PS))
    ax.annotate("predicted failure: "
                + r"$L_k\!\sim\!\mathcal{U}\{3,7\}$ vs $r_{\max}(8)=4$",
                xy=(0, lo), xytext=(12, -2), textcoords="offset points",
                fontsize=6.2, color="#8a2b2a", ha="left",
                arrowprops=dict(arrowstyle="-", color="#c08a89", lw=0.6,
                                shrinkA=1, shrinkB=2))

    ax.set_xticks(x)
    ax.set_xticklabels([str(N) for N in NS])
    ax.set_xlabel(r"array size $N$")
    ax.set_ylabel(r"$\Delta_{\rm HS}$ (dB)")
    ax.set_title(r"(b) aperture, same configuration", fontsize=8, pad=3)

    ax2 = ax.twinx()
    ax2.grid(False)
    # EM-GS sits at a different absolute level for each pilot count (-4.4 dB at
    # P=10, -13.0 dB at P=30), so both are drawn and the axis is scaled to hold
    # them. Averaging the two would cancel opposite small movements and make
    # EM-GS look flatter than it is.
    for P in PS:
        emn = [S[f"N{N}_P{P}"]["mean_em_gs_db"] for N in NS]
        ax2.plot(x, emn, "s:", color="#2a78d6", mew=0, lw=1.3, alpha=0.9,
                 zorder=3)
        ax2.annotate(rf"EM-GS, $P={P}$: {max(emn) - min(emn):.3f} dB across $N$",
                     xy=(x[2], emn[2]), xytext=(-6, 5 if P == 30 else -9),
                     textcoords="offset points", fontsize=6.0,
                     color="#2a78d6", ha="right")
    ax2.set_ylabel("EM-GS NMSE (dB)", color="#2a78d6")
    ax2.tick_params(axis="y", colors="#2a78d6")
    allem = [S[f"N{N}_P{P}"]["mean_em_gs_db"] for N in NS for P in PS]
    ax2.set_ylim(min(allem) - 1.6, max(allem) + 3.4)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left",
              handlelength=1.6, labelspacing=0.2, borderpad=0.1,
              columnspacing=0.9, handletextpad=0.4, ncol=2, fontsize=5.9)


def main():
    S = assemble()
    OUT.mkdir(parents=True, exist_ok=True)
    Path("results/p14").mkdir(parents=True, exist_ok=True)
    Path("results/p14/aperture_summary.json").write_text(
        json.dumps(S, indent=1) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.55))
    em, hs, ccrb, crlb = panel_a(axes[0])
    panel_b(axes[1], S)
    fig.subplots_adjust(wspace=0.34)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig1_twopanel.{ext}", facecolor="white")
    plt.close(fig)

    # A shorter copy for paper/paper1/haatim_hsgs_letter_restyled.tex. Same
    # panels and same data; the restyled prose runs longer and needs the
    # vertical space. The frozen letter keeps the 2.55 in version above.
    fig2, ax2s = plt.subplots(1, 2, figsize=(7.16, 1.95))
    panel_a(ax2s[0])
    panel_b(ax2s[1], S)
    fig2.subplots_adjust(wspace=0.34)
    fig2.savefig(OUT / "fig1_twopanel_compact.pdf", facecolor="white")
    plt.close(fig2)

    # ---- every number the manuscript quotes from these panels -------------
    d = em - hs
    print("PANEL (a), configuration A, N=32 P=30")
    print(f"  ratio-of-sums gain      {d.min():+.3f} .. {d.max():+.3f} dB")
    print(f"  EM-GS vs uncon. CRLB, SNR>=5, max |gap| "
          f"{np.max(np.abs(em[2:] - crlb[2:])):.3f} dB")
    print(f"  CCRB below uncon. CRLB  {(crlb - ccrb).min():.3f}"
          f" .. {(crlb - ccrb).max():.3f} dB")
    print(f"  HS-GS above CCRB        {(hs - ccrb).min():.3f}"
          f" .. {(hs - ccrb).max():.3f} dB")

    print("\nPANEL (b), configuration A, mean over the 12 operating points at "
          "each N")
    print(f"  {'N':>3} {'med':>8} {'ros':>8} {'EM-GS':>8} {'active':>7}")
    for N in NS:
        med = np.mean([S[f"N{N}_P{P}"]["mean_delta_median_db"] for P in PS])
        ros = np.mean([S[f"N{N}_P{P}"]["mean_delta_ratio_of_sums_db"]
                       for P in PS])
        emn = np.mean([S[f"N{N}_P{P}"]["mean_em_gs_db"] for P in PS])
        act = np.mean([S[f"N{N}_P{P}"]["mean_active_frac"] for P in PS])
        print(f"  {N:>3} {med:>+8.3f} {ros:>+8.3f} {emn:>+8.3f} {act:>7.3f}")
    ems = [np.mean([S[f"N{N}_P{P}"]["mean_em_gs_db"] for P in PS]) for N in NS]
    print(f"  EM-GS spread across N, pooled over P: {max(ems) - min(ems):.3f} dB")
    for P in PS:
        v = [S[f"N{N}_P{P}"]["mean_em_gs_db"] for N in NS]
        print(f"  EM-GS spread across N at P={P}: {max(v) - min(v):.3f} dB "
              f"({[round(t, 3) for t in v]})")
    print("  The pooled figure is the smaller one because the two pilot counts "
          "move in\n  opposite directions; quote the per-P figure.")
    print("\n  wrote", OUT / "fig1_twopanel.pdf")


if __name__ == "__main__":
    main()

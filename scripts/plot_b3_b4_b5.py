"""Track-B final figures: B3 (NMSE vs SNR), B4 (NMSE vs P), B5 (gain vs N).

Reads only the saved per-trial stores via summary.json; runs no Monte Carlo.
Style matches the Track-A final figures: thin lines, small open markers,
subtle grid, thin spines, compact legend, no titles.

Uncertainty is deliberately NOT drawn -- no error bars, no whiskers. The
bootstrap 95% intervals live in summary.json and in the tables. Every
plotted point is a computed value; nothing is smoothed or interpolated.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

REPO = Path(__file__).resolve().parent.parent
TB = REPO / "results/track_b"
# Track-B figures stay inside the Track-B worktree. Track A is frozen and
# must not be written into, not even with new untracked files.
FINAL = REPO / "results/track_b/final"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.labelsize": 9.5, "axes.titlesize": 9.5, "legend.fontsize": 8,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.3, "lines.markersize": 4.5,
    "lines.markeredgewidth": 0.9, "grid.linewidth": 0.4, "grid.alpha": 0.35,
    "legend.frameon": True, "legend.framealpha": 0.9,
    "legend.edgecolor": "0.7", "legend.borderpad": 0.4,
    "legend.labelspacing": 0.3, "legend.handlelength": 2.0,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

FIGSIZE = (3.6, 2.9)
STYLE = {
    "biased_gs": dict(label="Biased GS", c="#D9A404", ls="-", m="o"),
    "em_gs":     dict(label="EM-GS",     c="#C4451C", ls="-", m="s"),
    "hs_gs":     dict(label="HS-GS (proposed)", c="#1F6FB4", ls="-", m="^"),
}
ORDER = ["biased_gs", "em_gs", "hs_gs"]
CRLB_STYLE = dict(color="#2E7D32", ls="-.", lw=1.1, label="CRLB (unconstrained)")


def crlb():
    f = TB / "crlb.json"
    return json.loads(f.read_text()) if f.exists() else None
NCOL = {8: "#8C8C8C", 16: "#C4451C", 32: "#1F6FB4"}


def style_axes(ax, xstep=None):
    ax.grid(True, which="major", linestyle="-", linewidth=0.4, alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction="in", length=3)
    if xstep:
        ax.xaxis.set_major_locator(MultipleLocator(xstep))


def save(fig, stem):
    out = TB / "final"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  wrote {stem}.pdf/.png")


def load(store):
    return json.loads((TB / store / "summary.json").read_text())


# ------------------------------------------------------------------ B3 ----
def b3():
    rows = load("b3")
    Ns = sorted({r["N"] for r in rows})
    Ps = sorted({r["P"] for r in rows})
    # one panel per (N, P): separate panels keep each uncluttered
    for P in Ps:
        fig, axes = plt.subplots(1, len(Ns), figsize=(3.05 * len(Ns), 2.75),
                                 sharey=True)
        for ax, N in zip(np.atleast_1d(axes), Ns):
            sub = sorted((r for r in rows if r["N"] == N and r["P"] == P),
                         key=lambda r: r["snr_db"])
            for alg in ORDER:
                st = STYLE[alg]
                ax.plot([r["snr_db"] for r in sub],
                        [r["pooled_db"][alg] for r in sub],
                        color=st["c"], ls=st["ls"], marker=st["m"],
                        markerfacecolor="none", markeredgewidth=0.9,
                        label=st["label"])
            C = crlb()
            if C:
                xs = [r["snr_db"] for r in sub]
                ax.plot(xs, [C["b3"][f"N{N}_P{P}_snr{x:+.0f}"] for x in xs],
                        **CRLB_STYLE)
            style_axes(ax, xstep=5)
            ax.set_xlabel("SNR (dB)")
            ax.set_title(f"$N = {N}$", fontsize=9)
        np.atleast_1d(axes)[0].set_ylabel("Channel NMSE$_G$ (dB)")
        np.atleast_1d(axes)[0].legend(loc="lower left")
        fig.tight_layout()
        save(fig, f"b3_nmse_vs_snr_P{P}")

    # gain over EM-GS vs SNR, one line per N, one panel per P
    fig, axes = plt.subplots(1, len(Ps), figsize=(3.3 * len(Ps), 2.75),
                             sharey=True)
    for ax, P in zip(np.atleast_1d(axes), Ps):
        for N in Ns:
            sub = sorted((r for r in rows if r["N"] == N and r["P"] == P),
                         key=lambda r: r["snr_db"])
            ax.plot([r["snr_db"] for r in sub],
                    [r["gain_hs_vs_em_db"] for r in sub], color=NCOL[N],
                    marker="o", markerfacecolor="none", markeredgewidth=0.9,
                    label=f"$N = {N}$")
        ax.axhline(0.0, color="0.55", lw=0.7, ls=":")
        style_axes(ax, xstep=5)
        ax.set_xlabel("SNR (dB)")
        ax.set_title(f"$P = {P}$", fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel("HS-GS gain over EM-GS (dB)")
    np.atleast_1d(axes)[0].legend(loc="best")
    fig.tight_layout()
    save(fig, "b3_gain_vs_snr")


# ------------------------------------------------------------------ B4 ----
def b4():
    rows = sorted(load("b4"), key=lambda r: r["P"])
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for alg in ORDER:
        st = STYLE[alg]
        ax.plot([r["P"] for r in rows], [r["pooled_db"][alg] for r in rows],
                color=st["c"], ls=st["ls"], marker=st["m"],
                markerfacecolor="none", markeredgewidth=0.9, label=st["label"])
    C = crlb()
    if C:
        ax.plot([r["P"] for r in rows],
                [C["b4"][f"P{r['P']}"] for r in rows], **CRLB_STYLE)
    K = 3
    ax.axvline(2 * K, color="0.6", lw=0.7, ls=":")
    ax.annotate(f"$P = 2K = {2*K}$", xy=(2 * K, ax.get_ylim()[1]),
                xytext=(2, -8), textcoords="offset points", fontsize=7.5,
                color="0.4", va="top")
    style_axes(ax, xstep=10)
    ax.set_xlabel("Pilot length $P$")
    ax.set_ylabel("Channel NMSE$_G$ (dB)")
    ax.legend(loc="best")
    fig.tight_layout()
    save(fig, "b4_nmse_vs_pilots_N16")


# ------------------------------------------------------------------ B5 ----
def b5():
    rows = load("b3")
    Ns = sorted({r["N"] for r in rows})
    Ps = sorted({r["P"] for r in rows})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.75))
    for P in Ps:
        g = [np.mean([r["gain_hs_vs_em_db"]
                      for r in rows if r["N"] == N and r["P"] == P]) for N in Ns]
        ax.plot(Ns, g, marker="o", markerfacecolor="none", markeredgewidth=0.9,
                label=f"$P = {P}$")
        w = [np.mean([r["win_rate_vs_em"]
                      for r in rows if r["N"] == N and r["P"] == P]) for N in Ns]
        ax2.plot(Ns, [100 * v for v in w], marker="s", markerfacecolor="none",
                 markeredgewidth=0.9, label=f"$P = {P}$")
    ax.axhline(0.0, color="0.55", lw=0.7, ls=":")
    ax2.axhline(50.0, color="0.55", lw=0.7, ls=":")
    for a, yl in ((ax, "Mean HS-GS gain over EM-GS (dB)"),
                  (ax2, "Per-trial win rate vs EM-GS (%)")):
        style_axes(a)
        a.set_xscale("log", base=2)
        a.set_xticks(Ns)
        a.set_xticklabels([str(n) for n in Ns])
        a.set_xlabel("Array size $N$")
        a.set_ylabel(yl)
        a.legend(loc="best")
    fig.tight_layout()
    save(fig, "b5_gain_scaling_vs_N")




# ------------------------------------------------------------------ B6 ----
def b6():
    """RSR sweep: does the structural advantage survive a weak reference?

    Three panels rather than two: overlaying both array sizes on one NMSE
    axis needed a six-entry legend that covered the data.
    """
    rows = sorted(json.loads((TB / "b6/summary.json").read_text()),
                  key=lambda r: (r["N"], r["rsr_db"]))
    Ns = sorted({r["N"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.85))
    for ax, N in zip(axes[:2], Ns):
        sub = [r for r in rows if r["N"] == N]
        xs = [r["rsr_db"] for r in sub]
        for alg in ORDER:
            st = STYLE[alg]
            ax.plot(xs, [r["pooled_db"][alg] for r in sub], color=st["c"],
                    ls=st["ls"], marker=st["m"], markerfacecolor="none",
                    markeredgewidth=0.9, label=st["label"])
        style_axes(ax, xstep=6)
        ax.set_xlabel("RSR (dB)")
        ax.set_title(f"$N = {N}$", fontsize=9)
    axes[0].set_ylabel("Channel NMSE$_G$ (dB)")
    axes[0].legend(loc="best")
    ax2 = axes[2]
    for N in Ns:
        sub = [r for r in rows if r["N"] == N]
        ax2.plot([r["rsr_db"] for r in sub],
                 [r["gain_hs_vs_em_db"] for r in sub], color=NCOL[N],
                 marker="o", markerfacecolor="none", markeredgewidth=0.9,
                 label=f"$N = {N}$")
    ax2.axhline(0.0, color="0.55", lw=0.7, ls=":")
    style_axes(ax2, xstep=6)
    ax2.set_xlabel("RSR (dB)")
    ax2.set_ylabel("HS-GS gain over EM-GS (dB)")
    ax2.legend(loc="best")
    fig.tight_layout()
    save(fig, "b6_rsr_sweep")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["b3", "b4", "b5", "b6"]
    for w in which:
        print(f"{w}:")
        globals()[w]()

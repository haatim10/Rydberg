"""Complete figure set for Track B1 — Cui's estimators on the ULA channel.

NO Hankel constraint appears anywhere in these figures. Only biased GS and
EM-GS are plotted, plus the unconstrained CRLB where it is the relevant
reference. Everything is read from stored aggregates; no Monte Carlo is run.

Three of these did not exist before: the array-size control, the
distance-to-bound panel, and the EM-GS-advantage-vs-RSR panel. All three are
Track-B1 results (they involve no structural prior) that were previously
only reported as numbers.
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
OUT = TB / "final"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 9,
    "axes.labelsize": 9.5, "legend.fontsize": 8, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "axes.linewidth": 0.6, "lines.linewidth": 1.3,
    "lines.markersize": 4.5, "lines.markeredgewidth": 0.9,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})
GS = dict(label="Biased GS", c="#D9A404", ls="-", m="o")
EM = dict(label="EM-GS", c="#C4451C", ls="-", m="s")
CR = dict(color="#2E7D32", ls="-.", lw=1.1, label="Unconstrained CRLB")
NCOL = {8: "#8C8C8C", 16: "#C4451C", 32: "#1F6FB4"}


def ax_style(ax, xstep=None):
    ax.grid(True, which="major", ls="-", lw=0.4, alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(direction="in", length=3)
    if xstep:
        ax.xaxis.set_major_locator(MultipleLocator(xstep))


def save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  wrote {stem}.pdf/.png")


def line(ax, xs, ys, st, **kw):
    ax.plot(xs, ys, color=st["c"], ls=st["ls"], marker=st["m"],
            markerfacecolor="none", markeredgewidth=0.9, label=st["label"], **kw)


S3 = json.loads((TB / "b3/summary.json").read_text())
S6 = json.loads((TB / "b6/summary.json").read_text())
C = json.loads((TB / "crlb.json").read_text())


def fig_array_size_control():
    """THE Track-B1 result: the baseline does not improve with array size."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.85))
    for P, ax in ((10, a), (30, b)):
        for N in (8, 16, 32):
            sub = sorted((r for r in S3 if r["N"] == N and r["P"] == P),
                         key=lambda r: r["snr_db"])
            ax.plot([r["snr_db"] for r in sub],
                    [r["pooled_db"]["em_gs"] for r in sub], color=NCOL[N],
                    marker="s", markerfacecolor="none", markeredgewidth=0.9,
                    label=f"EM-GS, $N = {N}$")
        ax_style(ax, xstep=5)
        ax.set_xlabel("SNR (dB)")
        ax.set_title(f"$P = {P}$", fontsize=9)
    a.set_ylabel("Channel NMSE$_G$ (dB)")
    a.legend(loc="lower left")
    fig.tight_layout()
    save(fig, "b1_em_gs_vs_array_size")

    # the same data as a spread, which is the actual claim
    fig, ax = plt.subplots(figsize=(4.2, 2.85))
    for P in (10, 30):
        xs, ys = [], []
        for s in sorted({r["snr_db"] for r in S3}):
            v = [r["pooled_db"]["em_gs"] for r in S3
                 if r["P"] == P and r["snr_db"] == s]
            xs.append(s); ys.append(max(v) - min(v))
        ax.plot(xs, ys, marker="o", markerfacecolor="none",
                markeredgewidth=0.9, label=f"$P = {P}$")
    ax.axhline(0.2, color="0.55", lw=0.7, ls=":")
    ax.annotate("0.2 dB", xy=(20, 0.2), xytext=(-4, 3),
                textcoords="offset points", fontsize=7.5, color="0.4", ha="right")
    ax_style(ax, xstep=5)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("EM-GS spread across $N \\in \\{8,16,32\\}$ (dB)")
    ax.set_ylim(0, 0.32)
    ax.legend(loc="best")
    fig.tight_layout()
    save(fig, "b1_em_gs_spread_across_N")


def fig_distance_to_bound():
    """How close is the prior-free baseline to the best prior-free estimator?"""
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.85))
    N = 8
    sub10 = sorted((r for r in S3 if r["N"] == N and r["P"] == 10),
                   key=lambda r: r["snr_db"])
    sub30 = sorted((r for r in S3 if r["N"] == N and r["P"] == 30),
                   key=lambda r: r["snr_db"])
    for ax, sub, P in ((a, sub10, 10), (b, sub30, 30)):
        xs = [r["snr_db"] for r in sub]
        line(ax, xs, [r["pooled_db"]["biased_gs"] for r in sub], GS)
        line(ax, xs, [r["pooled_db"]["em_gs"] for r in sub], EM)
        ax.plot(xs, [C["b3"][f"N{N}_P{P}_snr{x:+.0f}"] for x in xs], **CR)
        ax_style(ax, xstep=5)
        ax.set_xlabel("SNR (dB)")
        ax.set_title(f"$N = 8$, $P = {P}$", fontsize=9)
    a.set_ylabel("Channel NMSE$_G$ (dB)")
    a.legend(loc="lower left")
    fig.tight_layout()
    save(fig, "b1_baseline_vs_unconstrained_crlb")

    fig, ax = plt.subplots(figsize=(4.2, 2.85))
    for P, mk in ((10, "o"), (30, "s")):
        for N in (8, 16, 32):
            sub = sorted((r for r in S3 if r["N"] == N and r["P"] == P),
                         key=lambda r: r["snr_db"])
            xs = [r["snr_db"] for r in sub]
            ax.plot(xs, [r["pooled_db"]["em_gs"]
                         - C["b3"][f"N{N}_P{P}_snr{x:+.0f}"]
                         for r, x in zip(sub, xs)], color=NCOL[N], marker=mk,
                    markerfacecolor="none", markeredgewidth=0.9,
                    ls="-" if P == 30 else "--",
                    label=f"$N={N}$, $P={P}$")
    ax_style(ax, xstep=5)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("EM-GS above the unconstrained CRLB (dB)")
    ax.legend(loc="upper left", fontsize=6.6, ncol=2)
    fig.tight_layout()
    save(fig, "b1_em_gs_gap_to_bound")


def fig_em_advantage_vs_rsr():
    """Where does EM-GS's Bessel weighting actually pay? R(kappa) -> 1."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.85))
    for N, ax in ((8, a), (32, b)):
        sub = sorted((r for r in S6 if r["N"] == N), key=lambda r: r["rsr_db"])
        xs = [r["rsr_db"] for r in sub]
        line(ax, xs, [r["pooled_db"]["biased_gs"] for r in sub], GS)
        line(ax, xs, [r["pooled_db"]["em_gs"] for r in sub], EM)
        ax_style(ax, xstep=6)
        ax.set_xlabel("RSR (dB)")
        ax.set_title(f"$N = {N}$", fontsize=9)
    a.set_ylabel("Channel NMSE$_G$ (dB)")
    a.legend(loc="best")
    fig.tight_layout()
    save(fig, "b1_gs_vs_em_gs_across_rsr")

    fig, ax = plt.subplots(figsize=(4.2, 2.85))
    for N in (8, 32):
        sub = sorted((r for r in S6 if r["N"] == N), key=lambda r: r["rsr_db"])
        ax.plot([r["rsr_db"] for r in sub],
                [r["pooled_db"]["biased_gs"] - r["pooled_db"]["em_gs"]
                 for r in sub], color=NCOL[N], marker="o",
                markerfacecolor="none", markeredgewidth=0.9, label=f"$N = {N}$")
    ax.axhline(0.0, color="0.55", lw=0.7, ls=":")
    ax_style(ax, xstep=6)
    ax.set_xlabel("RSR (dB)")
    ax.set_ylabel("EM-GS advantage over biased GS (dB)")
    ax.legend(loc="best")
    fig.tight_layout()
    save(fig, "b1_em_gs_advantage_vs_rsr")


if __name__ == "__main__":
    print("Track B1 figures (no Hankel constraint anywhere):")
    fig_array_size_control()
    fig_distance_to_bound()
    fig_em_advantage_vs_rsr()

"""Track-B figures with BOTH bounds drawn.

Three curves of reference, and they mean different things:

  Unconstrained CRLB (corrected)  bounds estimators that treat G as 2NK free
      real parameters -- biased GS and EM-GS. Each magnitude measurement
      contributes ONE real constraint, so its Fisher contribution is rank 1
      in real coordinates. Flat in N, as a row-separable problem requires.

  Constrained CRLB                bounds estimators that know G lies on the
      3*sum(L_k)-parameter geometric manifold -- the bound HS-GS should be
      judged against. Gorman-Hero / Stoica-Ng.

  (the previously plotted bound is NOT drawn: it credits each magnitude
   measurement with constraining both quadratures and is optimistic by up to
   4.2 dB. See the errata section of the report.)

Neither bound governs a biased estimator, and all three estimators here are
biased, so small excursions below are possible and are not violations.
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
    "axes.labelsize": 9.5, "legend.fontsize": 7.6, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "axes.linewidth": 0.6, "lines.linewidth": 1.3,
    "lines.markersize": 4.2, "lines.markeredgewidth": 0.9,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})
STYLE = {
    "biased_gs": dict(label="Biased GS", c="#D9A404", ls="-", m="o"),
    "em_gs":     dict(label="EM-GS", c="#C4451C", ls="-", m="s"),
    "hs_gs":     dict(label="HS-GS (proposed)", c="#1F6FB4", ls="-", m="^"),
}
ORDER = ["biased_gs", "em_gs", "hs_gs"]
UNC = dict(color="#2E7D32", ls="-.", lw=1.15, label="Unconstrained CRLB")
CON = dict(color="#6A3D9A", ls=":", lw=1.5, label="Constrained CRLB")
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


CC = json.loads((TB / "constrained_crlb.json").read_text())
S3 = json.loads((TB / "b3/summary.json").read_text())
S4 = json.loads((TB / "b4/summary.json").read_text())
S6 = json.loads((TB / "b6/summary.json").read_text())


def b3():
    for P in (10, 30):
        fig, axes = plt.subplots(1, 3, figsize=(9.4, 2.95), sharey=True)
        for ax, N in zip(axes, (8, 16, 32)):
            sub = sorted((r for r in S3 if r["N"] == N and r["P"] == P),
                         key=lambda r: r["snr_db"])
            xs = [r["snr_db"] for r in sub]
            for alg in ORDER:
                st = STYLE[alg]
                ax.plot(xs, [r["pooled_db"][alg] for r in sub], color=st["c"],
                        ls=st["ls"], marker=st["m"], markerfacecolor="none",
                        markeredgewidth=0.9, label=st["label"])
            keys = [f"N{N}_P{P}_snr{x:+.0f}" for x in xs]
            ax.plot(xs, [CC["unconstrained_rank1"]["b3"][k] for k in keys], **UNC)
            ax.plot(xs, [CC["constrained"]["b3"][k] for k in keys], **CON)
            ax_style(ax, xstep=5)
            ax.set_xlabel("SNR (dB)")
            ax.set_title(f"$N = {N}$", fontsize=9)
        axes[0].set_ylabel("Channel NMSE$_G$ (dB)")
        axes[0].legend(loc="lower left")
        fig.tight_layout()
        save(fig, f"b3_with_constrained_crlb_P{P}")

    # headroom: how far above the constrained bound is each estimator?
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9), sharey=True)
    for ax, P in zip(axes, (10, 30)):
        for N in (8, 16, 32):
            sub = sorted((r for r in S3 if r["N"] == N and r["P"] == P),
                         key=lambda r: r["snr_db"])
            xs = [r["snr_db"] for r in sub]
            gap = [r["pooled_db"]["hs_gs"]
                   - CC["constrained"]["b3"][f"N{N}_P{P}_snr{x:+.0f}"]
                   for r, x in zip(sub, xs)]
            ax.plot(xs, gap, color=NCOL[N], marker="^", markerfacecolor="none",
                    markeredgewidth=0.9, label=f"$N = {N}$")
        ax.axhline(0.0, color="0.55", lw=0.7, ls=":")
        ax_style(ax, xstep=5)
        ax.set_xlabel("SNR (dB)")
        ax.set_title(f"$P = {P}$", fontsize=9)
    axes[0].set_ylabel("HS-GS above the constrained CRLB (dB)")
    axes[0].legend(loc="best")
    fig.tight_layout()
    save(fig, "b3_hsgs_headroom_to_constrained")


def b4():
    rows = sorted(S4, key=lambda r: r["P"])
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    xs = [r["P"] for r in rows]
    for alg in ORDER:
        st = STYLE[alg]
        ax.plot(xs, [r["pooled_db"][alg] for r in rows], color=st["c"],
                ls=st["ls"], marker=st["m"], markerfacecolor="none",
                markeredgewidth=0.9, label=st["label"])
    ax.plot(xs, [CC["unconstrained_rank1"]["b4"][f"P{p}"] for p in xs], **UNC)
    ax.plot(xs, [CC["constrained"]["b4"][f"P{p}"] for p in xs], **CON)
    ax.axvline(6, color="0.6", lw=0.7, ls=":")
    ax_style(ax, xstep=10)
    ax.set_xlabel("Pilot length $P$")
    ax.set_ylabel("Channel NMSE$_G$ (dB)")
    ax.legend(loc="best")
    fig.tight_layout()
    save(fig, "b4_with_constrained_crlb")


def b6():
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.9))
    for ax, N in zip(axes, (8, 32)):
        sub = sorted((r for r in S6 if r["N"] == N), key=lambda r: r["rsr_db"])
        xs = [r["rsr_db"] for r in sub]
        for alg in ORDER:
            st = STYLE[alg]
            ax.plot(xs, [r["pooled_db"][alg] for r in sub], color=st["c"],
                    ls=st["ls"], marker=st["m"], markerfacecolor="none",
                    markeredgewidth=0.9, label=st["label"])
        keys = [f"N{N}_P30_snr+5_rsr{x:+.0f}" for x in xs]
        ax.plot(xs, [CC["unconstrained_rank1"]["b6"][k] for k in keys], **UNC)
        ax.plot(xs, [CC["constrained"]["b6"][k] for k in keys], **CON)
        ax_style(ax, xstep=6)
        ax.set_xlabel("RSR (dB)")
        ax.set_title(f"$N = {N}$", fontsize=9)
    axes[0].set_ylabel("Channel NMSE$_G$ (dB)")
    axes[0].legend(loc="best")
    fig.tight_layout()
    save(fig, "b6_with_constrained_crlb")


if __name__ == "__main__":
    print("Track-B figures with both bounds:")
    b3(); b4(); b6()

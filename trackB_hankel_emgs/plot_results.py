"""Four primary figures. Reads ONLY the verified stores written by
verify_results.py -- results/summary.json and results/diagnostic_spectrum.json.
No number is hard-coded here, so figures cannot drift from the tables.

    python verify_results.py && python plot_results.py

Style: publication plain. White background, no gradients, no titles inside the
axes, consistent estimator labels and colours across all four figures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "lines.linewidth": 1.3, "lines.markersize": 4.2,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
})
EM = dict(marker="o", color="C0", label="EM-GS")
HK = dict(marker="^", color="C3", label="Hankel-EM-GS")


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print("  wrote figures/" + name)


def fig1_snr(S):
    A = S.get("experiment_A")
    if not A:
        return
    snr = sorted(float(k) for k in A)
    g = lambda f: [A[f"{s:+.1f}"][f] for s in snr]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    ax[0].plot(snr, g("em_gs_db"), **EM)
    ax[0].plot(snr, g("hankel_db"), **HK)
    ax[0].set_xlabel("SNR (dB)"); ax[0].set_ylabel("channel NMSE (dB)")
    ax[0].legend(framealpha=1.0)

    gain, lo, hi = np.array(g("gain_db")), np.array(g("gain_ci_lo")), np.array(g("gain_ci_hi"))
    ax[1].axhline(0, color="0.5", lw=0.9, ls=":")
    ax[1].errorbar(snr, gain, yerr=[gain - lo, hi - gain], fmt="^-", color="C3",
                   capsize=2.5, elinewidth=0.9)
    ax[1].set_xlabel("SNR (dB)")
    ax[1].set_ylabel(r"gain $\Delta_{\mathrm{H}}$ (dB)")
    ax[1].text(0.03, 0.95, "above 0: Hankel helps", transform=ax[1].transAxes,
               va="top", fontsize=7.5, color="0.35")
    n = A[f"{snr[0]:+.1f}"]["trials"]
    for a in ax:
        a.set_xticks(snr)
    fig.text(0.5, -0.06, f"$N$=8, $K$=3, $P$=30, RSR=12 dB, "
             f"$L_k\\sim\\mathcal{{U}}\\{{3..7\\}}$, {n} paired trials/point; "
             f"error bars 95% paired bootstrap CI", ha="center", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig1_nmse_vs_snr")


def fig2_array(S):
    B = S.get("experiment_B")
    if not B:
        return
    Ns = sorted(int(k) for k in B)
    mean = [B[str(N)]["mean_gain_db"] for N in Ns]
    mx = [B[str(N)]["max_gain_db"] for N in Ns]
    mn = [B[str(N)]["min_gain_db"] for N in Ns]
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.axhline(0, color="0.5", lw=0.9, ls=":")
    ax.fill_between(Ns, mn, mx, color="C3", alpha=0.13, lw=0,
                    label="min-max over SNR grid")
    ax.plot(Ns, mean, "^-", color="C3", label="mean gain over SNR grid")
    ax.plot(Ns, mx, "v--", color="0.45", ms=3.4, lw=0.9, label="max single point")
    ax.set_xscale("log", base=2); ax.set_xticks(Ns)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("array size $N$")
    ax.set_ylabel(r"gain $\Delta_{\mathrm{H}}$ over EM-GS (dB)")
    ax.legend(framealpha=1.0, loc="upper left")
    for N, m in zip(Ns, mean):
        ax.annotate(f"$r_{{\\max}}$={B[str(N)]['r_max']}", (N, m), textcoords="offset points",
                    xytext=(0, -13), ha="center", fontsize=7, color="0.35")
    fig.tight_layout()
    save(fig, "fig2_gain_vs_array_size")


def fig3_pathcount(S):
    C = S.get("experiment_C")
    if not C:
        return
    L = sorted(int(k) for k in C)
    f = lambda k: np.array([C[str(x)][k] for x in L])
    gain, lo, hi = f("gain_db"), f("gain_ci_lo"), f("gain_ci_hi")
    r_max = C[str(L[0])]["r_max"]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))

    ax[0].axhline(0, color="0.5", lw=0.9, ls=":")
    ax[0].axvline(r_max, color="0.3", ls=":", lw=1.1)
    ax[0].errorbar(L, gain, yerr=[gain - lo, hi - gain], fmt="^-", color="C3",
                   capsize=2.5, elinewidth=0.9)
    ax[0].annotate(f"rank ceiling\n$r_{{\\max}}={r_max}$", xy=(r_max, 0.97),
                   xycoords=("data", "axes fraction"), xytext=(-5, 0),
                   textcoords="offset points", ha="right", va="top",
                   fontsize=7.5, color="0.3")
    ax[0].set_xlabel("true paths per user $L$")
    ax[0].set_ylabel(r"gain $\Delta_{\mathrm{H}}$ (dB)")

    ax[1].plot(L, L, "-", color="0.6", lw=0.9, label=r"$\hat L = L$")
    ax[1].plot(L, f("mean_L_hat"), "s-", color="C2", label=r"$E[\hat L]$")
    ax[1].axhline(r_max, color="0.3", ls=":", lw=1.1)
    ax[1].set_xlabel("true paths per user $L$")
    ax[1].set_ylabel(r"selected rank $\hat L$")
    ax[1].legend(framealpha=1.0, loc="upper left")

    for a in ax:
        a.set_xticks(L); a.margins(y=0.16)
    n = C[str(L[0])]["trials"]
    fig.text(0.5, -0.06, f"$N$=32 ($r_{{\\max}}$={r_max}), $K$=3, $P$=30, SNR=5 dB, "
             f"RSR=12 dB, $L$ fixed and identical across users, "
             f"{n} paired trials/point", ha="center", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig3_gain_vs_path_count")


def fig4_spectrum():
    p = RES / "diagnostic_spectrum.json"
    if not p.exists():
        return
    D = json.loads(p.read_text()); c = D["config"]
    k = np.arange(1, len(D["true_channel"]) + 1)
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    ax.semilogy(k, D["true_channel"], "o-", color="0.1",
                label=f"true channel ($L={c['L']}$)")
    ax.semilogy(k, D["em_gs_estimate"], "s-", color="C0", label="EM-GS estimate")
    ax.semilogy(k, D["after_cadzow"], "^-", color="C3",
                label=f"after Cadzow ($\\hat L={c['L']}$)")
    ax.axvline(c["L"] + 0.5, color="0.5", ls=":", lw=0.9)
    ax.set_xticks([1, 4, 8, 12, 16])
    ax.set_xlabel("Hankel singular value index")
    ax.set_ylabel("normalised singular value")
    ax.legend(framealpha=1.0, loc="center right", bbox_to_anchor=(1.0, 0.42))
    fig.tight_layout()
    save(fig, "fig4_hankel_spectrum")


def main() -> int:
    p = RES / "summary.json"
    if not p.exists():
        print("results/summary.json missing. Run: python verify_results.py")
        return 1
    S = json.loads(p.read_text())
    fig1_snr(S); fig2_array(S); fig3_pathcount(S); fig4_spectrum()
    return 0


if __name__ == "__main__":
    sys.exit(main())

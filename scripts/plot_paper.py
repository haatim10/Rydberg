"""IEEE-format figures for the HS-GS manuscript.

Four figures, each supporting one claim made in the paper. Sizes are IEEE
column widths (3.5 in single, 7.16 in double). All values are recomputed
from the per-trial stores; nothing is transcribed.
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
R = REPO / "results" / "track_b"
OUT = REPO / "paper" / "fig"
NBOOT, SEED = 2000, 987654321
SNRS = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "lines.linewidth": 1.1, "lines.markersize": 3.6,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
STY = {"biased_gs": ("s--", "0.40", "GS [1]"),
       "em_gs":     ("o-",  "C0",   "EM-GS [1]"),
       "hs_gs":     ("^-",  "C3",   "HS-GS (proposed)")}


def b3(N, P, snr):
    return np.load(R / "b3" / f"N{N}_P{P}_snr{snr:+05.1f}.npz")


def pooled(d, est):
    return 10 * np.log10(d[f"num_{est}"].sum() / d["denom"].sum())


def crlb():
    return json.loads((R / "constrained_crlb.json").read_text())


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", facecolor="white")
    plt.close(fig)
    print("  wrote", name)


# ---- Fig. 1: NMSE vs SNR at N=32, P=30, all three estimators + both bounds
def fig1():
    CC = crlb()
    fig, ax = plt.subplots(figsize=(3.4, 2.35))
    for est, (mk, c, lab) in STY.items():
        ax.plot(SNRS, [pooled(b3(32, 30, s), est) for s in SNRS], mk, color=c, label=lab)
    key = [f"N32_P30_snr{s:+.0f}" for s in SNRS]
    ax.plot(SNRS, [CC["unconstrained_rank1"]["b3"][k] for k in key], ":",
            color="0.15", lw=1.0, label="CRLB, unconstrained")
    ax.plot(SNRS, [CC["constrained"]["b3"][k] for k in key], "-.",
            color="C2", lw=1.0, label="CCRB, geometric")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel(r"NMSE$_G$ (dB)")
    ax.set_xticks(SNRS); ax.legend(framealpha=1.0, loc="lower left")
    save(fig, "fig1_nmse_vs_snr")


# ---- Fig. 2: paired gain vs array size -- per-operating-point, the headline pooling
def fig2():
    """Plots the per-operating-point gain, the same pooling as the 2.85 dB headline.

    Individual (SNR, P) points are shown as well as their mean, because at N = 8
    the per-point gains straddle zero and a mean alone would hide that.
    """
    fig, ax = plt.subplots(figsize=(3.4, 2.25))
    for P, mk, c in ((10, "o-", "C0"), (30, "s-", "C3")):
        means = []
        for N in (8, 16, 32):
            g = [10 * np.log10(b3(N, P, s)["num_em_gs"].sum()
                               / b3(N, P, s)["num_hs_gs"].sum()) for s in SNRS]
            ax.plot([N] * len(g), g, mk[0], color=c, ms=2.6, alpha=0.40,
                    linestyle="none", zorder=1)
            means.append(float(np.mean(g)))
        ax.plot([8, 16, 32], means, mk, color=c, label=f"$P={P}$, mean", zorder=3)
        print(f"    fig2 P={P}: per-point means {[round(v,3) for v in means]}")
    ax.axhline(0, color="0.5", lw=0.8, ls=":")
    ax.set_xscale("log", base=2); ax.set_xticks([8, 16, 32])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("array size $N$")
    ax.set_ylabel(r"$\Delta_{\mathrm{HS}}$ over EM-GS (dB)")
    ax.legend(framealpha=1.0, loc="upper left")
    save(fig, "fig2_gain_vs_N")


# ---- Fig. 3: controlled path count (the mechanism test)
def fig3():
    rows = []
    for f in sorted(glob.glob(str(R / "b7" / "L*.npz"))):
        d = np.load(f); L = int(Path(f).stem[1:])
        e, h, den = d["num_em_gs"], d["num_hs_gs"], d["denom"]
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, den.size, size=(NBOOT, den.size))
        bs = 10 * np.log10(e[idx].sum(1) / h[idx].sum(1))
        rows.append((L, 10 * np.log10(d["num_biased_gs"].sum() / den.sum()),
                     10 * np.log10(e.sum() / den.sum()),
                     10 * np.log10(h.sum() / den.sum()),
                     10 * np.log10(e.sum() / h.sum()),
                     np.percentile(bs, 2.5), np.percentile(bs, 97.5),
                     d["L_hat"].mean()))
    L = [r[0] for r in rows]; cap = 16
    fig, ax = plt.subplots(2, 1, figsize=(3.4, 3.15), sharex=True)

    ax[0].plot(L, [r[1] for r in rows], "s--", color="0.40", label="GS")
    ax[0].plot(L, [r[2] for r in rows], "o-", color="C0", label="EM-GS")
    ax[0].plot(L, [r[3] for r in rows], "^-", color="C3", label="HS-GS")
    ax[0].set_ylabel(r"NMSE$_G$ (dB)")
    ax[0].legend(framealpha=1.0, loc="lower right")
    ax[0].set_title("(a) NMSE vs path count", fontsize=7.5)

    ax[1].axhline(0, color="0.5", lw=0.8, ls=":")
    ax[1].plot(L, [r[4] for r in rows], "^-", color="C3")
    ax[1].fill_between(L, [r[5] for r in rows], [r[6] for r in rows],
                       color="C3", alpha=0.18, lw=0)
    ax[1].set_ylabel(r"$\Delta_{\mathrm{HS}}$ over EM-GS (dB)")
    ax[1].set_title("(b) gain vanishes at the rank cap", fontsize=7.5)

    for a in ax:
        a.margins(y=0.22)
        a.axvline(cap, color="0.25", ls=":", lw=1.0)
        a.annotate("rank cap $r_{\\mathrm{max}}\\!=\\!16$",
                   xy=(cap, 0.97), xycoords=("data", "axes fraction"),
                   xytext=(-4, 0), textcoords="offset points",
                   ha="right", va="top", fontsize=6.6, color="0.25")
        a.set_xticks(L)
    ax[1].set_xlabel("paths per user $L$")
    save(fig, "fig3_pathcount")


# ---- Fig. 4: Hankel spectrum -- why the prior exists and what Cadzow does
def fig4():
    import sys; sys.path.insert(0, str(REPO))
    from rydberg_sim.gs import em_gs_channel_rows
    from rydberg_sim.monte_carlo import generate_channel_estimation_trial
    from rydberg_sim.track_b_drivers import track_b_spec, TRACK_B_K, TRACK_B_RSR_DB
    from rydberg_sim.track_b_proposed import cadzow_project, best_pencil
    from rydberg_sim.track_b_structure import hankel_matrix

    N, L = 32, 3
    sp = track_b_spec(P=30, n_trials=1, N=N, K=TRACK_B_K, L=(L,) * TRACK_B_K,
                      experiment="paper_fig4")
    w = generate_channel_estimation_trial(sp, 0, 5.0, TRACK_B_RSR_DB)
    Ge = em_gs_channel_rows(w.S, w.Z, w.B, w.sigma2, max_iter=50).G_hat
    p = best_pencil(N)
    sv = lambda g: np.linalg.svd(hankel_matrix(g, p), compute_uv=False)
    norm = lambda s: s / s[0]
    gc = cadzow_project(Ge[:, 0], L, n_iter=4)

    fig, ax = plt.subplots(figsize=(3.4, 2.35))
    k = np.arange(1, min(hankel_matrix(w.G[:, 0], p).shape) + 1)
    ax.semilogy(k, norm(sv(w.G[:, 0])), "o-", color="0.10", label=f"true channel ($L={L}$)")
    ax.semilogy(k, norm(sv(Ge[:, 0])), "s-", color="C0", label="EM-GS estimate")
    ax.semilogy(k, norm(sv(gc)), "^-", color="C3", label=f"after Cadzow ($\\hat L={L}$)")
    ax.axvline(L + 0.5, color="0.5", ls=":", lw=0.9)
    ax.set_xlabel("singular value index"); ax.set_ylabel("normalised singular value")
    ax.set_xticks([1, 4, 8, 12, 16])
    # the empty mid-right band is the only region no curve passes through
    ax.legend(framealpha=1.0, loc="center right", bbox_to_anchor=(1.0, 0.42))
    save(fig, "fig4_hankel_spectrum")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig1(); fig2(); fig3(); fig4()

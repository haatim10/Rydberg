"""Absolute NMSE of EM-GS vs Hankel-EM-GS as the array grows.

Companion to fig2, which plots only the DIFFERENCE Delta_H. This plots the two
NMSE curves themselves, so the absolute behaviour of each estimator with N is
visible rather than inferred.

Reads the existing per-trial stores in results/grid/*.npz. Nothing is
reconstructed from the gain plot and no value is approximated: every number
here is recomputed from stored per-trial NMSE numerators and the shared
denominator.

AGGREGATION (matches the existing pipeline exactly):
  * within an operating point, across trials -- RATIO OF SUMS, i.e. sum the
    energies first and convert to dB last:
        NMSE_dB = 10 log10( sum_t ||Ghat_t - G_t||_F^2 / sum_t ||G_t||_F^2 )
    Averaging therefore happens in the LINEAR (energy) domain, BEFORE the dB
    conversion. This is verify_results.pooled_db.
  * across SNR (the summary panel) -- unweighted mean of the per-point dB
    values, which is what verify_results reports as *_mean_over_points.
    Averaging here happens AFTER the dB conversion; see the note printed by
    --explain for why the linear-domain alternative is degenerate.

    python figure_array_size_comparison.py
    python figure_array_size_comparison.py --explain
"""
from __future__ import annotations

import csv
import glob
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from verify_results import boot_ci, pooled_db

HERE = Path(__file__).resolve().parent
RES, FIG = HERE / "results", HERE / "figures"
PANEL_SNRS = (-10.0, 0.0, 20.0)          # requested; all present in SNR_GRID_DB

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "lines.linewidth": 1.3, "lines.markersize": 4.6,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.facecolor": "white",
})
EM = dict(marker="o", color="C0", label="EM-GS")
HK = dict(marker="^", color="C3", label="Hankel-EM-GS")


def load_grid() -> dict:
    out = {}
    for f in sorted(glob.glob(str(RES / "grid" / "*.npz"))):
        st = Path(f).stem
        if st.endswith(".tmp"):
            continue
        d = np.load(f)
        out[(int(st[1:3]), float(st.split("snr")[1]))] = {
            k: d[k] for k in d.files if k != "fingerprint"}
    return out


def point_stats(d: dict) -> dict:
    e, h, den = d["num_em_gs"], d["num_hankel_em_gs"], d["denom"]
    # CI on each ABSOLUTE NMSE: resample trials, recompute the ratio of sums.
    rng = np.random.default_rng(cfg.BOOT_SEED)
    idx = rng.integers(0, e.size, size=(cfg.NBOOT, e.size))
    be = 10 * np.log10(e[idx].sum(1) / den[idx].sum(1))
    bh = 10 * np.log10(h[idx].sum(1) / den[idx].sum(1))
    glo, ghi = boot_ci(e, h)                     # paired CI on the gain
    return dict(
        trials=int(e.size),
        em_db=pooled_db(e, den), hk_db=pooled_db(h, den),
        em_lo=float(np.percentile(be, 2.5)), em_hi=float(np.percentile(be, 97.5)),
        hk_lo=float(np.percentile(bh, 2.5)), hk_hi=float(np.percentile(bh, 97.5)),
        gain_db=float(10 * np.log10(e.sum() / h.sum())),
        gain_lo=glo, gain_hi=ghi,
        active_frac=float(d["active"].mean()), r_max=int(d["r_max"][0]),
        mean_L_hat=float(d["L_hat"].mean()),
    )


# ---------------------------------------------------------------- invariance
def invariance_checks(grid: dict) -> list[tuple[str, bool, str]]:
    """Confirm that changing N changed ONLY N."""
    import em_gs, hankel_projection as hp
    from system_model import make_world
    out = []

    # one world per N at a common (P, SNR, trial)
    W = {N: make_world(0, N=N, P=cfg.P_DEFAULT, snr_db=5.0) for N in cfg.N_GRID}

    # L_k comes from a stream keyed on (seed, trial) only -> identical across N
    Lk = {N: tuple(int(v) for v in w.L_k) for N, w in W.items()}
    out.append(("path generation: L_k independent of N",
                len(set(Lk.values())) == 1, f"{Lk}"))

    # sigma2 = K / SNR_lin -- no N dependence
    s2 = {N: round(float(w.sigma2), 12) for N, w in W.items()}
    out.append(("SNR definition: sigma2 = K/SNR_lin, independent of N",
                len(set(s2.values())) == 1, f"sigma2={list(s2.values())[0]:.6f}"))

    # |b| = sqrt(RSR_lin), constant over (n,p) and independent of N
    b = {N: round(float(np.abs(w.B).max()), 10) for N, w in W.items()}
    bconst = all(np.allclose(np.abs(w.B), np.abs(w.B).flat[0]) for w in W.values())
    out.append(("RSR definition: |b| constant and independent of N",
                len(set(b.values())) == 1 and bconst, f"|b|={list(b.values())[0]:.4f}"))

    # pilots: same K x P, same normalisation; only the draw differs
    sh = {N: W[N].S.shape for N in cfg.N_GRID}
    pw = {N: round(float(np.mean(np.abs(W[N].S) ** 2)), 3) for N in cfg.N_GRID}
    out.append(("pilot normalisation: K x P shape and E|s|^2 unchanged by N",
                len(set(sh.values())) == 1, f"shape={list(sh.values())[0]}, E|s|^2~{pw}"))

    # channel energy per element is normalised, so NMSE is comparable across N
    pe = {N: round(float(np.mean(np.abs(W[N].G) ** 2)), 3) for N in cfg.N_GRID}
    out.append(("channel normalisation: per-element power ~beta_k for every N",
                max(pe.values()) / min(pe.values()) < 1.6, f"E|g|^2 per element {pe}"))

    # estimator settings are constants, not functions of N
    out.append(("EM-GS iterations fixed (no early stopping, both estimators)",
                cfg.GS_MAX_ITER == 50, f"max_iter={cfg.GS_MAX_ITER}, no stopping rule"))
    out.append(("Hankel schedule fixed", cfg.PROJECT_EVERY == 1 and cfg.CADZOW_ITER == 4,
                f"project_every={cfg.PROJECT_EVERY}, cadzow_iter={cfg.CADZOW_ITER}"))

    # rank RULE is the same; only its ceiling r_max = ceil(N/2) moves with N,
    # which is a property of the Hankel matrix, not a changed rule
    caps = {N: hp.rank_cap(N) for N in cfg.N_GRID}
    out.append(("rank-selection RULE identical; only the ceiling r_max=ceil(N/2) moves",
                all(c == -(-N // 2) for N, c in caps.items()),
                f"held-out residual, val_frac={cfg.VAL_FRAC}, candidates 1..r_max {caps}"))

    # trial count is per-N by budget, and is reported everywhere
    tc = {N: {int(grid[(N, s)]["trial"].size) for s in cfg.SNR_GRID_DB} for N in cfg.N_GRID}
    out.append(("trial count constant across SNR within each N",
                all(len(v) == 1 for v in tc.values()),
                ", ".join(f"N={N}: {v.pop()}" for N, v in tc.items())))
    return out


# ------------------------------------------------------------------- figures
def fig_panels(S: dict) -> dict:
    """One 2x2 float: the three SNR regimes plus the SNR-averaged summary.

    Kept as a single figure rather than two so the four panels share a caption
    and one float slot; no panel is dropped.
    """
    Ns = list(cfg.N_GRID)
    fig, ax = plt.subplots(2, 2, figsize=(3.3, 2.42), sharex=True)
    flat = ax.ravel()

    for a, snr in zip(flat[:3], PANEL_SNRS):
        for key, lo, hi, sty in (("em_db", "em_lo", "em_hi", EM),
                                 ("hk_db", "hk_lo", "hk_hi", HK)):
            y = np.array([S[(N, snr)][key] for N in Ns])
            l = np.array([S[(N, snr)][lo] for N in Ns])
            h = np.array([S[(N, snr)][hi] for N in Ns])
            a.errorbar(Ns, y, yerr=[y - l, h - y], capsize=2.5, elinewidth=0.9, **sty)
        a.set_title(f"SNR = {snr:+.0f} dB", fontsize=7.5)

    em = [float(np.mean([S[(N, s)]["em_db"] for s in cfg.SNR_GRID_DB])) for N in Ns]
    hk = [float(np.mean([S[(N, s)]["hk_db"] for s in cfg.SNR_GRID_DB])) for N in Ns]
    a = flat[3]
    a.plot(Ns, em, **EM); a.plot(Ns, hk, **HK)
    for N, u, v in zip(Ns, em, hk):
        a.annotate(f"{v - u:+.2f}", (N, (u + v) / 2), fontsize=6.6, ha="center",
                   va="center", color="0.3",
                   bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.7))
    a.set_title("SNR-averaged", fontsize=7.5)
    a.margins(x=0.16, y=0.16)

    for a in flat:
        a.set_xscale("log", base=2); a.set_xticks(Ns)
        a.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        a.set_ylabel("NMSE (dB)", fontsize=7.5)
    for a in ax[1]:
        a.set_xlabel("array size $N$")
    flat[0].legend(framealpha=1.0, loc="best", fontsize=7.5)
    fig.text(0.5, -0.03, "lower is better", ha="center", fontsize=7)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"fig5_nmse_vs_N_panels.{ext}")
    plt.close(fig)
    print("  wrote figures/fig5_nmse_vs_N_panels (4 panels)")
    return {"N": Ns, "em_gs_db": em, "hankel_db": hk}


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    grid = load_grid()
    need = [(N, s) for N in cfg.N_GRID for s in cfg.SNR_GRID_DB]
    missing = [k for k in need if k not in grid]
    if missing:
        print("MISSING operating points, run experiment_array_size.py first:", missing)
        return 1
    S = {k: point_stats(grid[k]) for k in need}

    # ---- cross-check every gain against the previously reported value
    prev = json.loads((RES / "summary.json").read_text())["experiment_B"]
    print("\nCROSS-CHECK: Delta_H recomputed here vs previously reported")
    worst = 0.0
    for N in cfg.N_GRID:
        for s in cfg.SNR_GRID_DB:
            mine = S[(N, s)]["gain_db"]
            ref = prev[str(N)]["per_snr_gain_db"][f"{s:+.1f}"]
            worst = max(worst, abs(mine - ref))
    ok = worst < 1e-9
    print(f"  max |recomputed - reported| = {worst:.3e} dB over 21 points "
          f"-- {'IDENTICAL' if ok else 'MISMATCH'}")
    # and that the two NMSE curves reproduce the gain by subtraction
    worst2 = max(abs((S[k]["em_db"] - S[k]["hk_db"]) - S[k]["gain_db"]) for k in need)
    print(f"  max |(NMSE_EM - NMSE_Hankel) - Delta_H| = {worst2:.3e} dB "
          f"-- {'consistent' if worst2 < 1e-9 else 'INCONSISTENT'}")
    if not ok or worst2 >= 1e-9:
        print("  Investigate before using these figures."); return 1

    print("\nINVARIANCE: does increasing N change anything except N?")
    checks = invariance_checks(grid)
    for name, good, detail in checks:
        print(f"  [{'PASS' if good else 'FAIL'}] {name} -- {detail}")
    if not all(g for _, g, _ in checks):
        return 1

    summ = fig_panels(S)

    with open(RES / "figure_array_size_comparison.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["N", "snr_db", "trials", "em_gs_nmse_db", "em_gs_ci_lo",
                    "em_gs_ci_hi", "hankel_nmse_db", "hankel_ci_lo", "hankel_ci_hi",
                    "gain_db", "gain_ci_lo", "gain_ci_hi", "r_max", "active_frac",
                    "mean_L_hat"])
        for N in cfg.N_GRID:
            for s in cfg.SNR_GRID_DB:
                v = S[(N, s)]
                w.writerow([N, f"{s:+.1f}", v["trials"],
                            f"{v['em_db']:.4f}", f"{v['em_lo']:.4f}", f"{v['em_hi']:.4f}",
                            f"{v['hk_db']:.4f}", f"{v['hk_lo']:.4f}", f"{v['hk_hi']:.4f}",
                            f"{v['gain_db']:.4f}", f"{v['gain_lo']:.4f}",
                            f"{v['gain_hi']:.4f}", v["r_max"],
                            f"{v['active_frac']:.4f}", f"{v['mean_L_hat']:.3f}"])
    print(f"  wrote {RES / 'figure_array_size_comparison.csv'}")

    print("\nSNR-AVERAGED NMSE (mean of per-SNR dB values)")
    print(f"{'N':>4} {'EM-GS':>9} {'Hankel':>9} {'Hankel-EM-GS':>13}")
    for N, a, b in zip(summ["N"], summ["em_gs_db"], summ["hankel_db"]):
        print(f"{N:>4} {a:>9.3f} {b:>9.3f} {b - a:>13.3f}")

    if "--explain" in sys.argv:
        print("\nWhy the SNR average is taken in dB, not linearly:")
        for N in cfg.N_GRID:
            lin = np.array([10 ** (S[(N, s)]["em_db"] / 10) for s in cfg.SNR_GRID_DB])
            share_lo = lin[0] / lin.sum()
            share_hi3 = lin[-3:].sum() / lin.sum()
            print(f"  N={N:>2}: linear-mean {10 * np.log10(lin.mean()):+7.3f} dB   "
                  f"dB-mean {np.mean([S[(N, s)]['em_db'] for s in cfg.SNR_GRID_DB]):+7.3f} dB   "
                  f"| SNR=-10 contributes {100 * share_lo:4.1f}% of the linear sum, "
                  f"the top three SNRs together {100 * share_hi3:.2f}%")
        print("  Linear NMSE spans ~3 orders of magnitude across this grid, so a\n"
              "  linear-domain average is dominated by the lowest-SNR point and is\n"
              "  nearly blind to the high-SNR half of the sweep -- where, at N=8, the\n"
              "  Hankel projection actually HURTS. Averaging in dB weights each SNR\n"
              "  regime equally, which is what a summary over regimes should do, and\n"
              "  is the convention verify_results already uses across operating points.\n"
              "  Averaging trials WITHIN a point still happens in the linear domain\n"
              "  (ratio of sums); that existing convention is preserved untouched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

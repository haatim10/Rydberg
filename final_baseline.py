"""
Baseline study, frozen scope.

System model (unchanged, from SystemModel.pdf):

    g[n,k] = c * sum_{l=1}^{L_k} alpha[l,k] * exp(-1j (n-1) psi[l,k])
    Z = |G S + B + W|

Estimators: Cui's published biased GS (Alg. 1) and EM-GS (Alg. 2), unmodified.
Nothing else is run in this study: no Xu GD, no closed-form linearised LS, no
Cadzow, no ESPRIT, no Hankel projection, no learned methods, no CRB of any kind.

For every trial, biased GS and EM-GS are applied to the identical channel,
pilot, reference, and noise realization (same draw of ``simulate()``).

Failure definition (relative, stated explicitly): at a fixed operating point,
a trial is an "outlier" if its NMSE (dB) exceeds the median NMSE (dB) at that
point by more than FAIL_MARGIN_DB = 10 dB. No absolute NMSE threshold is used.

Array size: N=32, matching SystemModel.pdf. N=16 remains available only as a
debugging configuration via --N 16; it is not used for the reported figures.

Reference-gain calibration: the reference gain magnitude is now set
deterministically so every single realization achieves the requested RSR
exactly (previously it was drawn as complex Gaussian with the correct
*expected* power, which let individual realizations miss the target RSR by
several dB). Only the reference phase remains random per realization. Achieved
SNR and RSR are measured directly from each realization's G, S, B, W and
reported alongside the requested values at every operating point.

Outputs
-------
    figures/S1_nmse_vs_snr.png
    figures/S2_nmse_vs_rsr.png
    figures/S3_nmse_vs_pilots.png
    figures/S4_convergence.png
    figures/S5_diagnostic_nmse_vs_minlambda.png   (optional diagnostic)
    results.json   -- all numbers, including achieved SNR/RSR, for the summary
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rydberg_ce.system import SystemConfig, simulate
from rydberg_ce.algorithms import biased_gs, em_gs

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 200, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.3, "legend.framealpha": 0.95,
    "axes.spines.top": False, "axes.spines.right": False,
})

GS_LABEL = "biased GS (Cui Alg. 1)"
EM_LABEL = "EM-GS (Cui Alg. 2)"
FAIL_MARGIN_DB = 10.0   # explicit, relative outlier definition (see module docstring)
N_ITER = 100            # default iteration budget for the three sweeps

# Fixed defaults, stated once here and reused verbatim in every sweep.
# N=32 matches SystemModel.pdf and is used for all final figures.
# N=16 remains available only as a debugging configuration (pass --N 16).
N_ELEM = 32
K_USERS = 3
P_DEFAULT = 20
SNR_DEFAULT = 5.0
RSR_DEFAULT = 12.0


def db(x):
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=float), 1e-30))


# ---------------------------------------------------------------------------
# Core: one operating point
# ---------------------------------------------------------------------------

def run_point(cfg: SystemConfig, trials: int, rng: np.random.Generator,
              n_iter: int = N_ITER):
    """Run both estimators on `trials` realizations of one operating point.

    Both estimators see the same realization (same G, S, B, W) on every trial.
    Returns per-trial NMSE arrays, the per-trial min|lambda| diagnostic, and the
    per-trial signal/reference/noise powers used to verify achieved SNR and RSR.
    """
    nmse_gs = np.empty(trials)
    nmse_em = np.empty(trials)
    min_lambda = np.empty(trials)
    sig_pow = np.empty(trials)     # mean|(GS)_{n,p}|^2
    ref_pow = np.empty(trials)     # mean|B_{n,p}|^2
    gk_pow = np.empty(trials)      # mean|g_{n,k}|^2  (RSR denominator)
    noise_pow = np.empty(trials)   # mean|W_{n,p}|^2

    for t in range(trials):
        r = simulate(cfg, rng)                       # one draw, shared by both algos
        G_gs = biased_gs(r.Z, r.S, r.B, r.sigma2, n_iter=n_iter)
        G_em = em_gs(r.Z, r.S, r.B, r.sigma2, n_iter=n_iter)
        nmse_gs[t] = r.nmse(G_gs)
        nmse_em[t] = r.nmse(G_em)
        min_lambda[t] = np.abs(r.G @ r.S + r.B).min()
        sig_pow[t] = np.mean(np.abs(r.G @ r.S) ** 2)
        ref_pow[t] = np.mean(np.abs(r.B) ** 2)
        gk_pow[t] = np.mean(np.abs(r.G) ** 2)
        noise_pow[t] = np.mean(np.abs(r.W) ** 2)

    return dict(nmse_gs=nmse_gs, nmse_em=nmse_em, min_lambda=min_lambda,
               sig_pow=sig_pow, ref_pow=ref_pow, gk_pow=gk_pow, noise_pow=noise_pow)


def summarize(run):
    """Mean/median NMSE, relative failure rate, and achieved SNR/RSR at one point."""
    def stats(v):
        dbv = db(v)
        fail = float(np.mean(dbv > np.median(dbv) + FAIL_MARGIN_DB))
        return dict(mean_db=float(db(np.mean(v))), median_db=float(np.median(dbv)),
                   fail_rate=fail)

    achieved_snr_db = float(db(np.mean(run["sig_pow"]) / np.mean(run["noise_pow"])))
    achieved_rsr_db = float(db(np.mean(run["ref_pow"]) / np.mean(run["gk_pow"])))

    return {"gs": stats(run["nmse_gs"]), "em": stats(run["nmse_em"]),
           "achieved_snr_db": achieved_snr_db, "achieved_rsr_db": achieved_rsr_db}


def sweep(label, cfg_fn, xs, trials, seed, n_iter=N_ITER):
    rng = np.random.default_rng(seed)
    rows, t0 = [], time.time()
    for i, x in enumerate(xs):
        run = run_point(cfg_fn(x), trials, rng, n_iter)
        rows.append(summarize(run))
        print(f"  [{label}] {i+1}/{len(xs)}  x={x}   ({time.time()-t0:.0f}s)", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sweep(xs, rows, xlabel, title, path):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for key, label, color, marker, ls in [
        ("gs", GS_LABEL + " (mean)", "#888888", "v", "--"),
        ("em", EM_LABEL + " (mean)", "#1f4788", "o", "-"),
    ]:
        ax.plot(xs, [r[key]["mean_db"] for r in rows], label=label,
                color=color, marker=marker, ls=ls, lw=2.0, ms=5)
    for key, label, color, marker in [
        ("gs", GS_LABEL + " (median)", "#bbbbbb", "v"),
        ("em", EM_LABEL + " (median)", "#6c9bd2", "o"),
    ]:
        ax.plot(xs, [r[key]["median_db"] for r in rows], label=label,
                color=color, marker=marker, ls=":", lw=1.3, ms=3.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("NMSE (dB)")
    ax.set_title(title, fontsize=10.5)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def print_table(xs, rows, xlabel, requested_snr=None, requested_rsr=None):
    print(f"\n    {xlabel:>10} | {'GS mean':>9} {'EM mean':>9} | "
          f"{'GS med':>9} {'EM med':>9} | {'GS fail%':>9} {'EM fail%':>9} | "
          f"{'req SNR':>8} {'ach SNR':>8} | {'req RSR':>8} {'ach RSR':>8}")
    print("    " + "-" * 118)
    for x, r in zip(xs, rows):
        req_snr = requested_snr(x) if callable(requested_snr) else requested_snr
        req_rsr = requested_rsr(x) if callable(requested_rsr) else requested_rsr
        print(f"    {x:>10} | {r['gs']['mean_db']:9.2f} {r['em']['mean_db']:9.2f} | "
              f"{r['gs']['median_db']:9.2f} {r['em']['median_db']:9.2f} | "
              f"{100*r['gs']['fail_rate']:9.1f} {100*r['em']['fail_rate']:9.1f} | "
              f"{req_snr:8.2f} {r['achieved_snr_db']:8.2f} | "
              f"{req_rsr:8.2f} {r['achieved_rsr_db']:8.2f}")


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def do_snr_sweep(trials, out):
    print(f"\n[S1] NMSE vs SNR   (N={N_ELEM}, K={K_USERS}, P={P_DEFAULT}, "
          f"RSR={RSR_DEFAULT} dB)")
    xs = list(range(-5, 21, 5))
    rows = sweep("SNR", lambda s: SystemConfig(N=N_ELEM, K=K_USERS, P=P_DEFAULT,
                                               snr_db=s, rsr_db=RSR_DEFAULT),
                xs, trials, seed=101)
    plot_sweep(xs, rows, "SNR (dB)",
              f"NMSE vs SNR  (N={N_ELEM}, K={K_USERS}, P={P_DEFAULT}, RSR={RSR_DEFAULT} dB)",
              os.path.join(out, "S1_nmse_vs_snr.png"))
    print_table(xs, rows, "SNR (dB)", requested_snr=lambda x: x, requested_rsr=RSR_DEFAULT)
    return {"xlabel": "SNR (dB)", "xs": xs, "rows": rows,
           "fixed": dict(N=N_ELEM, K=K_USERS, P=P_DEFAULT, rsr_db=RSR_DEFAULT)}


def do_rsr_sweep(trials, out):
    print(f"\n[S2] NMSE vs RSR   (N={N_ELEM}, K={K_USERS}, P={P_DEFAULT}, "
          f"SNR={SNR_DEFAULT} dB)")
    xs = list(range(0, 26, 5))
    rows = sweep("RSR", lambda rr: SystemConfig(N=N_ELEM, K=K_USERS, P=P_DEFAULT,
                                                snr_db=SNR_DEFAULT, rsr_db=rr),
                xs, trials, seed=202)
    plot_sweep(xs, rows, "RSR (dB)",
              f"NMSE vs RSR  (N={N_ELEM}, K={K_USERS}, P={P_DEFAULT}, SNR={SNR_DEFAULT} dB)",
              os.path.join(out, "S2_nmse_vs_rsr.png"))
    print_table(xs, rows, "RSR (dB)", requested_snr=SNR_DEFAULT, requested_rsr=lambda x: x)
    return {"xlabel": "RSR (dB)", "xs": xs, "rows": rows,
           "fixed": dict(N=N_ELEM, K=K_USERS, P=P_DEFAULT, snr_db=SNR_DEFAULT)}


def do_pilot_sweep(trials, out):
    print(f"\n[S3] NMSE vs pilot length P   (N={N_ELEM}, K={K_USERS}, "
          f"SNR={SNR_DEFAULT} dB, RSR={RSR_DEFAULT} dB)")
    xs = [6, 8, 10, 14, 20, 30, 40]
    rows = sweep("P", lambda p: SystemConfig(N=N_ELEM, K=K_USERS, P=int(p),
                                             snr_db=SNR_DEFAULT, rsr_db=RSR_DEFAULT),
                xs, trials, seed=303)
    plot_sweep(xs, rows, "pilot length P",
              f"NMSE vs pilot length  (N={N_ELEM}, K={K_USERS}, SNR={SNR_DEFAULT} dB, "
              f"RSR={RSR_DEFAULT} dB)",
              os.path.join(out, "S3_nmse_vs_pilots.png"))
    print_table(xs, rows, "P", requested_snr=SNR_DEFAULT, requested_rsr=RSR_DEFAULT)
    return {"xlabel": "P", "xs": xs, "rows": rows,
           "fixed": dict(N=N_ELEM, K=K_USERS, snr_db=SNR_DEFAULT, rsr_db=RSR_DEFAULT)}


def do_convergence(trials, out, P=P_DEFAULT, snr_db=10.0, rsr_db=RSR_DEFAULT,
                   n_iter=80):
    print(f"\n[S4] Convergence   (N={N_ELEM}, K={K_USERS}, P={P}, SNR={snr_db} dB, "
          f"RSR={rsr_db} dB, {trials} trials)")
    cfg = SystemConfig(N=N_ELEM, K=K_USERS, P=P, snr_db=snr_db, rsr_db=rsr_db)
    rng = np.random.default_rng(404)

    traj_gs = np.empty((trials, n_iter))
    traj_em = np.empty((trials, n_iter))
    sig_pow = np.empty(trials)
    ref_pow = np.empty(trials)
    gk_pow = np.empty(trials)
    noise_pow = np.empty(trials)
    t0 = time.time()
    for t in range(trials):
        r = simulate(cfg, rng)
        sig_pow[t] = np.mean(np.abs(r.G @ r.S) ** 2)
        ref_pow[t] = np.mean(np.abs(r.B) ** 2)
        gk_pow[t] = np.mean(np.abs(r.G) ** 2)
        noise_pow[t] = np.mean(np.abs(r.W) ** 2)
        G_gs = G_em = None
        for it in range(n_iter):
            G_gs = biased_gs(r.Z, r.S, r.B, r.sigma2, n_iter=1, G0=G_gs, tol=0)
            G_em = em_gs(r.Z, r.S, r.B, r.sigma2, n_iter=1, G0=G_em, tol=0)
            traj_gs[t, it] = r.nmse(G_gs)
            traj_em[t, it] = r.nmse(G_em)
        if (t + 1) % 20 == 0:
            print(f"  [convergence] {t+1}/{trials}  ({time.time()-t0:.0f}s)", flush=True)

    achieved_snr_db = float(db(np.mean(sig_pow) / np.mean(noise_pow)))
    achieved_rsr_db = float(db(np.mean(ref_pow) / np.mean(gk_pow)))
    print(f"    requested SNR={snr_db:.2f} dB, achieved={achieved_snr_db:.2f} dB   |   "
          f"requested RSR={rsr_db:.2f} dB, achieved={achieved_rsr_db:.2f} dB")

    mean_gs = db(traj_gs.mean(axis=0))
    mean_em = db(traj_em.mean(axis=0))
    med_gs = np.median(db(traj_gs), axis=0)
    med_em = np.median(db(traj_em), axis=0)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    its = np.arange(1, n_iter + 1)
    ax.plot(its, mean_gs, label=GS_LABEL + " (mean)", color="#888888", ls="--", lw=1.8)
    ax.plot(its, mean_em, label=EM_LABEL + " (mean)", color="#1f4788", ls="-", lw=2.0)
    ax.plot(its, med_gs, label=GS_LABEL + " (median)", color="#bbbbbb", ls=":", lw=1.2)
    ax.plot(its, med_em, label=EM_LABEL + " (median)", color="#6c9bd2", ls=":", lw=1.4)
    ax.set_xlabel("iteration")
    ax.set_ylabel("NMSE (dB)")
    ax.set_title(f"Convergence  (N={N_ELEM}, K={K_USERS}, P={P}, SNR={snr_db} dB, "
                 f"RSR={rsr_db} dB)", fontsize=10.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out, "S4_convergence.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")

    return {"P": P, "snr_db": snr_db, "rsr_db": rsr_db, "n_iter": n_iter,
           "trials": trials, "achieved_snr_db": achieved_snr_db,
           "achieved_rsr_db": achieved_rsr_db,
           "final_mean_gs_db": float(mean_gs[-1]), "final_mean_em_db": float(mean_em[-1]),
           "final_median_gs_db": float(med_gs[-1]), "final_median_em_db": float(med_em[-1])}


def do_diagnostic(trials, out, P=30, snr_db=20.0, rsr_db=RSR_DEFAULT):
    """Optional diagnostic: trial NMSE vs min|GS+B|, one representative setting.

    Reported strictly as an observed association (see module docstring / README):
    no causal or mechanistic claim is made here.
    """
    print(f"\n[S5, diagnostic] NMSE vs min|GS+B|   (N={N_ELEM}, K={K_USERS}, P={P}, "
          f"SNR={snr_db} dB, RSR={rsr_db} dB, {trials} trials)")
    cfg = SystemConfig(N=N_ELEM, K=K_USERS, P=P, snr_db=snr_db, rsr_db=rsr_db)
    rng = np.random.default_rng(505)
    nmse_gs, nmse_em, min_lambda = run_point(cfg, trials, rng)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.scatter(min_lambda, db(nmse_gs), s=10, alpha=0.5, color="#888888",
              label=GS_LABEL, marker="v")
    ax.scatter(min_lambda, db(nmse_em), s=10, alpha=0.5, color="#1f4788",
              label=EM_LABEL, marker="o")
    ax.set_xlabel(r"$\min_{n,p} |\lambda_{n,p}|$,  $\lambda = GS + B$")
    ax.set_ylabel("trial NMSE (dB)")
    ax.set_title(f"Trial NMSE vs min|GS+B|  (N={N_ELEM}, K={K_USERS}, P={P}, "
                 f"SNR={snr_db} dB, RSR={rsr_db} dB)\n"
                 "diagnostic only -- association, not a proven mechanism",
                 fontsize=9.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out, "S5_diagnostic_nmse_vs_minlambda.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")

    corr_gs = float(np.corrcoef(min_lambda, db(nmse_gs))[0, 1])
    corr_em = float(np.corrcoef(min_lambda, db(nmse_em))[0, 1])

    def spearman(a, b):
        ra = np.argsort(np.argsort(a))
        rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

    rho_gs = spearman(min_lambda, db(nmse_gs))
    rho_em = spearman(min_lambda, db(nmse_em))

    # Decile comparison: is the outlier rate concentrated in the smallest-min|lambda| decile?
    order = np.argsort(min_lambda)
    decile = max(1, trials // 10)
    low_idx, rest_idx = order[:decile], order[decile:]
    med_em_db = np.median(db(nmse_em))
    fail_low = float(np.mean(db(nmse_em[low_idx]) > med_em_db + FAIL_MARGIN_DB))
    fail_rest = float(np.mean(db(nmse_em[rest_idx]) > med_em_db + FAIL_MARGIN_DB))

    print(f"    Pearson correlation (min|lambda|, NMSE dB):  GS={corr_gs:+.3f}  "
          f"EM-GS={corr_em:+.3f}")
    print(f"    Spearman rank correlation:                    GS={rho_gs:+.3f}  "
          f"EM-GS={rho_em:+.3f}")
    print(f"    EM-GS outlier rate, smallest-decile min|lambda| trials: {100*fail_low:5.1f}%")
    print(f"    EM-GS outlier rate, remaining trials:                   {100*fail_rest:5.1f}%")
    print("    (association only; no causal or mechanistic claim is made)")

    return {"P": P, "snr_db": snr_db, "rsr_db": rsr_db, "trials": trials,
           "pearson_gs": corr_gs, "pearson_em": corr_em,
           "spearman_gs": rho_gs, "spearman_em": rho_em,
           "fail_rate_smallest_decile": fail_low, "fail_rate_remaining": fail_rest}


def do_failure_vs_rsr(trials, out, P=30, snr_db=20.0,
                      rsr_list=(0, 6, 12, 18, 24, 30)):
    """Failure rate vs RSR at the diagnostic setting, reported as a correlation only."""
    print(f"\n[S5b] Failure rate vs RSR at the diagnostic setting  "
          f"(N={N_ELEM}, K={K_USERS}, P={P}, SNR={snr_db} dB)")
    print(f"\n    {'RSR (dB)':>9} | {'EM mean':>9} {'EM median':>10} | {'EM fail%':>9}")
    print("    " + "-" * 48)
    rows = []
    for rsr in rsr_list:
        cfg = SystemConfig(N=N_ELEM, K=K_USERS, P=P, snr_db=snr_db, rsr_db=rsr)
        rng = np.random.default_rng(606 + int(rsr))
        _, nmse_em, _ = run_point(cfg, trials, rng)
        s = summarize(nmse_em, nmse_em)["em"]
        rows.append({"rsr_db": rsr, **s})
        print(f"    {rsr:>9} | {s['mean_db']:9.2f} {s['median_db']:10.2f} | "
              f"{100*s['fail_rate']:9.1f}")
    return rows


# ---------------------------------------------------------------------------

def main():
    global N_ELEM
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=300,
                    help="trials per operating point for the three main sweeps")
    ap.add_argument("--conv_trials", type=int, default=100)
    ap.add_argument("--diag_trials", type=int, default=500)
    ap.add_argument("--out", type=str, default="figures")
    ap.add_argument("--only", type=str, default="all",
                    help="all | snr | rsr | pilots | conv | diag | failrsr")
    ap.add_argument("--N", type=int, default=N_ELEM,
                    help="number of receive elements; 32 (default) matches "
                         "SystemModel.pdf, 16 is a debugging configuration only")
    args = ap.parse_args()
    N_ELEM = args.N
    os.makedirs(args.out, exist_ok=True)

    print("=" * 78)
    print("Baseline study: Cui biased GS / EM-GS on the frozen ULA system model")
    print(f"N={N_ELEM}  K={K_USERS}  algorithms unmodified; identical realizations "
          f"per trial for both estimators")
    print("=" * 78)

    results = {"config": dict(N=N_ELEM, K=K_USERS, P_default=P_DEFAULT,
                             snr_default=SNR_DEFAULT, rsr_default=RSR_DEFAULT,
                             n_iter=N_ITER, fail_margin_db=FAIL_MARGIN_DB,
                             trials=args.trials,
                             note="N=32 matches SystemModel.pdf; N=16 is debug-only")}

    results_path = os.path.join(args.out, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results.update(json.load(f))

    if args.only in ("all", "snr"):
        results["snr_sweep"] = do_snr_sweep(args.trials, args.out)
    if args.only in ("all", "rsr"):
        results["rsr_sweep"] = do_rsr_sweep(args.trials, args.out)
    if args.only in ("all", "pilots"):
        results["pilot_sweep"] = do_pilot_sweep(args.trials, args.out)
    if args.only in ("all", "conv"):
        results["convergence"] = do_convergence(args.conv_trials, args.out)
    if args.only in ("all", "diag"):
        results["diagnostic"] = do_diagnostic(args.diag_trials, args.out)
    if args.only in ("all", "failrsr"):
        results["failure_vs_rsr"] = do_failure_vs_rsr(args.trials, args.out)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nresults written to {results_path}")
    print("done")


if __name__ == "__main__":
    main()

"""Independently reload the raw per-trial stores and recompute every headline
number, plus a suite of sanity checks on the implementation.

Reads ONLY results/**/*.npz. Nothing is hard-coded; plot_results.py consumes
the JSON/CSV this script writes, so figures and text cannot drift apart.

    python verify_results.py            # tables + checks
    python verify_results.py --quick    # skip the slow implementation checks
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

import config as cfg

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FAILS: list[str] = []
CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    if not ok:
        FAILS.append(f"{name}: {detail}")


# ---------------------------------------------------------------- statistics
def pooled_db(num: np.ndarray, den: np.ndarray) -> float:
    """NMSE in dB, ratio of sums: 10 log10( sum||Ghat-G||^2 / sum||G||^2 ).

    Estimates the ensemble NMSE. Factor 10 (not 20) because both arguments are
    already energies. Never a mean of per-trial decibels.
    """
    return float(10 * np.log10(num.sum() / den.sum()))


def boot_ci(a: np.ndarray, b: np.ndarray, *, nboot=cfg.NBOOT, seed=cfg.BOOT_SEED):
    """95% PAIRED bootstrap CI and SD on the gain 10log10(sum a / sum b).

    The resample index vector is drawn once and applied to both estimators, so
    the shared channel realisation cancels and the interval on the difference
    is far tighter than the marginals would suggest.

    Returns ``(lo, hi, sd)``. ``sd`` is the standard deviation of the
    bootstrap distribution of the POOLED ratio-of-sums gain -- i.e. the
    standard error of the statistic actually reported. It is NOT the standard
    error of the mean of per-trial decibel gains, which estimates a different
    quantity (a mean of ratios, not a ratio of sums) and is smaller by
    Jensen; reporting that as the SE of the pooled gain was an audit finding.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(nboot, a.size))
    g = 10 * np.log10(a[idx].sum(1) / b[idx].sum(1))
    return (float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5)),
            float(np.std(g, ddof=1)))


def summarise(d: dict) -> dict:
    """All statistics for one operating point, from its raw per-trial arrays."""
    e, h, den = d["num_em_gs"], d["num_hankel_em_gs"], d["denom"]
    r_e, r_h = e / den, h / den                       # per-trial NMSE (linear)
    lo, hi, boot_sd = boot_ci(e, h)
    n = e.size
    per_trial_gain_db = 10 * np.log10(e / h)
    return dict(
        trials=int(n),
        em_gs_db=pooled_db(e, den), hankel_db=pooled_db(h, den),
        em_gs_median_db=float(10 * np.log10(np.median(r_e))),
        hankel_median_db=float(10 * np.log10(np.median(r_h))),
        em_gs_sd_db=float(np.std(10 * np.log10(r_e), ddof=1)),
        hankel_sd_db=float(np.std(10 * np.log10(r_h), ddof=1)),
        gain_db=float(10 * np.log10(e.sum() / h.sum())),
        gain_ci_lo=lo, gain_ci_hi=hi,
        # SE of the POOLED ratio-of-sums gain, from the paired bootstrap.
        gain_boot_sd_db=boot_sd,
        # SE of the mean of per-trial dB gains. A DIFFERENT estimand (mean of
        # ratios, not ratio of sums) -- kept for reference under a name that
        # says what it is, never as the SE of the pooled gain.
        per_trial_gain_mean_se_db=float(np.std(per_trial_gain_db, ddof=1) / np.sqrt(n)),
        gain_median_db=float(np.median(per_trial_gain_db)),
        win_rate=float((h < e).mean()),
        tie_rate=float((h == e).mean()),
        active_frac=float(d["active"].mean()),
        mean_L_hat=float(d["L_hat"].mean()),
        r_max=int(d["r_max"][0]),
        mean_L_true_sum=float(d["L_true_sum"].mean()),
    )


def load(path: str) -> dict:
    d = np.load(path)
    out = {k: d[k] for k in d.files if k != "fingerprint"}
    if out["trial"].size == 0:
        raise ValueError(f"{path}: store holds no completed trials")
    return out


# ------------------------------------------------------------------ sweeps
def grid_points() -> dict:
    out = {}
    for f in sorted(glob.glob(str(RES / "grid" / "*.npz"))):
        stem = Path(f).stem                      # N08_P30_snr+05.0
        if stem.endswith(".tmp"):                # a flush in progress
            continue
        N = int(stem[1:3]); P = int(stem.split("_P")[1].split("_")[0])
        snr = float(stem.split("snr")[1])
        out[(N, P, snr)] = load(f)
    return out


def path_points() -> dict:
    out = {}
    for f in sorted(glob.glob(str(RES / "pathcount" / "L*.npz"))):
        stem = Path(f).stem
        if stem.endswith(".tmp"):
            continue
        out[int(stem[1:])] = load(f)
    return out


# ------------------------------------------------------- implementation checks
def implementation_checks(quick: bool = False) -> None:
    import em_gs
    import hankel_em_gs as hem
    import hankel_projection as hp
    from system_model import channel_nmse_parts, make_world

    # 1. baseline and Hankel share the SAME measurement update, exactly
    w = make_world(0, N=16, P=30, snr_db=5.0)
    base = em_gs.em_gs(w.S, w.Z, w.B, w.sigma2)
    off = hem.hankel_em_gs(w.S, w.Z, w.B, w.sigma2,
                           L_hat=hp.rank_cap(16)).G_hat
    check("1 projection-off == EM-GS bit-for-bit", np.array_equal(base, off),
          f"max|diff|={np.abs(base - off).max():.3e}")

    # 2. paired inputs: one frozen world object, and it is immutable
    names = ("G", "S", "B", "W", "Z")
    readonly = {nm: (not getattr(w, nm).flags.writeable) for nm in names}
    check("2 world arrays are read-only (pairing cannot be broken)",
          all(readonly.values()),
          "read-only: " + ", ".join(f"{k}={v}" for k, v in readonly.items()))

    # 3. NMSE formula matches ||Ghat-G||_F^2 / ||G||_F^2
    num, den = channel_nmse_parts(base, w.G)
    ref = float(np.linalg.norm(base - w.G, "fro") ** 2)
    refd = float(np.linalg.norm(w.G, "fro") ** 2)
    check("3 NMSE numerator/denominator formula",
          abs(num - ref) < 1e-9 and abs(den - refd) < 1e-9,
          f"num diff {abs(num - ref):.2e}, den diff {abs(den - refd):.2e}")

    # 4. anti-diagonal averaging is exact on an already-Hankel matrix
    g = np.arange(16) + 1j * np.arange(16)[::-1]
    check("4 unlift(lift(g)) == g (Hankel projector is exact)",
          np.allclose(hp.unlift(hp.lift(g)), g, atol=1e-12),
          f"max|diff|={np.abs(hp.unlift(hp.lift(g)) - g).max():.3e}")

    # 5. rank selection uses NO ground truth
    src = (HERE.parent / "rydberg_sim" / "track_b_proposed.py").read_text()
    body = src[src.index("def select_order_heldout"):src.index("def hs_gs(")]
    check("5 rank selection is not an oracle",
          all(t not in body for t in ("L_k", "world.G", "G_true", ".L_true")),
          "no ground-truth symbol appears in select_order_heldout")

    # 6. noiseless L-path channel has Hankel rank exactly L
    rng = np.random.default_rng(12345)
    ok6 = []
    for L in (2, 3):
        psi = rng.uniform(-np.pi, np.pi, L)
        a = rng.normal(size=L) + 1j * rng.normal(size=L)
        gg = (a[None, :] * np.exp(-1j * np.arange(32)[:, None] * psi[None, :])).sum(1)
        s = hp.singular_values(gg); s = s / s[0]
        ok6.append((int((s > 1e-10).sum()) == L, L, float(s[L])))
    check("6 noiseless L-path Hankel has numerical rank L",
          all(t[0] for t in ok6),
          "; ".join(f"L={L}: sv[L]/sv[0]={v:.2e}" for _, L, v in ok6))

    # 7. truncated SVD really produces the requested rank
    H = hp.lift(np.asarray(base[:, 0]))
    T = hp.truncate_rank(H, 3)
    sv = np.linalg.svd(T, compute_uv=False)
    check("7 truncate_rank(H,3) has numerical rank 3",
          int((sv > sv[0] * 1e-10).sum()) == 3, f"sv[3]/sv[0]={sv[3] / sv[0]:.2e}")

    # 8. Hankel dimensions and rank ceiling
    dims = {N: (hp.lift(np.zeros(N)).shape, hp.rank_cap(N)) for N in (8, 16, 32)}
    check("8 Hankel dims and ceiling r_max=ceil(N/2)",
          all(cap == -(-N // 2) and min(sh) == cap for N, (sh, cap) in dims.items()),
          "; ".join(f"N={N}: {sh} cap={c}" for N, (sh, c) in dims.items()))

    # 9. valid target rank in every stored trial
    bad = 0
    for d in list(grid_points().values()) + list(path_points().values()):
        bad += int(((d["L_hat"] < 1) | (d["L_hat"] > d["r_max"])).sum())
    check("9 every stored L_hat in [1, r_max]", bad == 0, f"{bad} violations")

    # 10. no NaN / Inf anywhere in the stores
    nn = 0
    for d in list(grid_points().values()) + list(path_points().values()):
        for k in ("num_em_gs", "num_hankel_em_gs", "denom"):
            nn += int((~np.isfinite(d[k])).sum())
    check("10 no NaN/Inf in any stored result", nn == 0, f"{nn} non-finite values")

    # 11. per-trial pairing guard recorded during the run
    tot = okp = 0
    for d in list(grid_points().values()) + list(path_points().values()):
        tot += d["paired_ok"].size; okp += int(d["paired_ok"].sum())
    check("11 per-trial pairing guard passed", tot > 0 and okp == tot,
          f"{okp}/{tot} trials confirmed Z==|GS+B+W| and outputs finite")

    if quick:
        return

    # 12. reproducibility: rerunning a trial reproduces it exactly
    w2 = make_world(7, N=16, P=30, snr_db=5.0)
    w3 = make_world(7, N=16, P=30, snr_db=5.0)
    r2 = hem.hankel_em_gs(w2.S, w2.Z, w2.B, w2.sigma2)
    r3 = hem.hankel_em_gs(w3.S, w3.Z, w3.B, w3.sigma2)
    check("12 fixed seed reproduces world and estimate exactly",
          np.array_equal(w2.G, w3.G) and np.array_equal(r2.G_hat, r3.G_hat),
          "same trial index -> identical G, S, B, W and identical output")

    # 13. stored numbers reproduce from a fresh rerun of the same trial
    f0 = sorted(glob.glob(str(RES / "grid" / "N08_P30_snr+05.0.npz")))
    if f0:
        d0 = load(f0[0]); i0 = int(np.argmin(d0["trial"])); t0 = int(d0["trial"][i0])
        ww = make_world(t0, N=8, P=30, snr_db=5.0)
        gb = em_gs.em_gs(ww.S, ww.Z, ww.B, ww.sigma2)
        rr = hem.hankel_em_gs(ww.S, ww.Z, ww.B, ww.sigma2)
        nb, db = channel_nmse_parts(gb, ww.G)
        nh, _ = channel_nmse_parts(rr.G_hat, ww.G)
        check("13 stored trial reproduces on rerun",
              abs(nb - d0["num_em_gs"][i0]) < 1e-9
              and abs(nh - d0["num_hankel_em_gs"][i0]) < 1e-9
              and abs(db - d0["denom"][i0]) < 1e-9,
              f"trial {t0}: recomputed vs stored agree to <1e-9")

    # 14. CHAINED EM-GS == a single max_iter=T call, bit for bit.
    # em_gs.py's docstring asserts this, and the whole fairness argument rests
    # on it, but check 1 compares two CHAINED paths and so never tested it.
    from rydberg_sim.gs import em_gs_channel_rows
    w14 = make_world(0, N=16, P=30, snr_db=5.0)
    worst14, all14 = 0.0, True
    for T in (1, 5, 50):
        single = em_gs_channel_rows(w14.S, w14.Z, w14.B, w14.sigma2,
                                    max_iter=T, ridge=cfg.RIDGE, G0=None).G_hat
        chained = em_gs.em_gs(w14.S, w14.Z, w14.B, w14.sigma2, max_iter=T)
        all14 &= bool(np.array_equal(single, chained))
        worst14 = max(worst14, float(np.abs(single - chained).max()))
    check("14 chained em_gs_step == single max_iter=T call", all14,
          f"T in (1,5,50): max|diff|={worst14:.3e}")

    # 15. Rank selection uses no oracle -- BEHAVIOURALLY, not by grep.
    # Re-run the selector on observables copied into fresh plain arrays that
    # are detached from the world object entirely, so no attribute path to
    # G, L_k or theta survives. A selector with any hidden route to ground
    # truth would have to change its answer.
    w15 = make_world(3, N=16, P=30, snr_db=0.0)
    full = hem.hankel_em_gs(w15.S, w15.Z, w15.B, w15.sigma2)
    S_d = np.array(w15.S, copy=True)
    Z_d = np.array(w15.Z, copy=True)
    B_d = np.array(w15.B, copy=True)
    s2_d = float(w15.sigma2)
    del w15
    detached, _ = hp.select_rank(S_d, Z_d, B_d, s2_d)
    check("15 rank selection reproduces from detached observables only",
          int(detached) == int(full.L_hat),
          f"L_hat detached={int(detached)} vs in-estimator={int(full.L_hat)}")

    # 16. Low Hankel rank is NECESSARY for the geometric model, not sufficient.
    # (a) an L-path unit-modulus channel has Hankel rank <= L; (b) a rank-L
    # Hankel lifting also arises from poles OFF the unit circle, which are not
    # ULA steering responses -- so the constraint is a relaxation of the
    # geometric manifold, not a characterisation of it.
    rng16 = np.random.default_rng(4242)
    nec_ok, suff_witness = True, 0.0
    for L in (2, 3, 4):
        psi16 = rng16.uniform(-np.pi, np.pi, L)
        a16 = rng16.normal(size=L) + 1j * rng16.normal(size=L)
        nn16 = np.arange(32)[:, None]
        g_phys = (a16[None, :] * np.exp(-1j * nn16 * psi16[None, :])).sum(1)
        s_phys = hp.singular_values(g_phys)
        nec_ok &= int((s_phys / s_phys[0] > 1e-10).sum()) <= L
        # same rank, radii != 1 => not a ULA channel
        z_off = np.exp(-1j * psi16) * (1.0 + 0.35 * rng16.random(L))
        g_off = (a16[None, :] * z_off[None, :] ** nn16).sum(1)
        s_off = hp.singular_values(g_off)
        if int((s_off / s_off[0] > 1e-10).sum()) == L:
            suff_witness = max(suff_witness, float(np.abs(np.abs(z_off) - 1).max()))
    check("16 Hankel rank<=L necessary, NOT sufficient (off-circle witness)",
          nec_ok and suff_witness > 0.0,
          f"rank-L witness with max||z|-1| = {suff_witness:.3f}")


# ----------------------------------------------------------------- reporting
def main() -> int:
    quick = "--quick" in sys.argv
    grid, paths = grid_points(), path_points()
    if not grid and not paths:
        print("No results found. Run the experiment scripts first."); return 1

    out: dict = {"fingerprint_config": {
        k: getattr(cfg, k) for k in
        ("MASTER_SEED", "K", "N_DEFAULT", "P_DEFAULT", "L_MIN", "L_MAX",
         "RSR_DB", "GS_MAX_ITER", "CADZOW_ITER", "PROJECT_EVERY",
         "SELECT_ITER", "VAL_FRAC", "N_TRIALS")}}

    # ---- Experiment A: N = 8 slice
    A = {snr: summarise(d) for (N, P, snr), d in sorted(grid.items()) if N == cfg.N_DEFAULT}
    if A:
        print(f"\nEXPERIMENT A -- NMSE vs SNR (N={cfg.N_DEFAULT}, P={cfg.P_DEFAULT}, "
              f"K={cfg.K}, RSR={cfg.RSR_DB} dB, L_k~U{{{cfg.L_MIN}..{cfg.L_MAX}}})")
        print(f"{'SNR':>6} {'trials':>7} {'EM-GS':>9} {'Hankel':>9} {'gain':>8} "
              f"{'95% CI':>18} {'SE':>6} {'win%':>6} {'act%':>6} {'L_hat':>6}")
        for snr, s in A.items():
            print(f"{snr:>6.0f} {s['trials']:>7d} {s['em_gs_db']:>9.3f} "
                  f"{s['hankel_db']:>9.3f} {s['gain_db']:>+8.3f} "
                  f"[{s['gain_ci_lo']:+6.3f},{s['gain_ci_hi']:+6.3f}] "
                  f"{s['gain_boot_sd_db']:>6.3f} {100 * s['win_rate']:>6.1f} "
                  f"{100 * s['active_frac']:>6.1f} {s['mean_L_hat']:>6.2f}")
        out["experiment_A"] = {f"{k:+.1f}": v for k, v in A.items()}

    # ---- Experiment B: gain vs N
    if grid:
        print(f"\nEXPERIMENT B -- gain vs array size (P={cfg.P_DEFAULT}, "
              f"mean and max over the SNR grid)")
        print(f"{'N':>4} {'r_max':>6} {'pts':>4} {'trials':>7} {'EM-GS':>9} "
              f"{'Hankel':>9} {'mean gain':>10} {'max gain':>9} {'at SNR':>7} "
              f"{'win%':>6} {'act%':>6}")
        B = {}
        for N in cfg.N_GRID:
            pts = {snr: summarise(d) for (n_, p_, snr), d in sorted(grid.items()) if n_ == N}
            if not pts:
                continue
            gains = {snr: s["gain_db"] for snr, s in pts.items()}
            snr_max = max(gains, key=gains.get)
            ne = np.concatenate([grid[(N, cfg.P_DEFAULT, s)]["num_em_gs"] for s in pts])
            nh = np.concatenate([grid[(N, cfg.P_DEFAULT, s)]["num_hankel_em_gs"] for s in pts])
            dn = np.concatenate([grid[(N, cfg.P_DEFAULT, s)]["denom"] for s in pts])
            B[N] = dict(
                r_max=pts[list(pts)[0]]["r_max"], points=len(pts),
                trials=int(sum(s["trials"] for s in pts.values())),
                em_gs_db_mean_over_points=float(np.mean([s["em_gs_db"] for s in pts.values()])),
                hankel_db_mean_over_points=float(np.mean([s["hankel_db"] for s in pts.values()])),
                mean_gain_db=float(np.mean(list(gains.values()))),
                max_gain_db=float(gains[snr_max]), max_gain_at_snr=float(snr_max),
                min_gain_db=float(min(gains.values())),
                pooled_gain_db=float(10 * np.log10(ne.sum() / nh.sum())),
                pooled_em_gs_db=pooled_db(ne, dn), pooled_hankel_db=pooled_db(nh, dn),
                win_rate=float(np.mean([s["win_rate"] for s in pts.values()])),
                active_frac=float(np.mean([s["active_frac"] for s in pts.values()])),
                per_snr_gain_db={f"{s:+.1f}": g for s, g in gains.items()})
            b = B[N]
            print(f"{N:>4} {b['r_max']:>6} {b['points']:>4} {b['trials']:>7} "
                  f"{b['em_gs_db_mean_over_points']:>9.3f} "
                  f"{b['hankel_db_mean_over_points']:>9.3f} "
                  f"{b['mean_gain_db']:>+10.3f} {b['max_gain_db']:>+9.3f} "
                  f"{b['max_gain_at_snr']:>7.0f} {100 * b['win_rate']:>6.1f} "
                  f"{100 * b['active_frac']:>6.1f}")
        print("  NOTE: 'mean gain' is the unweighted mean of the per-SNR gains; "
              "'max gain' is the single best operating point. They are different "
              "quantities and are never conflated.")
        out["experiment_B"] = {str(k): v for k, v in B.items()}

    # ---- Experiment C: gain vs true L
    if paths:
        r_max = int(list(paths.values())[0]["r_max"][0])
        print(f"\nEXPERIMENT C -- gain vs TRUE path count (N={cfg.EXP_C_N}, "
              f"r_max={r_max}, P={cfg.P_DEFAULT}, SNR={cfg.EXP_C_SNR} dB, L fixed)")
        print(f"{'L':>4} {'trials':>7} {'EM-GS':>9} {'Hankel':>9} {'gain':>8} "
              f"{'95% CI':>18} {'win%':>6} {'act%':>6} {'E[L_hat]':>9} {'L_hat-L':>8}")
        C = {}
        for L, d in sorted(paths.items()):
            s = summarise(d)
            s["L"] = L
            s["rank_error"] = float(d["L_hat"].mean() - L)
            C[L] = s
            print(f"{L:>4} {s['trials']:>7d} {s['em_gs_db']:>9.3f} "
                  f"{s['hankel_db']:>9.3f} {s['gain_db']:>+8.3f} "
                  f"[{s['gain_ci_lo']:+6.3f},{s['gain_ci_hi']:+6.3f}] "
                  f"{100 * s['win_rate']:>6.1f} {100 * s['active_frac']:>6.1f} "
                  f"{s['mean_L_hat']:>9.2f} {s['rank_error']:>+8.2f}")
        g = [C[L]["gain_db"] for L in sorted(C)]
        mono = all(g[i] >= g[i + 1] - 1e-9 for i in range(len(g) - 1))
        print(f"  gain monotonically non-increasing in L: {mono}")
        out["experiment_C"] = {str(k): v for k, v in C.items()}
        out["experiment_C_monotone"] = bool(mono)

    # ---- sanity checks
    implementation_checks(quick=quick)
    print("\nSANITY CHECKS")
    for name, ok, detail in CHECKS:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    out["checks"] = [{"name": n, "pass": o, "detail": d} for n, o, d in CHECKS]
    out["checks_passed"] = sum(1 for _, o, _ in CHECKS if o)
    out["checks_total"] = len(CHECKS)

    (RES / "summary.json").write_text(json.dumps(out, indent=1))
    _write_csvs(out)
    print(f"\n{out['checks_passed']}/{out['checks_total']} checks passed")
    print(f"wrote {RES / 'summary.json'} and results/*.csv")
    if FAILS:
        print("\nFAILURES:")
        for f in FAILS:
            print("  " + f)
        return 1
    return 0


def _write_csvs(out: dict) -> None:
    import csv
    if "experiment_A" in out:
        with open(RES / "experiment_A_snr.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["snr_db", "trials", "em_gs_db", "hankel_em_gs_db", "gain_db",
                        "gain_ci_lo", "gain_ci_hi", "gain_boot_sd_db", "win_rate",
                        "active_frac", "mean_L_hat"])
            for k, s in out["experiment_A"].items():
                w.writerow([k, s["trials"], f"{s['em_gs_db']:.4f}", f"{s['hankel_db']:.4f}",
                            f"{s['gain_db']:.4f}", f"{s['gain_ci_lo']:.4f}",
                            f"{s['gain_ci_hi']:.4f}", f"{s['gain_boot_sd_db']:.4f}",
                            f"{s['win_rate']:.4f}", f"{s['active_frac']:.4f}",
                            f"{s['mean_L_hat']:.3f}"])
    if "experiment_B" in out:
        with open(RES / "experiment_B_array_size.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["N", "r_max", "points", "trials", "em_gs_db_mean_over_points",
                        "hankel_db_mean_over_points", "mean_gain_db", "max_gain_db",
                        "max_gain_at_snr", "min_gain_db", "pooled_gain_db",
                        "win_rate", "active_frac"])
            for N, b in out["experiment_B"].items():
                w.writerow([N, b["r_max"], b["points"], b["trials"],
                            f"{b['em_gs_db_mean_over_points']:.4f}",
                            f"{b['hankel_db_mean_over_points']:.4f}",
                            f"{b['mean_gain_db']:.4f}", f"{b['max_gain_db']:.4f}",
                            f"{b['max_gain_at_snr']:.1f}", f"{b['min_gain_db']:.4f}",
                            f"{b['pooled_gain_db']:.4f}", f"{b['win_rate']:.4f}",
                            f"{b['active_frac']:.4f}"])
    if "experiment_C" in out:
        with open(RES / "experiment_C_path_count.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["L", "trials", "em_gs_db", "hankel_em_gs_db", "gain_db",
                        "gain_ci_lo", "gain_ci_hi", "win_rate", "active_frac",
                        "mean_L_hat", "rank_error"])
            for L, s in out["experiment_C"].items():
                w.writerow([L, s["trials"], f"{s['em_gs_db']:.4f}", f"{s['hankel_db']:.4f}",
                            f"{s['gain_db']:.4f}", f"{s['gain_ci_lo']:.4f}",
                            f"{s['gain_ci_hi']:.4f}", f"{s['win_rate']:.4f}",
                            f"{s['active_frac']:.4f}", f"{s['mean_L_hat']:.3f}",
                            f"{s['rank_error']:.3f}"])


if __name__ == "__main__":
    sys.exit(main())

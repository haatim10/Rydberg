"""PROMPT 9 Part A -- normalization re-analysis. No new training.

A1  structural compression factor, and whether K cancels
A2  re-index every measured configuration by effective rank, not L
A3  collapse the GS-vs-EM-GS gap onto kappa
A4  data scaling per parameter and per intrinsic dimension

Run:  PYTHONPATH=. python3 scratch/trackD_partA9_normalization.py
"""
from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from rydberg_sim.track_b_proposed import hankel_rank_cap
from rydberg_sim.track_b_structure import hankel_matrix
from trackD_urformer.config import TrackDConfig
from trackD_urformer.dataset import make_world

OUT = Path("reports/trackD_partA9_normalization.json")
TRACKB_C = Path("trackB_hankel_emgs/results/experiment_C_path_count.csv")
SWEEPS = Path("results/track_d/sweeps")


def eff_rank(g: np.ndarray, pencil=None) -> float:
    """Roy-Vetterli effective rank: exp of the spectral entropy of s/sum(s)."""
    s = np.linalg.svd(hankel_matrix(g, pencil), compute_uv=False)
    s = s[s > 0]
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(np.maximum(p, 1e-300)))))


def true_reff(cfg, *, N, L, n_trials=200, seed0=910_000) -> float:
    """Median effective rank of the NOISELESS channel columns."""
    vals = []
    for t in range(n_trials):
        s = make_world(seed0 + t, sysc=cfg.system, N=N, P=cfg.system.P,
                       snr_db=5.0, L=L)
        vals.extend(eff_rank(np.asarray(s.G_true)[:, k]) for k in range(s.K))
    return float(np.median(vals))


def sv_reff(N=32, n_clusters=4, rays=10, mode="clustered", n_cols=400,
            seed=20260903) -> float:
    rng = np.random.default_rng(seed)
    n = np.arange(N)
    vals = []
    for _ in range(n_cols):
        if mode == "literal":
            th = rng.uniform(-np.pi / 2, np.pi / 2, n_clusters * rays)
        else:
            ctr = rng.uniform(-np.pi / 2, np.pi / 2, n_clusters)
            off = np.deg2rad(rng.uniform(-5.0, 5.0, (n_clusters, rays)))
            th = np.clip(ctr[:, None] + off, -np.pi / 2, np.pi / 2).ravel()
        D = th.size
        a = (rng.standard_normal(D) + 1j * rng.standard_normal(D)) / np.sqrt(2 * D)
        vals.append(eff_rank((a[None, :] * np.exp(
            -1j * (np.pi * np.sin(th))[None, :] * n[:, None])).sum(1)))
    return float(np.median(vals))


def median_kappa(cfg, *, snr_db, P, n_trials=200, seed0=920_000) -> float:
    """Median kappa = 2 Z |Y| / sigma^2 at the TRUE channel.

    A property of the operating point, computable with no estimator: it is
    what the EM filter's Bessel ratio actually receives.
    """
    from rydberg_sim.forward import exact_forward
    from rydberg_sim.rng import get_operating_point_rngs

    vals = []
    for t in range(n_trials):
        s = make_world(seed0 + t, sysc=cfg.system, N=cfg.system.N, P=P,
                       snr_db=snr_db)
        rngs = get_operating_point_rngs(cfg.system.master_seed, s.trial,
                                        s.snr_db, s.rsr_db)
        ex = exact_forward(s.G_true, s.S, s.B, s.sigma2, rng_noise=rngs.noise)
        Y = np.asarray(ex.E)
        vals.append(np.median(2.0 * np.asarray(s.Z) * np.abs(Y) / s.sigma2))
    return float(np.median(vals))


def paired_median_db(mode, key_a, key_b):
    """Paired median of (a - b) in dB per sweep point, from stored num/den."""
    out = []
    for f in sorted(SWEEPS.glob(f"{mode}_*.json")):
        r = json.loads(f.read_text())
        if r["n"] < 100:
            continue
        a = np.array(r["num"][key_a]) / np.array(r["den"][key_a])
        b = np.array(r["num"][key_b]) / np.array(r["den"][key_b])
        out.append({"snr_db": r["snr_db"], "P": r["P"],
                    "median_diff_db": float(np.median(10 * np.log10(a)
                                                      - 10 * np.log10(b)))})
    return out


def main() -> int:
    cfg = TrackDConfig()
    res = {}

    # ---------------------------------------------------------------- A1
    print("=== A1: structural compression factor ===")
    N, K = cfg.system.N, cfg.system.K
    a1 = {"source": "SystemModel.pdf, 'Parameter count' paragraph",
          "quote": ("Treated as unstructured, H possesses 2NK real degrees of "
                    "freedom. Under the geometric model it is described by "
                    "3 sum_k L_k real parameters, comprising two per complex "
                    "path gain and one per angle of arrival."),
          "ratio_general": "3 * sum_k L_k / (2 N K)",
          "ratio_equal_L": "3 K Lbar / (2 N K) = 3 Lbar / (2 N)",
          "K_cancels": True}
    # numeric check that K really cancels
    checks = []
    for Kx in (2, 3, 4, 6):
        for Lb in (3, 5, 7):
            gen = 3 * Kx * Lb / (2 * N * Kx)
            simp = 3 * Lb / (2 * N)
            checks.append(abs(gen - simp) < 1e-15)
    a1["numeric_check_all_K"] = bool(all(checks))
    a1["compression_at_default"] = 3 * 5 / (2 * N)     # Lbar = 5
    print(f"  3*Lbar/(2N) at Lbar=5, N=32: {a1['compression_at_default']:.4f}")
    print(f"  K cancels for all tested (K,Lbar): {a1['numeric_check_all_K']}")
    res["A1"] = a1

    # ---------------------------------------------------------------- A2
    print("\n=== A2: re-index by effective rank ===")
    tb = [dict(r) for r in csv.DictReader(open(TRACKB_C))]
    cap32 = hankel_rank_cap(32)
    rows = []
    for r in tb:
        L = int(r["L"])
        re_ = true_reff(cfg, N=32, L=L)
        rows.append({"source": "trackB_expC", "N": 32, "L": L,
                     "r_eff": re_, "r_eff_over_cap": re_ / cap32,
                     "L_over_cap": L / cap32,
                     "delta_hs_db": float(r["gain_db"]),
                     "ci": [float(r["gain_ci_lo"]), float(r["gain_ci_hi"])],
                     "rank_rule": "adaptive (held-out residual)",
                     "snr_db": 5.0})
        print(f"  L={L:2d}  r_eff {re_:5.2f}  r_eff/cap {re_/cap32:5.3f}  "
              f"Delta_HS {float(r['gain_db']):+6.3f}")
    # the B1 configurations, precomputed so Part B can index them directly
    pre = {}
    for Nx, Ls in ((16, (2, 4, 7)), (64, (8, 14, 29))):
        for L in Ls:
            re_ = true_reff(cfg, N=Nx, L=L, n_trials=120)
            pre[f"N{Nx}_L{L}"] = {"N": Nx, "L": L, "cap": hankel_rank_cap(Nx),
                                  "r_eff": re_,
                                  "r_eff_over_cap": re_ / hankel_rank_cap(Nx),
                                  "L_over_cap": L / hankel_rank_cap(Nx)}
            print(f"  [B1 pre] N={Nx:2d} L={L:2d}  r_eff {re_:5.2f}  "
                  f"r_eff/cap {re_/hankel_rank_cap(Nx):5.3f}")
    xiao = {m: sv_reff(mode=m) for m in ("clustered", "literal")}
    for m, v in xiao.items():
        print(f"  [Xiao SV {m:9s}] r_eff {v:5.2f}  r_eff/cap {v/cap32:5.3f}")
    res["A2"] = {"cap_N32": cap32, "trackB_reindexed": rows,
                 "B1_preindex": pre, "xiao_sv_r_eff": xiao,
                 "note": ("r_eff is of the NOISELESS channel. The ESTIMATE's "
                          "r_eff cannot index this plot: PROMPT 8 C1 measured "
                          "it at 8.91-11.77 across L=1..16 at 5 dB, i.e. it is "
                          "set by the noise floor, not by the channel.")}

    # ---------------------------------------------------------------- A3
    print("\n=== A3: GS - EM-GS against kappa ===")
    gs_snr = paired_median_db("snr", "GS", "EM-GS")
    gs_pil = paired_median_db("pilots", "GS", "EM-GS")
    for row in gs_snr:
        row["median_kappa"] = median_kappa(cfg, snr_db=row["snr_db"],
                                           P=row["P"], n_trials=120)
        print(f"  SNR {row['snr_db']:+6.1f}  kappa {row['median_kappa']:9.2f}  "
              f"GS-EMGS {row['median_diff_db']:+7.3f} dB")
    for row in gs_pil:
        row["median_kappa"] = median_kappa(cfg, snr_db=row["snr_db"],
                                           P=row["P"], n_trials=120)
    res["A3"] = {"snr_family": gs_snr, "pilot_family": gs_pil,
                 "kappa_definition": "median over (n,p) of 2 Z |Y| / sigma^2 "
                                     "at the true channel"}

    # ---------------------------------------------------------------- A4
    print("\n=== A4: data scaling per parameter ===")
    params = 1_586_900
    intrinsic = 3 * cfg.system.K * 5          # 3 * sum_k L_k, Lbar = 5
    a4 = []
    for n, nmse in ((20_000, -9.480), (40_000, -10.357), (80_000, -10.831)):
        a4.append({"n_train": n, "test_median_nmse_db": nmse,
                   "samples_per_parameter": n / params,
                   "samples_per_intrinsic_dim": n / intrinsic,
                   "real_measurements_per_parameter":
                       n * 2 * cfg.system.N * cfg.system.P / params})
        print(f"  {n:6d} samples  {n/params:.4f} per param  "
              f"{n/intrinsic:7.1f} per intrinsic dim  "
              f"{n*2*cfg.system.N*cfg.system.P/params:6.2f} real meas/param")
    res["A4"] = {"params": params, "intrinsic_dim_3sumL": intrinsic,
                 "rows": a4}

    # ---------------------------------------------------------------- A5
    res["A5"] = {
        "claim": ("Xiao et al. Fig. 4 is described only as 'evaluated at a "
                  "fixed SNR of 5 dB'. The paper does not state whether the "
                  "networks were retrained per pilot count P."),
        "tag": "[FACT] about the paper's text, not an accusation",
        "why_it_matters": ("Our PROMPT 9 C2/C3 matched-pilot runs separate "
                           "pilot EFFICIENCY (matched training at each P) from "
                           "pilot-count GENERALIZATION (one P=20 model "
                           "evaluated OOD) -- a distinction the paper does not "
                           "draw either way."),
    }

    OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

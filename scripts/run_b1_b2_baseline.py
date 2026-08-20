"""Preliminary Track-B baseline: Cui biased GS and EM-GS on the ULA channel.

Only the two exact-model baselines are run. No structural projection.
"""
import json
from pathlib import Path

import numpy as np

from rydberg_sim.gs import biased_gs_channel_rows, em_gs_channel_rows
from rydberg_sim.metrics import channel_nmse
from rydberg_sim.track_b_drivers import (
    B1_SNR_DB,
    B2_P,
    B2_SNR_DB,
    TRACK_B_K,
    TRACK_B_N,
    TRACK_B_RSR_DB,
    track_b_world,
)

ALGS = ("biased_gs", "em_gs")
N_TRIALS = 400
BOOT = 400
OUT = Path(__file__).resolve().parent.parent / "results" / "track_b"


def estimate(w, alg):
    if alg == "biased_gs":
        return biased_gs_channel_rows(w.S, w.Z, w.B, max_iter=50).G_hat
    return em_gs_channel_rows(w.S, w.Z, w.B, w.sigma2, max_iter=50).G_hat


def point(P, snr, n_trials=N_TRIALS):
    """Per-trial (error, true) energies for each algorithm on shared worlds."""
    err = {a: np.zeros(n_trials) for a in ALGS}
    tru = np.zeros(n_trials)
    for t in range(n_trials):
        w = track_b_world(t, int(P), float(snr))
        tru[t] = float(np.linalg.norm(w.G, ord="fro") ** 2)
        for a in ALGS:
            err[a][t] = channel_nmse(estimate(w, a), w.G).error_energy
    return err, tru


def pooled_db(e, t):
    return 10 * np.log10(e.sum() / t.sum())


def boot_ci(e, t, rng, n=BOOT):
    idx = rng.integers(0, e.size, size=(n, e.size))
    vals = 10 * np.log10(e[idx].sum(1) / t[idx].sum(1))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def sweep(name, points, make):
    rng = np.random.default_rng(0)
    rows = []
    print(f"\n=== {name} ===")
    print(f"{'x':>5} | " + " ".join(f"{a:>10} {'95% CI':>16}" for a in ALGS)
          + f" | {'GS-EMGS':>8}")
    for x in points:
        err, tru = point(*make(x))
        cells, db = [], {}
        for a in ALGS:
            d = pooled_db(err[a], tru)
            lo, hi = boot_ci(err[a], tru, rng)
            db[a] = d
            cells.append(f"{d:10.2f} [{lo:6.2f},{hi:6.2f}]")
            rows.append(dict(sweep=name, x=float(x), algorithm=a, nmse_db=d,
                             ci_low=lo, ci_high=hi,
                             median_db=float(10 * np.log10(np.median(err[a] / tru))),
                             n_trials=int(err[a].size)))
        gap = db["biased_gs"] - db["em_gs"]
        print(f"{x:5g} | " + " ".join(cells) + f" | {gap:+8.2f}")
    return rows


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Track-B baseline: N={TRACK_B_N}, K={TRACK_B_K}, L_k~U{{3..7}}, "
          f"RSR={TRACK_B_RSR_DB} dB, t0=50, {N_TRIALS} trials/point")
    print("NMSE_G = sum||Ghat-G||_F^2 / sum||G||_F^2 (ratio of sums), 10log10")
    all_rows = []
    for P in (10, 30):
        all_rows += sweep(f"B1 (P={P})", B1_SNR_DB, lambda s, P=P: (P, s))
    all_rows += sweep(f"B2 (SNR={B2_SNR_DB} dB)", B2_P,
                      lambda p: (p, B2_SNR_DB))
    (OUT / "baseline_preliminary.json").write_text(json.dumps(
        {"N": TRACK_B_N, "K": TRACK_B_K, "L_k": "U{3..7} per realization",
         "rsr_db": TRACK_B_RSR_DB, "max_iter": 50, "n_trials": N_TRIALS,
         "rows": all_rows}, indent=2))
    print(f"\nwrote {OUT/'baseline_preliminary.json'}")

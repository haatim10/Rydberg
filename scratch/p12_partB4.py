"""PROMPT 12 B4 -- coarse-to-fine order selection, as an OPTION.

Order selection is 57.7-85.5% of HS-GS runtime and the manuscript describes the
fix rather than doing it. This implements it and scores the two gates.

Coarse-to-fine: score a coarse grid (stride s over 1..cap), take the argmin,
then refine over a local window of +/-(s-1) around it. Candidate count falls
from cap to roughly cap/s + 2(s-1), minimised near s = sqrt(cap/2).

GATES, both required before it may be reported:
  G1  returns the same L_hat as exhaustive search in >= 90% of trials
  G2  Delta_HS degrades by no more than 0.05 dB, with a CI supporting that

If either gate fails, report the failure and leave exhaustive search as the
method.

Run:  PYTHONPATH=. python3 scratch/p12_partB4.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from rydberg_sim.track_b_proposed import (
    hankel_rank_cap, hs_gs, magnitude_residual, select_order_heldout,
)
from scratch.p12_partB import (
    TD_KW, boot_ci_median, em_gs_only, nmse_parts, trackd_world,
)

OUT = Path("results/p12")


def select_order_coarse_to_fine(S, Z, B, sigma2, *, stride=3,
                                exact_step="em_gs", max_iter=25, val_frac=0.3,
                                pencil=None, cadzow_iter=4):
    """Coarse grid, then refine in a local window. Returns (L_hat, n_evaluated)."""
    S = np.asarray(S, dtype=np.complex128)
    P = S.shape[1]
    n_val = max(1, int(round(val_frac * P)))
    val = np.arange(P - n_val, P)
    fit = np.arange(0, P - n_val)
    N = np.asarray(Z).shape[0]
    cap = hankel_rank_cap(N, pencil)

    scores: dict[int, float] = {}

    def score(L):
        if L in scores:
            return scores[L]
        r = hs_gs(S[:, fit], np.asarray(Z)[:, fit], np.asarray(B)[:, fit],
                  sigma2, L_hat=L, exact_step=exact_step, max_iter=max_iter,
                  pencil=pencil, cadzow_iter=cadzow_iter)
        scores[L] = magnitude_residual(r.G_hat, S, B, Z, cols=val)
        return scores[L]

    coarse = list(range(1, cap + 1, stride))
    if coarse[-1] != cap:
        coarse.append(cap)
    for L in coarse:
        score(L)
    best = min(scores, key=scores.get)
    lo, hi = max(1, best - (stride - 1)), min(cap, best + (stride - 1))
    for L in range(lo, hi + 1):
        score(L)
    L_best = min(scores, key=scores.get)
    return int(L_best), len(scores), cap


def run(n_trials=300, stride=3):
    rows = []
    same, n_ex, n_ctf = [], [], []
    num_ex, num_ctf, num_em, den = [], [], [], []
    t_ex = t_ctf = 0.0
    rng = np.random.default_rng(np.random.SeedSequence([640_000, stride]))
    for t in range(n_trials):
        snr = float(np.round(rng.uniform(-10, 20), 3))
        w = trackd_world(640_000 + t, N=32, P=20, snr_db=snr)
        G = np.asarray(w.G_true)

        t0 = time.time()
        L_ex, sc = select_order_heldout(w.S, w.Z, w.B, w.sigma2,
                                        exact_step="em_gs",
                                        max_iter=TD_KW["select_iter"],
                                        cadzow_iter=4)
        t_ex += time.time() - t0
        n_ex.append(len(sc))

        t0 = time.time()
        L_cf, n_ev, cap = select_order_coarse_to_fine(
            w.S, w.Z, w.B, w.sigma2, stride=stride,
            max_iter=TD_KW["select_iter"], cadzow_iter=4)
        t_ctf += time.time() - t0
        n_ctf.append(n_ev)

        same.append(int(L_ex == L_cf))

        e = em_gs_only(w, TD_KW["max_iter"])
        gex = hs_gs(w.S, w.Z, w.B, w.sigma2, L_hat=L_ex, exact_step="em_gs",
                    max_iter=TD_KW["max_iter"], cadzow_iter=4).G_hat
        gcf = hs_gs(w.S, w.Z, w.B, w.sigma2, L_hat=L_cf, exact_step="em_gs",
                    max_iter=TD_KW["max_iter"], cadzow_iter=4).G_hat
        a, b = nmse_parts(e, G); num_em.append(a); den.append(b)
        num_ex.append(nmse_parts(gex, G)[0])
        num_ctf.append(nmse_parts(gcf, G)[0])

        if (t + 1) % 50 == 0:
            print(f"    {t+1}/{n_trials}  agree {100*np.mean(same):.1f}%  "
                  f"cands {np.mean(n_ex):.1f} -> {np.mean(n_ctf):.1f}", flush=True)

    den = np.asarray(den)
    d_ex = 10 * np.log10(np.asarray(num_em) / den) - \
        10 * np.log10(np.asarray(num_ex) / den)
    d_cf = 10 * np.log10(np.asarray(num_em) / den) - \
        10 * np.log10(np.asarray(num_ctf) / den)
    degrade = d_ex - d_cf                      # >0 means coarse-to-fine is worse
    ci = boot_ci_median(degrade)

    agree = float(np.mean(same))
    g1 = agree >= 0.90
    # G2: degradation no more than 0.05 dB, with the CI supporting that.
    g2 = bool(ci[1] <= 0.05)

    out = {
        "cell": "B4_coarse_to_fine_order_selection",
        "config": {"N": 32, "K": 3, "P": 20, "rsr_db": 10.0,
                   "n_trials": n_trials, "stride": stride,
                   "snr": "U[-10,20]", "operator": TD_KW},
        "agreement_rate": agree,
        "candidates_exhaustive_mean": float(np.mean(n_ex)),
        "candidates_coarse_to_fine_mean": float(np.mean(n_ctf)),
        "candidate_reduction": float(1 - np.mean(n_ctf) / np.mean(n_ex)),
        "selection_seconds_exhaustive": round(t_ex, 1),
        "selection_seconds_coarse_to_fine": round(t_ctf, 1),
        "selection_speedup": float(t_ex / t_ctf) if t_ctf else None,
        "delta_hs_exhaustive_median_db": float(np.median(d_ex)),
        "delta_hs_coarse_to_fine_median_db": float(np.median(d_cf)),
        "degradation_median_db": float(np.median(degrade)),
        "degradation_boot_ci95": list(ci),
        "GATE_1_agreement_ge_90pct": bool(g1),
        "GATE_2_degradation_le_0p05dB": g2,
        "BOTH_GATES_PASS": bool(g1 and g2),
    }
    return out


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "B4.json"
    if f.exists():
        print(f"skip (done): {f}")
        raise SystemExit(0)
    print("=== PROMPT 12 B4: coarse-to-fine order selection ===", flush=True)
    r = run()
    f.write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in r.items() if k != "config"}, indent=1))
    print(f"wrote {f}")

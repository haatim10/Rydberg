"""FIX 11 — the unconstrained CRLB for the Track-B estimation problem.

Sec. 4.6 already implements the Rician Fisher information; Sec. 1.3's
canonical mapping makes it directly applicable to the estimation role. Per
receive element n, the canonical problem is

    z = |M^H u + b|,  M = S,  u = conj(g_n),  b = conj(B[n,:])

so the row-n CRLB is tr(F_n^-1) and the whole-array bound is the sum over n.
Normalising by E||G||_F^2 = N*K*beta puts it on the same axis as NMSE_G.

IMPORTANT CAVEAT, stated in every caption that uses this curve: this is the
UNCONSTRAINED bound. It bounds GS and EM-GS, which use no structural prior.
HS-GS exploits a rank constraint and is NOT bounded by it; a constrained
bound would need the tangent space of the structured manifold and is not
computed here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from rydberg_sim.crlb import cui_crlb
from rydberg_sim.track_b_drivers import TRACK_B_K, track_b_world

N_TRIALS = 20          # the bound is smooth; it needs far fewer trials than NMSE


def crlb_nmse_db(N: int, P: int, snr_db: float, rsr_db: float = 12.0,
                 n_trials: int = N_TRIALS) -> float:
    """Averaged unconstrained CRLB, normalised like NMSE_G, in dB."""
    num, den = 0.0, 0.0
    for t in range(n_trials):
        w = track_b_world(t, P, float(snr_db), rsr_db=rsr_db, N=N)
        tr = 0.0
        for n in range(N):
            u = np.conjugate(w.G[n])
            b = np.conjugate(w.B[n])
            r = cui_crlb(w.S, u, b, w.sigma2, expected_u_energy=float(TRACK_B_K))
            tr += float(np.real(np.trace(r.crlb)))
        num += tr
        den += float(np.linalg.norm(w.G, "fro") ** 2)
    return float(10 * np.log10(num / den))


def check_high_snr_gap(N: int = 8, P: int = 30, snr_db: float = 40.0) -> float:
    """Track-A's own check, reused: at high SNR the CRLB must sit
    10log10(2) = 3.0103 dB above the genie-ZF covariance."""
    w = track_b_world(0, P, snr_db, N=N)
    crlb_tr = zf_tr = 0.0
    for n in range(N):
        u, b = np.conjugate(w.G[n]), np.conjugate(w.B[n])
        crlb_tr += float(np.real(np.trace(
            cui_crlb(w.S, u, b, w.sigma2, expected_u_energy=float(TRACK_B_K)).crlb)))
        MMh = w.S @ w.S.conj().T
        zf_tr += float(np.real(np.trace(w.sigma2 * np.linalg.inv(MMh))))
    return 10 * np.log10(crlb_tr / zf_tr)


def _job(a):
    kind, N, P, s = a
    return kind, f"N{N}_P{P}_snr{s:+.0f}" if kind == "b3" else f"P{P}", \
        crlb_nmse_db(N, P, s)


def main():
    import multiprocessing as mp
    import os

    gap = check_high_snr_gap()
    print(f"high-SNR CRLB / genie-ZF gap: {gap:.4f} dB "
          f"(must be 10log10(2) = 3.0103)", flush=True)
    assert abs(gap - 3.0103) < 0.05, (
        f"estimation-role CRLB has a convention error: gap {gap:.4f} dB, "
        "expected 3.0103 -- most likely real-vs-complex Fisher information")

    jobs = [("b3", N, P, s) for N in (8, 16, 32) for P in (10, 30)
            for s in (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)]
    jobs += [("b4", 16, P, 5.0) for P in (6, 10, 14, 20, 30, 40)]
    jobs.sort(key=lambda j: -j[1])
    out = {"caveat": "UNCONSTRAINED bound: valid for GS and EM-GS; HS-GS uses a "
                     "structural prior and is not bounded by it",
           "n_trials": N_TRIALS, "high_snr_gap_vs_genie_zf_db": gap,
           "b3": {}, "b4": {}}
    with mp.Pool(int(os.environ.get("CRLB_PROCS", "3"))) as pool:
        for kind, key, val in pool.imap_unordered(_job, jobs):
            out[kind][key] = val
            print(f"  {kind} {key}: {val:7.2f} dB", flush=True)
    (REPO / "results/track_b/crlb.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {REPO/'results/track_b/crlb.json'}", flush=True)


if __name__ == "__main__":
    main()

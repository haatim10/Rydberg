"""FIX 4 (recommended addition) — how tight is the Hankel relaxation?

Rank <= L admits sums of exponentials z_i^n with ARBITRARY non-zero complex
z_i; the ULA model requires |z_i| = 1. Cadzow enforces no unit-modulus
condition, so the feasible set strictly contains the ULA set. That is an
honest weakness of the method -- but it is measurable rather than merely
arguable: run ESPRIT on the projected channel, recover the modes, and look
at how far | |z_i| - 1 | actually is.

If the modes sit near the unit circle the relaxation is tight in practice
and we can say so with a number. If they don't, that is a real finding about
where the remaining slack lives.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from rydberg_sim.track_b_drivers import track_b_world
from rydberg_sim.track_b_proposed import hs_gs_auto
from rydberg_sim.track_b_structure import hankel_matrix

N_TRIALS = 80


def modes(g: np.ndarray, L: int) -> np.ndarray:
    """ESPRIT modes z_i of a length-N sequence, via the shift-invariance of
    the Hankel range space."""
    g = np.asarray(g, dtype=np.complex128).ravel()
    p = max(range(1, g.size), key=lambda q: min(g.size - q, q + 1))
    H = hankel_matrix(g, p)
    U = np.linalg.svd(H, full_matrices=False)[0][:, :L]
    if U.shape[0] < 2:
        return np.array([])
    psi = np.linalg.pinv(U[:-1]) @ U[1:]
    return np.linalg.eigvals(psi)


def main():
    N, P, snr = 32, 30, 5.0
    dev_proj, dev_base, allL = [], [], []
    for t in range(N_TRIALS):
        w = track_b_world(t, P, snr, N=N)
        r = hs_gs_auto(w.S, w.Z, w.B, w.sigma2, exact_step="em_gs",
                       max_iter=50, select_iter=20)
        if not r.constraint_active:
            continue
        L = int(r.L_hat)
        allL.append(L)
        for k in range(r.G_hat.shape[1]):
            z = modes(r.G_hat[:, k], L)
            if z.size:
                dev_proj.extend(np.abs(np.abs(z) - 1.0))
            zt = modes(w.G[:, k], L)          # true channel, same order
            if zt.size:
                dev_base.extend(np.abs(np.abs(zt) - 1.0))
    d = np.asarray(dev_proj); b = np.asarray(dev_base)
    out = {
        "config": {"N": N, "P": P, "snr_db": snr, "n_trials": N_TRIALS,
                   "n_active": len(allL), "mean_L_hat": float(np.mean(allL))},
        "projected": {"n_modes": int(d.size), "median": float(np.median(d)),
                      "p90": float(np.percentile(d, 90)),
                      "p99": float(np.percentile(d, 99)), "max": float(d.max()),
                      "frac_within_0p05": float(np.mean(d < 0.05)),
                      "frac_within_0p10": float(np.mean(d < 0.10))},
        "true_channel_reference": {
            "n_modes": int(b.size), "median": float(np.median(b)),
            "p90": float(np.percentile(b, 90))},
        "note": "|z|-1 deviation of ESPRIT modes. The true-channel column is "
                "the floor: it is what this ESPRIT estimator reports on data "
                "that is exactly on the ULA manifold, so it separates "
                "relaxation slack from estimator error.",
    }
    (REPO / "results/track_b/modulus_tightness.json").write_text(
        json.dumps(out, indent=2))
    print(f"N={N}, P={P}, SNR={snr} dB, {out['config']['n_active']} active trials, "
          f"mean L_hat {out['config']['mean_L_hat']:.2f}")
    print(f"{'':22}{'median':>9}{'p90':>9}{'p99':>9}{'max':>9}")
    pj = out["projected"]
    print(f"  {'HS-GS projected':<20}{pj['median']:9.4f}{pj['p90']:9.4f}"
          f"{pj['p99']:9.4f}{pj['max']:9.4f}")
    tr = out["true_channel_reference"]
    print(f"  {'true channel (floor)':<20}{tr['median']:9.4f}{tr['p90']:9.4f}")
    print(f"\n  modes within 0.05 of the unit circle: {pj['frac_within_0p05']:.1%}")
    print(f"  modes within 0.10 of the unit circle: {pj['frac_within_0p10']:.1%}")
    print(f"\nwrote {REPO/'results/track_b/modulus_tightness.json'}")


if __name__ == "__main__":
    main()

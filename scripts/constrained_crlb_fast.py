"""Constrained + corrected-unconstrained CRLB, at the full 400-trial budget.

Same mathematics as constrained_crlb.py. The only change is speed, so the
bound can be averaged over the SAME 400 channel realisations the estimator
curves use instead of 10. At 10 trials the per-point Monte-Carlo jitter is
+-0.31 dB, which is what made the earlier curves visibly wavy; at 400 it is
+-0.05 dB.

Two optimisations, neither of which changes the result:

1. beta(a, sigma^2) is a smooth 1-D function of a at fixed sigma^2, and
   sigma^2 is constant within an operating point. So one interpolation table
   per point replaces N*P*n_trials quadratures with ~400. Measured accuracy
   of the table: max relative error 2.7e-4, median 6e-7 -- three orders of
   magnitude below the sampling noise it removes.
2. J is block diagonal across receive elements, so D^T J D is assembled
   blockwise as sum_n D_n^T J_n D_n rather than by forming the full
   2NK x 2NK matrix.

The bound depends only on (G, S, B, sigma^2) -- no noise realisation -- so
the only variability is the channel draw.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from rydberg_sim.crlb import rician_fisher_scalar
from rydberg_sim.track_b_drivers import track_b_world

N_TRIALS = 400
TABLE_PTS = 400


def beta_table(a_lo, a_hi, sigma2, n=TABLE_PTS):
    """Interpolation table for beta over |lambda| at fixed sigma^2."""
    pad = 0.02 * max(a_hi - a_lo, 1e-9)
    grid = np.linspace(max(a_lo - pad, 1e-9), a_hi + pad, n)
    vals = np.array([rician_fisher_scalar(float(a), sigma2).beta for a in grid])
    return grid, vals


def point_bounds(N, P, snr_db, rsr_db, n_trials=N_TRIALS):
    """Return summed traces (unconstrained, constrained), channel energy, and
    diagnostics, over trials 0..n_trials-1 -- the same worlds the estimators saw."""
    worlds = [track_b_world(t, P, float(snr_db), rsr_db=float(rsr_db), N=N)
              for t in range(n_trials)]
    K = worlds[0].G.shape[1]
    sigma2 = float(worlds[0].sigma2)
    lam = [w.G @ w.S + w.B for w in worlds]
    amps = np.concatenate([np.abs(l).ravel() for l in lam])
    grid, vals = beta_table(amps.min(), amps.max(), sigma2)

    unc = con = den = 0.0
    ranks, conds = [], []
    nn = np.arange(N)
    for w, L in zip(worlds, lam):
        A = np.abs(L)                                   # (N, P)
        beta = np.interp(A, grid, vals)                 # (N, P)
        ph = np.exp(-1j * np.angle(L))                  # (N, P)

        # ---- manifold Jacobian D, rows ordered [Re G[n,:], Im G[n,:]] ----
        m = int(sum(3 * len(a) for a in w.alpha))
        Dc = np.zeros((N, K, m), dtype=np.complex128)   # complex dG[n,k]/dtheta
        col = 0
        for k in range(K):
            psi, al = np.asarray(w.theta[k]), np.asarray(w.alpha[k])
            for l in range(len(al)):
                e = np.exp(-1j * nn * psi[l])
                Dc[:, k, col] = al[l] * (-1j * nn) * e
                Dc[:, k, col + 1] = e
                Dc[:, k, col + 2] = 1j * e
                col += 3
        # real form per receive element: (2K, m)
        Dn = np.concatenate([Dc.real, Dc.imag], axis=1)

        JD = np.zeros((m, m))
        for n in range(N):
            # J_n = sum_p 4 beta_np grad grad^T,  grad = [Re c ; -Im c]
            c = ph[n][None, :] * w.S                     # (K, P)
            g = np.concatenate([c.real, -c.imag], axis=0)  # (2K, P)
            Jn = (g * (4.0 * beta[n])[None, :]) @ g.T      # (2K, 2K)
            unc += float(np.trace(np.linalg.inv(Jn)))
            JD += Dn[n].T @ Jn @ Dn[n]
        # The path parametrisation is OVERCOMPLETE whenever 3*sum(L_k) exceeds
        # the ambient real dimension 2NK -- true in ~42% of trials at N=8.
        # There the map theta -> G is not injective and D^T J D is genuinely
        # singular. The CCRB is still well defined, because it depends on the
        # TANGENT SPACE range(D) and not on theta being identifiable, but it
        # must be formed from an ORTHONORMAL basis of that tangent space
        # rather than by pseudo-inverting D^T J D (condition numbers up to
        # 7e17 otherwise). Gorman-Hero / Stoica-Ng in stable form:
        #     U = orthonormal basis of range(D),  CCRB = U (U^T J U)^{-1} U^T
        Dfull = Dn.reshape(N * 2 * K, m)
        Uf, sv, _ = np.linalg.svd(Dfull, full_matrices=False)
        keep = sv > sv[0] * 1e-9
        U = Uf[:, keep].reshape(N, 2 * K, int(keep.sum()))
        ranks.append(int(keep.sum()))
        UJU = np.zeros((int(keep.sum()),) * 2)
        for n in range(N):
            c = ph[n][None, :] * w.S
            g = np.concatenate([c.real, -c.imag], axis=0)
            UJU += U[n].T @ ((g * (4.0 * beta[n])[None, :]) @ g.T) @ U[n]
        conds.append(float(np.linalg.cond(UJU)))
        Minv = np.linalg.inv(UJU)
        con += float(sum(np.trace(U[n] @ Minv @ U[n].T) for n in range(N)))
        den += float(np.linalg.norm(w.G, "fro") ** 2)
    return unc, con, den, ranks, conds


def _job(a):
    kind, N, P, s, rsr, nt = a
    unc, con, den, ranks, conds = point_bounds(N, P, s, rsr, nt)
    key = (f"P{P}" if kind == "b4" else
           f"N{N}_P{P}_snr{s:+.0f}_rsr{rsr:+.0f}" if kind == "b6" else
           f"N{N}_P{P}_snr{s:+.0f}")
    return (kind, key, 10 * np.log10(unc / den), 10 * np.log10(con / den),
            float(np.mean(ranks)), float(np.median(conds)), nt)


def main():
    import multiprocessing as mp
    nt = int(os.environ.get("CC_TRIALS", str(N_TRIALS)))
    jobs = [("b3", N, P, s, 12.0, nt) for N in (8, 16, 32) for P in (10, 30)
            for s in (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)]
    jobs += [("b4", 16, P, 5.0, 12.0, nt) for P in (6, 10, 14, 20, 30, 40)]
    jobs += [("b6", N, 30, 5.0, r, nt) for N in (8, 32)
             for r in (0.0, 6.0, 12.0, 18.0, 24.0)]
    jobs.sort(key=lambda j: -j[1] * j[2])
    out = {k: {"b3": {}, "b4": {}, "b6": {}} for k in
           ("unconstrained_rank1", "constrained", "jacobian_rank", "jacobian_cond")}
    out["n_trials"] = nt
    out["note"] = (
        "Averaged over the SAME trial indices the estimator curves use, so the "
        "comparison is paired on channel realisations. unconstrained_rank1: "
        "each magnitude measurement contributes ONE real constraint (rank-1 "
        "Fisher in real coordinates). constrained: Gorman-Hero / Stoica-Ng "
        "bound on the 3*sum(L_k)-parameter geometric manifold. Neither bound "
        "governs a biased estimator.")
    print(f"{nt} trials/point, {len(jobs)} points", flush=True)
    with mp.Pool(int(os.environ.get("CC_PROCS", "4"))) as pool:
        for kind, key, u, c, r, cond, _ in pool.imap_unordered(_job, jobs):
            out["unconstrained_rank1"][kind][key] = u
            out["constrained"][kind][key] = c
            out["jacobian_rank"][kind][key] = r
            out["jacobian_cond"][kind][key] = cond
            print(f"  {kind} {key}: unconstr {u:7.2f}  constr {c:7.2f}"
                  f"  (rank {r:.1f}, cond {cond:.1e})", flush=True)
    (REPO / "results/track_b/constrained_crlb.json").write_text(
        json.dumps(out, indent=2))
    print(f"\nwrote {REPO/'results/track_b/constrained_crlb.json'}", flush=True)


if __name__ == "__main__":
    main()

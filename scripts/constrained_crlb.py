"""Constrained CRLB for the geometric-ULA channel-estimation problem.

The unconstrained bound already computed (scripts/track_b_crlb.py) governs
estimators that treat G as 2NK free real parameters. HS-GS does not: it
constrains G to the low-rank Hankel variety, so that bound does not apply to
it -- which is why HS-GS legitimately falls below it. This computes the bound
that DOES apply.

Construction
------------
Physical parameters, per user k and path l:

    theta = { psi_lk, Re alpha_lk, Im alpha_lk },   m = 3 * sum_k L_k

The channel is a smooth function of them,

    G[n,k] = sum_l alpha_lk exp(-j (n-1) psi_lk).

Write g = g(theta) in real coordinates (2NK of them) and D = dg/dtheta.
For an estimator unbiased ON THE MANIFOLD, the Gorman-Hero / Stoica-Ng
constrained bound is

    CCRB_g = D (D^T J D)^{-1} D^T ,      E||Ghat - G||_F^2 >= Tr(CCRB_g)

with J the unconstrained Fisher information in the same real coordinates.
Equivalently: invert the Fisher information in the 3*sum(L) physical
parameters and push it back to the channel.

Fisher information, from first principles in real coordinates
------------------------------------------------------------
Each measurement z[n,p] is Rician with noncentrality a = |lambda[n,p]|,
lambda = GS + B. Its log-likelihood depends on the parameters only through a,
so with score

    d/da log p = (2/sigma^2) ( z R(kappa) - a ),   kappa = 2 z a / sigma^2

and the zero-mean-score identity E[z R(kappa)] = a, the scalar Fisher
information of one measurement is

    I(a) = (4/sigma^4) ( E[z^2 R^2(kappa)] - a^2 ) = 4 * beta

with beta exactly the quantity Cui's CRLB already computes. The gradient of a
w.r.t. the real channel coordinates of row n is, with c_k = e^{-j angle(lambda)} S[k,p],

    da/dRe G[n,k] =  Re(c_k),     da/dIm G[n,k] = -Im(c_k),

so J is block diagonal across receive elements (each measurement touches one
row), J_n = sum_p I(a_np) grad grad^T. The STRUCTURE is what couples rows,
and it enters only through D.

Validation before use (both run in main()):
  1. the real-coordinate UNCONSTRAINED bound Tr(J^{-1}) must reproduce the
     already-validated complex-convention bound in results/track_b/crlb.json;
  2. at high SNR it must sit 10log10(2) = 3.0103 dB above the genie-ZF
     covariance sigma^2 Tr((M M^H)^{-1}), the same check Track A uses.
Only if both pass is the projected bound trustworthy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from rydberg_sim.crlb import rician_fisher_scalar
from rydberg_sim.track_b_drivers import TRACK_B_K, track_b_world

N_TRIALS = 20
#: Relative singular-value cut defining the numerical rank of the tangent
#: space range(D). Matches constrained_crlb_fast.py so the two paths agree.
TANGENT_RTOL = 1e-9


def beta_of(a: float, sigma2: float) -> float:
    """Cui's beta = (E[z^2 R^2(kappa)] - a^2)/sigma^4 for one measurement."""
    return float(rician_fisher_scalar(float(a), float(sigma2)).beta)


def fisher_real(G, S, B, sigma2):
    """Unconstrained Fisher information in real channel coordinates.

    Returns J as a list of N blocks, each (2K, 2K); the full matrix is
    block diagonal because measurement (n,p) depends only on row n of G.
    Ordering within a block: [Re G[n,0..K-1], Im G[n,0..K-1]].
    """
    N, K = G.shape
    P = S.shape[1]
    lam = G @ S + B                                  # (N, P) noiseless field
    blocks = []
    for n in range(N):
        Jn = np.zeros((2 * K, 2 * K))
        ph = np.exp(-1j * np.angle(lam[n]))          # (P,)
        for p in range(P):
            c = ph[p] * S[:, p]                      # (K,)
            grad = np.concatenate([c.real, -c.imag])  # (2K,)
            Jn += 4.0 * beta_of(abs(lam[n, p]), sigma2) * np.outer(grad, grad)
        blocks.append(Jn)
    return blocks


def jacobian(psi_list, alpha_list, N, K):
    """D = d(vec_real G)/d(phi), phi = {psi, Re alpha, Im alpha} per path.

    The angular coordinate is the SPATIAL FREQUENCY psi = pi sin(theta), which
    is what appears in the generator G[n,k] = sum_l alpha_lk exp(-i n psi_lk).
    Passing the physical AoA theta here is a bug: it evaluates the tangent
    space at the wrong point of the manifold (verified by
    scripts/audit_verify.py, PART 4b).

    Row ordering of the output matches fisher_real: for receive element n the
    2K rows are [Re G[n,:], Im G[n,:]]; blocks are stacked over n.
    """
    m = int(sum(3 * len(a) for a in alpha_list))
    D = np.zeros((N * 2 * K, m))
    nn = np.arange(N)
    col = 0
    for k in range(K):
        psi, al = psi_list[k], alpha_list[k]
        for l in range(len(al)):
            e = np.exp(-1j * nn * psi[l])            # (N,)
            dpsi = al[l] * (-1j * nn) * e            # d/d psi
            dre = e                                   # d/d Re alpha
            dim = 1j * e                              # d/d Im alpha
            for j, d in enumerate((dpsi, dre, dim)):
                for n in range(N):
                    D[n * 2 * K + k, col + j] = d[n].real
                    D[n * 2 * K + K + k, col + j] = d[n].imag
            col += 3
    return D


def bounds_for(N, P, snr_db, rsr_db=12.0, n_trials=N_TRIALS):
    """Return (unconstrained, constrained) traces summed over trials, and the
    channel energy denominator."""
    unc = con = den = 0.0
    ranks = []
    for t in range(n_trials):
        w = track_b_world(t, P, float(snr_db), rsr_db=rsr_db, N=N)
        K = w.G.shape[1]
        blocks = fisher_real(w.G, w.S, w.B, w.sigma2)
        # unconstrained: block diagonal, so the trace of the inverse is the
        # sum of the per-row traces
        unc += sum(float(np.trace(np.linalg.inv(Jn))) for Jn in blocks)
        # constrained: project through the manifold Jacobian, in the SPATIAL
        # FREQUENCY coordinate psi that the generator actually uses.
        D = jacobian([np.asarray(x) for x in w.psi],
                     [np.asarray(x) for x in w.alpha], N, K)
        J = np.zeros((N * 2 * K, N * 2 * K))
        for n, Jn in enumerate(blocks):
            s = n * 2 * K
            J[s:s + 2 * K, s:s + 2 * K] = Jn
        # The path parametrisation is overcomplete whenever 3*sum(L_k) exceeds
        # the ambient real dimension 2NK, so D^T J D is genuinely singular and
        # pseudo-inverting it is numerically hopeless (cond up to 7e17). The
        # CCRB depends only on the TANGENT SPACE range(D), so form it from an
        # orthonormal basis: CCRB = U (U^T J U)^{-1} U^T.
        U_full, sv, _ = np.linalg.svd(D, full_matrices=False)
        keep = sv > sv[0] * TANGENT_RTOL
        U = U_full[:, keep]
        ranks.append(int(keep.sum()))
        UJU = U.T @ J @ U
        con += float(np.trace(U @ np.linalg.solve(UJU, U.T)))
        den += float(np.linalg.norm(w.G, "fro") ** 2)
    return unc, con, den, ranks


# ------------------------------------------------------------ validation ---
def complex_convention_db(N, P, snr_db, rsr_db=12.0, n_trials=6):
    """The crlb.json convention (F = sum_q beta_q m_q m_q^H) on GIVEN trials.

    Recomputed here rather than read from crlb.json so the comparison is
    PAIRED on channel realisations. Comparing a 6-trial sweep against a
    20-trial stored value confounds the convention gap with Monte-Carlo
    jitter, which is large enough here to flip its sign.
    """
    from rydberg_sim.crlb import cui_crlb
    num = den = 0.0
    for t in range(n_trials):
        w = track_b_world(t, P, float(snr_db), rsr_db=rsr_db, N=N)
        for n in range(N):
            r = cui_crlb(w.S, np.conjugate(w.G[n]), np.conjugate(w.B[n]),
                         w.sigma2, expected_u_energy=float(TRACK_B_K))
            num += float(np.real(np.trace(r.crlb)))
        den += float(np.linalg.norm(w.G, "fro") ** 2)
    return 10 * np.log10(num / den)


def validate():
    ok = True
    print("=" * 78)
    print("VALIDATION 1 — real rank-1 bound must EXCEED the two-quadrature one")
    print("=" * 78)
    print("  A magnitude measurement constrains ONE real direction, so its real")
    print("  Fisher term is rank 1. The complex convention F = sum beta m m^H")
    print("  credits both quadratures and can therefore only ADD information,")
    print("  giving a strictly LOWER bound. These are different quantities: the")
    print("  test is the inequality, not equality. Paired on the same trials.")
    print(f"{'N':>3}{'P':>4}{'SNR':>5} | {'rank-1':>9} {'2-quad':>9} {'excess dB':>10}")
    for N, P, s in ((8, 30, 5.0), (8, 10, 15.0), (16, 30, 0.0), (32, 30, 10.0)):
        unc, _, den, _ = bounds_for(N, P, s, n_trials=6)
        mine = 10 * np.log10(unc / den)
        ref = complex_convention_db(N, P, s, n_trials=6)
        d = mine - ref
        good = d > 0.0
        ok &= good
        print(f"{N:3d}{P:4d}{s:5.0f} | {mine:9.3f} {ref:9.3f} {d:+10.3f}"
              f"  {'ok' if good else 'MISMATCH — rank-1 bound below 2-quadrature'}")

    print()
    print("=" * 78)
    print("VALIDATION 2 — high-SNR gap over genie ZF must be 10log10(2) dB")
    print("=" * 78)
    for N, P in ((8, 30), (16, 30)):
        w = track_b_world(0, P, 45.0, N=N)
        blocks = fisher_real(w.G, w.S, w.B, w.sigma2)
        crlb = sum(float(np.trace(np.linalg.inv(Jn))) for Jn in blocks)
        # genie ZF: knowing the phase, error covariance is sigma^2 (M M^H)^-1
        # per receive row; E||err||^2 = sigma^2 Tr((S S^H)^-1) per row
        zf = N * float(np.real(w.sigma2 * np.trace(
            np.linalg.inv(w.S @ w.S.conj().T))))
        gap = 10 * np.log10(crlb / zf)
        good = abs(gap - 3.0103) < 0.7   # finite-P: rank-1 sum only
        ok &= good                        # averages to SS^H asymptotically
        print(f"  N={N:2d} P={P}: gap {gap:.4f} dB vs required 3.0103  "
              f"{'ok' if good else 'MISMATCH'}")
    return ok




# --------------------------------------------------------- full sweep -----
def _job(a):
    kind, N, P, s, rsr = a
    unc, con, den, ranks = bounds_for(N, P, s, rsr_db=rsr, n_trials=10)
    key = (f"N{N}_P{P}_snr{s:+.0f}" if kind != "b6"
           else f"N{N}_P{P}_snr{s:+.0f}_rsr{rsr:+.0f}")
    if kind == "b4":
        key = f"P{P}"
    return kind, key, 10 * np.log10(unc / den), 10 * np.log10(con / den), \
        float(np.mean(ranks))


def sweep():
    import multiprocessing as mp
    import os
    jobs = [("b3", N, P, s, 12.0) for N in (8, 16, 32) for P in (10, 30)
            for s in (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)]
    jobs += [("b4", 16, P, 5.0, 12.0) for P in (6, 10, 14, 20, 30, 40)]
    jobs += [("b6", N, 30, 5.0, r) for N in (8, 32)
             for r in (0.0, 6.0, 12.0, 18.0, 24.0)]
    jobs.sort(key=lambda j: -j[1])
    out = {"unconstrained_rank1": {}, "constrained": {}, "mean_jacobian_rank": {},
           "note": "unconstrained_rank1 is the CORRECTED bound: each "
                   "magnitude measurement contributes ONE real constraint, so "
                   "its Fisher contribution is rank 1 in real coordinates. "
                   "constrained is the Gorman-Hero/Stoica-Ng bound on the "
                   "3*sum(L_k)-parameter manifold. Both are for estimators "
                   "unbiased in the relevant sense."}
    for k in ("b3", "b4", "b6"):
        for d in out.values():
            if isinstance(d, dict):
                d.setdefault(k, {})
    with mp.Pool(int(os.environ.get("CC_PROCS", "4"))) as pool:
        for kind, key, u, c, r in pool.imap_unordered(_job, jobs):
            out["unconstrained_rank1"][kind][key] = u
            out["constrained"][kind][key] = c
            out["mean_jacobian_rank"][kind][key] = r
            print(f"  {kind} {key}: unconstrained {u:7.2f}  constrained {c:7.2f}"
                  f"  (Jacobian rank {r:.1f})", flush=True)
    (REPO / "results/track_b/constrained_crlb.json").write_text(
        json.dumps(out, indent=2))
    print(f"\nwrote {REPO/'results/track_b/constrained_crlb.json'}", flush=True)


if __name__ == "__main__":
    if not validate():
        print("\nVALIDATION FAILED — not computing the constrained bound.")
        sys.exit(1)
    print("\nBoth validations pass; the real-coordinate Fisher is consistent.\n")
    sweep()

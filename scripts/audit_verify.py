"""Independent numerical audit of the Rician FIM, the real-coordinate channel
FIM, and the geometric-manifold Jacobian.

Nothing here reuses the implementation's own algebra: every claim is checked
against central finite differences of the ACTUAL generator/forward model, or
against direct quadrature of the ACTUAL likelihood. Run before and after any
fix; the Jacobian block is expected to FAIL on the pre-fix implementation.

    python scripts/audit_verify.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from rydberg_sim.crlb import _rician_pdf_scalar, rician_fisher_scalar
from rydberg_sim.gs import bessel_ratio
from rydberg_sim.track_b_drivers import track_b_world

RESULTS: list[tuple[str, bool, float, str]] = []


def record(name: str, ok: bool, err: float, note: str = "") -> None:
    RESULTS.append((name, bool(ok), float(err), note))
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] {name:<52} max err {err:.3e}  {note}")


# ---------------------------------------------------------------- PART 1 ---
def log_p(z: float, a: float, s2: float) -> float:
    """log of the ACTUAL implemented Rician density."""
    return float(np.log(_rician_pdf_scalar(z, a, s2)))


def score_analytic(z: float, a: float, s2: float) -> float:
    kappa = 2.0 * z * a / s2
    return (2.0 / s2) * (z * float(np.asarray(bessel_ratio(kappa))) - a)


def part1_score() -> None:
    """d/da log p(z|a) == (2/s2)(z R(kappa) - a), by central differences."""
    print("\nPART 1a - Rician score, finite differences vs analytic")
    worst = 0.0
    for s2 in (0.05, 0.3, 1.0, 4.0):
        for a in (0.15, 0.7, 2.0, 5.0):
            for z in (0.2, 0.9, 2.5, 6.0):
                h = 1e-6 * max(a, 1.0)
                fd = (log_p(z, a + h, s2) - log_p(z, a - h, s2)) / (2 * h)
                an = score_analytic(z, a, s2)
                worst = max(worst, abs(fd - an) / max(abs(an), 1.0))
    record("d log p / da  (central FD, rel)", worst < 1e-6, worst)


def part1_fisher() -> None:
    """E[(d log p/da)^2] == 4 * beta_code, by direct quadrature."""
    print("\nPART 1b - scalar Fisher information vs 4*beta_code")
    worst = 0.0
    detail = ""
    for s2 in (0.05, 0.3, 1.0, 4.0):
        for a in (0.15, 0.7, 2.0, 5.0):
            sig = np.sqrt(s2)
            lo, hi = max(0.0, a - 10 * sig), a + 10 * sig

            def integrand(z: float) -> float:
                if z <= 0.0:
                    return 0.0
                return score_analytic(z, a, s2) ** 2 * _rician_pdf_scalar(z, a, s2)

            val, _ = quad(integrand, lo, hi, epsabs=1e-13, epsrel=1e-11, limit=400)
            beta = rician_fisher_scalar(a, s2).beta
            rel = abs(val - 4.0 * beta) / max(abs(val), 1e-12)
            if rel > worst:
                worst, detail = rel, f"(a={a}, s2={s2})"
    record("I(a) == 4*beta_code  (quadrature, rel)", worst < 1e-6, worst, detail)


def part1_zero_score() -> None:
    """The identity the derivation leans on: E[z R(kappa)] == a."""
    print("\nPART 1c - zero-mean score identity E[z R] = a")
    worst = 0.0
    for s2 in (0.05, 0.3, 1.0, 4.0):
        for a in (0.15, 0.7, 2.0, 5.0):
            sig = np.sqrt(s2)
            lo, hi = max(0.0, a - 10 * sig), a + 10 * sig

            def integrand(z: float) -> float:
                if z <= 0.0:
                    return 0.0
                k = 2.0 * z * a / s2
                return z * float(np.asarray(bessel_ratio(k))) * _rician_pdf_scalar(z, a, s2)

            val, _ = quad(integrand, lo, hi, epsabs=1e-13, epsrel=1e-11, limit=400)
            worst = max(worst, abs(val - a) / max(a, 1e-12))
    record("E[z R(kappa)] == a  (quadrature, rel)", worst < 1e-6, worst)


# ---------------------------------------------------------------- PART 2 ---
def part2_channel_grad() -> None:
    """da/d[Re G, Im G] == [Re c, -Im c],  c = e^{-i angle(lam)} S[:,p].

    Finite-differenced through the ACTUAL forward map a = |G S + B|.
    """
    print("\nPART 2a - da/d(real channel coords) vs finite differences")
    worst = 0.0
    for trial, N, P in ((0, 8, 10), (3, 16, 30), (5, 8, 30)):
        w = track_b_world(trial, P, 5.0, N=N)
        G, S, B = np.array(w.G), np.array(w.S), np.array(w.B)
        K = G.shape[1]
        lam = G @ S + B
        ph = np.exp(-1j * np.angle(lam))
        for n in (0, N // 2, N - 1):
            for p in (0, P // 2, P - 1):
                c = ph[n, p] * S[:, p]
                analytic = np.concatenate([c.real, -c.imag])
                fd = np.zeros(2 * K)
                for k in range(K):
                    for j, bump in enumerate((1.0, 1j)):
                        h = 1e-7
                        Gp, Gm = G.copy(), G.copy()
                        Gp[n, k] += h * bump
                        Gm[n, k] -= h * bump
                        ap = abs((Gp @ S + B)[n, p])
                        am = abs((Gm @ S + B)[n, p])
                        fd[j * K + k] = (ap - am) / (2 * h)
                worst = max(worst, np.max(np.abs(fd - analytic)) /
                            max(np.max(np.abs(analytic)), 1.0))
    record("da/d[ReG,ImG] == [Re c, -Im c]  (FD, rel)", worst < 1e-6, worst)


def part2_rank_and_block() -> None:
    """One magnitude measurement contributes a rank-1 real Fisher term, and
    the assembled unconstrained J is block diagonal across receive rows."""
    print("\nPART 2b - rank-1 per measurement, block diagonal across rows")
    w = track_b_world(0, 10, 5.0, N=8)
    G, S, B = np.array(w.G), np.array(w.S), np.array(w.B)
    N, K = G.shape
    P = S.shape[1]
    lam = G @ S + B
    ph = np.exp(-1j * np.angle(lam))

    worst_rank = 0.0
    for n in range(N):
        for p in range(P):
            c = ph[n, p] * S[:, p]
            g = np.concatenate([c.real, -c.imag])
            M = np.outer(g, g)
            sv = np.linalg.svd(M, compute_uv=False)
            # rank 1 <=> second singular value is numerically zero
            worst_rank = max(worst_rank, sv[1] / max(sv[0], 1e-300))
    record("single measurement Fisher term is rank 1", worst_rank < 1e-12, worst_rank)

    # block diagonality: build the full 2NK x 2NK real FIM from scratch by
    # differentiating every a[n,p] w.r.t. EVERY channel coordinate.
    dim = 2 * N * K

    def coord_index(n: int, k: int, imag: bool) -> int:
        return n * 2 * K + (K if imag else 0) + k

    J = np.zeros((dim, dim))
    for n in range(N):
        for p in range(P):
            grad = np.zeros(dim)
            c = ph[n, p] * S[:, p]
            for k in range(K):
                grad[coord_index(n, k, False)] = c[k].real
                grad[coord_index(n, k, True)] = -c[k].imag
            J += 4.0 * rician_fisher_scalar(abs(lam[n, p]), w.sigma2).beta * np.outer(grad, grad)
    off = 0.0
    for n in range(N):
        for mrow in range(N):
            if n == mrow:
                continue
            blk = J[n * 2 * K:(n + 1) * 2 * K, mrow * 2 * K:(mrow + 1) * 2 * K]
            off = max(off, np.max(np.abs(blk)))
    record("unconstrained J block diagonal across rows", off == 0.0, off)


# ---------------------------------------------------------------- PART 4 ---
def build_G(psi_list, alpha_list, N, K) -> np.ndarray:
    """Reference generator: G[n,k] = sum_l alpha_lk exp(-i n psi_lk)."""
    nn = np.arange(N)
    G = np.zeros((N, K), dtype=np.complex128)
    for k in range(K):
        for l in range(len(alpha_list[k])):
            G[:, k] += alpha_list[k][l] * np.exp(-1j * nn * psi_list[k][l])
    return G


def part4_generator_convention() -> None:
    """The stored world must satisfy G = sum alpha exp(-i n psi), psi = pi sin theta.

    This pins down WHICH variable belongs in the exponent before any Jacobian
    is differentiated.
    """
    print("\nPART 4a - which variable is in the exponent of the real generator")
    worst_psi = worst_theta = 0.0
    for trial, N in ((0, 8), (2, 16), (4, 32)):
        w = track_b_world(trial, 30, 5.0, N=N)
        K = w.G.shape[1]
        G_psi = build_G([np.asarray(x) for x in w.psi],
                        [np.asarray(x) for x in w.alpha], N, K)
        G_theta = build_G([np.asarray(x) for x in w.theta],
                          [np.asarray(x) for x in w.alpha], N, K)
        scale = max(np.max(np.abs(w.G)), 1e-12)
        worst_psi = max(worst_psi, np.max(np.abs(G_psi - w.G)) / scale)
        worst_theta = max(worst_theta, np.max(np.abs(G_theta - w.G)) / scale)
    record("G == sum alpha exp(-i n PSI)", worst_psi < 1e-12, worst_psi)
    record("G == sum alpha exp(-i n THETA)  [must FAIL]",
           worst_theta > 1e-3, worst_theta, "(sanity: theta is NOT the exponent)")


def part4_jacobian(jac_fn, label: str) -> None:
    """Central-difference every column of the manifold Jacobian.

    Perturbs psi / Re alpha / Im alpha and re-runs the ACTUAL generator.
    """
    print(f"\nPART 4b - manifold Jacobian finite differences [{label}]")
    worst = 0.0
    detail = ""
    for trial, N in ((0, 8), (2, 16), (1, 8), (4, 32)):
        w = track_b_world(trial, 30, 5.0, N=N)
        K = w.G.shape[1]
        psi = [np.array(x, dtype=float) for x in w.psi]
        alpha = [np.array(x, dtype=complex) for x in w.alpha]
        D = jac_fn(w, N, K)

        col = 0
        for k in range(K):
            for l in range(len(alpha[k])):
                for j in range(3):
                    h = 1e-6
                    pp = [x.copy() for x in psi]
                    ap = [x.copy() for x in alpha]
                    pm = [x.copy() for x in psi]
                    am = [x.copy() for x in alpha]
                    if j == 0:
                        pp[k][l] += h
                        pm[k][l] -= h
                    elif j == 1:
                        ap[k][l] += h
                        am[k][l] -= h
                    else:
                        ap[k][l] += 1j * h
                        am[k][l] -= 1j * h
                    Gp = build_G(pp, ap, N, K)
                    Gm = build_G(pm, am, N, K)
                    dG = (Gp - Gm) / (2 * h)
                    fd = np.zeros(N * 2 * K)
                    for n in range(N):
                        for kk in range(K):
                            fd[n * 2 * K + kk] = dG[n, kk].real
                            fd[n * 2 * K + K + kk] = dG[n, kk].imag
                    an = D[:, col + j]
                    denom = max(np.max(np.abs(fd)), 1.0)
                    rel = np.max(np.abs(fd - an)) / denom
                    if rel > worst:
                        worst, detail = rel, f"(N={N}, trial={trial}, k={k}, l={l}, par={j})"
                col += 3
    record(f"manifold Jacobian vs FD [{label}]", worst < 1e-6, worst, detail)


# ---------------------------------------------------------------- PART 5 ---
def ccrb_from_D(D, J):
    """Gorman-Hero / Stoica-Ng tangent-space CCRB from a Jacobian and FIM."""
    U_full, sv, _ = np.linalg.svd(D, full_matrices=False)
    U = U_full[:, sv > sv[0] * 1e-9]
    return float(np.trace(U @ np.linalg.solve(U.T @ J @ U, U.T))), U.shape[1]


def part5_tangent_properties() -> None:
    """Symmetry, PSD, and conditioning of the tangent-space CCRB."""
    from constrained_crlb import fisher_real, jacobian
    print("\nPART 5a - CCRB matrix properties")
    worst_sym = 0.0
    min_eig = np.inf
    worst_cond = 0.0
    for trial, N in ((0, 8), (2, 16)):
        w = track_b_world(trial, 30, 5.0, N=N)
        K = w.G.shape[1]
        blocks = fisher_real(w.G, w.S, w.B, w.sigma2)
        J = np.zeros((N * 2 * K, N * 2 * K))
        for n, Jn in enumerate(blocks):
            s = n * 2 * K
            J[s:s + 2 * K, s:s + 2 * K] = Jn
        D = jacobian([np.asarray(x) for x in w.psi],
                     [np.asarray(x) for x in w.alpha], N, K)
        U_full, sv, _ = np.linalg.svd(D, full_matrices=False)
        U = U_full[:, sv > sv[0] * 1e-9]
        UJU = U.T @ J @ U
        C = U @ np.linalg.solve(UJU, U.T)
        worst_sym = max(worst_sym, np.max(np.abs(C - C.T)) / max(np.abs(C).max(), 1e-12))
        min_eig = min(min_eig, float(np.linalg.eigvalsh(0.5 * (C + C.T))[0]))
        worst_cond = max(worst_cond, float(np.linalg.cond(UJU)))
    record("CCRB symmetric", worst_sym < 1e-9, worst_sym)
    record("CCRB positive semidefinite", min_eig > -1e-12, abs(min(min_eig, 0.0)),
           f"(min eig {min_eig:.2e})")
    record("U^T J U well conditioned (<1e6)", worst_cond < 1e6, worst_cond,
           f"(cond {worst_cond:.1e})")


def part5_invariance() -> None:
    """CCRB must be invariant to the angular parameterization.

    Parameterizing by theta instead of psi scales each angular Jacobian column
    by pi cos(theta) (chain rule). A per-column scaling leaves range(D)
    unchanged, so the tangent space -- and therefore the CCRB -- must be
    identical away from endfire, where cos(theta) -> 0.
    """
    from constrained_crlb import fisher_real, jacobian
    print("\nPART 5b - parameterization invariance: psi vs theta")
    worst = 0.0
    for trial, N in ((0, 8), (2, 16), (4, 32)):
        w = track_b_world(trial, 30, 5.0, N=N)
        K = w.G.shape[1]
        blocks = fisher_real(w.G, w.S, w.B, w.sigma2)
        J = np.zeros((N * 2 * K, N * 2 * K))
        for n, Jn in enumerate(blocks):
            s = n * 2 * K
            J[s:s + 2 * K, s:s + 2 * K] = Jn
        D_psi = jacobian([np.asarray(x) for x in w.psi],
                         [np.asarray(x) for x in w.alpha], N, K)
        # theta parameterization: d g / d theta = pi cos(theta) * d g / d psi
        D_theta = D_psi.copy()
        col = 0
        for k in range(K):
            th = np.asarray(w.theta[k])
            for l in range(len(np.asarray(w.alpha[k]))):
                D_theta[:, col] = D_psi[:, col] * (np.pi * np.cos(th[l]))
                col += 3
        c_psi, r_psi = ccrb_from_D(D_psi, J)
        c_th, r_th = ccrb_from_D(D_theta, J)
        rel = abs(c_psi - c_th) / max(abs(c_psi), 1e-12)
        assert r_psi == r_th, f"rank changed: {r_psi} vs {r_th}"
        worst = max(worst, rel)
    record("CCRB invariant to psi vs theta parameterization", worst < 1e-8, worst)


def main() -> int:
    print("=" * 78)
    print("INDEPENDENT NUMERICAL AUDIT")
    print("=" * 78)
    part1_score()
    part1_fisher()
    part1_zero_score()
    part2_channel_grad()
    part2_rank_and_block()
    part4_generator_convention()

    from constrained_crlb import jacobian as jac

    # the production path: both CCRB implementations now feed w.psi
    def with_psi(w, N, K):
        return jac([np.asarray(x) for x in w.psi],
                   [np.asarray(x) for x in w.alpha], N, K)

    part4_jacobian(with_psi, "production (w.psi)")

    # negative control: the pre-audit binding must still be detected as wrong
    print("\nPART 4c - negative control: the pre-audit binding (w.theta)")
    ctrl_worst = 0.0
    for trial, N in ((0, 8), (2, 16)):
        w = track_b_world(trial, 30, 5.0, N=N)
        K = w.G.shape[1]
        Dt = jac([np.asarray(x) for x in w.theta],
                 [np.asarray(x) for x in w.alpha], N, K)
        Dp = jac([np.asarray(x) for x in w.psi],
                 [np.asarray(x) for x in w.alpha], N, K)
        ctrl_worst = max(ctrl_worst, float(np.abs(Dt - Dp).max()))
    record("w.theta binding differs from w.psi  [must be LARGE]",
           ctrl_worst > 1e-3, ctrl_worst, "(regression guard)")

    part5_tangent_properties()
    part5_invariance()

    print("\n" + "=" * 78)
    n_fail = sum(1 for _, ok, _, _ in RESULTS if not ok)
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    for name, ok, err, _ in RESULTS:
        if not ok:
            print(f"  FAILED: {name}  (max err {err:.3e})")
    print("=" * 78)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())

"""
Channel estimation algorithms for the Rydberg atomic MIMO receiver.

Implemented (labels follow the implementation plan):

  A1  biased_gs           Cui Alg. 1, exact model, unstructured
  A2  em_gs               Cui Alg. 2, exact model, unstructured
  A3  em_gs(projector=..) Cui's iteration + Xu's geometric projection  <- the method
  A4  xu_gd               Gradient descent on Xu's linearised objective
  A5  linear_ls           Exact closed-form solution of the same linearised problem
  A6  (see structured.py) Gauss-Newton over {theta, alpha}

All estimators share the signature ``f(Z, S, B, ...) -> G_hat`` (shape (N, K))
and never see the ground-truth channel.

Model recap
-----------
    Z = |G S + B + W|,   G: (N,K),  S: (K,P),  B: (N,P),  W ~ CN(0, sigma2)

Row n transposes to Cui's detection model with the dictionary

    Cui A^H  <->  S^T        Cui s  <->  g_n        Cui N  <->  P

so that Cui's ``(A A^H)^{-1} A`` becomes ``pinv(S^T)``.  In matrix form the
row-wise least squares collapses to ``G = V @ pinv(S)``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ive

__all__ = ["bessel_ratio", "spectral_init", "biased_gs", "em_gs",
           "xu_gd", "linear_ls", "estimate"]


# ----------------------------------------------------------------------------
# The EM high-pass filter
# ----------------------------------------------------------------------------

def bessel_ratio(x: np.ndarray) -> np.ndarray:
    """R(x) = I1(x) / I0(x), computed stably for large x.

    ``ive(v, x) = Iv(x) * exp(-|x|)``, so the exponential factor cancels in the
    ratio and no overflow occurs.  R is monotone increasing from R(0)=0 toward 1;
    Cui's high-pass interpretation: it suppresses low-SNR observations and
    degenerates to the identity (i.e. biased GS) as SNR grows.
    """
    x = np.abs(np.asarray(x, dtype=float))
    num, den = ive(1, x), ive(0, x)
    return np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)


# ----------------------------------------------------------------------------
# Spectral initialization (Cui Alg. 1/2, steps 1-4), batched over elements
# ----------------------------------------------------------------------------

def spectral_init(Z: np.ndarray, S: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Spectral initializer for all N elements at once.

    Per element n, Cui augments the model with the known reference so it becomes
    reference-free:  z_n = |Abar_n^H gbar_n|, with Abar_n^H = [S^T, b_n] and
    gbar_n = [g_n; 1].  The leading eigenvector of  M_n = Abar_n diag(z_n) Abar_n^H
    gives the direction; a scalar least squares gives the magnitude; forcing the
    phase of the last entry to zero removes the global phase.
    """
    N, P = Z.shape
    K = S.shape[0]

    # Abar_n^H = [S^T, b_n]  ->  (N, P, K+1)
    AbarH = np.empty((N, P, K + 1), dtype=complex)
    AbarH[:, :, :K] = S.T[None, :, :]
    AbarH[:, :, K] = B
    Abar = AbarH.conj().transpose(0, 2, 1)              # (N, K+1, P)

    # M_n = Abar_n diag(z_n) Abar_n^H
    M = (Abar * Z[:, None, :]) @ AbarH                  # (N, K+1, K+1)
    M = 0.5 * (M + M.conj().transpose(0, 2, 1))         # symmetrize (numerical)

    w, V = np.linalg.eigh(M)                            # ascending eigenvalues
    v = V[:, :, -1]                                     # (N, K+1) principal

    proj = np.einsum('nkp,nk->np', AbarH.conj().transpose(0, 2, 1), v.conj())
    proj = np.abs(np.einsum('npk,nk->np', AbarH, v))    # |Abar^H v|, shape (N,P)

    num = np.sum(proj * Z, axis=1)
    den = np.sum(proj ** 2, axis=1)
    rbar = num / np.maximum(den, 1e-300)                # (N,)

    gbar = rbar[:, None] * v                            # (N, K+1)
    phase = np.exp(-1j * np.angle(gbar[:, K]))          # zero the last entry's phase
    return gbar[:, :K] * phase[:, None]                 # (N, K)


# ----------------------------------------------------------------------------
# Cui's algorithms, with optional geometric projection
# ----------------------------------------------------------------------------

def _gs_core(Z, S, B, sigma2, use_em, n_iter, projector, L, damping,
             G0, tol, track):
    """Shared iteration for biased GS (use_em=False) and EM-GS (use_em=True)."""
    N, P = Z.shape
    K = S.shape[0]

    pinvS = np.linalg.pinv(S)                           # (P, K), computed once
    G = spectral_init(Z, S, B) if G0 is None else G0.astype(complex).copy()

    history = []
    for _ in range(n_iter):
        G_prev = G

        Lam = G @ S + B                                 # (N, P)
        V = Z * np.exp(1j * np.angle(Lam))
        if use_em:
            V = V * bessel_ratio((2.0 / sigma2) * Z * np.abs(Lam))
        V = V - B

        G_new = V @ pinvS                               # unstructured LS step

        if damping < 1.0:
            G_new = (1.0 - damping) * G_prev + damping * G_new

        if projector is not None:                       # <- Xu's structure
            for k in range(K):
                G_new[:, k] = projector(G_new[:, k], int(L[k]))

        G = G_new
        if track:
            history.append(np.sum((Z - np.abs(G @ S + B)) ** 2))
        if np.linalg.norm(G - G_prev) ** 2 < tol * max(np.linalg.norm(G_prev) ** 2, 1e-30):
            break

    return (G, np.array(history)) if track else G


def biased_gs(Z, S, B, sigma2=None, n_iter=50, projector=None, L=None,
              damping=1.0, G0=None, tol=1e-12, track=False):
    """A1 -- Cui Algorithm 1 (least squares / biased Gerchberg-Saxton).

    Set ``projector`` (and ``L``) to obtain the structured variant.
    """
    return _gs_core(Z, S, B, sigma2, False, n_iter, projector, L,
                    damping, G0, tol, track)


def em_gs(Z, S, B, sigma2, n_iter=50, projector=None, L=None,
          damping=1.0, G0=None, tol=1e-12, track=False):
    """A2 / A3 -- Cui Algorithm 2 (maximum likelihood / EM-GS).

    With ``projector`` supplied this is the proposed method: Cui's exact-model
    iteration constrained at every step to Xu's geometric structure.
    """
    return _gs_core(Z, S, B, sigma2, True, n_iter, projector, L,
                    damping, G0, tol, track)


# ----------------------------------------------------------------------------
# Xu's linearised model (valid only when the reference dominates)
# ----------------------------------------------------------------------------

def _linearize(Z, B):
    """Return (Y, Psi) of the high-RSR expansion  Y ~ Re{Psi o (GS)} + Nbar."""
    Y = Z - np.abs(B)
    Psi = np.exp(-1j * np.angle(B))
    return Y, Psi


def xu_gd(Z, S, B, n_iter=200, step=None, G0=None, projector=None, L=None,
          tol=1e-12, track=False):
    """A4 -- gradient descent on Xu's linearised objective.

        min_G || Y - Re{Psi o (GS)} ||_F^2

    Backtracking line search is used so the comparison against the closed-form
    solution is not confounded by step-size tuning.
    """
    Y, Psi = _linearize(Z, B)
    N, P = Z.shape
    K = S.shape[0]

    G = np.zeros((N, K), dtype=complex) if G0 is None else G0.astype(complex).copy()

    def obj(Gx):
        return np.sum((Y - np.real(Psi * (Gx @ S))) ** 2)

    eta = step if step is not None else 1.0 / max(np.linalg.norm(S, 2) ** 2, 1e-12)
    f = obj(G)
    history = [f]

    for _ in range(n_iter):
        E = Y - np.real(Psi * (G @ S))
        grad = -(E * Psi.conj()) @ S.conj().T           # d/dG* of the objective

        for _ls in range(40):                           # backtracking
            G_new = G - eta * grad
            if projector is not None:
                for k in range(K):
                    G_new[:, k] = projector(G_new[:, k], int(L[k]))
            f_new = obj(G_new)
            if f_new <= f or projector is not None:
                break
            eta *= 0.5

        if abs(f - f_new) < tol * max(abs(f), 1e-30):
            G, f = G_new, f_new
            history.append(f)
            break
        G, f = G_new, f_new
        eta *= 1.1                                      # mild step growth
        if track:
            history.append(f)

    return (G, np.array(history)) if track else G


def linear_ls(Z, S, B):
    """A5 -- exact closed-form solution of the linearised problem.

    SystemModel.pdf section 11: the linearised problem separates across elements,

        y_n = Phi_n gcheck_n + nbar_n,   gcheck_n = [Re g_n; Im g_n] in R^{2K},

    with rows  phi_{n,p} = [Re(psi_{n,p} s_p); -Im(psi_{n,p} s_p)].  Hence

        gcheck_n = (Phi_n^T Phi_n)^{-1} Phi_n^T y_n.

    Any iterative descent on the same objective can at best attain this, which is
    what makes the A4-vs-A5 comparison informative.
    """
    Y, Psi = _linearize(Z, B)
    N, P = Z.shape
    K = S.shape[0]

    # U[n,p,k] = Psi[n,p] * S[k,p]
    U = Psi[:, :, None] * S.T[None, :, :]               # (N, P, K)
    Phi = np.concatenate([U.real, -U.imag], axis=2)     # (N, P, 2K)

    A = Phi.transpose(0, 2, 1) @ Phi                    # (N, 2K, 2K)
    b = np.einsum('npk,np->nk', Phi, Y)                 # (N, 2K)

    gcheck = np.linalg.solve(A, b[:, :, None])[:, :, 0]  # (N, 2K)
    return gcheck[:, :K] + 1j * gcheck[:, K:]


# ----------------------------------------------------------------------------
# Convenience dispatcher
# ----------------------------------------------------------------------------

def estimate(name, real, projector=None, **kwargs):
    """Run an estimator by name on a :class:`Realization`."""
    Z, S, B, s2, L = real.Z, real.S, real.B, real.sigma2, real.L
    name = name.lower()
    if name in ("a1", "gs", "biased_gs"):
        return biased_gs(Z, S, B, s2, projector=projector, L=L, **kwargs)
    if name in ("a2", "a3", "emgs", "em_gs"):
        return em_gs(Z, S, B, s2, projector=projector, L=L, **kwargs)
    if name in ("a4", "xu_gd", "gd"):
        return xu_gd(Z, S, B, projector=projector, L=L, **kwargs)
    if name in ("a5", "linear_ls", "ls"):
        return linear_ls(Z, S, B)
    raise ValueError(f"unknown estimator: {name}")

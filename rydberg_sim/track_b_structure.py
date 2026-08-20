"""ULA structural representations and projections for Track B (Step B3).

This module is **side-effect free** and defines no estimator. It provides
the structural vocabulary that a structure-aware estimator can project
onto, and nothing here ever touches the observation ``Z`` or the exact
forward model. In particular it never linearises anything.

Structure being represented
---------------------------
Each user's channel column is a sum of ``L_k`` complex exponentials in the
element index (SystemModel.pdf §3–5, and :mod:`rydberg_sim.channel`):

    g_k[n] = Σ_ℓ α_{ℓ,k} exp(-j (n-1) ψ_{ℓ,k}),   ψ = π sin θ

so, exactly,

    g_k = A(θ_k) α_k,     A(θ_k) ∈ C^{N × L_k},  α_k ∈ C^{L_k}

Three projections onto that structure are offered, deliberately kept
interchangeable so they can be compared later:

``angular``
    Sparse synthesis on an oversampled angle grid ``D`` (orthogonal
    matching pursuit). Assumes paths lie near grid points.
``hankel``
    Cadzow / structured low-rank: a sum of ``L`` exponentials makes the
    Hankel matrix of ``g`` exactly rank ``L``, whatever the angles are —
    no grid, no on-grid assumption.
``esprit``
    Parametric: recover ``ψ`` by rotational invariance, then re-fit the
    gains by least squares. Off-grid, but commits to a path count.

None of these is claimed as the proposed algorithm. They are the modular
alternatives the prototype in :mod:`rydberg_sim.track_b_prototype`
switches between.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channel import spatial_frequency, steering_matrix

DEFAULT_GRID = 8  # angle-grid oversampling relative to N


# ---------------------------------------------------------------------------
# Representation g_k = A(theta_k) alpha_k
# ---------------------------------------------------------------------------


def synthesize_from_paths(
    theta: np.ndarray, alpha: np.ndarray, N: int
) -> np.ndarray:
    """``g = A(θ) α`` — the exact generative form, for diagnostics and tests."""
    theta = np.asarray(theta, dtype=np.float64).ravel()
    alpha = np.asarray(alpha, dtype=np.complex128).ravel()
    if theta.size != alpha.size:
        raise ValueError(
            f"theta and alpha must match, got {theta.size} and {alpha.size}"
        )
    return steering_matrix(theta, N) @ alpha


def angle_grid(n_grid: int) -> np.ndarray:
    """Angles uniform in ``sin θ`` — i.e. uniform in spatial frequency ψ.

    Uniform-in-ψ is the right grid: the array manifold is a function of
    ``ψ = π sin θ``, so a grid uniform in θ would be dense at broadside and
    sparse at endfire.
    """
    if n_grid < 2:
        raise ValueError(f"n_grid must be >= 2, got {n_grid}")
    sin_theta = np.linspace(-1.0, 1.0, int(n_grid), endpoint=False)
    return np.arcsin(np.clip(sin_theta, -1.0, 1.0))


def angle_dictionary(N: int, n_grid: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Overcomplete dictionary ``D ∈ C^{N × n_grid}`` with unit-norm columns.

    Returns ``(D, theta_grid)``. ``g_k ≈ D x_k`` with ``x_k`` sparse.
    """
    if N < 2:
        raise ValueError(f"N must be >= 2, got {N}")
    n_grid = int(n_grid) if n_grid is not None else DEFAULT_GRID * int(N)
    theta = angle_grid(n_grid)
    D = steering_matrix(theta, N)
    D = D / np.linalg.norm(D, axis=0, keepdims=True)
    return D, theta


# ---------------------------------------------------------------------------
# Projection A — sparse in angle (OMP)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AngularFit:
    g_hat: np.ndarray
    support: np.ndarray
    theta_hat: np.ndarray
    coeffs: np.ndarray
    residual: float


def angular_project(
    g: np.ndarray,
    n_paths: int,
    *,
    D: np.ndarray | None = None,
    theta_grid: np.ndarray | None = None,
    n_grid: int | None = None,
) -> AngularFit:
    """Orthogonal matching pursuit of ``g`` onto the angle dictionary.

    Greedy: repeatedly take the grid atom most correlated with the current
    residual, then re-solve least squares on the whole selected support.
    """
    g = np.asarray(g, dtype=np.complex128).ravel()
    N = g.size
    if D is None or theta_grid is None:
        D, theta_grid = angle_dictionary(N, n_grid)
    n_paths = max(1, min(int(n_paths), N))

    residual = g.copy()
    support: list[int] = []
    coeffs = np.zeros(0, dtype=np.complex128)
    for _ in range(n_paths):
        corr = np.abs(D.conj().T @ residual)
        corr[support] = -np.inf
        j = int(np.argmax(corr))
        support.append(j)
        Dsub = D[:, support]
        coeffs, *_ = np.linalg.lstsq(Dsub, g, rcond=None)
        residual = g - Dsub @ coeffs
    sup = np.asarray(support, dtype=int)
    return AngularFit(
        g_hat=D[:, sup] @ coeffs,
        support=sup,
        theta_hat=np.asarray(theta_grid)[sup],
        coeffs=coeffs,
        residual=float(np.linalg.norm(residual)),
    )


# ---------------------------------------------------------------------------
# Projection B — Hankel structured low rank (Cadzow)
# ---------------------------------------------------------------------------


def hankel_matrix(g: np.ndarray, pencil: int | None = None) -> np.ndarray:
    """Hankel matrix of ``g``; exactly rank ``L`` for ``L`` exponentials."""
    g = np.asarray(g, dtype=np.complex128).ravel()
    N = g.size
    p = int(pencil) if pencil is not None else N // 2
    if not (1 <= p <= N - 1):
        raise ValueError(f"pencil must satisfy 1 <= p <= N-1, got {p}")
    rows = N - p
    return np.lib.stride_tricks.sliding_window_view(g, p + 1)[:rows]


def hankel_to_vector(Hk: np.ndarray) -> np.ndarray:
    """Inverse of :func:`hankel_matrix` by averaging each anti-diagonal."""
    Hk = np.asarray(Hk, dtype=np.complex128)
    rows, cols = Hk.shape
    N = rows + cols - 1
    out = np.zeros(N, dtype=np.complex128)
    count = np.zeros(N, dtype=np.float64)
    ii, jj = np.indices((rows, cols))
    np.add.at(out, (ii + jj).ravel(), Hk.ravel())
    np.add.at(count, (ii + jj).ravel(), 1.0)
    return out / count


def hankel_rank(g: np.ndarray, *, tol: float = 1e-8, pencil: int | None = None) -> int:
    """Numerical rank of the Hankel matrix — the effective path count."""
    s = np.linalg.svd(hankel_matrix(g, pencil), compute_uv=False)
    if s.size == 0 or s[0] == 0.0:
        return 0
    return int(np.sum(s > tol * s[0]))


def estimate_order(
    g: np.ndarray,
    *,
    pencil: int | None = None,
    max_order: int | None = None,
    method: str = "mdl",
) -> int:
    """Estimate the number of exponentials in ``g`` from its Hankel spectrum.

    Assuming a fixed path count is the dominant failure mode of every
    projection here: under-modelling is catastrophic and over-modelling
    degrades to a no-op, so the order must come from the data.

    ``method="mdl"`` applies the Wax–Kailath minimum-description-length
    criterion to the Hankel singular values; ``method="gap"`` takes the
    largest multiplicative gap in the spectrum.
    """
    Hk = hankel_matrix(g, pencil)
    s = np.linalg.svd(Hk, compute_uv=False)
    n = s.size
    cap = min(int(max_order) if max_order is not None else n - 1, n - 1)
    if cap < 1 or s[0] <= 0.0:
        return 1
    lam = np.maximum(s**2, np.finfo(float).tiny)
    if method == "gap":
        ratios = lam[:cap] / lam[1:cap + 1]
        return int(np.argmax(ratios) + 1)
    if method != "mdl":
        raise ValueError(f"unknown order method {method!r}")
    rows = Hk.shape[0]
    best_k, best_val = 1, np.inf
    for k in range(0, cap + 1):
        tail = lam[k:]
        m = tail.size
        if m <= 0:
            break
        geo = np.exp(np.mean(np.log(tail)))
        ari = np.mean(tail)
        if ari <= 0:
            continue
        ll = -rows * m * np.log(geo / ari)
        pen = 0.5 * k * (2 * n - k) * np.log(rows)
        val = ll + pen
        if val < best_val:
            best_val, best_k = val, max(1, k)
    return int(best_k)


def hankel_project(
    g: np.ndarray,
    rank: int,
    *,
    pencil: int | None = None,
    n_iter: int = 8,
) -> np.ndarray:
    """Cadzow: alternate truncation to ``rank`` with re-Hankelisation.

    Grid-free — it enforces "``g`` is a sum of ``rank`` exponentials"
    without ever naming the angles.
    """
    g = np.asarray(g, dtype=np.complex128).ravel()
    rank = max(1, int(rank))
    cur = g.copy()
    for _ in range(max(1, int(n_iter))):
        Hk = hankel_matrix(cur, pencil)
        U, s, Vh = np.linalg.svd(Hk, full_matrices=False)
        r = min(rank, s.size)
        Hk_lr = (U[:, :r] * s[:r]) @ Vh[:r]
        cur = hankel_to_vector(Hk_lr)
    return cur


# ---------------------------------------------------------------------------
# Projection C — ESPRIT (parametric, off-grid)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EspritFit:
    g_hat: np.ndarray
    psi_hat: np.ndarray
    theta_hat: np.ndarray
    alpha_hat: np.ndarray


def esprit_project(
    g: np.ndarray, n_paths: int, *, pencil: int | None = None
) -> EspritFit:
    """Estimate ``ψ`` by rotational invariance, then re-fit gains by LS.

    ``ψ`` is recovered modulo 2π and mapped back through
    ``θ = arcsin(ψ/π)``; values outside the visible region are clipped,
    which is the standard ULA convention.
    """
    g = np.asarray(g, dtype=np.complex128).ravel()
    N = g.size
    n_paths = max(1, min(int(n_paths), N // 2))
    Hk = hankel_matrix(g, pencil)
    U, _s, _Vh = np.linalg.svd(Hk, full_matrices=False)
    Us = U[:, :n_paths]
    if Us.shape[0] < 2:
        raise ValueError("not enough Hankel rows for ESPRIT")
    Phi, *_ = np.linalg.lstsq(Us[:-1], Us[1:], rcond=None)
    eig = np.linalg.eigvals(Phi)
    # generator uses exp(-j psi) per element step
    psi = -np.angle(eig)
    psi = np.mod(psi + np.pi, 2.0 * np.pi) - np.pi
    theta = np.arcsin(np.clip(psi / np.pi, -1.0, 1.0))
    A = steering_matrix(theta, N)
    alpha, *_ = np.linalg.lstsq(A, g, rcond=None)
    return EspritFit(g_hat=A @ alpha, psi_hat=psi, theta_hat=theta, alpha_hat=alpha)


# ---------------------------------------------------------------------------
# Uniform entry point
# ---------------------------------------------------------------------------

PROJECTIONS = ("none", "angular", "hankel", "esprit")


def project_column(
    g: np.ndarray,
    method: str,
    n_paths: int,
    *,
    D: np.ndarray | None = None,
    theta_grid: np.ndarray | None = None,
    cadzow_iter: int = 8,
) -> np.ndarray:
    """Project one channel column onto the ULA multipath manifold."""
    if method == "none":
        return np.asarray(g, dtype=np.complex128).ravel()
    if method == "angular":
        return angular_project(g, n_paths, D=D, theta_grid=theta_grid).g_hat
    if method == "hankel":
        return hankel_project(g, n_paths, n_iter=cadzow_iter)
    if method == "esprit":
        return esprit_project(g, n_paths).g_hat
    raise ValueError(f"unknown projection {method!r}; choose from {PROJECTIONS}")


def project_matrix(
    G: np.ndarray,
    method: str,
    n_paths: int | np.ndarray | str,
    *,
    n_grid: int | None = None,
    cadzow_iter: int = 8,
    max_order: int | None = None,
) -> np.ndarray:
    """Column-wise :func:`project_column` over ``G ∈ C^{N × K}``.

    ``n_paths="auto"`` selects each column's order with
    :func:`estimate_order` instead of assuming one.
    """
    G = np.asarray(G, dtype=np.complex128)
    if G.ndim != 2:
        raise ValueError(f"G must be 2-D (N, K), got {G.shape}")
    N, K = G.shape
    D = theta_grid = None
    if method == "angular":
        D, theta_grid = angle_dictionary(N, n_grid)
    if isinstance(n_paths, str):
        if n_paths != "auto":
            raise ValueError(f"n_paths must be an int, array, or 'auto'")
        counts = np.array([estimate_order(G[:, k], max_order=max_order)
                           for k in range(K)], dtype=int)
    else:
        counts = (np.full(K, int(n_paths)) if np.isscalar(n_paths)
                  else np.asarray(n_paths, dtype=int).ravel())
    if counts.size != K:
        raise ValueError(f"n_paths must be scalar or length-K, got {counts.size}")
    out = np.empty_like(G)
    for k in range(K):
        out[:, k] = project_column(
            G[:, k], method, int(counts[k]), D=D, theta_grid=theta_grid,
            cadzow_iter=cadzow_iter,
        )
    return out


__all__ = [
    "AngularFit", "EspritFit", "PROJECTIONS", "angle_dictionary", "angle_grid",
    "angular_project", "esprit_project", "estimate_order", "hankel_matrix",
    "hankel_project",
    "hankel_rank", "hankel_to_vector", "project_column", "project_matrix",
    "spatial_frequency", "synthesize_from_paths",
]

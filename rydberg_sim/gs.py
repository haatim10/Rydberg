"""Cui biased Gerchberg–Saxton (Algorithm 1) in the canonical convention.

Canonical model
---------------
    z = |M^H u + b + w|

    M ∈ C^{D × Q},   u ∈ C^D,   z, b, w length Q
    (z is stored as a real nonnegative amplitude)

This solver is completely generic. It does **not** know whether ``u`` is
Cui's unknown symbol vector or ``conj(g_n)`` from the channel-estimation
adapter. Those mappings live in :func:`biased_gs_channel_rows`.

Algorithm (Cui Alg. 1, biased GS)
---------------------------------
Default initialization is the Step-8 spectral initializer
(:func:`rydberg_sim.spectral.spectral_initialize`). An explicit ``u0``
may be supplied for tests (spectral vs random). Spectral remains the
production default.

At iteration ``t = 1, …, max_iter``, given ``u^{t-1}``::

    λ^{t-1} = M^H u^{t-1} + b
    θ^t     = angle(λ^{t-1})
    y^t     = z ⊙ exp(1j θ^t)          # measured |z|, estimated phase only
    r^t     = y^t - b
    u^t     = argmin_u ||M^H u - r^t||_2^2

For full-row-rank ``M`` the LS step is the normal equation

    (M M^H) u^t = M r^t

solved with ``np.linalg.solve`` (no explicit inverse). If ``ridge > 0``::

    (M M^H + ridge I) u^t = M r^t

``M`` is constant, so ``M^H`` and ``M M^H`` are formed once. Exactly
``max_iter`` iterations are run; there is no extra early-stopping rule.
``max_iter`` is a required argument (Table I later uses 50; that value
is not hard-coded here).

The magnitude-domain objective recorded after initialization and after
every update is

    J(u) = ||z - |M^H u + b|||_2^2

What this module does **not** implement (Step 10+)
-------------------------------------------------
EM-GS, Bessel ratio I1/I0, CRLB, GD/PGD, Monte Carlo figure sweeps, BER,
QAM projection inside the iteration. Optional nearest-neighbour QAM
projection is a separate detection-layer helper and is never called by
the continuous solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .qam import QAMConstellation, build_qam_constellation
from .spectral import spectral_initialize

DEFAULT_RIDGE = 0.0


@dataclass(frozen=True, eq=False)
class BiasedGSResult:
    """Canonical biased-GS output and debugging diagnostics.

    Attributes
    ----------
    u_hat
        Estimate after ``max_iter`` updates, shape ``(D,)``.
    u0
        Initialization actually used, shape ``(D,)``.
    objective_history
        ``[J_0, J_1, …, J_{max_iter}]``, length ``max_iter + 1``.
    n_iter
        Number of GS updates performed (equals ``max_iter``).
    ridge
        Tikhonov parameter actually used.
    regularization_used
        ``True`` iff ``ridge > 0``.
    init_source
        ``"spectral"`` or ``"given"``.
    iterates
        ``(max_iter + 1, D)`` if ``store_iterates`` was set, else ``None``.
        Row 0 is ``u0``; row t is the estimate after iteration t.
    """

    u_hat: np.ndarray
    u0: np.ndarray
    objective_history: np.ndarray
    n_iter: int
    ridge: float
    regularization_used: bool
    init_source: Literal["spectral", "given"]
    iterates: np.ndarray | None


@dataclass(frozen=True, eq=False)
class ChannelBiasedGSResult:
    """Per-element channel-row adapter around :func:`biased_gs`.

    Physical model for receive element ``n``::

        z_n = |S^T g_n + b_n + w_n|

    Canonical mapping (conjugation of the whole observation)::

        M = S
        u = conj(g_n)
        b_solver = conj(b_n)

    Then ``G_hat[n] = conj(u_hat)``. Each row calls the **same** canonical
    :func:`biased_gs`, with its own ``z_n``, ``b_n``, and therefore its
    own Step-8 spectral initialization. ``M = S`` is shared; one
    initializer is never reused across ``N``.
    """

    G_hat: np.ndarray
    G0: np.ndarray
    row_results: tuple[BiasedGSResult, ...]


def _require_finite(arr: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")


def _as_complex_matrix(value: object, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.complex128)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {arr.shape}")
    _require_finite(arr, name)
    return np.array(arr, dtype=np.complex128, copy=True)


def _as_complex_vector(value: object, name: str, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.complex128)
    if arr.ndim == 2 and arr.shape in ((length, 1), (1, length)):
        arr = arr.reshape(length)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size != length:
        raise ValueError(f"{name} must have length {length}, got {arr.size}")
    _require_finite(arr, name)
    return np.array(arr, dtype=np.complex128, copy=True)


def _as_real_vector(value: object, name: str, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 2 and arr.shape in ((length, 1), (1, length)):
        arr = arr.reshape(length)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size != length:
        raise ValueError(f"{name} must have length {length}, got {arr.size}")
    _require_finite(arr, name)
    return np.array(arr, dtype=np.float64, copy=True)


def _as_ridge(value: object) -> float:
    try:
        ridge = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"ridge must be a real number, got {value!r}") from exc
    if not np.isfinite(ridge):
        raise ValueError(f"ridge must be finite, got {value!r}")
    if ridge < 0.0:
        raise ValueError(f"ridge must be >= 0, got {ridge}")
    return ridge


def _as_max_iter(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value:
        raise TypeError(f"max_iter must be an integer, got {value!r}")
    max_iter = int(value)
    if max_iter <= 0:
        raise ValueError(f"max_iter must be > 0, got {max_iter}")
    return max_iter


def magnitude_objective(
    M: np.ndarray,
    u: np.ndarray,
    b: np.ndarray,
    z: np.ndarray,
    *,
    M_H: np.ndarray | None = None,
) -> float:
    """``J(u) = ||z - |M^H u + b|||_2^2``."""
    M_arr = _as_complex_matrix(M, "M")
    d, q = M_arr.shape
    u_arr = _as_complex_vector(u, "u", d)
    b_arr = _as_complex_vector(b, "b", q)
    z_arr = _as_real_vector(z, "z", q)
    mh = M_arr.conj().T if M_H is None else M_H
    pred = np.abs(mh @ u_arr + b_arr)
    return float(np.sum((z_arr - pred) ** 2))


def random_complex_initialization(
    D: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Unit-energy CN random vector. **Not** the production default.

    Uses an explicit ``numpy.random.Generator``. Callers who want
    trial-addressable draws should pass
    ``get_trial_rngs(master_seed, trial_index).solver``. That stream is
    the *sixth* SeedSequence child and does not retune channel, pilots,
    reference, noise, or data.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy Generator, got {type(rng)!r}")
    if isinstance(D, (bool, np.bool_)) or int(D) != D:
        raise TypeError(f"D must be an integer, got {D!r}")
    d = int(D)
    if d < 1:
        raise ValueError(f"D must be >= 1, got {d}")
    real = rng.standard_normal(d)
    imag = rng.standard_normal(d)
    vec = (real + 1j * imag).astype(np.complex128, copy=False)
    nrm = float(np.linalg.norm(vec))
    if nrm == 0.0:
        vec = np.ones(d, dtype=np.complex128)
        nrm = float(np.linalg.norm(vec))
    return vec / nrm


def project_to_qam(
    u: np.ndarray,
    constellation: QAMConstellation | int,
) -> np.ndarray:
    """Optional detection-layer nearest-neighbour projection onto Step-4 QAM.

    **Not** part of the continuous biased-GS iteration. Cui Algorithm 1
    as implemented here is the unconstrained LS update; call this only
    after the iterations if a discrete symbol estimate is required.
    """
    const = (
        constellation
        if isinstance(constellation, QAMConstellation)
        else build_qam_constellation(int(constellation))
    )
    u_arr = np.asarray(u, dtype=np.complex128).reshape(-1)
    _require_finite(u_arr, "u")
    delta = u_arr[:, np.newaxis] - const.points[np.newaxis, :]
    idx = np.argmin(np.abs(delta), axis=1)
    return const.points[idx].astype(np.complex128, copy=False)


def _solve_ls(A: np.ndarray, rhs: np.ndarray, ridge: float) -> np.ndarray:
    """Solve ``(A + ridge I) u = rhs`` without forming an explicit inverse."""
    gram = np.array(A, dtype=np.complex128, copy=True)
    d = gram.shape[0]
    if ridge != 0.0:
        gram.flat[:: d + 1] += ridge
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "M M^H is singular with the requested ridge; biased GS needs "
            "full-row-rank M or ridge > 0. Refusing to switch algorithms."
        ) from exc


def biased_gs(
    M: np.ndarray,
    z: np.ndarray,
    b: np.ndarray,
    *,
    max_iter: int,
    u0: np.ndarray | None = None,
    ridge: float = DEFAULT_RIDGE,
    store_iterates: bool = False,
) -> BiasedGSResult:
    """Cui biased GS (Algorithm 1) for ``z = |M^H u + b + w|``.

    Parameters
    ----------
    M, z, b
        Canonical dictionary, observed amplitudes, known reference.
        Shapes ``(D, Q)``, ``(Q,)``, ``(Q,)``.
    max_iter
        Number of GS updates. Required. Run exactly this many iterations.
    u0
        Optional initialization of shape ``(D,)``. Default is Step-8
        spectral initialization. Random initialization is **not** the
        default; pass an explicit vector from
        :func:`random_complex_initialization` for the comparison test.
    ridge
        Optional Tikhonov parameter, default ``0`` (no regularisation).
        Nonzero ridge is never applied silently.
    store_iterates
        If True, also store every ``u`` including ``u0``.
    """
    M_arr = _as_complex_matrix(M, "M")
    d, q = M_arr.shape
    z_arr = _as_real_vector(z, "z", q)
    b_arr = _as_complex_vector(b, "b", q)
    if np.any(z_arr < 0.0):
        raise ValueError("z must be nonnegative amplitudes")
    n_iter = _as_max_iter(max_iter)
    ridge_val = _as_ridge(ridge)

    if u0 is None:
        u = spectral_initialize(M_arr, z_arr, b_arr).u0
        init_source: Literal["spectral", "given"] = "spectral"
    else:
        u = _as_complex_vector(u0, "u0", d)
        init_source = "given"

    # Precompute once: M does not change across iterations.
    m_h = M_arr.conj().T
    gram = M_arr @ m_h
    if ridge_val != 0.0:
        gram = gram + ridge_val * np.eye(d, dtype=np.complex128)

    history = np.empty(n_iter + 1, dtype=np.float64)
    history[0] = magnitude_objective(M_arr, u, b_arr, z_arr, M_H=m_h)
    iterates: np.ndarray | None
    if store_iterates:
        iterates = np.empty((n_iter + 1, d), dtype=np.complex128)
        iterates[0] = u
    else:
        iterates = None

    u_work = np.array(u, dtype=np.complex128, copy=True)
    for t in range(1, n_iter + 1):
        lam = m_h @ u_work + b_arr
        # Keep measured magnitudes z; update phases only.
        y = z_arr * np.exp(1j * np.angle(lam))
        r = y - b_arr
        rhs = M_arr @ r
        u_work = _solve_ls(gram, rhs, ridge=0.0)
        history[t] = magnitude_objective(M_arr, u_work, b_arr, z_arr, M_H=m_h)
        if iterates is not None:
            iterates[t] = u_work

    return BiasedGSResult(
        u_hat=np.asarray(u_work, dtype=np.complex128).reshape(d),
        u0=np.asarray(u, dtype=np.complex128).reshape(d),
        objective_history=history,
        n_iter=n_iter,
        ridge=ridge_val,
        regularization_used=ridge_val > 0.0,
        init_source=init_source,
        iterates=iterates,
    )


def biased_gs_channel_rows(
    S: np.ndarray,
    Z: np.ndarray,
    B: np.ndarray,
    *,
    max_iter: int,
    ridge: float = DEFAULT_RIDGE,
    G0: np.ndarray | None = None,
    store_iterates: bool = False,
) -> ChannelBiasedGSResult:
    """Channel-estimation adapter: loop over receive elements.

    Does **not** duplicate GS equations. For each ``n`` it calls
    :func:`biased_gs` with ``M = S``, ``z = Z[n]``,
    ``b = conj(B[n])``, then ``G_hat[n] = conj(u_hat)``.

    If ``G0`` is omitted, every row uses its own Step-8 spectral
    initialization (different ``z_n``, ``b_n`` ⇒ different ``M_spec``).
    """
    S_arr = _as_complex_matrix(S, "S")
    Z_arr = np.asarray(Z, dtype=np.float64)
    B_arr = _as_complex_matrix(B, "B")
    if Z_arr.ndim != 2:
        raise ValueError(f"Z must be 2-D (N, P), got shape {Z_arr.shape}")
    _require_finite(Z_arr, "Z")
    n_rx, n_pilots = Z_arr.shape
    n_users, p_s = S_arr.shape
    if p_s != n_pilots:
        raise ValueError(
            f"incompatible Z and S: Z.shape={Z_arr.shape}, S.shape={S_arr.shape}"
        )
    if B_arr.shape != (n_rx, n_pilots):
        raise ValueError(
            f"incompatible B: B.shape={B_arr.shape}, expected {(n_rx, n_pilots)}"
        )
    g0_arr: np.ndarray | None
    if G0 is None:
        g0_arr = None
    else:
        g0_arr = _as_complex_matrix(G0, "G0")
        if g0_arr.shape != (n_rx, n_users):
            raise ValueError(
                f"G0.shape={g0_arr.shape}, expected {(n_rx, n_users)}"
            )

    G_hat = np.empty((n_rx, n_users), dtype=np.complex128)
    G0_out = np.empty((n_rx, n_users), dtype=np.complex128)
    rows: list[BiasedGSResult] = []
    for n in range(n_rx):
        u0_n = None if g0_arr is None else np.conjugate(g0_arr[n])
        row = biased_gs(
            S_arr,
            Z_arr[n],
            np.conjugate(B_arr[n]),
            max_iter=max_iter,
            u0=u0_n,
            ridge=ridge,
            store_iterates=store_iterates,
        )
        G_hat[n] = np.conjugate(row.u_hat)
        G0_out[n] = np.conjugate(row.u0)
        rows.append(row)
    return ChannelBiasedGSResult(
        G_hat=G_hat,
        G0=G0_out,
        row_results=tuple(rows),
    )

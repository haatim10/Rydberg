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

Algorithm 2 (EM-GS) uses the same phase and LS steps, but multiplies the
restored observation by the Bessel ratio ``R(κ) = I₁(κ)/I₀(κ)`` with

    κ = (2 / σ²) z ⊙ |λ|.

``bessel_ratio`` is computed via ``scipy.special.ive`` (and an
asymptotic tail for ``κ > 1e4``). It is **not** ``i1/i0``.

What this module does **not** implement (Step 15+)
-------------------------------------------------
Monte Carlo figure sweeps, Track-C execution, machine learning. Cui's
CRLB lives in :mod:`rydberg_sim.crlb`. QAM projection is not applied
inside GS or EM-GS iterations; :func:`rydberg_sim.qam.project_to_qam`
is re-exported here for the optional detection-layer helper.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.special import ive

from .baselines import rician_log_likelihood
from .qam import project_to_qam
from .spectral import spectral_initialize

DEFAULT_RIDGE = 0.0
# Plan: use 1 - 1/(2x) - 1/(8x^2) for x > 1e4 (high-SNR kappa).
BESSEL_RATIO_ASYMP_X = 1.0e4
IVE_DENOM_FLOOR = 1.0e-300


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
            "M M^H is singular with the requested ridge; GS/EM-GS needs "
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


# ---------------------------------------------------------------------------
# Bessel ratio R(x) = I1(x)/I0(x)  (Cui Alg. 2)
# ---------------------------------------------------------------------------

def _bessel_ratio_asymptotic(x: np.ndarray) -> np.ndarray:
    """``1 - 1/(2x) - 1/(8x^2)`` for large positive ``x``."""
    return 1.0 - 1.0 / (2.0 * x) - 1.0 / (8.0 * x * x)


def bessel_ratio(x: np.ndarray | float) -> np.ndarray:
    """Stable ``R(x) = I_1(x) / I_0(x)`` for ``x >= 0``.

    Uses ``scipy.special.ive`` so the shared ``exp(-|x|)`` factor in
    ``I_ν(x)`` cancels. Do **not** form ``i1(x)/i0(x)``: both overflow
    for realistic high-SNR ``κ``.

    For ``x > 1e4`` the implementation plan's expansion
    ``R(x) ≈ 1 - 1/(2x) - 1/(8x^2)`` is used. ``R(0) = 0`` exactly.
    Output lies in ``[0, 1]``.
    """
    arr = np.asarray(x, dtype=np.float64)
    orig_shape = arr.shape
    ax = np.abs(arr).reshape(-1)
    out = np.empty(ax.shape, dtype=np.float64)
    large = ax > BESSEL_RATIO_ASYMP_X
    small = ~large
    if np.any(small):
        xs = ax[small]
        num = ive(1, xs)
        den = ive(0, xs)
        ratio = np.zeros(xs.shape, dtype=np.float64)
        finite_den = np.isfinite(den) & (den > 0.0)
        ratio[finite_den] = num[finite_den] / np.maximum(
            den[finite_den], IVE_DENOM_FLOOR
        )
        # Exact identity: I1(0)=0, I0(0)=1.
        ratio[xs == 0.0] = 0.0
        out[small] = ratio
    if np.any(large):
        out[large] = _bessel_ratio_asymptotic(ax[large])
    np.clip(out, 0.0, 1.0, out=out)
    return out.reshape(orig_shape)


def _as_sigma2_positive(value: object) -> float:
    try:
        sigma2 = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"sigma2 must be a real number, got {value!r}") from exc
    if not np.isfinite(sigma2):
        raise ValueError(f"sigma2 must be finite, got {value!r}")
    if sigma2 <= 0.0:
        raise ValueError(
            f"sigma2 must be > 0 for EM-GS, got {sigma2}; "
            "κ contains 1/sigma2. Use biased GS for the noiseless problem."
        )
    return sigma2


def em_kappa(z: np.ndarray, lam: np.ndarray, sigma2: float) -> np.ndarray:
    """``κ = (2 / σ²) z ⊙ |λ|``. Factor 2 and ``sigma2`` (not ``sigma``)."""
    sigma2_val = _as_sigma2_positive(sigma2)
    z_arr = np.asarray(z, dtype=np.float64)
    lam_arr = np.asarray(lam)
    if z_arr.shape != np.shape(lam_arr):
        raise ValueError(
            f"z and lambda shapes must match, got {z_arr.shape} and {np.shape(lam_arr)}"
        )
    return (2.0 / sigma2_val) * z_arr * np.abs(lam_arr)


@dataclass(frozen=True, eq=False)
class EMGSResult:
    """Canonical EM-GS (Algorithm 2) output.

    ``objective_history`` is the magnitude LS ``J(u)``. It is **not**
    required to be monotone: EM-GS maximises a Rician likelihood, not
    ``J``. ``loglik_history`` uses the Step-7
    :func:`~rydberg_sim.baselines.rician_log_likelihood` convention.
    """

    u_hat: np.ndarray
    u0: np.ndarray
    n_iter: int
    sigma2: float
    ridge: float
    regularization_used: bool
    init_source: Literal["spectral", "given"]
    objective_history: np.ndarray
    loglik_history: np.ndarray
    kappa_mean: np.ndarray
    kappa_final: np.ndarray
    iterates: np.ndarray | None


@dataclass(frozen=True, eq=False)
class ChannelEMGSResult:
    """Per-element channel-row adapter around :func:`em_gs`."""

    G_hat: np.ndarray
    G0: np.ndarray
    row_results: tuple[EMGSResult, ...]


def em_gs(
    M: np.ndarray,
    z: np.ndarray,
    b: np.ndarray,
    sigma2: float,
    *,
    max_iter: int,
    u0: np.ndarray | None = None,
    ridge: float = DEFAULT_RIDGE,
    store_iterates: bool = False,
) -> EMGSResult:
    """Cui EM-GS (Algorithm 2) for ``z = |M^H u + b + w|``.

    Same phase and LS update as :func:`biased_gs`, with the extra
    Bessel weight::

        λ = M^H u + b
        κ = (2 / σ²) z ⊙ |λ|
        y_EM = z ⊙ exp(1j angle(λ)) ⊙ R(κ)
        r = y_EM - b
        (M M^H + ridge I) u_new = M r

    ``sigma2 > 0`` is required. This function has no channel or QAM
    special case.
    """
    M_arr = _as_complex_matrix(M, "M")
    d, q = M_arr.shape
    z_arr = _as_real_vector(z, "z", q)
    b_arr = _as_complex_vector(b, "b", q)
    if np.any(z_arr < 0.0):
        raise ValueError("z must be nonnegative amplitudes")
    sigma2_val = _as_sigma2_positive(sigma2)
    n_iter = _as_max_iter(max_iter)
    ridge_val = _as_ridge(ridge)

    if u0 is None:
        u = spectral_initialize(M_arr, z_arr, b_arr).u0
        init_source: Literal["spectral", "given"] = "spectral"
    else:
        u = _as_complex_vector(u0, "u0", d)
        init_source = "given"

    m_h = M_arr.conj().T
    gram = M_arr @ m_h
    if ridge_val != 0.0:
        gram = gram + ridge_val * np.eye(d, dtype=np.complex128)

    def _lam(u_vec: np.ndarray) -> np.ndarray:
        return m_h @ u_vec + b_arr

    def _ll(u_vec: np.ndarray) -> float:
        return rician_log_likelihood(z_arr, _lam(u_vec), sigma2_val)

    history = np.empty(n_iter + 1, dtype=np.float64)
    loglik = np.empty(n_iter + 1, dtype=np.float64)
    kappa_mean = np.empty(n_iter, dtype=np.float64)
    history[0] = magnitude_objective(M_arr, u, b_arr, z_arr, M_H=m_h)
    loglik[0] = _ll(u)
    iterates: np.ndarray | None
    if store_iterates:
        iterates = np.empty((n_iter + 1, d), dtype=np.complex128)
        iterates[0] = u
    else:
        iterates = None

    u_work = np.array(u, dtype=np.complex128, copy=True)
    kappa = np.empty(q, dtype=np.float64)
    for t in range(1, n_iter + 1):
        lam = _lam(u_work)
        # Same phase as biased GS; extra R(κ) weight is the EM correction.
        kappa = em_kappa(z_arr, lam, sigma2_val)
        kappa_mean[t - 1] = float(np.mean(kappa))
        y_em = z_arr * np.exp(1j * np.angle(lam)) * bessel_ratio(kappa)
        r = y_em - b_arr
        u_work = _solve_ls(gram, M_arr @ r, ridge=0.0)
        history[t] = magnitude_objective(M_arr, u_work, b_arr, z_arr, M_H=m_h)
        loglik[t] = _ll(u_work)
        if iterates is not None:
            iterates[t] = u_work

    kappa_final = em_kappa(z_arr, _lam(u_work), sigma2_val)
    return EMGSResult(
        u_hat=np.asarray(u_work, dtype=np.complex128).reshape(d),
        u0=np.asarray(u, dtype=np.complex128).reshape(d),
        n_iter=n_iter,
        sigma2=sigma2_val,
        ridge=ridge_val,
        regularization_used=ridge_val > 0.0,
        init_source=init_source,
        objective_history=history,
        loglik_history=loglik,
        kappa_mean=kappa_mean,
        kappa_final=kappa_final,
        iterates=iterates,
    )


def em_gs_channel_rows(
    S: np.ndarray,
    Z: np.ndarray,
    B: np.ndarray,
    sigma2: float,
    *,
    max_iter: int,
    ridge: float = DEFAULT_RIDGE,
    G0: np.ndarray | None = None,
    store_iterates: bool = False,
) -> ChannelEMGSResult:
    """Channel-estimation adapter: loop of canonical :func:`em_gs`.

    ``M = S``, ``b_solver = conj(B[n])``, ``G_hat[n] = conj(u_hat)``.
    Does not copy EM equations. Each row has its own spectral init.
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
    rows: list[EMGSResult] = []
    for n in range(n_rx):
        u0_n = None if g0_arr is None else np.conjugate(g0_arr[n])
        row = em_gs(
            S_arr,
            Z_arr[n],
            np.conjugate(B_arr[n]),
            sigma2,
            max_iter=max_iter,
            u0=u0_n,
            ridge=ridge,
            store_iterates=store_iterates,
        )
        G_hat[n] = np.conjugate(row.u_hat)
        G0_out[n] = np.conjugate(row.u0)
        rows.append(row)
    return ChannelEMGSResult(
        G_hat=G_hat,
        G0=G0_out,
        row_results=tuple(rows),
    )


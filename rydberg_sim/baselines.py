"""Debugging / reference baselines (Step 7).

This module is **not** the future GS / EM-GS solver. Keep Gerchberg–Saxton,
spectral initialization, EM, CRLB, and gradient methods out of here.

The estimators below are reference solutions that later iterative algorithms
must be checked against. Some of them are *not* valid magnitude-only
receivers (the ZF genie is given the true phase).

Canonical biased phase-retrieval form
-------------------------------------
    z = |M^H u + b + w|

with explicit dimensions, validated on every call:

    M ∈ C^{D × Q}
    u ∈ C^D
    z, b, w ∈ C^Q   (z is stored as a real nonnegative amplitude)

Channel-estimation mapping (one receive row): ``u = g_n`` (so ``D = K``)
and ``M^H = S^T`` i.e. ``M = conj(S)`` with ``Q = P``.

What is implemented
-------------------
A. Genie-aided ZF with known phase (not a valid z-only estimator).
B. Closed-form least squares for the Section 11 linearised model.
C. CM-ZF: **not implemented** (Cui's channel-magnitude ZF is not specified
   in the available source; do not invent an approximation).
D. Exhaustive QAM search for Track-A *detection* (not channel estimation):
   magnitude-domain LS, and Rician ML for the Step-5 observation model.

What is **not** implemented (Step 13+)
-------------------------------------
GD/PGD, Monte Carlo estimator sweeps, figures, BER experiments.
Cui CRLB lives in :mod:`rydberg_sim.crlb`. The linearised channel CRLB
lives in :mod:`rydberg_sim.linearised_crlb` (derived, not copied from Xu).

Future acceptance test (do **not** implement GD here to satisfy it):

    At high RSR, GD converges to within 1e-6 relative Frobenius distance
    of the closed-form linearised LS solution.

Tikhonov / ridge
----------------
Optional ``ridge >= 0`` is the same documented convention intended for later
solvers: replace ``A x = rhs`` with ``(A + ridge I) x = rhs``. The default
is ``ridge = 0`` (no regularisation). A nonzero ridge is never applied
silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal

import numpy as np
from scipy.special import i0e

from .qam import QAMConstellation, build_qam_constellation

ObservationSource = Literal["exact_magnitude", "ideal_linear"]

DEFAULT_MAX_CANDIDATES = 65_536
DEFAULT_RIDGE = 0.0

# Marked for a later step. This module must not grow a GD implementation
# just to satisfy that comparison.
FUTURE_GD_VS_CLOSED_FORM_TEST = (
    "At high RSR, GD converges to within 1e-6 relative Frobenius distance "
    "of the closed-form linearised LS solution. GD is Step 8+; not implemented."
)


class ExhaustiveSearchTooLargeError(ValueError):
    """Raised when ``M_qam ** D`` exceeds the configured candidate cap."""


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
    if value is None:
        raise TypeError(
            "ridge must be an explicit real number; omit the argument to use 0"
        )
    try:
        ridge = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"ridge must be a real number, got {value!r}") from exc
    if not np.isfinite(ridge):
        raise ValueError(f"ridge must be finite, got {value!r}")
    if ridge < 0.0:
        raise ValueError(f"ridge must be >= 0, got {ridge}")
    return ridge


def _as_sigma2_positive(value: object) -> float:
    try:
        sigma2 = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"sigma2 must be a real number, got {value!r}") from exc
    if not np.isfinite(sigma2):
        raise ValueError(f"sigma2 must be finite, got {value!r}")
    if sigma2 <= 0.0:
        raise ValueError(
            f"sigma2 must be > 0 for Rician ML, got {sigma2}; "
            "the noiseless limit is handled by exhaustive LS"
        )
    return sigma2


def _gram_solve(A: np.ndarray, rhs: np.ndarray, ridge: float) -> np.ndarray:
    """Solve ``(A + ridge I) x = rhs`` without forming an explicit inverse."""
    gram = np.array(A, dtype=A.dtype, copy=True)
    if ridge != 0.0:
        n = gram.shape[0]
        gram.flat[:: n + 1] += ridge
    return np.linalg.solve(gram, rhs)


def canonical_M_from_pilots(S: np.ndarray) -> np.ndarray:
    """Dictionary ``M`` such that ``M^H u = S.T @ u`` for a channel row ``u``.

    ``S`` has shape ``(K, P)``. Returns ``M = conj(S)`` of shape ``(K, P)``
    so that ``M^H`` is ``S.T``.
    """
    S_arr = _as_complex_matrix(S, "S")
    return np.conjugate(S_arr)


def pack_gtilde(g: np.ndarray) -> np.ndarray:
    """Stack ``[Re(g); Im(g)]`` into a real vector of length ``2K``."""
    g_arr = np.asarray(g, dtype=np.complex128).reshape(-1)
    _require_finite(g_arr, "g")
    return np.concatenate([g_arr.real, g_arr.imag]).astype(np.float64, copy=False)


def unpack_gtilde(gtilde: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pack_gtilde`."""
    arr = np.asarray(gtilde, dtype=np.float64).reshape(-1)
    _require_finite(arr, "gtilde")
    if arr.size % 2 != 0:
        raise ValueError(f"gtilde length must be even, got {arr.size}")
    k = arr.size // 2
    return (arr[:k] + 1j * arr[k:]).astype(np.complex128, copy=False)


# ---------------------------------------------------------------------------
# Part A — genie-aided ZF with known phase
# ---------------------------------------------------------------------------

def reconstruct_complex_observation(z: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """``z * exp(1j * theta)``. Recovers ``M^H u + b + w`` when ``theta`` is exact."""
    z_arr = np.asarray(z, dtype=np.float64)
    theta_arr = np.asarray(theta, dtype=np.float64)
    if z_arr.shape != theta_arr.shape:
        raise ValueError(
            f"z and theta shapes must match, got {z_arr.shape} and {theta_arr.shape}"
        )
    _require_finite(z_arr, "z")
    _require_finite(theta_arr, "theta")
    return (z_arr * np.exp(1j * theta_arr)).astype(np.complex128, copy=False)


def true_complex_observation(
    M: np.ndarray,
    u: np.ndarray,
    b: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """``lambda = M^H u + b + w``, shape ``(Q,)``."""
    M_arr = _as_complex_matrix(M, "M")
    d, q = M_arr.shape
    u_arr = _as_complex_vector(u, "u", d)
    b_arr = _as_complex_vector(b, "b", q)
    w_arr = _as_complex_vector(w, "w", q)
    return (M_arr.conj().T @ u_arr + b_arr + w_arr).astype(np.complex128, copy=False)


def zf_known_phase(
    M: np.ndarray,
    z: np.ndarray,
    theta: np.ndarray,
    b: np.ndarray,
    *,
    ridge: float = DEFAULT_RIDGE,
) -> np.ndarray:
    """Genie-aided ZF with the **true** complex phase.

    This baseline uses

        theta = angle(M^H u_true + b + w)

    which a magnitude-only receiver does **not** observe. Therefore it is
    **not** a valid estimator operating only on ``z``. It is a lower /
    reference benchmark.

    It is allowed to lie below Cui's CRLB, because that CRLB applies to
    estimators based on magnitude-only observations, whereas this genie is
    given additional phase information. That is not a bug.

    Reconstruction and solve (no explicit matrix inverse)::

        r = z * exp(1j * theta) - b
        (M M^H + ridge I) u_hat = M r

    ``ridge`` defaults to 0. Nonzero Tikhonov regularisation is applied
    only when the caller passes ``ridge > 0``.
    """
    M_arr = _as_complex_matrix(M, "M")
    d, q = M_arr.shape
    z_arr = _as_real_vector(z, "z", q)
    theta_arr = _as_real_vector(theta, "theta", q)
    b_arr = _as_complex_vector(b, "b", q)
    ridge_val = _as_ridge(ridge)

    r = reconstruct_complex_observation(z_arr, theta_arr) - b_arr
    gram = M_arr @ M_arr.conj().T
    rhs = M_arr @ r
    u_hat = _gram_solve(gram, rhs, ridge_val)
    return np.asarray(u_hat, dtype=np.complex128).reshape(d)


def zf_known_phase_from_truth(
    M: np.ndarray,
    u_true: np.ndarray,
    b: np.ndarray,
    w: np.ndarray,
    *,
    ridge: float = DEFAULT_RIDGE,
) -> np.ndarray:
    """Genie ZF when the caller has ``u_true`` and the pre-magnitude noise ``w``.

    Computes ``lambda = M^H u_true + b + w``, then ``z = |lambda|``,
    ``theta = angle(lambda)``, and calls :func:`zf_known_phase`.
    """
    lam = true_complex_observation(M, u_true, b, w)
    return zf_known_phase(M, np.abs(lam), np.angle(lam), b, ridge=ridge)


def theoretical_zf_error_covariance(
    M: np.ndarray,
    sigma2: float,
    *,
    ridge: float = DEFAULT_RIDGE,
) -> np.ndarray:
    """``Cov(u_hat - u) = sigma2 (M M^H + ridge I)^{-1}``.

    Computed by solving ``(M M^H + ridge I) X = I``, not by calling
    ``np.linalg.inv`` inside the estimator. This matrix is for tests and
    later CRLB comparisons, not for forming ``u_hat``.
    """
    M_arr = _as_complex_matrix(M, "M")
    ridge_val = _as_ridge(ridge)
    try:
        sigma2_val = float(sigma2)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"sigma2 must be a real number, got {sigma2!r}") from exc
    if not np.isfinite(sigma2_val) or sigma2_val < 0.0:
        raise ValueError(f"sigma2 must be finite and >= 0, got {sigma2!r}")
    d = M_arr.shape[0]
    gram = M_arr @ M_arr.conj().T
    eye = np.eye(d, dtype=np.complex128)
    gram_inv = _gram_solve(gram, eye, ridge_val)
    return (sigma2_val * gram_inv).astype(np.complex128, copy=False)


# ---------------------------------------------------------------------------
# Part B — linearised closed-form LS (SystemModel.pdf Section 11)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class LinearisedLSResult:
    """Per-element closed-form LS on a linearised observation ``Y``.

    ``observation_source`` records which ``Y`` the caller supplied. It does
    **not** convert ``Z - |B|`` into the ideal linear model. Approximation
    error in the exact-magnitude case is therefore not hidden.

    Attributes
    ----------
    G_hat
        Stacked channel estimate, shape ``(N, K)``, complex128.
    gtilde_hat
        Real packed rows ``[Re g_n, Im g_n]``, shape ``(N, 2K)``, float64.
    Phi
        Design matrices, shape ``(N, P, 2K)``, float64.
        ``Phi[n]`` is ``Phi_n`` of shape ``(P, 2K)``.
    observation_source
        ``"exact_magnitude"`` if ``Y = Z - |B|`` from Step 5, or
        ``"ideal_linear"`` if ``Y`` was generated as
        ``Re{Psi ⊙ GS} + nbar`` for algebra / covariance tests.
    ridge
        Tikhonov parameter actually used (0 means none).
    """

    G_hat: np.ndarray
    gtilde_hat: np.ndarray
    Phi: np.ndarray
    observation_source: ObservationSource
    ridge: float


def linearised_design_matrix(psi_n: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Build ``Phi_n ∈ R^{P × 2K}`` for one receive element.

    For pilot ``p``::

        phi_{n,p} = [ Re(psi_{n,p} s_p) ;  -Im(psi_{n,p} s_p) ]

    The minus sign on the imaginary block is required. Using ``+Im``
    would break ``Phi_n @ gtilde_n = Re{psi_n ⊙ (g_n @ S)}``.
    """
    S_arr = _as_complex_matrix(S, "S")
    k, p = S_arr.shape
    psi = _as_complex_vector(psi_n, "psi_n", p)
    # U[p, k] = psi_n[p] * S[k, p]
    U = psi[:, np.newaxis] * S_arr.T
    phi = np.concatenate([U.real, -U.imag], axis=1)
    if phi.shape != (p, 2 * k):
        raise RuntimeError(f"internal Phi shape {phi.shape}, expected {(p, 2 * k)}")
    return np.asarray(phi, dtype=np.float64)


def make_ideal_linear_y(
    G: np.ndarray,
    S: np.ndarray,
    Psi: np.ndarray,
    sigma2: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Ideal linear-Gaussian observation, **not** ``Z - |B|``.

        Y_linear = Re{Psi ⊙ (G @ S)} + nbar,    nbar ~ N(0, sigma2 / 2)

    This exists so the closed-form LS algebra and covariance can be tested
    without the Taylor remainder of the exact magnitude model. Callers must
    still pass ``observation_source="ideal_linear"`` into the estimator.
    """
    G_arr = _as_complex_matrix(G, "G")
    S_arr = _as_complex_matrix(S, "S")
    Psi_arr = _as_complex_matrix(Psi, "Psi")
    n_rx, n_users = G_arr.shape
    k_s, n_pilots = S_arr.shape
    if k_s != n_users:
        raise ValueError(
            f"incompatible G and S: G.shape={G_arr.shape}, S.shape={S_arr.shape}"
        )
    if Psi_arr.shape != (n_rx, n_pilots):
        raise ValueError(
            f"incompatible Psi: Psi.shape={Psi_arr.shape}, expected {(n_rx, n_pilots)}"
        )
    try:
        sigma2_val = float(sigma2)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"sigma2 must be a real number, got {sigma2!r}") from exc
    if not np.isfinite(sigma2_val) or sigma2_val < 0.0:
        raise ValueError(f"sigma2 must be finite and >= 0, got {sigma2!r}")

    signal = np.real(Psi_arr * (G_arr @ S_arr)).astype(np.float64, copy=False)
    if sigma2_val == 0.0:
        return signal
    if rng is None:
        raise ValueError("sigma2 > 0 requires an explicit numpy Generator rng")
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy Generator, got {type(rng)!r}")
    nbar = rng.normal(loc=0.0, scale=np.sqrt(sigma2_val / 2.0), size=signal.shape)
    return np.asarray(signal + nbar, dtype=np.float64)


def _solve_real_ls(phi: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    """Solve ``min ||Phi gtilde - y||`` without inverting ``Phi^T Phi``."""
    if ridge == 0.0:
        gtilde, _, _, _ = np.linalg.lstsq(phi, y, rcond=None)
        return np.asarray(gtilde, dtype=np.float64)
    gram = phi.T @ phi
    rhs = phi.T @ y
    return np.asarray(_gram_solve(gram, rhs, ridge), dtype=np.float64)


def linearised_closed_form_ls(
    Y: np.ndarray,
    S: np.ndarray,
    Psi: np.ndarray,
    *,
    observation_source: ObservationSource,
    ridge: float = DEFAULT_RIDGE,
) -> LinearisedLSResult:
    """Exact closed-form LS for the Section 11 linearised model.

    Independently for each receive element ``n = 0, …, N-1``::

        gtilde_n = [Re(g_n); Im(g_n)] ∈ R^{2K}
        Phi_n ∈ R^{P × 2K}
        gtilde_hat_n  solves  Phi_n gtilde = y_n    (LS / ridge)

        g_hat_n = gtilde_hat_n[:K] + 1j * gtilde_hat_n[K:]

    ``observation_source`` is required and is recorded on the result:

    * ``"exact_magnitude"`` — ``Y`` is Step 5 ``Z - |B|``. The linear model
      is then approximate; finite-RSR error is expected and is not a
      solver bug.
    * ``"ideal_linear"`` — ``Y`` was generated as ``Re{Psi ⊙ GS} + nbar``.
      Noiseless recovery of ``G`` is then an algebra test.

    GD is **not** implemented here. The comparison
    ``||G_GD - G_LS||_F / ||G_LS||_F < 1e-6`` at high RSR is a future
    acceptance test (see ``FUTURE_GD_VS_CLOSED_FORM_TEST``).
    """
    if observation_source not in ("exact_magnitude", "ideal_linear"):
        raise ValueError(
            "observation_source must be 'exact_magnitude' (Y = Z - |B|) or "
            f"'ideal_linear' (Y = Re{{Psi ⊙ GS}} + nbar), got {observation_source!r}"
        )
    Y_arr = np.asarray(Y, dtype=np.float64)
    if Y_arr.ndim != 2:
        raise ValueError(f"Y must be 2-D, got shape {Y_arr.shape}")
    _require_finite(Y_arr, "Y")
    S_arr = _as_complex_matrix(S, "S")
    Psi_arr = _as_complex_matrix(Psi, "Psi")
    ridge_val = _as_ridge(ridge)

    n_rx, n_pilots = Y_arr.shape
    n_users, p_s = S_arr.shape
    if p_s != n_pilots:
        raise ValueError(
            f"incompatible Y and S: Y.shape={Y_arr.shape}, S.shape={S_arr.shape}"
        )
    if Psi_arr.shape != (n_rx, n_pilots):
        raise ValueError(
            f"incompatible Psi: Psi.shape={Psi_arr.shape}, expected {(n_rx, n_pilots)}"
        )

    two_k = 2 * n_users
    phi_all = np.empty((n_rx, n_pilots, two_k), dtype=np.float64)
    gtilde_hat = np.empty((n_rx, two_k), dtype=np.float64)
    y_real = np.real(Y_arr).astype(np.float64, copy=False)

    for n in range(n_rx):
        phi_n = linearised_design_matrix(Psi_arr[n], S_arr)
        phi_all[n] = phi_n
        gtilde_hat[n] = _solve_real_ls(phi_n, y_real[n], ridge_val)

    G_hat = gtilde_hat[:, :n_users] + 1j * gtilde_hat[:, n_users:]
    return LinearisedLSResult(
        G_hat=np.asarray(G_hat, dtype=np.complex128),
        gtilde_hat=gtilde_hat,
        Phi=phi_all,
        observation_source=observation_source,
        ridge=ridge_val,
    )


def theoretical_linearised_ls_gtilde_covariance(
    Phi_n: np.ndarray,
    sigma2: float,
    *,
    ridge: float = DEFAULT_RIDGE,
) -> np.ndarray:
    """``Cov(gtilde_hat_n - gtilde_n) = (sigma2 / 2) (Phi^T Phi + ridge I)^{-1}``.

    This is the standard real LS covariance under ``nbar ~ N(0, sigma2/2)``.
    Step 12 (:func:`rydberg_sim.linearised_crlb.linearised_row_crlb`)
    derives the same no-ridge covariance from that model; no Xu
    prefactor is copied. Ridge here is only for the LS estimator.
    """
    phi = np.asarray(Phi_n, dtype=np.float64)
    if phi.ndim != 2:
        raise ValueError(f"Phi_n must be 2-D, got shape {phi.shape}")
    _require_finite(phi, "Phi_n")
    ridge_val = _as_ridge(ridge)
    try:
        sigma2_val = float(sigma2)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"sigma2 must be a real number, got {sigma2!r}") from exc
    if not np.isfinite(sigma2_val) or sigma2_val < 0.0:
        raise ValueError(f"sigma2 must be finite and >= 0, got {sigma2!r}")
    two_k = phi.shape[1]
    gram = phi.T @ phi
    gram_inv = _gram_solve(gram, np.eye(two_k, dtype=np.float64), ridge_val)
    return ((sigma2_val / 2.0) * gram_inv).astype(np.float64, copy=False)


# ---------------------------------------------------------------------------
# Part C — CM-ZF (not implemented)
# ---------------------------------------------------------------------------

def cm_zf(*args: object, **kwargs: object) -> np.ndarray:
    """Cui channel-magnitude ZF.

    **Not implemented.** SystemModel.pdf and the rest of this repository do
    not give unambiguous CM-ZF equations. The implementation plan says this
    baseline must be reconstructed from Cui's cited channel-magnitude
    reference, and explicitly allows dropping it if it cannot be pinned
    down.

    Do not substitute a homemade least-squares or ZF-on-|G| approximation
    and call it CM-ZF.
    """
    raise NotImplementedError(
        "CM-ZF is unimplemented: Cui's channel-magnitude ZF estimator is "
        "not specified in SystemModel.pdf or this repository. Do not invent "
        "an approximation. This baseline may be dropped later if the cited "
        "equations cannot be recovered. TODO: implement only after the "
        "source equations are identified unambiguously."
    )


# ---------------------------------------------------------------------------
# Part D — exhaustive QAM LS / ML (Track A detection, not channel estimation)
# ---------------------------------------------------------------------------

def qam_candidate_count(constellation_size: int, D: int) -> int:
    """``M_qam ** D``. Integer power; does not allocate the candidate list."""
    if isinstance(constellation_size, (bool, np.bool_)) or int(constellation_size) != constellation_size:
        raise TypeError(f"constellation_size must be an integer, got {constellation_size!r}")
    if isinstance(D, (bool, np.bool_)) or int(D) != D:
        raise TypeError(f"D must be an integer, got {D!r}")
    m = int(constellation_size)
    d = int(D)
    if m < 2 or d < 1:
        raise ValueError(f"need constellation_size >= 2 and D >= 1, got {m}, {d}")
    return m**d


def exhaustive_search_complexity_gate(
    constellation_size: int,
    D: int,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> int:
    """Return ``M**D`` or raise :class:`ExhaustiveSearchTooLargeError`.

    Default ``max_candidates`` is 65536. That admits 4-QAM with ``D=3``
    (64 candidates) and refuses 16-QAM with ``D=6`` (16_777_216).
    """
    n = qam_candidate_count(constellation_size, D)
    if isinstance(max_candidates, (bool, np.bool_)) or int(max_candidates) != max_candidates:
        raise TypeError(f"max_candidates must be an integer, got {max_candidates!r}")
    cap = int(max_candidates)
    if cap < 1:
        raise ValueError(f"max_candidates must be >= 1, got {cap}")
    if n > cap:
        raise ExhaustiveSearchTooLargeError(
            f"exhaustive search has {constellation_size}**{D} = {n} candidates, "
            f"which exceeds max_candidates={cap}. Refusing to run "
            "(16-QAM with D=6 is 16_777_216 and is not a routine Monte Carlo)."
        )
    return n


def _as_step4_constellation(constellation: QAMConstellation | int) -> QAMConstellation:
    """Always the Step-4 Gray unit-energy table; never a second alphabet."""
    if isinstance(constellation, QAMConstellation):
        # Re-fetch from the Step-4 cache so a mutated copy cannot sneak in.
        cached = build_qam_constellation(constellation.M)
        if constellation.points is not cached.points:
            raise ValueError(
                "constellation is not the Step-4 cached Gray unit-energy QAM table; "
                "exhaustive search does not accept a separately defined alphabet"
            )
        return cached
    return build_qam_constellation(int(constellation))


def enumerate_qam_symbol_vectors(
    constellation: QAMConstellation | int,
    D: int,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> np.ndarray:
    """All ``u ∈ constellation^D``, shape ``(M**D, D)``, Step-4 points only."""
    const = _as_step4_constellation(constellation)
    n = exhaustive_search_complexity_gate(const.M, D, max_candidates=max_candidates)
    # product over the cached points. Gate already ran, so this is bounded.
    candidates = np.array(list(product(const.points, repeat=int(D))), dtype=np.complex128)
    if candidates.shape != (n, int(D)):
        raise RuntimeError(
            f"internal candidate shape {candidates.shape}, expected {(n, int(D))}"
        )
    return candidates


def log_bessel_i0(x: np.ndarray | float) -> np.ndarray:
    """Numerically stable ``log I_0(x)`` via ``i0e``: ``log(i0e(|x|)) + |x|``."""
    ax = np.abs(np.asarray(x, dtype=np.float64))
    # i0e underflows to 0 only for huge |x|; the +|x| term still dominates.
    i0e_val = i0e(ax)
    tiny = np.finfo(np.float64).tiny
    return np.log(np.maximum(i0e_val, tiny)) + ax


def rician_log_likelihood(
    z: np.ndarray,
    lam: np.ndarray,
    sigma2: float,
) -> float:
    """Sum of Rician log-likelihood terms that **depend on** ``λ``.

    Observation model (Step 5): ``z = |λ + w|`` with ``w ~ CN(0, σ²)``,
    ``E[|w|²] = σ²``. The amplitude is Rician:

        p(z|λ) = (2z/σ²) exp(-(z²+|λ|²)/σ²) I₀(2 z |λ| / σ²),   z ≥ 0.

    Terms that do not depend on ``λ`` (hence not on ``u``) are omitted:

        ℓ(λ) = Σ_q [ −|λ_q|²/σ² + log I₀(2 z_q |λ_q| / σ²) ].

    This is the unique ML metric for the already-implemented magnitude
    model. It is also the likelihood whose EM surrogate uses the Bessel
    ratio of argument ``2 z |λ| / σ²`` (Cui Algorithm 2).

    Evaluated in the log domain; probabilities are never multiplied.
    """
    sigma2_val = _as_sigma2_positive(sigma2)
    z_arr = np.asarray(z, dtype=np.float64)
    lam_arr = np.asarray(lam, dtype=np.complex128)
    if z_arr.shape != lam_arr.shape:
        raise ValueError(
            f"z and lambda shapes must match, got {z_arr.shape} and {lam_arr.shape}"
        )
    _require_finite(z_arr, "z")
    _require_finite(lam_arr, "lambda")
    abs_lam = np.abs(lam_arr)
    arg = (2.0 * z_arr * abs_lam) / sigma2_val
    ell = - (abs_lam**2) / sigma2_val + log_bessel_i0(arg)
    return float(np.sum(ell))


@dataclass(frozen=True, eq=False)
class ExhaustiveSearchResult:
    """One exhaustive QAM search over ``constellation^D``."""

    u_hat: np.ndarray
    metric: float
    index: int
    num_candidates: int
    criterion: Literal["ls", "ml"]
    constellation: QAMConstellation
    unique_minimizer: bool


def _validate_detection_inputs(
    M: np.ndarray,
    b: np.ndarray,
    z: np.ndarray,
    constellation: QAMConstellation | int,
    max_candidates: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, QAMConstellation, np.ndarray, int]:
    M_arr = _as_complex_matrix(M, "M")
    d, q = M_arr.shape
    b_arr = _as_complex_vector(b, "b", q)
    z_arr = _as_real_vector(z, "z", q)
    const = _as_step4_constellation(constellation)
    candidates = enumerate_qam_symbol_vectors(const, d, max_candidates=max_candidates)
    return M_arr, b_arr, z_arr, const, candidates, d


def exhaustive_magnitude_ls(
    M: np.ndarray,
    b: np.ndarray,
    z: np.ndarray,
    constellation: QAMConstellation | int,
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> ExhaustiveSearchResult:
    """Exhaustive magnitude-domain LS over QAM vectors.

        J_LS(u) = || z - |M^H u + b| ||_2^2

    Track A *detection*, not channel estimation. Candidates are the
    Step-4 unit-energy Gray QAM alphabet raised to ``D = M.shape[0]``.
    """
    M_arr, b_arr, z_arr, const, candidates, d = _validate_detection_inputs(
        M, b, z, constellation, max_candidates
    )
    # lambda_c = M^H u_c + b  →  (Q, C)
    lam = M_arr.conj().T @ candidates.T + b_arr[:, np.newaxis]
    z_hat = np.abs(lam)
    j_ls = np.sum((z_arr[:, np.newaxis] - z_hat) ** 2, axis=0)
    best = int(np.argmin(j_ls))
    best_val = float(j_ls[best])
    # Uniqueness to a tight absolute tolerance (noiseless tests need this).
    n_tie = int(np.count_nonzero(np.abs(j_ls - best_val) <= 1e-12 * max(1.0, best_val)))
    return ExhaustiveSearchResult(
        u_hat=np.asarray(candidates[best], dtype=np.complex128).reshape(d),
        metric=best_val,
        index=best,
        num_candidates=int(candidates.shape[0]),
        criterion="ls",
        constellation=const,
        unique_minimizer=n_tie == 1,
    )


def exhaustive_magnitude_ml(
    M: np.ndarray,
    b: np.ndarray,
    z: np.ndarray,
    constellation: QAMConstellation | int,
    sigma2: float,
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> ExhaustiveSearchResult:
    """Exhaustive Rician ML over QAM vectors (log-domain).

    LS and ML are **not** assumed identical. This maximises

        Σ_q [ −|λ_q|²/σ² + log I₀(2 z_q |λ_q| / σ²) ]

    with ``λ = M^H u + b``, which is the u-dependent part of the Rician
    likelihood for ``z = |λ + w|``, ``w ~ CN(0, σ²)``.

    Requires ``sigma2 > 0``. The noiseless problem is exhaustive LS.
    """
    sigma2_val = _as_sigma2_positive(sigma2)
    M_arr, b_arr, z_arr, const, candidates, d = _validate_detection_inputs(
        M, b, z, constellation, max_candidates
    )
    lam = M_arr.conj().T @ candidates.T + b_arr[:, np.newaxis]
    abs_lam = np.abs(lam)
    arg = (2.0 * z_arr[:, np.newaxis] * abs_lam) / sigma2_val
    ell = - (abs_lam**2) / sigma2_val + log_bessel_i0(arg)
    loglik = np.sum(ell, axis=0)
    best = int(np.argmax(loglik))
    best_val = float(loglik[best])
    n_tie = int(np.count_nonzero(np.abs(loglik - best_val) <= 1e-10 * max(1.0, abs(best_val))))
    return ExhaustiveSearchResult(
        u_hat=np.asarray(candidates[best], dtype=np.complex128).reshape(d),
        metric=best_val,
        index=best,
        num_candidates=int(candidates.shape[0]),
        criterion="ml",
        constellation=const,
        unique_minimizer=n_tie == 1,
    )

"""Linearised (strong-reference) channel CRLB (Step 12).

This is **not** Cui's exact magnitude-only Rician CRLB (Step 11) and it
is **not** a copy of Xu's published prefactor.

Real linear model (same as Step 7)
----------------------------------
For receive element ``n``::

    y_n = Phi_n gtilde_n + nbar_n

    gtilde_n = [Re(g_n); Im(g_n)] ∈ R^{2K}
    Phi_n ∈ R^{P × 2K}
    nbar_n ~ N(0, (σ²/2) I_P)

``sigma2 = σ²`` is the **complex** pre-magnitude noise variance used
everywhere in this project. The real linearised noise variance is
therefore ``sigma2 / 2``.

``Phi_n`` is **exactly** :func:`rydberg_sim.baselines.linearised_design_matrix`::

    phi_{n,p} = [ Re(ψ_{n,p} s_p) ;  -Im(ψ_{n,p} s_p) ]
    ψ_{n,p}   = exp(-1j * angle(B[n,p]))

Same Ψ sign, same ``-Im`` block, same ``[Re g; Im g]`` packing as the
Step-7 closed-form LS estimator.

Fisher / CRLB
-------------
This implementation derives the covariance from
``nbar ~ N(0, sigma2/2 I)``, giving

    F_n    = (2 / sigma2) * Phi_n^T Phi_n
    CRLB_n = (sigma2 / 2) * (Phi_n^T Phi_n)^{-1}

No prefactor is copied from Xu.

``CRLB_n`` is formed with ``solve(Phi^T Phi, I)``, never
``inv(Phi^T Phi)``. No ridge by default. Rank-deficient ``Phi_n``
raises rather than a silent pseudo-inverse.

Channel MSE
-----------
``E||g_hat_n - g_n||_2^2 = E||gtilde_hat_n - gtilde_n||_2^2`` is bounded
by ``trace(CRLB_n)``. The whole-channel bound is ``Σ_n trace(CRLB_n)``.

Normalized bound::

    normalized_crlb_G = (Σ_n trace(CRLB_n)) / E||G||_F^2

with the Step-6 energy ``E||G||_F^2 = c² N Σ_k β_k`` (not a per-trial
``||G||_F^2``).

Validity vs information
-----------------------
``F_n`` depends on the **phase** of ``B`` through ``Ψ``. It does not
depend on ``|B|`` once the linearised model is formed. Whether that
linearised model is a good approximation of ``Z = |GS+B+W|`` is an RSR
question and is **not** this Fisher information.

Do not compare this bound to Step 11's Rician CRLB as if they described
the same statistical model at arbitrary RSR.

What this module does **not** implement (Step 13+)
-------------------------------------------------
metrics framework, Monte Carlo harness, GD/PGD, figure sweeps, BER,
Track-C, machine learning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .baselines import linearised_design_matrix
from .forward import reference_phase_matrix

PHI_RANK_SV_REL_TOL = 1e-8


class RankDeficientPhiError(np.linalg.LinAlgError):
    """``Phi_n`` is not full column rank; the linearised CRLB is undefined."""


def _require_finite(arr: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")


def _as_sigma2_positive(value: object) -> float:
    try:
        sigma2 = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"sigma2 must be a real number, got {value!r}") from exc
    if not np.isfinite(sigma2):
        raise ValueError(f"sigma2 must be finite, got {value!r}")
    if sigma2 <= 0.0:
        raise ValueError(
            f"sigma2 must be > 0, got {sigma2}; "
            "the linearised Fisher contains 2/sigma2."
        )
    return sigma2


def _as_c_positive(value: object) -> float:
    try:
        c_val = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"c must be a real number, got {value!r}") from exc
    if not np.isfinite(c_val) or c_val <= 0.0:
        raise ValueError(f"c must be finite and > 0, got {value!r}")
    return c_val


def _as_beta_k(value: object) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"beta_k must be a non-empty 1-D sequence, got {np.shape(value)}")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"every beta_k must be finite and > 0, got {value!r}")
    return np.array(arr, dtype=np.float64, copy=True)


def _as_real_matrix(value: object, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {arr.shape}")
    _require_finite(arr, name)
    return np.array(arr, dtype=np.float64, copy=True)


def _require_full_column_rank_phi(phi: np.ndarray) -> None:
    p, two_k = phi.shape
    if two_k < 1 or two_k % 2 != 0:
        raise ValueError(f"Phi must have 2K columns with K>=1, got shape {phi.shape}")
    if p < two_k:
        raise RankDeficientPhiError(
            f"Phi has shape {phi.shape}: P={p} < 2K={two_k}, so Phi^T Phi "
            "is singular. The project requires P >= 2K for channel estimation. "
            "Refusing to pseudo-invert an unidentifiable linearised model."
        )
    svals = np.linalg.svd(phi, compute_uv=False)
    smax = float(svals[0])
    smin = float(svals[-1])
    if smax == 0.0 or smin < PHI_RANK_SV_REL_TOL * smax:
        raise RankDeficientPhiError(
            f"Phi is rank deficient (σ_min/σ_max={smin / max(smax, 1e-300):.3g}). "
            "Refusing to silently pseudo-invert."
        )


def expected_channel_frobenius_energy(
    N: int,
    beta_k: object,
    c: float = 1.0,
) -> float:
    """``E||G||_F^2 = c² N Σ_k β_k`` from Step 6 (not a per-trial ``||G||_F^2``)."""
    if isinstance(N, (bool, np.bool_)) or int(N) != N:
        raise TypeError(f"N must be an integer, got {N!r}")
    n_rx = int(N)
    if n_rx < 1:
        raise ValueError(f"N must be >= 1, got {n_rx}")
    betas = _as_beta_k(beta_k)
    c_val = _as_c_positive(c)
    return float((c_val**2) * n_rx * np.sum(betas))


@dataclass(frozen=True, eq=False)
class LinearisedRowCRLBResult:
    """Per-element linearised Fisher / CRLB.

    ``F`` is real symmetric ``(2K, 2K)``. ``crlb`` is
    ``(sigma2/2) (Phi^T Phi)^{-1}``.
    """

    Phi: np.ndarray
    F: np.ndarray
    crlb: np.ndarray
    trace: float
    sigma2: float


@dataclass(frozen=True, eq=False)
class LinearisedChannelCRLBResult:
    """Block-per-element linearised channel CRLB.

    ``crlb[n]`` is ``CRLB_n``, shape ``(N, 2K, 2K)``. The implementation
    does **not** form ``I_N ⊗ CRLB_row`` because ``Phi_n`` can differ
    with ``n`` through the phase of ``B``.
    """

    Phi: np.ndarray
    F: np.ndarray
    crlb: np.ndarray
    Psi: np.ndarray
    trace_per_row: np.ndarray
    mse_bound: float
    expected_channel_energy: float
    normalized_crlb: float
    sigma2: float


def linearised_row_fisher(Phi: np.ndarray, sigma2: float) -> np.ndarray:
    """``F = (2 / σ²) Phi^T Phi`` for ``nbar ~ N(0, σ²/2 I)``."""
    phi = _as_real_matrix(Phi, "Phi")
    sigma2_val = _as_sigma2_positive(sigma2)
    _require_full_column_rank_phi(phi)
    gram = phi.T @ phi
    F = (2.0 / sigma2_val) * gram
    F = 0.5 * (F + F.T)
    return F


def linearised_row_crlb(Phi: np.ndarray, sigma2: float) -> LinearisedRowCRLBResult:
    """``CRLB = (σ²/2) (Phi^T Phi)^{-1}`` via ``solve``, no explicit inverse.

    This implementation derives the covariance from
    ``nbar ~ N(0, sigma2/2 I)``, giving
    ``Cov = (sigma2/2)(Phi^T Phi)^{-1}``.
    No prefactor is copied from Xu.
    """
    phi = _as_real_matrix(Phi, "Phi")
    sigma2_val = _as_sigma2_positive(sigma2)
    _require_full_column_rank_phi(phi)
    two_k = phi.shape[1]
    gram = phi.T @ phi
    F = (2.0 / sigma2_val) * gram
    F = 0.5 * (F + F.T)
    try:
        gram_inv = np.linalg.solve(gram, np.eye(two_k, dtype=np.float64))
    except np.linalg.LinAlgError as exc:
        raise RankDeficientPhiError(
            "Phi^T Phi is singular; no silent ridge or pseudo-inverse."
        ) from exc
    crlb = (sigma2_val / 2.0) * gram_inv
    crlb = 0.5 * (crlb + crlb.T)
    return LinearisedRowCRLBResult(
        Phi=phi,
        F=F,
        crlb=crlb,
        trace=float(np.trace(crlb)),
        sigma2=sigma2_val,
    )


def linearised_channel_crlb(
    S: np.ndarray,
    B: np.ndarray,
    sigma2: float,
    beta_k: object,
    *,
    c: float = 1.0,
) -> LinearisedChannelCRLBResult:
    """Per-element linearised CRLB stacked over receive rows.

    ``Psi = exp(-1j angle(B))`` from :func:`reference_phase_matrix`.
    Each ``Phi_n`` is :func:`linearised_design_matrix` — no second Φ
    convention. ``beta_k`` and ``c`` enter **only** the expected-energy
    denominator, not ``F_n``.
    """
    S_arr = np.asarray(S, dtype=np.complex128)
    if S_arr.ndim != 2:
        raise ValueError(f"S must be 2-D (K, P), got shape {S_arr.shape}")
    _require_finite(S_arr, "S")
    Psi = reference_phase_matrix(B)
    n_rx, n_pilots = Psi.shape
    n_users, p_s = S_arr.shape
    if p_s != n_pilots:
        raise ValueError(
            f"incompatible B/Psi and S: Psi.shape={Psi.shape}, S.shape={S_arr.shape}"
        )
    betas = _as_beta_k(beta_k)
    if betas.size == 1 and n_users > 1:
        betas = np.full(n_users, float(betas[0]), dtype=np.float64)
    if betas.size != n_users:
        raise ValueError(
            f"beta_k has length {betas.size}, expected K={n_users} (or a scalar)"
        )
    c_val = _as_c_positive(c)
    sigma2_val = _as_sigma2_positive(sigma2)

    two_k = 2 * n_users
    phi_all = np.empty((n_rx, n_pilots, two_k), dtype=np.float64)
    F_all = np.empty((n_rx, two_k, two_k), dtype=np.float64)
    crlb_all = np.empty((n_rx, two_k, two_k), dtype=np.float64)
    traces = np.empty(n_rx, dtype=np.float64)
    for n in range(n_rx):
        phi_n = linearised_design_matrix(Psi[n], S_arr)
        row = linearised_row_crlb(phi_n, sigma2_val)
        phi_all[n] = row.Phi
        F_all[n] = row.F
        crlb_all[n] = row.crlb
        traces[n] = row.trace

    energy = expected_channel_frobenius_energy(n_rx, betas, c=c_val)
    mse_bound = float(np.sum(traces))
    return LinearisedChannelCRLBResult(
        Phi=phi_all,
        F=F_all,
        crlb=crlb_all,
        Psi=np.array(Psi, dtype=np.complex128, copy=True),
        trace_per_row=traces,
        mse_bound=mse_bound,
        expected_channel_energy=energy,
        normalized_crlb=float(mse_bound / energy),
        sigma2=sigma2_val,
    )

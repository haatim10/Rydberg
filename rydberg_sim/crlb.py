"""Cui Fisher information / CRLB for the magnitude-only Rician model (Step 11).

Canonical model (same as Steps 8–10)
------------------------------------
    z = |M^H u + b + w|

    M ∈ C^{D × Q},   u ∈ C^D,   b, w length Q
    w ~ CN(0, σ² I),  E[|w_q|²] = σ² = ``sigma2``

    λ_q = m_q^H u + b_q

Each amplitude ``z_q`` is Rician with the Step-5 / Step-7 density

    p(z|λ) = (2z/σ²) exp(-(z²+|λ|²)/σ²) I₀(2 z |λ| / σ²),   z ≥ 0.

This module is generic. It does **not** know whether ``u`` is Cui's
transmitted symbol vector or ``conj(g_n)`` from a channel-estimation
adapter. There is no G/S/QAM special case.

Fisher information (Cui)
------------------------
    F = Σ_q β_q m_q m_q^H ∈ C^{D × D}

    β_q = (E[z_q² R(κ_q)²] − |λ_q|²) / σ⁴

with ``σ⁴ = (sigma2)²`` — the code name is ``sigma2_squared``, never a
bare ``sigma`` — and

    κ_q = 2 z_q |λ_q| / σ²
    R(x) = I₁(x)/I₀(x)     (the Step-10 :func:`rydberg_sim.gs.bessel_ratio`)

High-SNR analytic limit
-----------------------
    β_q → 1 / (2 σ²)

    F → (1/(2 σ²)) M M^H
    CRLB → 2 σ² (M M^H)^{-1}

Known-phase ZF (Step 7) has covariance ``σ² (M M^H)^{-1}``. The
high-SNR magnitude-only CRLB is therefore **3.0103 dB** (exactly
``10 log10 2``) above that genie covariance. ZF-known-phase is given the
true noisy complex phase, so it is **allowed** to sit below this CRLB.
That is not a violation: this CRLB applies to estimators that see only
``z``.

Quadrature
----------
The plan's one-sided interval ``[0, |λ| + 8 sqrt(σ²)]`` is equivalent
when ``|λ|`` is not large compared with ``σ``, but at high SNR QUADPACK
can miss a spike of width ``~σ`` sitting at the right end of a huge
``[0, |λ|]`` interval (the integrand evaluates to ~0 on almost the whole
domain). This implementation therefore integrates on the two-sided
window

    [max(0, |λ| − 8 σ),  |λ| + 8 σ],   σ = sqrt(σ²)

split at ``|λ|``. When ``|λ| < 8σ`` the lower limit is 0, recovering the
plan. The truncated left tail is the same 8σ event the plan already
accepts on the right. ``β`` is **not** clipped to ``1/(2σ²)``.

What this module does **not** implement (Step 12+)
-------------------------------------------------
Xu's closed-form channel CRLB, GD/PGD, figure sweeps, BER, Track-C,
machine learning. Do not import or copy Xu's constant here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.special import i0e

from .gs import bessel_ratio

# Plan: integrate a neighbourhood of |λ| of half-width 8 σ, σ = sqrt(σ²).
QUAD_TAIL_SIGMAS = 8.0
QUAD_EPSABS = 1e-14
QUAD_EPSREL = 1e-12
QUAD_LIMIT = 400
# Windowed mass should be 1 to many digits; a miss of the high-SNR spike
# shows up as mass ≪ 1.
PDF_MASS_MIN = 0.999
PDF_MASS_MAX = 1.001


class NegativeFisherBetaError(ValueError):
    """A quadrature ``β_q`` came out negative. Not clipped."""


class SingularFisherError(np.linalg.LinAlgError):
    """Fisher matrix is not invertible; no silent ridge."""


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
            "the Rician density and κ contain 1/sigma2."
        )
    return sigma2


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


def _as_lambda_abs(value: object) -> np.ndarray:
    arr = np.abs(np.asarray(value, dtype=np.float64))
    if arr.ndim > 1:
        raise ValueError(f"lambda_abs must be scalar or 1-D, got shape {arr.shape}")
    _require_finite(arr, "lambda_abs")
    return np.array(arr, dtype=np.float64, copy=True).reshape(-1)


def high_snr_fisher_beta_limit(sigma2: float) -> float:
    """Analytic high-SNR limit ``β → 1 / (2 σ²)``."""
    sigma2_val = _as_sigma2_positive(sigma2)
    return 1.0 / (2.0 * sigma2_val)


def cui_crlb_high_snr_limit(M: np.ndarray, sigma2: float) -> np.ndarray:
    """``2 σ² (M M^H)^{-1}`` via ``solve``, no explicit inverse."""
    M_arr = _as_complex_matrix(M, "M")
    sigma2_val = _as_sigma2_positive(sigma2)
    d = M_arr.shape[0]
    gram = M_arr @ M_arr.conj().T
    try:
        return (2.0 * sigma2_val) * np.linalg.solve(
            gram, np.eye(d, dtype=np.complex128)
        )
    except np.linalg.LinAlgError as exc:
        raise SingularFisherError(
            "M M^H is singular; the high-SNR CRLB limit is undefined."
        ) from exc


def rician_amplitude_pdf(
    z: np.ndarray | float,
    lambda_abs: float,
    sigma2: float,
) -> np.ndarray:
    """Rician amplitude density of ``z = |λ + w|``, ``w ~ CN(0, σ²)``.

    Stable evaluation of the Step-7 density::

        p(z) = (2z/σ²) exp(-(z²+|λ|²)/σ²) I₀(2 z |λ| / σ²)
             = (2z/σ²) exp(-(z-|λ|)²/σ²) i0e(2 z |λ| / σ²)

    ``i0e`` is the same scaled Bessel used by
    :func:`rydberg_sim.baselines.log_bessel_i0`. Do **not** form
    ``i0(kappa)``: it overflows at high SNR.
    """
    sigma2_val = _as_sigma2_positive(sigma2)
    abs_lam = abs(float(lambda_abs))
    z_arr = np.asarray(z, dtype=np.float64)
    out = np.zeros(np.shape(z_arr), dtype=np.float64)
    z_flat = np.reshape(z_arr, -1)
    out_flat = np.reshape(out, -1)
    pos = z_flat > 0.0
    if np.any(pos):
        zp = z_flat[pos]
        kappa = (2.0 * zp * abs_lam) / sigma2_val
        # (z-|λ|)² / σ² ≥ 0, so the exponential is in (0, 1].
        expo = np.exp(-((zp - abs_lam) ** 2) / sigma2_val)
        out_flat[pos] = (2.0 * zp / sigma2_val) * expo * i0e(kappa)
    if np.ndim(z) == 0:
        return np.asarray(out_flat[0])
    return out


def _rician_pdf_scalar(z: float, abs_lam: float, sigma2: float) -> float:
    if z <= 0.0:
        return 0.0
    kappa = (2.0 * z * abs_lam) / sigma2
    expo = np.exp(-((z - abs_lam) ** 2) / sigma2)
    return float((2.0 * z / sigma2) * expo * i0e(kappa))


def _quadrature_window(abs_lam: float, sigma2: float) -> tuple[float, float]:
    """``[max(0, |λ| − 8σ), |λ| + 8σ]`` with ``σ = sqrt(σ²)``."""
    sigma = float(np.sqrt(sigma2))
    lo = max(0.0, abs_lam - QUAD_TAIL_SIGMAS * sigma)
    hi = abs_lam + QUAD_TAIL_SIGMAS * sigma
    return lo, hi


def _integrate_kind(
    abs_lam: float,
    sigma2: float,
    kind: str,
) -> tuple[float, float, float, float]:
    """Adaptive 1-D quadrature of mass / E[z² R²] / (E[z² R²]−|λ|²)."""
    a = float(abs_lam)
    s2 = float(sigma2)
    lo, hi = _quadrature_window(a, s2)

    def integrand(z: float) -> float:
        if z <= 0.0:
            return 0.0
        pdf = _rician_pdf_scalar(z, a, s2)
        if kind == "mass":
            return pdf
        kappa = (2.0 * z * a) / s2
        ratio = float(np.asarray(bessel_ratio(kappa)))
        z2_r2 = z * z * ratio * ratio
        if kind == "expectation":
            return z2_r2 * pdf
        return (z2_r2 - a * a) * pdf

    def _q(left: float, right: float) -> tuple[float, float]:
        if right <= left:
            return 0.0, 0.0
        val, err = quad(
            integrand,
            left,
            right,
            epsabs=QUAD_EPSABS,
            epsrel=QUAD_EPSREL,
            limit=QUAD_LIMIT,
        )
        return float(val), float(err)

    if lo < a < hi:
        v1, e1 = _q(lo, a)
        v2, e2 = _q(a, hi)
        return v1 + v2, e1 + e2, lo, hi
    val, err = _q(lo, hi)
    return val, err, lo, hi


@dataclass(frozen=True, eq=False)
class RicianFisherScalar:
    """Quadrature of one Rician measurement's Fisher weight.

    ``beta`` is the raw ``(E − |λ|²) / σ⁴``. It is **not** clipped to
    ``1/(2σ²)``. ``pdf_mass`` is the integrated density on the quadrature
    window (should be ~1).
    """

    lambda_abs: float
    sigma2: float
    expectation_z2_r2: float
    beta: float
    pdf_mass: float
    z_low: float
    z_high: float


def rician_fisher_scalar(lambda_abs: float, sigma2: float) -> RicianFisherScalar:
    """Compute ``E[z² R(κ)²]`` and ``β`` for one nonnegative ``|λ|``."""
    sigma2_val = _as_sigma2_positive(sigma2)
    abs_lam = abs(float(lambda_abs))
    if not np.isfinite(abs_lam):
        raise ValueError(f"lambda_abs must be finite, got {lambda_abs!r}")

    mass, _, lo, hi = _integrate_kind(abs_lam, sigma2_val, "mass")
    if not (PDF_MASS_MIN <= mass <= PDF_MASS_MAX):
        raise ValueError(
            f"Rician pdf mass on [{lo}, {hi}] is {mass}; expected ≈ 1. "
            "The high-SNR spike was probably missed by the quadrature."
        )

    expectation, _, _, _ = _integrate_kind(abs_lam, sigma2_val, "expectation")
    # Integrate the difference to avoid subtracting two large nearly-equal
    # numbers after the fact. Not a clip: this is the same integral.
    delta, _, _, _ = _integrate_kind(abs_lam, sigma2_val, "difference")
    sigma2_squared = sigma2_val * sigma2_val  # σ⁴
    beta = delta / sigma2_squared
    if not np.isfinite(expectation) or not np.isfinite(beta):
        raise ValueError(
            f"non-finite Fisher terms for |λ|={abs_lam}, sigma2={sigma2_val}: "
            f"E={expectation}, beta={beta}"
        )
    return RicianFisherScalar(
        lambda_abs=abs_lam,
        sigma2=sigma2_val,
        expectation_z2_r2=float(expectation),
        beta=float(beta),
        pdf_mass=float(mass),
        z_low=float(lo),
        z_high=float(hi),
    )


def fisher_expectation_z2_r2(
    lambda_abs: np.ndarray | float,
    sigma2: float,
) -> np.ndarray:
    """``E_q = E[z² R(2 z |λ| / σ²)²]`` for each ``|λ|``."""
    abs_arr = _as_lambda_abs(lambda_abs)
    out = np.empty(abs_arr.shape, dtype=np.float64)
    for i, a in enumerate(abs_arr):
        out[i] = rician_fisher_scalar(float(a), sigma2).expectation_z2_r2
    if np.ndim(lambda_abs) == 0:
        return np.asarray(out[0])
    return out


def fisher_beta(
    lambda_abs: np.ndarray | float,
    sigma2: float,
) -> np.ndarray:
    """Cui Fisher weights ``β_q = (E[z² R(κ)²] − |λ|²) / σ⁴``.

    Accepts a scalar or 1-D ``|λ|``. Does **not** clip negative values.
    """
    abs_arr = _as_lambda_abs(lambda_abs)
    out = np.empty(abs_arr.shape, dtype=np.float64)
    for i, a in enumerate(abs_arr):
        out[i] = rician_fisher_scalar(float(a), sigma2).beta
    if np.ndim(lambda_abs) == 0:
        return np.asarray(out[0])
    return out


@dataclass(frozen=True, eq=False)
class CuiFisherResult:
    """Canonical Cui Fisher information.

    ``F`` has already been Hermitian-symmetrized for roundoff only.
    ``beta`` is the raw per-measurement weight (not clipped).
    """

    F: np.ndarray
    beta: np.ndarray
    expectation_z2_r2: np.ndarray
    lambda_abs: np.ndarray
    pdf_mass: np.ndarray
    sigma2: float


@dataclass(frozen=True, eq=False)
class CuiCRLBResult:
    """Canonical Cui CRLB and the normalized detection metric.

    ``normalized_crlb = Tr(CRLB).real / expected_u_energy``.

    For unit-average-energy detection symbols the caller should pass
    ``expected_u_energy=D`` (equal to ``K`` when ``D=K``), **not** a
    per-trial ``||u||²``. Omitting the argument uses ``D``. That
    convention is **not** automatically the channel-estimation CRLB
    (Step 12).
    """

    crlb: np.ndarray
    F: np.ndarray
    beta: np.ndarray
    normalized_crlb: float
    expected_u_energy: float
    sigma2: float
    lambda_abs: np.ndarray


def cui_fisher_information(
    M: np.ndarray,
    u: np.ndarray,
    b: np.ndarray,
    sigma2: float,
) -> CuiFisherResult:
    """``F = Σ_q β_q m_q m_q^H`` for ``λ = M^H u + b``.

    No G/S/QAM arguments. Negative ``β_q`` is raised, not clipped.
    """
    M_arr = _as_complex_matrix(M, "M")
    d, q = M_arr.shape
    u_arr = _as_complex_vector(u, "u", d)
    b_arr = _as_complex_vector(b, "b", q)
    sigma2_val = _as_sigma2_positive(sigma2)

    lam = M_arr.conj().T @ u_arr + b_arr
    abs_lam = np.abs(lam)
    beta = np.empty(q, dtype=np.float64)
    expectation = np.empty(q, dtype=np.float64)
    mass = np.empty(q, dtype=np.float64)
    for i in range(q):
        term = rician_fisher_scalar(float(abs_lam[i]), sigma2_val)
        beta[i] = term.beta
        expectation[i] = term.expectation_z2_r2
        mass[i] = term.pdf_mass

    if np.any(~np.isfinite(beta)):
        raise ValueError(f"non-finite β_q: {beta}")
    if np.any(beta < 0.0):
        bad = np.where(beta < 0.0)[0]
        raise NegativeFisherBetaError(
            f"β_q < 0 at indices {bad.tolist()} with values {beta[bad]!r}. "
            "This is a quadrature failure, not clipped to 1/(2 σ²)."
        )

    # F = M diag(β) M^H = Σ_q β_q m_q m_q^H.
    F = (M_arr * beta) @ M_arr.conj().T
    F = 0.5 * (F + F.conj().T)
    return CuiFisherResult(
        F=F,
        beta=beta,
        expectation_z2_r2=expectation,
        lambda_abs=abs_lam,
        pdf_mass=mass,
        sigma2=sigma2_val,
    )


def cui_crlb(
    M: np.ndarray,
    u: np.ndarray,
    b: np.ndarray,
    sigma2: float,
    *,
    expected_u_energy: float | None = None,
) -> CuiCRLBResult:
    """Cui CRLB ``F^{-1}`` via ``solve(F, I)``. No default ridge.

    ``expected_u_energy`` is the denominator of the normalized metric
    ``Tr(CRLB)/E||u||²``. Default is ``D``, the unit-energy detection
    convention, not a per-trial ``||u||²``.
    """
    fim = cui_fisher_information(M, u, b, sigma2)
    d = fim.F.shape[0]
    if expected_u_energy is None:
        energy = float(d)
    else:
        try:
            energy = float(expected_u_energy)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"expected_u_energy must be a real number, got {expected_u_energy!r}"
            ) from exc
        if not np.isfinite(energy) or energy <= 0.0:
            raise ValueError(
                f"expected_u_energy must be > 0, got {energy}; "
                "do not substitute a per-trial ||u||² silently."
            )

    evals = np.linalg.eigh(fim.F)[0]
    if evals[0] <= 0.0:
        raise SingularFisherError(
            f"Fisher matrix is not positive definite (min eig={evals[0]!r}). "
            "Refusing to invert or regularize."
        )
    try:
        crlb = np.linalg.solve(fim.F, np.eye(d, dtype=np.complex128))
    except np.linalg.LinAlgError as exc:
        raise SingularFisherError(
            "Fisher matrix solve failed; no silent ridge."
        ) from exc

    normalized = float(np.trace(crlb).real / energy)
    return CuiCRLBResult(
        crlb=crlb,
        F=fim.F,
        beta=fim.beta,
        normalized_crlb=normalized,
        expected_u_energy=energy,
        sigma2=fim.sigma2,
        lambda_abs=fim.lambda_abs,
    )

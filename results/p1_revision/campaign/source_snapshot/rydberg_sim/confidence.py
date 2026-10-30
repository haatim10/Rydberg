"""Confidence-interval helpers for Monte Carlo summaries (Step 14).

These functions are **side-effect free**. They do not generate trials,
run estimators, or average per-trial dB / per-trial BER.

NMSE
----
Aggregate NMSE is a **ratio of sums** of linear energies, not the mean
of per-trial dB and not (in general) the mean of per-trial ratios.
The standard error treats trial-level ``(error_energy, true_energy)``
pairs as i.i.d. and applies the delta method to ``R = μ_e / μ_t``.

BER
---
Wilson intervals are computed from the **pooled** counts
``(total_bit_errors, total_bit_count)``. Do not average per-trial
Wilson intervals.

Zero errors
-----------
Empirical BER is ``0``, but that is not a claim that the true rate is
known to be zero. The Wilson upper bound is positive, and
:func:`rule_of_three` exposes the classic ``~3 / n`` 95% intuition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Standard normal quantile for a two-sided 95% interval.
WILSON_Z_95: float = 1.959963984540054


def _as_nonneg_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value:  # type: ignore[arg-type]
        raise TypeError(f"{name} must be an integer, got {value!r}")
    n = int(value)
    if n < 0:
        raise ValueError(f"{name} must be >= 0, got {n}")
    return n


@dataclass(frozen=True)
class WilsonInterval:
    """Wilson score interval for a binomial proportion.

    ``ber`` is the empirical rate ``k / n`` (0 when ``k = 0``).
    ``low`` / ``high`` are the interval endpoints in ``[0, 1]``.
    ``rule_of_three`` is ``3 / n`` when ``k = 0``, else ``None``.
    """

    bit_errors: int
    bit_count: int
    ber: float
    z: float
    low: float
    high: float
    rule_of_three: float | None


def rule_of_three(bit_count: int) -> float:
    """``3 / n`` upper-bound intuition for zero observed errors (~95%)."""
    n = _as_nonneg_int(bit_count, "bit_count")
    if n <= 0:
        raise ValueError(f"bit_count must be > 0, got {n}")
    return 3.0 / float(n)


def wilson_interval(
    bit_errors: int,
    bit_count: int,
    *,
    z: float = WILSON_Z_95,
) -> WilsonInterval:
    """Wilson score interval from pooled ``(k, n)``.

    When ``bit_errors = 0``, ``ber = 0`` but ``high > 0``. Do not treat
    a zero count as an exact known rate of zero.
    """
    k = _as_nonneg_int(bit_errors, "bit_errors")
    n = _as_nonneg_int(bit_count, "bit_count")
    if n <= 0:
        raise ValueError(f"bit_count must be > 0, got {n}")
    if k > n:
        raise ValueError(f"need bit_errors <= bit_count, got {k} > {n}")
    try:
        z_val = float(z)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"z must be a real number, got {z!r}") from exc
    if not np.isfinite(z_val) or z_val <= 0.0:
        raise ValueError(f"z must be finite and > 0, got {z!r}")

    p = k / n
    z2 = z_val * z_val
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    rad = z_val * np.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    low = float(max(0.0, center - rad))
    high = float(min(1.0, center + rad))
    # The algebraic Wilson endpoints are 0 and 1 at the boundaries;
    # enforce that exactly so floating-point cancellation cannot leak
    # a ~1e-18 residual that looks like a strictly positive lower bound.
    if k == 0:
        low = 0.0
    if k == n:
        high = 1.0
    rot = rule_of_three(n) if k == 0 else None
    return WilsonInterval(
        bit_errors=k,
        bit_count=n,
        ber=float(p),
        z=z_val,
        low=low,
        high=high,
        rule_of_three=rot,
    )


@dataclass(frozen=True)
class NmseUncertainty:
    """Delta-method SE for the ratio-of-sums NMSE.

    ``nmse_linear = sum(error) / sum(energy)``.
    ``se_linear`` is the standard error of that ratio, not of per-trial
    dB and not of the mean of per-trial ratios.
    """

    n_trials: int
    total_error_energy: float
    total_true_energy: float
    nmse_linear: float
    se_linear: float


def nmse_ratio_standard_error(
    error_energies: np.ndarray | object,
    true_energies: np.ndarray | object,
) -> NmseUncertainty:
    """Standard error of ``R = mean(e) / mean(t)`` via the delta method.

    Trial-level pairs ``(e_i, t_i)`` are treated as i.i.d. With
    ``n`` trials,

        Var(R) ≈ (1/n) μ_t^{-2} (s_e² + R² s_t² − 2 R s_{et})

    using unbiased sample variances / covariance (``ddof=1``) for
    ``n >= 2``. For ``n = 1`` the SE is reported as ``nan``.
    """
    e = np.asarray(error_energies, dtype=np.float64).reshape(-1)
    t = np.asarray(true_energies, dtype=np.float64).reshape(-1)
    if e.shape != t.shape:
        raise ValueError(
            f"error_energies and true_energies length mismatch: {e.size} vs {t.size}"
        )
    n = int(e.size)
    if n == 0:
        raise ValueError("need at least one trial")
    if not np.all(np.isfinite(e)) or np.any(e < 0.0):
        raise ValueError("error_energies must be finite and >= 0")
    if not np.all(np.isfinite(t)) or np.any(t <= 0.0):
        raise ValueError("true_energies must be finite and > 0")

    sum_e = float(np.sum(e))
    sum_t = float(np.sum(t))
    r = sum_e / sum_t
    if n == 1:
        se = float("nan")
    else:
        mean_e = sum_e / n
        mean_t = sum_t / n
        # Unbiased sample moments of the trial-level pairs.
        e_c = e - mean_e
        t_c = t - mean_t
        var_e = float(np.dot(e_c, e_c) / (n - 1))
        var_t = float(np.dot(t_c, t_c) / (n - 1))
        cov_et = float(np.dot(e_c, t_c) / (n - 1))
        var_r = (var_e + (r**2) * var_t - 2.0 * r * cov_et) / (n * mean_t**2)
        se = float(np.sqrt(max(var_r, 0.0)))
    return NmseUncertainty(
        n_trials=n,
        total_error_energy=sum_e,
        total_true_energy=sum_t,
        nmse_linear=r,
        se_linear=se,
    )

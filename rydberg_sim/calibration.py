"""SNR and RSR power calibration (Step 6).

Maps ``(SNR_dB, RSR_dB)`` to the forward-model parameters ``sigma2``
and ``|alpha_b|``. These functions are independent of ``forward.py``:
the observation model never decides SNR or RSR.

SNR (Cui / SystemModel, total signal power)
-------------------------------------------
    SNR = E[|(GS)_{n,p}|^2] / E[|W_{n,p}|^2]

With ``E|s_{k,p}|^2 = 1``, independent users, and
``E|g_{n,k}|^2 = c^2 beta_k``:

    E[|(GS)_{n,p}|^2] = c^2 * sum_k beta_k
    E[|W_{n,p}|^2]    = sigma2

    SNR_lin = 10**(SNR_dB / 10)
    sigma2  = c^2 * sum_k beta_k / SNR_lin

SNR uses **total** signal power from all K users. For the baseline
``c = 1``, ``beta_k = 1`` this is ``sigma2 = K / SNR_lin``.
Fixed total SNR does **not** imply fixed per-user SNR: doubling K at
fixed SNR_dB doubles ``sigma2``.

RSR (Cui, single-user denominator)
----------------------------------
    RSR = E[|B_{n,p}|^2] / E[|g_{n,k} s_{k,p}|^2]

The denominator is **one user's** contribution, not ``E[|(GS)_{n,p}|^2]``.
Do not sum over K here.

    E[|g_{n,k} s_{k,p}|^2] = c^2 * beta_k
    E[|B_{n,p}|^2]         = c^2 * |alpha_b|^2 * E[|s_b|^2]

so ``c`` cancels and

    |alpha_b| = sqrt( RSR_lin * beta_ref / E[|s_b|^2] )

``beta_ref`` is the large-scale power of the **explicitly chosen**
reference user. It is never silently replaced by ``mean(beta_k)``.

Baseline ``beta_ref = 1``, ``E[|s_b|^2] = 1`` (``s_b[p] = 1``):

    |alpha_b| = sqrt(RSR_lin)

independent of K. It is **not** ``sqrt(K * RSR_lin)`` or
``sqrt(RSR_lin / K)``.

``alpha_b = |alpha_b| * exp(j phi_b)``. Only the magnitude is fixed by
RSR; ``phi_b`` does not change RSR.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def db_to_linear(x_db: float) -> float:
    """``10**(x_db / 10)``."""
    x = float(x_db)
    if not np.isfinite(x):
        raise ValueError(f"dB value must be finite, got {x_db!r}")
    return float(10.0 ** (x / 10.0))


def linear_to_db(x_lin: float) -> float:
    """``10 * log10(x_lin)``."""
    x = float(x_lin)
    if not np.isfinite(x) or x <= 0.0:
        raise ValueError(f"linear value must be finite and > 0, got {x_lin!r}")
    return float(10.0 * np.log10(x))


def _as_beta_vector(beta_k: object) -> np.ndarray:
    arr = np.atleast_1d(np.asarray(beta_k, dtype=np.float64))
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"beta_k must be a non-empty 1-D sequence, got {np.shape(beta_k)}")
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError(f"every beta_k must be finite and > 0, got {beta_k!r}")
    return arr


def _as_positive_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must not be a boolean, got {value!r}")
    x = float(value)  # type: ignore[arg-type]
    if not np.isfinite(x) or x <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return x


def reference_user_beta(beta_k: object, user_index: int) -> float:
    """Return ``beta_k[user_index]``. Never substitutes ``mean(beta_k)``."""
    betas = _as_beta_vector(beta_k)
    if isinstance(user_index, (bool, np.bool_)) or int(user_index) != user_index:
        raise TypeError(f"user_index must be an integer, got {user_index!r}")
    idx = int(user_index)
    if idx < 0 or idx >= betas.size:
        raise ValueError(
            f"user_index={idx} out of range for K={betas.size}"
        )
    return float(betas[idx])


def snr_db_to_sigma2(
    snr_db: float,
    beta_k: object,
    c: float = 1.0,
) -> float:
    """Complex noise variance from target SNR in dB.

    ``sigma2 = c^2 * sum(beta_k) / 10**(SNR_dB/10)``.

    ``beta_k`` is the per-user sequence (length K). A scalar is treated
    as **one** user, not as ``K`` implicit copies — pass ``(1,)*K`` or
    ``np.ones(K)`` for the equal-beta baseline.
    """
    snr_lin = db_to_linear(snr_db)
    betas = _as_beta_vector(beta_k)
    c_val = _as_positive_scalar(c, "c")
    return float((c_val**2) * float(np.sum(betas)) / snr_lin)


def rsr_db_to_alpha_magnitude(
    rsr_db: float,
    beta_ref: float,
    *,
    e_s_b_sq: float = 1.0,
) -> float:
    """``|alpha_b|`` from target RSR in dB and a single-user ``beta_ref``.

    ``|alpha_b| = sqrt( RSR_lin * beta_ref / E[|s_b|^2] )``.

    ``beta_ref`` is the large-scale power of the chosen reference user,
    **not** ``sum(beta_k)`` or ``mean(beta_k)``. ``c`` does not appear
    because it cancels between ``E|B|^2`` and ``E|g_k s_k|^2``.
    """
    rsr_lin = db_to_linear(rsr_db)
    b_ref = _as_positive_scalar(beta_ref, "beta_ref")
    esb = _as_positive_scalar(e_s_b_sq, "e_s_b_sq")
    return float(np.sqrt(rsr_lin * b_ref / esb))


def make_alpha_b(magnitude: float, phi_b: float = 0.0) -> complex:
    """``alpha_b = |alpha_b| * exp(1j * phi_b)``.

    Only ``magnitude`` is fixed by RSR. Baseline ``phi_b = 0``.
    """
    mag = _as_positive_scalar(magnitude, "magnitude")
    phase = float(phi_b)
    if not np.isfinite(phase):
        raise ValueError(f"phi_b must be finite, got {phi_b!r}")
    return complex(mag * np.exp(1j * phase))


@dataclass(frozen=True)
class MeasuredSNR:
    """Empirical SNR from generated ``G, S, W`` (linear powers, then dB)."""

    signal_power: float
    noise_power: float
    snr_lin: float
    snr_db: float


@dataclass(frozen=True)
class MeasuredRSR:
    """Empirical RSR from generated ``B, G, S`` with a single-user denominator."""

    reference_power: float
    single_user_power: float
    user_index: int
    rsr_lin: float
    rsr_db: float


def mean_abs_sq(arr: np.ndarray) -> float:
    """``mean(|arr|^2)`` over all entries."""
    a = np.asarray(arr)
    if a.size == 0:
        raise ValueError("cannot average an empty array")
    return float(np.mean(np.abs(a) ** 2))


def signal_mean_power(G: np.ndarray, S: np.ndarray) -> float:
    """``mean(|(G @ S)_{n,p}|^2)`` — total multi-user signal power."""
    return mean_abs_sq(np.matmul(G, S))


def single_user_mean_power(G: np.ndarray, S: np.ndarray, user_index: int) -> float:
    """``mean(|G[:, k] S[k, :]|^2)`` — Cui single-user RSR denominator."""
    G = np.asarray(G)
    S = np.asarray(S)
    if G.ndim != 2 or S.ndim != 2:
        raise ValueError("G and S must be 2-D")
    n_users = G.shape[1]
    if S.shape[0] != n_users:
        raise ValueError(f"G.shape[1]={n_users} != S.shape[0]={S.shape[0]}")
    idx = int(user_index)
    if idx < 0 or idx >= n_users:
        raise ValueError(f"user_index={idx} out of range for K={n_users}")
    return mean_abs_sq(G[:, idx : idx + 1] @ S[idx : idx + 1, :])


def measure_snr(G: np.ndarray, S: np.ndarray, W: np.ndarray) -> MeasuredSNR:
    """Estimate SNR from actual generated matrices (not from the dB target)."""
    sig = signal_mean_power(G, S)
    noise = mean_abs_sq(W)
    if noise <= 0.0:
        raise ValueError("noise power must be > 0 to measure SNR")
    snr_lin = sig / noise
    return MeasuredSNR(
        signal_power=sig,
        noise_power=noise,
        snr_lin=snr_lin,
        snr_db=linear_to_db(snr_lin),
    )


def measure_rsr(
    B: np.ndarray,
    G: np.ndarray,
    S: np.ndarray,
    user_index: int = 0,
) -> MeasuredRSR:
    """Estimate RSR against **one** user, not against full ``G @ S``."""
    ref = mean_abs_sq(B)
    user = single_user_mean_power(G, S, user_index)
    if user <= 0.0:
        raise ValueError("single-user signal power must be > 0 to measure RSR")
    rsr_lin = ref / user
    return MeasuredRSR(
        reference_power=ref,
        single_user_power=user,
        user_index=int(user_index),
        rsr_lin=rsr_lin,
        rsr_db=linear_to_db(rsr_lin),
    )

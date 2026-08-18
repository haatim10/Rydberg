"""Exact and linearised observation models (Step 5).

This module implements the **observation equations only**. It does not
choose SNR, RSR, ``sigma2`` from SNR, or ``alpha_b`` from RSR. Those
belong to Step 6. Here ``sigma2`` is an explicit input and ``B`` is
supplied by the caller (Step 3).

Part A — exact model (SystemModel.pdf §7–8)
------------------------------------------
    E = G @ S + B + W
    Z = |E|                 (elementwise amplitude, not |E|²)

with ``vec(W) ~ CN(0, sigma2 I)``, i.e. each entry satisfies
``E[|W[n,p]|²] = sigma2``. Generation uses the Step-1 **noise** stream:

    W = sqrt(sigma2 / 2) * (X + j Y),    X, Y i.i.d. N(0, 1).

``sigma2 = 0`` is supported and yields ``W = 0``.

Part B — strong-reference linearisation (SystemModel.pdf §11)
-------------------------------------------------------------
Starting from the **same** exact ``Z`` (never a second independent
draw of ``Nbar`` on top of ``Z``):

    Y = Z - |B|
    Ψ = exp(-j ∠B)                 (elementwise; this sign is required)
    Y_linear_signal = Re{ Ψ ⊙ (G @ S) }

The first-order model is ``Y ≈ Y_linear_signal + Nbar`` with
``Nbar ~ N(0, sigma2/2)`` describing the *effective* residual noise
after linearisation. That statement is not permission to add another
noise realization.

There is **no** synthetic linearised generator here. Strong-reference
behaviour including RSR = 30 dB is checked with Step-6 calibration;
this module still does not compute RSR or SNR itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rng import get_trial_rngs


@dataclass(frozen=True, eq=False)
class ExactObservation:
    """One exact magnitude-only observation ``Z = |G S + B + W|``.

    Attributes
    ----------
    Z
        Elementwise amplitude, shape ``(N, P)``, float64, nonnegative.
        This is ``|E|``, not ``|E|²``.
    E
        Complex field ``G @ S + B + W``, shape ``(N, P)``, complex128.
    signal
        ``G @ S``, shape ``(N, P)``, complex128.
    B
        Reference, shape ``(N, P)``, complex128.
    W
        Noise, shape ``(N, P)``, complex128. All zeros if ``sigma2 == 0``.
    sigma2
        Complex noise variance: ``E[|W[n,p]|²] = sigma2``.
    G, S
        Copies of the inputs, shapes ``(N, K)`` and ``(K, P)``.
    """

    Z: np.ndarray
    E: np.ndarray
    signal: np.ndarray
    B: np.ndarray
    W: np.ndarray
    sigma2: float
    G: np.ndarray
    S: np.ndarray


@dataclass(frozen=True, eq=False)
class LinearisedObservation:
    """Strong-reference linearisation of an exact observation.

    ``Y`` is always ``Z - |B|`` from the exact model. No extra ``Nbar``
    is added.

    Attributes
    ----------
    Y
        Debiased observation ``Z - |B|``, shape ``(N, P)``, float64.
    Psi
        ``exp(-1j * angle(B))``, shape ``(N, P)``, complex128, unit modulus.
    Y_linear_signal
        ``Re{Psi ⊙ (G @ S)}``, shape ``(N, P)``, float64.
    residual
        ``Y - Y_linear_signal`` (Taylor remainder plus effective noise).
    relative_frobenius_error
        ``||residual||_F / ||Y_linear_signal||_F``.
    """

    Y: np.ndarray
    Psi: np.ndarray
    Y_linear_signal: np.ndarray
    residual: np.ndarray
    relative_frobenius_error: float


def _require_finite(arr: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")


def _as_complex_matrix(value: object, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.complex128)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {arr.shape}")
    _require_finite(arr, name)
    out = np.array(arr, dtype=np.complex128, copy=True)
    out.flags.writeable = False
    return out


def _as_sigma2(value: object) -> float:
    try:
        sigma2 = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"sigma2 must be a real number, got {value!r}") from exc
    if not np.isfinite(sigma2):
        raise ValueError(f"sigma2 must be finite, got {value!r}")
    if sigma2 < 0.0:
        raise ValueError(f"sigma2 must be >= 0, got {sigma2}")
    return sigma2


def _freeze(arr: np.ndarray) -> np.ndarray:
    arr.flags.writeable = False
    return arr


def reference_phase_matrix(B: np.ndarray) -> np.ndarray:
    """``Ψ = exp(-1j ∠B)``, unit-modulus, same sign as SystemModel.pdf §11.

    Undefined if any entry of ``B`` is zero. The linearised Fisher
    information depends on this phase, not on ``|B|``.
    """
    B_arr = _as_complex_matrix(B, "B")
    if np.any(B_arr == 0):
        raise ValueError(
            "reference_phase_matrix requires B[n,p] != 0 for all n,p "
            "(angle(B) is undefined at zeros)"
        )
    psi = np.exp(-1j * np.angle(B_arr)).astype(np.complex128, copy=False)
    return _freeze(np.array(psi, dtype=np.complex128, copy=True))


def exact_forward(
    G: np.ndarray,
    S: np.ndarray,
    B: np.ndarray,
    sigma2: float,
    rng_noise: np.random.Generator | None = None,
    *,
    master_seed: int | None = None,
    trial_index: int | None = None,
) -> ExactObservation:
    """Exact observation ``Z = |G @ S + B + W|``.

    Parameters
    ----------
    G, S, B
        Channel, pilots, and reference. Shapes ``(N, K)``, ``(K, P)``,
        ``(N, P)``.
    sigma2
        Complex noise variance ``E[|W|²]``. ``0`` yields ``W = 0`` and
        does not consume an RNG.
    rng_noise
        Optional Step-1 **noise** ``Generator``. Used when ``sigma2 > 0``.
    master_seed, trial_index
        Address ``get_trial_rngs(...).noise`` when ``rng_noise`` is
        omitted and ``sigma2 > 0``.

    Notes
    -----
    ``Z`` is the elementwise **amplitude**. This function never returns
    ``|E|²``. It does not interpret SNR or RSR.
    """
    G_arr = _as_complex_matrix(G, "G")
    S_arr = _as_complex_matrix(S, "S")
    B_arr = _as_complex_matrix(B, "B")
    sigma2_val = _as_sigma2(sigma2)

    n_rx, n_users = G_arr.shape
    k_s, n_pilots = S_arr.shape
    if k_s != n_users:
        raise ValueError(
            f"incompatible G and S: G.shape={G_arr.shape}, S.shape={S_arr.shape} "
            "(expected S.shape[0] == G.shape[1])"
        )
    if B_arr.shape != (n_rx, n_pilots):
        raise ValueError(
            f"incompatible B: B.shape={B_arr.shape}, expected ({n_rx}, {n_pilots}) "
            "to match (G.shape[0], S.shape[1])"
        )

    signal = np.matmul(G_arr, S_arr, dtype=np.complex128)

    if sigma2_val == 0.0:
        W = np.zeros((n_rx, n_pilots), dtype=np.complex128)
    else:
        if rng_noise is None:
            if master_seed is None or trial_index is None:
                raise ValueError(
                    "sigma2 > 0 requires rng_noise or both master_seed and trial_index"
                )
            rng_noise = get_trial_rngs(int(master_seed), int(trial_index)).noise
        elif not isinstance(rng_noise, np.random.Generator):
            raise TypeError(
                f"rng_noise must be a numpy Generator, got {type(rng_noise)!r}"
            )
        scale = np.sqrt(sigma2_val / 2.0)
        real = rng_noise.standard_normal((n_rx, n_pilots))
        imag = rng_noise.standard_normal((n_rx, n_pilots))
        W = (scale * real + 1j * scale * imag).astype(np.complex128, copy=False)

    E = np.asarray(signal + B_arr + W, dtype=np.complex128)
    Z = np.abs(E).astype(np.float64, copy=False)

    return ExactObservation(
        Z=_freeze(Z),
        E=_freeze(E),
        signal=_freeze(signal),
        B=B_arr,
        W=_freeze(W),
        sigma2=sigma2_val,
        G=G_arr,
        S=S_arr,
    )


def linearised_observation(exact: ExactObservation) -> LinearisedObservation:
    """Debias an exact observation: ``Y = Z - |B|``, no extra noise.

    Uses ``Psi = exp(-1j * angle(B))`` exactly as in SystemModel.pdf
    Section 11. The linear signal term is ``Re{Psi ⊙ (G @ S)}``.
    """
    if np.any(exact.B == 0):
        raise ValueError(
            "linearised observation requires B[n,p] != 0 for all n,p "
            "(angle(B) is undefined at zeros)"
        )

    abs_B = np.abs(exact.B)
    Y = np.asarray(exact.Z - abs_B, dtype=np.float64)
    # Required sign: minus, not plus. Shared with the linearised CRLB.
    Psi = reference_phase_matrix(exact.B)
    Y_linear_signal = np.real(Psi * exact.signal).astype(np.float64, copy=False)
    residual = np.asarray(Y - Y_linear_signal, dtype=np.float64)

    denom = float(np.linalg.norm(Y_linear_signal, ord="fro"))
    numer = float(np.linalg.norm(residual, ord="fro"))
    if denom == 0.0:
        rel = float("inf") if numer > 0.0 else 0.0
    else:
        rel = numer / denom

    return LinearisedObservation(
        Y=_freeze(Y),
        Psi=_freeze(Psi),
        Y_linear_signal=_freeze(Y_linear_signal),
        residual=_freeze(residual),
        relative_frobenius_error=rel,
    )

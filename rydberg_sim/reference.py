"""Known reference field from SystemModel.pdf Section 6 (Step 3).

The reference is a single line-of-sight path from a **fixed, known**
geometry at angle ``vartheta``. It is fully known at the receiver
(A9, A11). This module does **not** draw ``vartheta`` randomly, does
**not** consume the Step-1 ``reference`` RNG stream, and does **not**
calibrate ``alpha_b`` from RSR (that is Step 6).

Model
-----
With ``d = λ/2``,

    ψ_b = π sin(ϑ)

    a_b = [1, exp(-j ψ_b), …, exp(-j (N-1) ψ_b)]^T   ∈ ℂ^N

    b_{n,p} = c · α_b · s_{b,p} · exp(-j (n-1) ψ_b)

In zero-based Python indexing ``n = 0, …, N-1`` this is

    B[n, p] = c * alpha_b * s_b[p] * exp(-j * n * psi_b)

or, equivalently, the outer product

    B = c * alpha_b * a_b * s_b^T    ∈ ℂ^{N × P}

Baseline waveform
-----------------
``s_b[p] = 1`` for every pilot instant ``p``. All columns of ``B`` are
therefore identical. The API accepts an explicit known complex vector
``s_b`` of shape ``(P,)`` for a later ablation; that ablation is **not**
performed here. In all cases ``s_b`` is known at the receiver.

``c = 1`` is the same numerical normalization as Step 2: it does not
mean the physical atomic conversion gain equals 1.

``alpha_b`` is an explicit nonzero input. It is not derived from RSR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channel import spatial_frequency, steering_vector


@dataclass(frozen=True, eq=False)
class ReferenceField:
    """One known reference-field realization.

    Attributes
    ----------
    B
        Reference matrix, shape ``(N, P)``, complex128.
        ``B[n, p] = c * alpha_b * s_b[p] * exp(-1j * n * psi_b)``.
    a_b
        Spatial steering vector, shape ``(N,)``, complex128.
    s_b
        Known reference symbols, shape ``(P,)``, complex128.
        Baseline: ``s_b[p] = 1``.
    alpha_b
        Complex reference-path coefficient. Nonzero by construction.
    vartheta
        Known reference arrival angle in radians (fixed, not redrawn).
    psi_b
        Spatial frequency ``π sin(vartheta)``.
    c
        Common known conversion gain. Default ``1.0`` is a numerical
        normalization, not a physical claim.
    """

    B: np.ndarray
    a_b: np.ndarray
    s_b: np.ndarray
    alpha_b: complex
    vartheta: float
    psi_b: float
    c: float


def _as_positive_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    try:
        value_int = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer, got {value!r}") from exc
    if value_int != value:
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if value_int <= 0:
        raise ValueError(f"{name} must be > 0, got {value_int}")
    return value_int


def _as_positive_c(value: object) -> float:
    c = float(value)
    if not np.isfinite(c) or c <= 0.0:
        raise ValueError(f"c must be finite and > 0, got {c}")
    return c


def _as_alpha_b(value: object) -> complex:
    alpha = complex(value)
    if not np.isfinite(alpha.real) or not np.isfinite(alpha.imag):
        raise ValueError(f"alpha_b must be finite, got {value!r}")
    if alpha == 0:
        raise ValueError("alpha_b must be nonzero")
    return alpha


def _as_vartheta(value: object) -> float:
    theta = float(value)
    if not np.isfinite(theta):
        raise ValueError(f"vartheta must be finite, got {value!r}")
    return theta


def _as_s_b(s_b: np.ndarray | None, P: int) -> np.ndarray:
    if s_b is None:
        # Baseline: fixed known waveform s_{b,p} = 1.
        out = np.ones(P, dtype=np.complex128)
        out.flags.writeable = False
        return out
    arr = np.asarray(s_b)
    if arr.ndim != 1 or arr.shape[0] != P:
        raise ValueError(
            f"s_b must have shape (P,) with P={P}, got shape {arr.shape}"
        )
    out = np.array(arr, dtype=np.complex128, copy=True)
    if not np.all(np.isfinite(out)):
        raise ValueError("s_b must be finite")
    if np.any(out == 0):
        raise ValueError("s_b must be nonzero for every pilot instant")
    out.flags.writeable = False
    return out


def generate_reference_field(
    *,
    N: int,
    P: int,
    alpha_b: complex,
    vartheta: float,
    c: float = 1.0,
    s_b: np.ndarray | None = None,
) -> ReferenceField:
    """Build the known reference matrix ``B ∈ ℂ^{N × P}``.

    Parameters
    ----------
    N
        Number of receive ULA elements.
    P
        Number of pilot instants.
    alpha_b
        Complex reference-path coefficient. Must be nonzero. Not
        calibrated from RSR in this step.
    vartheta
        Fixed known arrival angle of the reference (radians).
    c
        Common known conversion factor. Default ``1.0`` is a numerical
        normalization, not a physical conversion-gain claim.
    s_b
        Optional known reference waveform of shape ``(P,)``. If omitted,
        the baseline ``s_b[p] = 1`` is used. A caller-supplied vector is
        supported for a later ablation and is **not** used as the
        baseline. ``s_b`` is known at the receiver in either case.

    Returns
    -------
    ReferenceField
        Structured ground-truth reference quantities, including ``B``.

    Notes
    -----
    This function is deterministic given its arguments. It does not
    consume ``get_trial_rngs(...).reference``.
    """
    N = _as_positive_int(N, "N")
    P = _as_positive_int(P, "P")
    c = _as_positive_c(c)
    alpha = _as_alpha_b(alpha_b)
    theta = _as_vartheta(vartheta)
    symbols = _as_s_b(s_b, P)

    psi_b = float(spatial_frequency(theta))
    a_b = np.array(steering_vector(theta, N), dtype=np.complex128, copy=True)
    a_b.flags.writeable = False

    B = np.outer(c * alpha * a_b, symbols).astype(np.complex128, copy=False)
    B.flags.writeable = False

    return ReferenceField(
        B=B,
        a_b=a_b,
        s_b=symbols,
        alpha_b=alpha,
        vartheta=theta,
        psi_b=psi_b,
        c=c,
    )

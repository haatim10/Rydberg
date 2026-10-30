"""Complex Gaussian channel-estimation pilots (Step 4, Part A).

This module generates the known pilot matrix

    S ∈ ℂ^{K × P},    s_{k,p} ~ CN(0, 1) i.i.d.

It is **not** a QAM generator. Estimation pilots are circularly
symmetric Gaussians; finite-alphabet QAM lives in ``qam.py`` and is
used later for Track A / Track C detection, not for estimating G.

Model
-----
A proper CN(0, 1) sample is

    s = (x + j y) / √2,    x, y ~ N(0, 1) independent

which yields ``E[|s|²] = 1``.

Identifiability (SystemModel.pdf §10)
-------------------------------------
The unstructured per-row counting bound requires ``P ≥ 2K`` together
with ``S`` of full row rank (``rank(S) = K``). This generator enforces
both.

Rank criterion
--------------
``S`` (shape ``(K, P)``) is accepted iff it has full row rank under the
relative singular-value test

    σ_min(S) ≥ PILOT_RANK_SV_REL_TOL * σ_max(S)

with ``PILOT_RANK_SV_REL_TOL = 1e-8``. This is the same style of
criterion as Step 2, **not** ``numpy.linalg.matrix_rank``'s default
tolerance. A degenerate draw is discarded and a new matrix is sampled
from the **same** pilot RNG stream. ``S`` is never silently modified
(no QR / orthogonalization / replacement of entries).

RNG
---
Randomness comes from the **pilots** substream of
``get_trial_rngs(master_seed, trial_index)`` unless an explicit
``Generator`` is injected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channel import is_full_column_rank
from .rng import get_trial_rngs

# Relative singular-value floor for full row rank of S.
# Not NumPy's matrix_rank default (max(K, P) * eps * sigma_max).
PILOT_RANK_SV_REL_TOL: float = 1e-8

MAX_PILOT_DRAWS: int = 64


@dataclass(frozen=True, eq=False)
class PilotMatrix:
    """Known complex Gaussian estimation pilots.

    Attributes
    ----------
    S
        Pilot matrix, shape ``(K, P)``, complex128, i.i.d. CN(0, 1),
        full row rank. Known at the receiver.
    K
        Number of users (rows).
    P
        Number of pilot instants (columns). ``P >= 2K``.
    """

    S: np.ndarray
    K: int
    P: int


def is_full_row_rank(
    S: np.ndarray, *, rel_tol: float = PILOT_RANK_SV_REL_TOL
) -> bool:
    """Return True iff ``S`` has full row rank under the SV criterion.

    For ``S`` of shape ``(K, P)`` with ``P ≥ K``, this is equivalent to
    full column rank of ``S.T`` (rank is transposition-invariant).
    """
    S = np.asarray(S)
    if S.ndim != 2:
        raise ValueError(f"S must be 2-D, got shape {S.shape}")
    return is_full_column_rank(S.T, rel_tol=rel_tol)


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


def _complex_normal_unit(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """i.i.d. ``CN(0, 1)`` with ``E[|s|²] = 1``."""
    scale = 1.0 / np.sqrt(2.0)
    real = rng.standard_normal(shape)
    imag = rng.standard_normal(shape)
    return (scale * real + 1j * scale * imag).astype(np.complex128, copy=False)


def generate_gaussian_pilots(
    *,
    K: int,
    P: int,
    master_seed: int | None = None,
    trial_index: int | None = None,
    rng: np.random.Generator | None = None,
) -> PilotMatrix:
    """Draw a known CN(0, 1) pilot matrix ``S ∈ ℂ^{K × P}``.

    Parameters
    ----------
    K
        Number of users (rows).
    P
        Number of pilot instants (columns). Must satisfy ``P ≥ 2K``.
    master_seed, trial_index
        Address the Step-1 **pilots** substream when ``rng`` is omitted.
    rng
        Optional injected ``Generator``. Intended for tests. When
        omitted, ``get_trial_rngs(master_seed, trial_index).pilots`` is
        used.

    Notes
    -----
    These pilots are **not** QAM symbols. Do not pass a QAM sequence as
    ``S`` for channel estimation.
    """
    K = _as_positive_int(K, "K")
    P = _as_positive_int(P, "P")
    if P < 2 * K:
        raise ValueError(
            f"estimation pilots require P >= 2K (identifiability), got P={P}, K={K}"
        )

    if rng is None:
        if master_seed is None or trial_index is None:
            raise ValueError("provide rng or both master_seed and trial_index")
        rng = get_trial_rngs(int(master_seed), int(trial_index)).pilots

    for _ in range(MAX_PILOT_DRAWS):
        S = _complex_normal_unit(rng, (K, P))
        if is_full_row_rank(S, rel_tol=PILOT_RANK_SV_REL_TOL):
            S.flags.writeable = False
            return PilotMatrix(S=S, K=K, P=P)

    raise RuntimeError(
        f"Failed to draw a full-row-rank {K}x{P} pilot matrix in "
        f"{MAX_PILOT_DRAWS} attempts (PILOT_RANK_SV_REL_TOL={PILOT_RANK_SV_REL_TOL})"
    )

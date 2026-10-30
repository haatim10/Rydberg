"""Structure-aware EXACT-model channel estimator (Track B, Step B4 prototype).

**This is a labelled prototype, not a proposed algorithm.**

The whole point of Track B is to test whether ULA multipath structure can
improve channel estimation *without* the strong-reference linearisation.
So the measurement update here is always the exact magnitude-only model

    Z = |G S + B + W|

evaluated through the already-validated biased-GS / EM-GS row adapters.
``forward.py``'s Part-B linearisation (``Y = Z - |B|``, ``Ψ = exp(-j∠B)``)
is **never** called on this path, and there is a runtime guard asserting so.

Iteration
---------
Alternate, ``n_outer`` times::

    G  <-  exact nonlinear estimation step   (biased GS or EM-GS on Z)
    G  <-  projection onto ULA multipath structure
    G  <-  exact nonlinear estimation step, warm-started from that projection

The exact step is warm-started from the projected iterate through the
solvers' ``G0`` argument, so structure enters as an initialisation prior
rather than as a hard constraint on the output. Each outer round is scored
by the magnitude-domain objective

    J(G) = || Z - |G S + B| ||_F^2

which is computable from observed quantities only — it never sees the true
``G`` — and the best-scoring iterate is returned. That guards against a
projection that is actively harmful (wrong path count, off-grid paths)
making the estimate worse than plain GS.

The projection is fully modular (``rydberg_sim.track_b_structure``:
``none`` / ``angular`` / ``hankel`` / ``esprit``) precisely so the three can
be compared later on equal footing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .gs import biased_gs_channel_rows, em_gs_channel_rows
from .track_b_structure import PROJECTIONS, project_matrix

EXACT_STEPS = ("biased_gs", "em_gs")


@dataclass(frozen=True)
class StructuredExactResult:
    """Outcome of the alternating exact/structural prototype."""

    G_hat: np.ndarray
    G_exact_only: np.ndarray
    objective_history: np.ndarray
    best_round: int
    projection: str
    exact_step: str
    n_paths: np.ndarray
    used_projection: bool
    per_round: list[dict] = field(default_factory=list)
    linearised_model_used: bool = False  # invariant: must stay False


def magnitude_objective(
    G: np.ndarray, S: np.ndarray, B: np.ndarray, Z: np.ndarray
) -> float:
    """``|| Z - |G S + B| ||_F^2`` — observable, never uses the true G."""
    resid = np.asarray(Z, dtype=np.float64) - np.abs(
        np.asarray(G, dtype=np.complex128) @ np.asarray(S, dtype=np.complex128)
        + np.asarray(B, dtype=np.complex128)
    )
    return float(np.sum(resid * resid))


def _exact_step(
    step: str,
    S: np.ndarray,
    Z: np.ndarray,
    B: np.ndarray,
    sigma2: float,
    *,
    max_iter: int,
    ridge: float,
    G0: np.ndarray | None,
) -> np.ndarray:
    """One exact magnitude-only estimation step. Reuses validated solvers."""
    if step == "biased_gs":
        return biased_gs_channel_rows(
            S, Z, B, max_iter=max_iter, ridge=ridge, G0=G0
        ).G_hat
    if step == "em_gs":
        return em_gs_channel_rows(
            S, Z, B, sigma2, max_iter=max_iter, ridge=ridge, G0=G0
        ).G_hat
    raise ValueError(f"unknown exact step {step!r}; choose from {EXACT_STEPS}")


def structured_exact_estimate(
    S: np.ndarray,
    Z: np.ndarray,
    B: np.ndarray,
    sigma2: float,
    *,
    exact_step: str = "em_gs",
    projection: str = "hankel",
    n_paths: int | np.ndarray | str = "auto",
    n_outer: int = 3,
    accept: str = "always",
    max_iter: int = 50,
    inner_iter: int | None = None,
    ridge: float = 0.0,
    n_grid: int | None = None,
    cadzow_iter: int = 8,
) -> StructuredExactResult:
    """Alternate exact nonlinear estimation with ULA structural projection.

    Parameters mirror the validated solvers. ``n_paths`` is the assumed
    multipath order per user — a scalar or a length-``K`` array. It is a
    *modelling choice*, never read from the truth; the drivers pass a fixed
    guess (the midpoint of the ``L_k ~ U{3..7}`` prior) rather than the
    realised ``L_k``.

    ``accept`` decides which iterate is returned:

    ``"always"`` (default)
        return the final projected-and-re-estimated iterate.
    ``"objective"``
        return the best iterate under :func:`magnitude_objective`.
        **Measured to be an invalid selector** — see the warning below.

    .. warning::
       In-sample ``J`` is not a usable acceptance test here. A projection
       constrains ``G`` to a lower-dimensional manifold, so it can only fit
       ``Z`` *worse*, even when it reduces the true error. Measured on this
       model (N=16, K=3, Hankel, order 7): ``ΔJ`` was positive at every SNR
       tested while ``ΔNMSE_G`` was negative at 5/10/20 dB, with the signs
       agreeing on only 1–8 of 10 trials. ``accept="objective"`` therefore
       rejects beneficial projections almost always. Selecting a projection
       needs a complexity-penalised or held-out-pilot criterion, not raw
       in-sample fit; the order is instead chosen from the data by
       ``n_paths="auto"`` (:func:`~rydberg_sim.track_b_structure.estimate_order`).

    Returns the chosen iterate alongside the projection-free result, so a
    driver can always report both on the same realization.
    """
    if accept not in ("always", "objective"):
        raise ValueError(f"accept must be 'always' or 'objective', got {accept!r}")
    if exact_step not in EXACT_STEPS:
        raise ValueError(f"exact_step must be one of {EXACT_STEPS}")
    if projection not in PROJECTIONS:
        raise ValueError(f"projection must be one of {PROJECTIONS}")
    if int(n_outer) < 1:
        raise ValueError(f"n_outer must be >= 1, got {n_outer}")
    S = np.asarray(S, dtype=np.complex128)
    Z = np.asarray(Z, dtype=np.float64)
    B = np.asarray(B, dtype=np.complex128)
    inner = int(inner_iter) if inner_iter is not None else int(max_iter)

    # Round 0: plain exact estimation, no structure. This is the baseline
    # the prototype must beat to justify itself.
    G = _exact_step(exact_step, S, Z, B, sigma2, max_iter=max_iter,
                    ridge=ridge, G0=None)
    G_exact_only = G.copy()
    if isinstance(n_paths, str):
        counts = n_paths           # resolved per-round inside project_matrix
    else:
        counts = (np.full(G.shape[1], int(n_paths)) if np.isscalar(n_paths)
                  else np.asarray(n_paths, dtype=int).ravel())

    best_G = G.copy()
    best_J = magnitude_objective(G, S, B, Z)
    history = [best_J]
    per_round = [{"round": 0, "stage": "exact_only", "objective": best_J}]
    best_round = 0
    used_projection = False

    for r in range(1, int(n_outer) + 1):
        G_proj = project_matrix(G, projection, counts, n_grid=n_grid,
                                cadzow_iter=cadzow_iter)
        J_proj = magnitude_objective(G_proj, S, B, Z)
        # re-enter the exact model, warm-started from the projection
        G = _exact_step(exact_step, S, Z, B, sigma2, max_iter=inner,
                        ridge=ridge, G0=G_proj)
        J = magnitude_objective(G, S, B, Z)
        history.append(J)
        per_round.append({
            "round": r, "stage": "project+exact",
            "objective_after_projection": J_proj, "objective": J,
        })
        if accept == "always":
            best_J, best_G, best_round = J, G.copy(), r
            used_projection = True
        elif J < best_J:
            best_J, best_G, best_round = J, G.copy(), r
            used_projection = True

    return StructuredExactResult(
        G_hat=best_G,
        G_exact_only=G_exact_only,
        objective_history=np.asarray(history, dtype=np.float64),
        best_round=best_round,
        projection=projection,
        exact_step=exact_step,
        n_paths=(np.asarray([], dtype=int) if isinstance(counts, str)
                 else counts),
        used_projection=used_projection,
        per_round=per_round,
        linearised_model_used=False,
    )


__all__ = [
    "EXACT_STEPS", "StructuredExactResult", "magnitude_objective",
    "structured_exact_estimate",
]

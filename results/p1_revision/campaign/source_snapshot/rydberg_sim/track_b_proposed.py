"""Proposed Track-B estimator: Hankel-structured exact GS / EM-GS (HS-GS).

Combines Xu-style geometric ULA structure with Cui's EXACT nonlinear
magnitude model. There is no strong-reference linearisation anywhere on
this path: every measurement update is a genuine Cui GS/EM-GS iteration on
``Z = |G S + B + W|``.

Optimisation problem being approximated
--------------------------------------
    minimise_G   J(G) = || Z - |G S + B| ||_F^2
    subject to   rank( H(g_k) ) <= L_k ,   k = 1..K

``H(·)`` is the Hankel operator on a channel column. The constraint is an
*exact* algebraic characterisation of the generative model, not a
surrogate: by Kronecker's theorem a length-``N`` sequence is a sum of
``L`` complex exponentials

    g_k[n] = Σ_ℓ α_{ℓk} e^{-j(n-1)ψ_{ℓk}},   ψ = π sin θ

**iff** its Hankel matrix has rank ``L`` (for distinct ψ and ``L`` within
the pencil bound). So the feasible set is precisely the set of channels
the ULA model can generate — the angles and gains never appear, and no
grid is introduced.

Why Hankel/Cadzow rather than ESPRIT or an angular grid
-------------------------------------------------------
*Angular OMP* discretises ψ onto a grid, so off-grid paths leave an
irreducible bias — measured at ~6.9 % residual even on noiseless,
exactly-structured data — and the oversampled dictionary is highly
coherent, making greedy support recovery unreliable. It is not grid-free.

*ESPRIT* is grid-free and exact in the noiseless case (recovered angles to
7.6e-15 here), but it is a *parameter estimator*, not a projection: it is
not idempotent, has no variational characterisation, and commits hard to
``L̂`` angles. Measured failure: with the pencil bound exceeded it collapsed
(-3.45 dB). There is no meaningful notion of "the ESPRIT projection of the
current iterate", so it cannot be composed into a convergent alternating
scheme.

*Cadzow* is grid-free **and** a genuine alternating projection between two
closed sets, each with an exact projector:

    - rank <= L matrices: SVD truncation, the exact Frobenius-norm
      projection by Eckart-Young-Mirsky;
    - Hankel matrices: anti-diagonal averaging, the exact orthogonal
      projection onto that linear subspace.

That gives the structural step a variational meaning the other two lack,
which is what allows it to be interleaved with GS as a POCS iteration.
Cadzow is therefore the most defensible choice, and is what is used here.

Why the structure must be enforced *inside* the iteration
---------------------------------------------------------
Cui's row adapter is separable across receive elements: element ``n`` is
estimated from ``z_n`` alone. The ULA structure is a coupling *along* ``n``,
so an unstructured GS sweep cannot see it.

Applying the projection once and then running GS to convergence is
measured to do nothing: the gain decays monotonically with the number of
unconstrained iterations after projection — +1.30 dB at 1, +0.06 at 10,
0.00 at 50. That is an EMPIRICAL observation on this configuration. No
contraction or convergence property is claimed: GS is not known to be a
contraction, alternating-projection schemes of this kind are not
contractions in general, and the fixed points of the composed map are not
characterised here.

The defensible mechanism for interleaving is the dependence chain

    G^(t)  ->  lambda^(t) = G^(t) S + B  ->  kappa^(t) = 2 Z |lambda^(t)| / sigma^2

The EM-GS update is driven by the Bessel ratio R(kappa), so the phase and
per-measurement reliability used at iteration t+1 are functions of the
iterate at t. Enforcing structure at t therefore changes the information
the next measurement update consumes, which a single post-hoc projection
cannot do. Whether that helps is an empirical question, answered by the
schedule ablation (interleaved vs post-hoc) and not by a theorem.

Identifiability caveat
----------------------
The Hankel matrix of a length-``N`` column has rank at most
``max_p min(N-p, p+1) = ceil(N/2)``. At the frozen Track-B size ``N = 8``
the cap is 4, so for ``L_k >= 5`` — 60 % of the ``U{3..7}`` prior — the true
channel is already full-rank and the constraint is **vacuous**. The
degrees-of-freedom reduction there is ``2NK/3ΣL_k = 48/45 = 1.07x``. The
prior only carries real information for ``N >= 16`` (2.13x) and above.
This is a property of the configuration, not of the algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .gs import biased_gs_channel_rows, em_gs_channel_rows
from .track_b_structure import hankel_matrix, hankel_to_vector

EXACT_STEPS = ("biased_gs", "em_gs")


def hankel_rank_cap(N: int, pencil: int | None = None) -> int:
    """Largest rank a length-``N`` Hankel matrix can have."""
    N = int(N)
    if pencil is None:
        return max(min(N - p, p + 1) for p in range(1, N))
    return min(N - int(pencil), int(pencil) + 1)


def best_pencil(N: int) -> int:
    """Pencil maximising the representable rank."""
    return max(range(1, int(N)), key=lambda p: min(int(N) - p, p + 1))


def cadzow_project(g: np.ndarray, rank: int, *, pencil: int | None = None,
                   n_iter: int = 4) -> np.ndarray:
    """One Cadzow projection of a single channel column.

    Alternates the two exact projectors described in the module docstring.
    ``rank`` is clipped to the pencil bound, above which the step is a
    no-op by construction.
    """
    g = np.asarray(g, dtype=np.complex128).ravel()
    p = best_pencil(g.size) if pencil is None else int(pencil)
    cap = hankel_rank_cap(g.size, p)
    r = int(np.clip(int(rank), 1, cap))
    if r >= cap:
        return g                      # constraint inactive; do not perturb
    cur = g
    for _ in range(max(1, int(n_iter))):
        Hk = hankel_matrix(cur, p)
        U, s, Vh = np.linalg.svd(Hk, full_matrices=False)
        cur = hankel_to_vector((U[:, :r] * s[:r]) @ Vh[:r])
    return cur


def _exact_iteration(step, S, Z, B, sigma2, G0, ridge):
    """Exactly one validated Cui measurement update.

    Chaining ``max_iter=1`` calls with ``G0`` reproduces a single
    ``max_iter=t`` call bit-for-bit (verified), so no GS mathematics is
    reimplemented here.
    """
    if step == "biased_gs":
        return biased_gs_channel_rows(S, Z, B, max_iter=1, ridge=ridge, G0=G0).G_hat
    if step == "em_gs":
        return em_gs_channel_rows(S, Z, B, sigma2, max_iter=1, ridge=ridge,
                                  G0=G0).G_hat
    raise ValueError(f"unknown exact step {step!r}")


def magnitude_residual(G, S, B, Z, cols=None) -> float:
    """``|| Z - |G S + B| ||_F^2`` on selected pilot columns (observable)."""
    S = S if cols is None else S[:, cols]
    B = B if cols is None else B[:, cols]
    Z = Z if cols is None else Z[:, cols]
    r = np.asarray(Z, float) - np.abs(np.asarray(G, complex) @ S + B)
    return float(np.sum(r * r))


@dataclass(frozen=True)
class HSGSResult:
    G_hat: np.ndarray
    L_hat: int
    exact_step: str
    n_iter: int
    residual_history: np.ndarray
    order_scores: dict = field(default_factory=dict)
    constraint_active: bool = True
    linearised_model_used: bool = False


def hs_gs(
    S: np.ndarray,
    Z: np.ndarray,
    B: np.ndarray,
    sigma2: float,
    *,
    L_hat: int,
    exact_step: str = "em_gs",
    max_iter: int = 50,
    project_every: int = 1,
    cadzow_iter: int = 4,
    ridge: float = 0.0,
    pencil: int | None = None,
) -> HSGSResult:
    """Hankel-structured exact GS: ``T = P_S ∘ T_GS`` iterated ``max_iter`` times."""
    if exact_step not in EXACT_STEPS:
        raise ValueError(f"exact_step must be one of {EXACT_STEPS}")
    S = np.asarray(S, dtype=np.complex128)
    Z = np.asarray(Z, dtype=np.float64)
    B = np.asarray(B, dtype=np.complex128)
    N = Z.shape[0]
    cap = hankel_rank_cap(N, pencil)
    active = int(L_hat) < cap

    G = None
    hist = []
    for t in range(1, int(max_iter) + 1):
        G = _exact_iteration(exact_step, S, Z, B, sigma2, G, ridge)
        if active and t % max(1, int(project_every)) == 0:
            for k in range(G.shape[1]):
                G[:, k] = cadzow_project(G[:, k], int(L_hat), pencil=pencil,
                                         n_iter=cadzow_iter)
        hist.append(magnitude_residual(G, S, B, Z))
    return HSGSResult(
        G_hat=G, L_hat=int(L_hat), exact_step=exact_step, n_iter=int(max_iter),
        residual_history=np.asarray(hist), constraint_active=bool(active),
        linearised_model_used=False,
    )


def select_order_heldout(
    S, Z, B, sigma2, *, candidates=None, exact_step="em_gs", max_iter=25,
    val_frac=0.3, ridge=0.0, pencil=None, cadzow_iter=4,
) -> tuple[int, dict]:
    """Pick ``L̂`` by held-out pilot residual. Uses no ground truth.

    The in-sample residual cannot be used: a projection constrains ``G`` to a
    smaller set, so it can only fit the fitted pilots worse, and the
    in-sample criterion therefore always prefers the largest ``L``.
    Splitting the pilot columns and scoring on the held-out half penalises
    over-modelling correctly, and every quantity involved (``S``, ``Z``,
    ``B``) is observable.
    """
    S = np.asarray(S, dtype=np.complex128)
    P = S.shape[1]
    n_val = max(1, int(round(val_frac * P)))
    val = np.arange(P - n_val, P)
    fit = np.arange(0, P - n_val)
    N = np.asarray(Z).shape[0]
    cap = hankel_rank_cap(N, pencil)
    cands = list(candidates) if candidates is not None else list(range(1, cap + 1))

    scores = {}
    for L in cands:
        r = hs_gs(S[:, fit], np.asarray(Z)[:, fit], np.asarray(B)[:, fit], sigma2,
                  L_hat=L, exact_step=exact_step, max_iter=max_iter,
                  ridge=ridge, pencil=pencil, cadzow_iter=cadzow_iter)
        scores[int(L)] = magnitude_residual(r.G_hat, S, B, Z, cols=val)
    L_best = min(scores, key=scores.get)
    return int(L_best), scores


def hs_gs_auto(
    S, Z, B, sigma2, *, exact_step="em_gs", max_iter=50, select_iter=25,
    candidates=None, project_every=1, cadzow_iter=4, ridge=0.0, pencil=None,
) -> HSGSResult:
    """HS-GS with the model order chosen from held-out pilots."""
    L_hat, scores = select_order_heldout(
        S, Z, B, sigma2, candidates=candidates, exact_step=exact_step,
        max_iter=select_iter, ridge=ridge, pencil=pencil, cadzow_iter=cadzow_iter,
    )
    res = hs_gs(S, Z, B, sigma2, L_hat=L_hat, exact_step=exact_step,
                max_iter=max_iter, project_every=project_every,
                cadzow_iter=cadzow_iter, ridge=ridge, pencil=pencil)
    return HSGSResult(
        G_hat=res.G_hat, L_hat=L_hat, exact_step=exact_step, n_iter=res.n_iter,
        residual_history=res.residual_history, order_scores=scores,
        constraint_active=res.constraint_active, linearised_model_used=False,
    )


__all__ = [
    "EXACT_STEPS", "HSGSResult", "best_pencil", "cadzow_project",
    "hankel_rank_cap", "hs_gs", "hs_gs_auto", "magnitude_residual",
    "select_order_heldout",
]

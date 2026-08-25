"""Hankel lifting, rank selection, and the Cadzow projection.

WHY A HANKEL MATRIX
-------------------
One channel column of a geometric ULA is a sum of L complex exponentials,

    g_n = sum_{l=1..L} alpha_l z_l^n ,   z_l = exp(-j psi_l).

Lift it to the Hankel matrix H with H[q, r] = g_{q+r}. Then

    H[q, r] = sum_l alpha_l z_l^{q+r} = sum_l alpha_l z_l^q z_l^r ,

which separates in q and r, so with v_l = [1, z_l, ...]^T (length N-p) and
w_l = [1, z_l, ...]^T (length p+1),

    H = sum_{l=1..L} alpha_l v_l w_l^T      ==>   rank(H) <= L.

Each propagation path contributes exactly one rank-one term. Path sparsity
becomes low-rank matrix structure, with no angular grid and no angle
estimation. Verified numerically by test 6 in verify_results.py.

RANK CEILING
------------
H is (N-p) x (p+1), so rank(H) <= min(N-p, p+1), maximised over the pencil
parameter p at r_max = ceil(N/2): 4, 8, 16 for N = 8, 16, 32. If L >= r_max
the constraint "rank <= L" is satisfied by every vector and carries NO
information. ``cadzow_project`` returns its input unchanged in that case, so
the Hankel estimator degenerates to the EM-GS baseline exactly.

All functions here delegate to the audited implementations in
``rydberg_sim/track_b_structure.py`` and ``rydberg_sim/track_b_proposed.py``.
Nothing is reimplemented.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rydberg_sim.track_b_proposed import (
    best_pencil, cadzow_project, hankel_rank_cap, magnitude_residual,
    select_order_heldout,
)
from rydberg_sim.track_b_structure import hankel_matrix, hankel_to_vector

import config as cfg


# --------------------------------------------------------------------------
# lifting / reconstruction
# --------------------------------------------------------------------------
def lift(g: np.ndarray, pencil: int | None = None) -> np.ndarray:
    """Channel column -> Hankel matrix, shape ``(N-p, p+1)``.

    ``pencil=None`` uses ``best_pencil(N)``, the p maximising the attainable
    rank -- the same default the audited estimator uses.
    """
    g = np.asarray(g, dtype=np.complex128).ravel()
    p = best_pencil(g.size) if pencil is None else int(pencil)
    return hankel_matrix(g, p)


def unlift(H: np.ndarray) -> np.ndarray:
    """Hankel matrix -> channel column, by averaging each anti-diagonal.

    This is the exact projection onto the Hankel subspace: for a matrix that
    is already Hankel it is the identity (test 4), and for one that is not it
    returns the nearest Hankel matrix in Frobenius norm.
    """
    return hankel_to_vector(np.asarray(H, dtype=np.complex128))


def truncate_rank(H: np.ndarray, rank: int) -> np.ndarray:
    """Best rank-``r`` approximation of ``H`` by truncated SVD.

    Optimal in Frobenius norm by Eckart-Young. This is the second exact
    projector Cadzow alternates with :func:`unlift`.
    """
    H = np.asarray(H, dtype=np.complex128)
    r = int(max(1, min(int(rank), min(H.shape))))
    U, s, Vh = np.linalg.svd(H, full_matrices=False)
    return (U[:, :r] * s[:r]) @ Vh[:r]


def rank_cap(N: int, pencil: int | None = None) -> int:
    """``r_max = ceil(N/2)`` -- the largest rank a length-N Hankel can have."""
    return hankel_rank_cap(int(N), pencil)


def project(g: np.ndarray, rank: int, *, n_iter: int = cfg.CADZOW_ITER,
            pencil: int | None = None) -> np.ndarray:
    """Cadzow projection of one channel column onto "rank <= ``rank``".

    Alternates the two exact projectors for ``n_iter`` sweeps:
        1. lift to Hankel;
        2. truncated SVD to ``rank``;
        3. anti-diagonal averaging back to Hankel structure;
        4. repeat.

    The intersection of the two sets is a NON-CONVEX variety, and ``n_iter``
    sweeps do not reach its exact nearest point, so this is an APPROXIMATE
    projection. ``n_iter = 4`` is the audited default and is not tuned here.

    Returns ``g`` unchanged when ``rank >= r_max`` (constraint vacuous).
    """
    return cadzow_project(np.asarray(g, dtype=np.complex128).ravel(),
                          int(rank), pencil=pencil, n_iter=int(n_iter))


def select_rank(S, Z, B, sigma2, *, pencil: int | None = None) -> tuple[int, dict]:
    """Choose L_hat from a HELD-OUT PILOT RESIDUAL. Uses no ground truth.

    The in-sample residual cannot be used: constraining G to a smaller set can
    only worsen the fit to the pilots used for fitting, so an in-sample
    criterion always prefers the largest candidate. Splitting the pilot
    columns and scoring on the held-out fraction (``VAL_FRAC = 0.3``)
    penalises over-modelling correctly, and every quantity involved -- S, Z, B
    -- is observable at the receiver.

    Candidates are ``1 .. r_max``. Returns a SINGLE SCALAR L_hat that is
    applied to all K user columns; this is a known limitation of the audited
    implementation, preserved here deliberately (see AUDIT.md).

    NOT AN ORACLE: the true L_k never enters. Asserted by test 5.
    """
    return select_order_heldout(
        S, Z, B, sigma2, candidates=None, exact_step=cfg.EXACT_STEP,
        max_iter=cfg.SELECT_ITER, val_frac=cfg.VAL_FRAC, ridge=cfg.RIDGE,
        pencil=pencil, cadzow_iter=cfg.CADZOW_ITER,
    )


def singular_values(g: np.ndarray, pencil: int | None = None) -> np.ndarray:
    """Hankel singular values of one channel column (for the diagnostic)."""
    return np.linalg.svd(lift(g, pencil), compute_uv=False)


__all__ = ["lift", "unlift", "truncate_rank", "rank_cap", "project",
           "select_rank", "singular_values", "best_pencil",
           "magnitude_residual"]

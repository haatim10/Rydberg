"""Hankel-EM-GS = EM-GS + a structural projection. Nothing else differs.

Put the two side by side:

    BASELINE (em_gs.py)                  HANKEL VARIANT (this file)
    ------------------------------       ------------------------------
    G = None                             L_hat = select_rank(...)
    for t in 1..T:                       G = None
        G = em_gs_step(G)                for t in 1..T:
                                             G = em_gs_step(G)          <-- SAME
    return G                                 if active and t % PE == 0:
                                                 for k in columns:
                                                     G[:,k] = project(G[:,k])
                                         return G

The measurement update ``em_gs_step`` is literally the same function object in
both. The ONLY difference is the projection line. When ``L_hat >= r_max`` the
projection is a no-op by construction, ``active`` is False, and this function
returns exactly what ``em_gs.em_gs`` returns -- bit for bit, verified in
test 1 of verify_results.py.

INTERLEAVED, NOT POST-HOC. ``PROJECT_EVERY = 1``, so the projection is applied
after EVERY one of the T = 50 measurement updates, not once at the end. This
matters: the EM-GS update's quality depends on its current iterate through
kappa in the Bessel ratio, so a structurally cleaner iterate produces a better
phase estimate, which produces a better next iterate. This is the audited
behaviour of ``rydberg_sim.track_b_proposed.hs_gs`` and is preserved here.
Post-hoc projection is available as an ABLATION (ablation_schedule.py), not as
the baseline configuration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config as cfg
import hankel_projection as hp
from em_gs import em_gs_step


@dataclass(frozen=True)
class HankelResult:
    G_hat: np.ndarray
    L_hat: int                 # the selected rank
    active: bool               # False => projection was a no-op => == EM-GS
    r_max: int                 # rank ceiling ceil(N/2)
    order_scores: dict         # held-out residual per candidate rank


def hankel_em_gs(S, Z, B, sigma2, *, max_iter: int = cfg.GS_MAX_ITER,
                 project_every: int = cfg.PROJECT_EVERY,
                 cadzow_iter: int = cfg.CADZOW_ITER,
                 ridge: float = cfg.RIDGE,
                 L_hat: int | None = None,
                 pencil: int | None = None) -> HankelResult:
    """EM-GS with a Cadzow low-rank Hankel projection interleaved.

    Parameters
    ----------
    L_hat
        ``None`` (default) selects the rank from held-out pilots, using no
        ground truth. An explicit value is used only by the oracle-rank
        ablation, never by experiments A-C.
    """
    N = np.asarray(Z).shape[0]
    r_max = hp.rank_cap(N, pencil)

    if L_hat is None:
        L_hat, scores = hp.select_rank(S, Z, B, sigma2, pencil=pencil)
    else:
        L_hat, scores = int(L_hat), {}

    active = int(L_hat) < r_max

    G = None
    for t in range(1, int(max_iter) + 1):
        G = em_gs_step(S, Z, B, sigma2, G, ridge=ridge)      # identical to baseline
        if active and t % max(1, int(project_every)) == 0:
            for k in range(G.shape[1]):
                G[:, k] = hp.project(G[:, k], L_hat, n_iter=cadzow_iter,
                                     pencil=pencil)
    return HankelResult(G_hat=G, L_hat=int(L_hat), active=bool(active),
                        r_max=int(r_max), order_scores=scores)


__all__ = ["hankel_em_gs", "HankelResult"]

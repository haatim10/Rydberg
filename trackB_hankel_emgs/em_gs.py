"""Baseline EM-GS. No Hankel processing anywhere in this file.

EM-GS is Cui's Algorithm 2: Gerchberg-Saxton on the biased phase-retrieval
problem Z = |GS + B + W|, with the hard modulus substitution replaced by the
exact conditional mean of the phase factor under the Rician measurement law.
That introduces the Bessel ratio

    R(kappa) = I1(kappa) / I0(kappa),   kappa = (2/sigma2) * Z .* |GS + B|

so the restored observation is shrunk by R in [0,1) according to how reliable
each measurement is. The magnitude observation is NEVER linearised.

The implementation is the audited one in ``rydberg_sim/gs.py``; this module
only fixes the calling convention so that the baseline and the Hankel variant
are driven identically.

KEY FAIRNESS PROPERTY (checked by verify_results.py, test 1):
``em_gs_step`` applied T times with the iterate carried forward is bit-for-bit
identical to a single ``max_iter=T`` call. The Hankel estimator is built from
exactly these steps, so with the projection disabled it reduces to this
baseline EXACTLY -- not approximately.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rydberg_sim.gs import em_gs_channel_rows

import config as cfg


def em_gs_step(S, Z, B, sigma2, G0=None, *, ridge: float = cfg.RIDGE) -> np.ndarray:
    """Exactly ONE EM-GS measurement update, starting from ``G0``.

    ``G0=None`` triggers the module's spectral initialisation. This is the
    single shared primitive: the baseline chains it, and the Hankel variant
    chains it and projects in between.
    """
    return em_gs_channel_rows(S, Z, B, sigma2, max_iter=1, ridge=ridge,
                              G0=G0).G_hat


def em_gs(S, Z, B, sigma2, *, max_iter: int = cfg.GS_MAX_ITER,
          ridge: float = cfg.RIDGE) -> np.ndarray:
    """Baseline EM-GS: ``max_iter`` measurement updates, nothing else.

    Written as a chain of :func:`em_gs_step` rather than a single call so
    that it is structurally identical to :func:`hankel_em_gs.hankel_em_gs`
    with the projection removed. The two forms are bit-identical (test 1).
    """
    G = None
    for _ in range(int(max_iter)):
        G = em_gs_step(S, Z, B, sigma2, G, ridge=ridge)
    return G


__all__ = ["em_gs", "em_gs_step"]

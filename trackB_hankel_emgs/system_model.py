"""System model: geometric ULA channel, pilots, reference, noise, observation.

This is a THIN WRAPPER over the audited Track-B implementation in
``rydberg_sim/``. It does not reimplement any of the model. Its job is to
expose one function, :func:`make_world`, that returns a single frozen trial
containing everything both estimators consume, so that pairing is guaranteed
by construction rather than by convention.

Model (rydberg_sim/channel.py, verified in AUDIT.md):

    g_{n,k} = c * sum_{l=1..L_k} alpha_{l,k} exp(-j (n-1) psi_{l,k})
    psi_{l,k} = pi sin(theta_{l,k})            (half-wavelength ULA spacing)
    theta_{l,k} ~ U[-pi/2, pi/2]  i.i.d.       (uniform in THETA, not sin theta)
    alpha_{l,k} ~ CN(0, beta_k / L_k)          (equal average power per path)
    beta_k = 1, c = 1                          (so G == H exactly)

Pilots (rydberg_sim/pilots.py):
    S ~ CN(0,1), K x P, redrawn every trial, rejection-sampled to full row
    rank, P >= 2K enforced by the generator. NOT orthogonal.

Observation (rydberg_sim/forward.py):
    Z = |G S + B + W|,  W ~ CN(0, sigma2) i.i.d. added INSIDE the modulus.
    sigma2 = K / SNR_lin          (Cui eq. 36)
    |b|    = sqrt(RSR_lin)        (Cui eq. 37, SINGLE-user denominator)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rydberg_sim.monte_carlo import generate_channel_estimation_trial
from rydberg_sim.track_b_drivers import draw_L_k, track_b_spec

import config as cfg


def make_world(trial: int, *, N: int, P: int, snr_db: float,
               L: int | None = None, rsr_db: float | None = None,
               experiment: str = "hankel_emgs"):
    """One frozen trial world.

    Parameters
    ----------
    trial
        Trial index. Together with ``(snr_db, rsr_db)`` this keys the RNG, so
        the returned world is reproducible and identical for every estimator.
    L
        If given, ALL K users get exactly this many paths (experiment C).
        If ``None``, ``L_k ~ U{L_MIN..L_MAX}`` i.i.d. per user, which is the
        established Track-B behaviour used by experiments A and B.

    Returns
    -------
    A frozen (read-only) trial object carrying ``G, S, B, W, Z, sigma2, L_k``.
    Both estimators are handed this same object, so they cannot diverge in
    channel, pilots, reference or noise.
    """
    rsr = cfg.RSR_DB if rsr_db is None else float(rsr_db)
    if L is None:
        # Established Track-B behaviour: L_k drawn from a dedicated substream
        # keyed (master_seed, trial, 0x4C4B), independent of channel/pilot/noise.
        L_k = draw_L_k(trial, cfg.K, master_seed=cfg.MASTER_SEED,
                       L_min=cfg.L_MIN, L_max=cfg.L_MAX)
    else:
        L_k = (int(L),) * cfg.K
    spec = track_b_spec(P=P, n_trials=trial + 1, N=N, K=cfg.K, L=L_k,
                        master_seed=cfg.MASTER_SEED, experiment=experiment)
    return generate_channel_estimation_trial(spec, trial, float(snr_db), rsr)


def channel_nmse_parts(G_hat: np.ndarray, G: np.ndarray) -> tuple[float, float]:
    """Return ``(||G_hat - G||_F^2, ||G||_F^2)`` -- numerator and denominator.

    Stored separately per trial so that any pooling (ratio-of-sums, mean of
    ratios, median, bootstrap over any subset) can be reconstructed later
    without rerunning the Monte Carlo. See verify_results.py.
    """
    err = float(np.sum(np.abs(np.asarray(G_hat) - np.asarray(G)) ** 2))
    den = float(np.sum(np.abs(np.asarray(G)) ** 2))
    return err, den


__all__ = ["make_world", "channel_nmse_parts"]

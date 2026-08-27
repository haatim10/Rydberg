"""Thin adapters over the repository's existing, validated estimators.

**Nothing here reimplements an algorithm.** Every function delegates to
`rydberg_sim`. The audit confirmed all three initializers exist and are
validated, so all three are required (PROMPT 2 sec. 8):

    init in {"random", "spectral", "linearized_ls"}

* ``random``        - normalized random complex Gaussian, as the paper uses
* ``spectral``      - rydberg_sim/spectral.py:330 (14 validated tests)
* ``linearized_ls`` - rydberg_sim/baselines.py:416 (6 validated tests)

Never compare estimators with different initializers without labeling. The
``EM-GS random`` row is not optional: the paper random-inits everything
including its own baselines, so that row is what reproduces their claim, and
the ``spectral``/``linearized_ls`` rows are the honest control.
"""
from __future__ import annotations

import numpy as np

from rydberg_sim.baselines import linearised_closed_form_ls
from rydberg_sim.forward import reference_phase_matrix
from rydberg_sim.gs import biased_gs_channel_rows, em_gs_channel_rows
from rydberg_sim.metrics import channel_nmse
from rydberg_sim.spectral import spectral_initialize_channel_rows

__all__ = ["make_initial_G", "run_gs", "run_em_gs", "run_linearised_ls",
           "nmse_parts", "INITIALIZERS"]

INITIALIZERS = ("random", "spectral", "linearized_ls")


def make_initial_G(
    init: str,
    *,
    S: np.ndarray,
    Z: np.ndarray,
    B: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Build ``G^(0)`` by one of the three validated schemes.

    ``random`` is normalized so ``mean|G0|^2 = 1``, matching the paper's
    "random complex Gaussian matrix with zero mean and normalized variance".
    """
    N, P = np.asarray(Z).shape
    K = np.asarray(S).shape[0]

    if init == "random":
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x1234]))
        g = (rng.standard_normal((N, K)) + 1j * rng.standard_normal((N, K)))
        return (g / np.sqrt(2.0)).astype(np.complex128)

    if init == "spectral":
        return np.asarray(
            spectral_initialize_channel_rows(S, Z, B).G0, dtype=np.complex128
        )

    if init == "linearized_ls":
        # Section-11 debiased observation: Y = Z - |B|, Psi = exp(-j angle(B)).
        Y = np.asarray(Z, dtype=np.float64) - np.abs(np.asarray(B))
        Psi = reference_phase_matrix(np.asarray(B))
        return np.asarray(
            linearised_closed_form_ls(
                Y, S, Psi, observation_source="exact_magnitude"
            ).G_hat,
            dtype=np.complex128,
        )

    raise ValueError(f"unknown init {init!r}; expected one of {INITIALIZERS}")


def run_gs(sample, *, max_iter: int, init: str, seed: int, ridge: float = 0.0
           ) -> np.ndarray:
    """Classical biased GS. Delegates to rydberg_sim/gs.py:348."""
    G0 = make_initial_G(init, S=sample.S, Z=sample.Z, B=sample.B, seed=seed)
    return biased_gs_channel_rows(
        sample.S, sample.Z, sample.B, max_iter=max_iter, G0=G0, ridge=ridge
    ).G_hat


def run_em_gs(sample, *, max_iter: int, init: str, seed: int, ridge: float = 0.0
              ) -> np.ndarray:
    """Classical EM-GS. Delegates to rydberg_sim/gs.py:618."""
    G0 = make_initial_G(init, S=sample.S, Z=sample.Z, B=sample.B, seed=seed)
    return em_gs_channel_rows(
        sample.S, sample.Z, sample.B, sample.sigma2,
        max_iter=max_iter, G0=G0, ridge=ridge
    ).G_hat


def run_linearised_ls(sample, *, ridge: float = 0.0) -> np.ndarray:
    """Section-11 linearised closed-form LS as a standalone estimator."""
    Y = np.asarray(sample.Z, dtype=np.float64) - np.abs(np.asarray(sample.B))
    Psi = reference_phase_matrix(np.asarray(sample.B))
    return linearised_closed_form_ls(
        Y, sample.S, Psi, observation_source="exact_magnitude", ridge=ridge
    ).G_hat


def nmse_parts(G_hat: np.ndarray, G: np.ndarray) -> tuple[float, float]:
    """``(||G_hat - G||_F^2, ||G||_F^2)`` -- numerator and denominator.

    Stored separately per trial so any pooling (ratio-of-sums, mean of ratios,
    median, bootstrap) can be reconstructed later without rerunning the Monte
    Carlo. Matches the repository's aggregation convention: the reported
    aggregate is RATIO-OF-SUMS (metrics.py:386), not a mean of per-trial
    ratios. Cross-checked against ``rydberg_sim.metrics.channel_nmse``.
    """
    res = channel_nmse(G_hat, G)
    return float(res.error_energy), float(res.true_energy)

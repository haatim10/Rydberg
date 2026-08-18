"""Steps 1–2 of the Rydberg atomic MIMO simulation stack.

This package implements:

* frozen simulation configuration and a deterministic per-trial RNG policy
* the geometric ULA channel generator from SystemModel.pdf

It does **not** implement later stages (Cui GS/EM-GS, spectral initialization,
pilots, reference, noise, SNR/RSR calibration, BER, Monte Carlo sweeps, ...).

The conversion/polarisation factor ``c = ℘/ℏ`` is a common known
positive scalar (A5, A15). For normalized simulations ``c = 1``. That
choice is a numerical normalization, not a claim that the physical
atomic conversion gain equals 1.
"""

from .channel import (
    PSI_SEP_MIN,
    RANK_SV_REL_TOL,
    ChannelRealization,
    generate_ula_channel,
    is_full_column_rank,
    min_circular_psi_separation,
    spatial_frequency,
    steering_matrix,
    steering_vector,
)
from .config import SimulationConfig
from .rng import TrialRNGs, get_trial_rngs

__all__ = [
    "ChannelRealization",
    "PSI_SEP_MIN",
    "RANK_SV_REL_TOL",
    "SimulationConfig",
    "TrialRNGs",
    "generate_ula_channel",
    "get_trial_rngs",
    "is_full_column_rank",
    "min_circular_psi_separation",
    "spatial_frequency",
    "steering_matrix",
    "steering_vector",
]

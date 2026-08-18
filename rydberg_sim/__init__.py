"""Steps 1–5 of the Rydberg atomic MIMO simulation stack.

This package implements:

* frozen simulation configuration and a deterministic per-trial RNG policy
* the geometric ULA channel generator from SystemModel.pdf
* the known reference field B from SystemModel.pdf Section 6
* complex Gaussian estimation pilots S and a separate Gray-mapped QAM
  data generator
* the exact magnitude observation Z = |GS+B+W| and the strong-reference
  linearisation Y = Z - |B|

It does **not** implement later stages (SNR/RSR calibration, Cui GS/EM-GS,
detection, BER, Monte Carlo sweeps, ...).

Gaussian pilots ``S`` and QAM data symbols are distinct: ``S ~ CN(0,1)``
is known and used for channel estimation; QAM is a finite alphabet for
later detection. QAM is never used as the estimation pilot matrix.

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
from .forward import (
    ExactObservation,
    LinearisedObservation,
    exact_forward,
    linearised_observation,
)
from .pilots import (
    PILOT_RANK_SV_REL_TOL,
    PilotMatrix,
    generate_gaussian_pilots,
    is_full_row_rank,
)
from .qam import (
    QAMConstellation,
    QAMSequence,
    bits_to_qam,
    build_qam_constellation,
    generate_qam,
    qam_to_bits,
)
from .reference import ReferenceField, generate_reference_field
from .rng import TrialRNGs, get_trial_rngs

__all__ = [
    "ChannelRealization",
    "ExactObservation",
    "LinearisedObservation",
    "PILOT_RANK_SV_REL_TOL",
    "PSI_SEP_MIN",
    "QAMConstellation",
    "QAMSequence",
    "RANK_SV_REL_TOL",
    "ReferenceField",
    "PilotMatrix",
    "SimulationConfig",
    "TrialRNGs",
    "bits_to_qam",
    "build_qam_constellation",
    "exact_forward",
    "generate_gaussian_pilots",
    "generate_qam",
    "generate_reference_field",
    "generate_ula_channel",
    "get_trial_rngs",
    "is_full_column_rank",
    "is_full_row_rank",
    "linearised_observation",
    "min_circular_psi_separation",
    "qam_to_bits",
    "spatial_frequency",
    "steering_matrix",
    "steering_vector",
]

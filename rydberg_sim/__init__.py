"""Steps 1–7 of the Rydberg atomic MIMO simulation stack.

This package implements:

* frozen simulation configuration and a deterministic per-trial RNG policy
* the geometric ULA channel generator from SystemModel.pdf
* the known reference field B from SystemModel.pdf Section 6
* complex Gaussian estimation pilots S and a separate Gray-mapped QAM
  data generator
* the exact magnitude observation Z = |GS+B+W| and the strong-reference
  linearisation Y = Z - |B|
* SNR/RSR power calibration to sigma2 and |alpha_b|
* debugging/reference baselines (genie ZF, linearised closed-form LS,
  exhaustive QAM LS/ML). CM-ZF is explicitly unimplemented.

It does **not** implement later stages (spectral init, biased GS, EM-GS,
Cui/Xu CRLB, GD/PGD, Monte Carlo estimator sweeps, figures, BER).

Gaussian pilots ``S`` and QAM data symbols are distinct: ``S ~ CN(0,1)``
is known and used for channel estimation; QAM is a finite alphabet for
later detection. QAM is never used as the estimation pilot matrix.

The conversion/polarisation factor ``c = ℘/ℏ`` is a common known
positive scalar (A5, A15). For normalized simulations ``c = 1``. That
choice is a numerical normalization, not a claim that the physical
atomic conversion gain equals 1.
"""

from .baselines import (
    DEFAULT_MAX_CANDIDATES,
    ExhaustiveSearchResult,
    ExhaustiveSearchTooLargeError,
    LinearisedLSResult,
    cm_zf,
    enumerate_qam_symbol_vectors,
    exhaustive_magnitude_ls,
    exhaustive_magnitude_ml,
    exhaustive_search_complexity_gate,
    linearised_closed_form_ls,
    qam_candidate_count,
    zf_known_phase,
    zf_known_phase_from_truth,
)
from .calibration import (
    MeasuredRSR,
    MeasuredSNR,
    db_to_linear,
    linear_to_db,
    make_alpha_b,
    measure_rsr,
    measure_snr,
    reference_user_beta,
    rsr_db_to_alpha_magnitude,
    snr_db_to_sigma2,
)
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
    "DEFAULT_MAX_CANDIDATES",
    "ChannelRealization",
    "ExhaustiveSearchResult",
    "ExhaustiveSearchTooLargeError",
    "LinearisedLSResult",
    "ExactObservation",
    "LinearisedObservation",
    "MeasuredRSR",
    "MeasuredSNR",
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
    "cm_zf",
    "db_to_linear",
    "enumerate_qam_symbol_vectors",
    "exact_forward",
    "exhaustive_magnitude_ls",
    "exhaustive_magnitude_ml",
    "exhaustive_search_complexity_gate",
    "generate_gaussian_pilots",
    "generate_qam",
    "generate_reference_field",
    "generate_ula_channel",
    "get_trial_rngs",
    "is_full_column_rank",
    "is_full_row_rank",
    "linearised_closed_form_ls",
    "linearised_observation",
    "linear_to_db",
    "make_alpha_b",
    "measure_rsr",
    "measure_snr",
    "min_circular_psi_separation",
    "qam_candidate_count",
    "qam_to_bits",
    "reference_user_beta",
    "rsr_db_to_alpha_magnitude",
    "snr_db_to_sigma2",
    "spatial_frequency",
    "steering_matrix",
    "steering_vector",
    "zf_known_phase",
    "zf_known_phase_from_truth",
]

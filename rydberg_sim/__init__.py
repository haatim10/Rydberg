"""Steps 1–13 of the Rydberg atomic MIMO simulation stack.

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
* canonical Cui spectral initialization for ``z = |M^H u + b + w|``
* Cui biased Gerchberg–Saxton (Algorithm 1) and EM-GS (Algorithm 2)
* Cui Fisher information / CRLB for the magnitude-only Rician model
* linearised channel CRLB from the Step-7 real Gaussian model
  (derived from nbar ~ N(0, sigma2/2 I); not copied from Xu)
* deterministic NMSE / BER metrics (no simulation generation)

It does **not** implement later stages (Monte Carlo harness, GD/PGD,
estimator sweeps, figures).

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
from .crlb import (
    CuiCRLBResult,
    CuiFisherResult,
    cui_crlb,
    cui_crlb_high_snr_limit,
    cui_fisher_information,
    fisher_beta,
    fisher_expectation_z2_r2,
    high_snr_fisher_beta_limit,
    rician_amplitude_pdf,
    rician_fisher_scalar,
)
from .gs import (
    BiasedGSResult,
    ChannelBiasedGSResult,
    ChannelEMGSResult,
    EMGSResult,
    biased_gs,
    biased_gs_channel_rows,
    bessel_ratio,
    em_gs,
    em_gs_channel_rows,
    random_complex_initialization,
)
from .forward import (
    ExactObservation,
    LinearisedObservation,
    exact_forward,
    linearised_observation,
    reference_phase_matrix,
)
from .linearised_crlb import (
    LinearisedChannelCRLBResult,
    LinearisedRowCRLBResult,
    expected_channel_frobenius_energy,
    linearised_channel_crlb,
    linearised_row_crlb,
    linearised_row_fisher,
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
    nearest_qam_indices,
    project_to_qam,
    qam_to_bits,
)
from .metrics import (
    BERResult,
    BerAccumulator,
    ChannelNMSEResult,
    DetectionNMSEResult,
    NmseAccumulator,
    channel_nmse,
    decoded_bits,
    detection_ber,
    detection_nmse,
    nmse_to_db,
    phase_align_channel_rows,
)
from .reference import ReferenceField, generate_reference_field
from .rng import TrialRNGs, get_trial_rngs
from .spectral import (
    ChannelSpectralInitResult,
    SpectralInitResult,
    spectral_initialize,
    spectral_initialize_channel_rows,
)

__all__ = [
    "DEFAULT_MAX_CANDIDATES",
    "BERResult",
    "BerAccumulator",
    "ChannelNMSEResult",
    "DetectionNMSEResult",
    "NmseAccumulator",
    "ChannelRealization",
    "ExhaustiveSearchResult",
    "ExhaustiveSearchTooLargeError",
    "LinearisedLSResult",
    "ChannelSpectralInitResult",
    "SpectralInitResult",
    "BiasedGSResult",
    "ChannelBiasedGSResult",
    "EMGSResult",
    "ChannelEMGSResult",
    "CuiCRLBResult",
    "CuiFisherResult",
    "LinearisedChannelCRLBResult",
    "LinearisedRowCRLBResult",
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
    "biased_gs",
    "biased_gs_channel_rows",
    "bessel_ratio",
    "build_qam_constellation",
    "channel_nmse",
    "cm_zf",
    "cui_crlb",
    "cui_crlb_high_snr_limit",
    "cui_fisher_information",
    "db_to_linear",
    "decoded_bits",
    "detection_ber",
    "detection_nmse",
    "em_gs",
    "em_gs_channel_rows",
    "enumerate_qam_symbol_vectors",
    "exact_forward",
    "exhaustive_magnitude_ls",
    "exhaustive_magnitude_ml",
    "exhaustive_search_complexity_gate",
    "expected_channel_frobenius_energy",
    "fisher_beta",
    "fisher_expectation_z2_r2",
    "generate_gaussian_pilots",
    "generate_qam",
    "generate_reference_field",
    "generate_ula_channel",
    "get_trial_rngs",
    "high_snr_fisher_beta_limit",
    "is_full_column_rank",
    "is_full_row_rank",
    "linearised_channel_crlb",
    "linearised_closed_form_ls",
    "linearised_observation",
    "linearised_row_crlb",
    "linearised_row_fisher",
    "linear_to_db",
    "make_alpha_b",
    "measure_rsr",
    "measure_snr",
    "min_circular_psi_separation",
    "nearest_qam_indices",
    "nmse_to_db",
    "qam_candidate_count",
    "phase_align_channel_rows",
    "project_to_qam",
    "qam_to_bits",
    "random_complex_initialization",
    "reference_phase_matrix",
    "reference_user_beta",
    "rician_amplitude_pdf",
    "rician_fisher_scalar",
    "rsr_db_to_alpha_magnitude",
    "snr_db_to_sigma2",
    "spectral_initialize",
    "spectral_initialize_channel_rows",
    "spatial_frequency",
    "steering_matrix",
    "steering_vector",
    "zf_known_phase",
    "zf_known_phase_from_truth",
]

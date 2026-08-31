"""Track D (URformer) configuration.

Every value is serialized into every checkpoint and every report. Values
inherited from elsewhere carry a provenance comment naming the source; values
introduced here are marked ``[TRACK D]`` and are justified in README.md.

Nothing in this file may be changed in response to a result.

Reference
---------
J. Xiao, J. Wang, M. Zeng, H. Xu, X. Li and A. Nallanathan, "Channel
Estimation for Rydberg Atomic Quantum Receivers: Unrolled Phase Retrieval From
Holographic Snapshots," IEEE Signal Processing Letters, vol. 33,
pp. 1696-1700, 2026.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Base commit this package was built against (PROMPT 2 sec. 0).
# ---------------------------------------------------------------------------
TRACK_D_BASE_SHA = "54c88d5d5888923d973b6ff6a429e51da75b61df"
TRACK_D_AUDIT_COMMIT = "74984c7"

# ---------------------------------------------------------------------------
# Paper Table I, recorded verbatim so divergences are explicit, never silent.
# ---------------------------------------------------------------------------
PAPER_TABLE_I: dict[str, object] = {
    "K_users": 4,
    "M_atomic_antennas": 32,
    "L_clusters": 4,
    "C_l_subrays": 10,
    "doa_distribution": "U(-pi/2, pi/2)",
    "T_UR_layers": 10,
    "L_enc_encoders": 3,
    "d_model": 64,
    "training_samples": 20000,
    "batch_size": 32,
    "initial_lr": 1e-3,
    "optimizer": "Adam",
    "scheduler": "cosine annealing",
    "epochs_T_max": 50,
    "T_GS_classical_baselines": 100,
    "P_for_snr_sweep": 20,
    "snr_db_for_pilot_sweep": 5.0,
    "rsr_db_paper_convention": 10.0,
}

# Divergences from Table I, each with a reason. Rendered into every report.
PAPER_DIVERGENCES: tuple[dict[str, str], ...] = (
    {
        "item": "K (users)",
        "paper": "4",
        "ours": "3",
        "reason": "Track B's frozen K=3 (trackB_hankel_emgs/config.py:K). The "
                  "repository wins over the paper so the eventual "
                  "EM-GS/URformer/HS-EM-GS/HS-URformer ablation is like-for-like.",
    },
    {
        "item": "channel model",
        "paper": "clustered Saleh-Valenzuela, L=4 clusters x C_l=10 subrays, sqrt(M/N_ray)",
        "ours": "geometric specular ULA, L_k ~ U{3..7} i.i.d. per user, alpha ~ CN(0, beta_k/L_k)",
        "reason": "PROMPT 2 sec. 0: we keep our model. Clusters/subrays do not apply.",
    },
    {
        "item": "RSR denominator",
        "paper": "multi-user  E|H s_p|^2",
        "ours": "single-user  E|g_nk s_kp|^2  (Cui eq. 37)",
        "reason": "Seventh discrepancy found by the audit. Differs by exactly K. "
                  "Both fields stored on every row: rsr_ours_dB and "
                  "rsr_paper_equiv_dB = rsr_ours_dB - 10*log10(K).",
    },
    {
        "item": "steering sign",
        "paper": "e^{+j 2pi/lambda d sin(theta)}",
        "ours": "e^{-j n psi}, psi = pi sin(theta)",
        "reason": "rydberg_sim/channel.py:156, confirmed by audit gate (c).",
    },
    {
        "item": "pilot orientation",
        "paper": "S in C^{PxK}, writes H S^T, M-step (S^T)pinv",
        "ours": "S in C^{KxP}, G S, M-step G = R S^H (S S^H)^{-1}",
        "reason": "rydberg_sim/gs.py:326-331. Never mechanically port (S.T).pinv().",
    },
    {
        "item": "transduction gain",
        "paper": "G scalar outside the magnitude: z = G|Hs+b+w|",
        "ours": "c folded in, G = cH, Z = |GS+B+W|, c = 1.0",
        "reason": "rydberg_sim/channel.py:285. Known real scalar, so NMSE is identical.",
    },
    {
        "item": "R(kappa) definition",
        "paper": "never defined - says only 'the ratio of modified Bessel functions'",
        "ours": "I1(kappa)/I0(kappa) via exponentially scaled ive/i1e-i0e",
        "reason": "rydberg_sim/gs.py:426, inherited from Cui et al. Repository is the authority.",
    },
    {
        "item": "torch wheel",
        "paper": "n/a",
        "ours": "PyPI torch (CUDA-linked build) running CPU-only",
        "reason": "PROMPT 2 sec. 0 asked for the CPU-only index "
                  "(download.pytorch.org), but that host is blocked by this "
                  "environment's proxy policy (403 on CONNECT). PyPI is the only "
                  "reachable index. torch.cuda.is_available() is False; execution "
                  "is CPU-only either way.",
    },
)


# ---------------------------------------------------------------------------
# System model - INHERITED, do not retune.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SystemConfig:
    """Physical configuration. Every field inherited from Track B."""

    K: int = 3                      # trackB_hankel_emgs/config.py:K
    N: int = 32                     # Track B N_GRID max; paper Table I M=32
    P: int = 20                     # paper Table I P for the SNR sweep
    L_min: int = 3                  # trackB_hankel_emgs/config.py:L_MIN
    L_max: int = 7                  # trackB_hankel_emgs/config.py:L_MAX
    beta: float = 1.0               # rydberg_sim/track_b_drivers.py:track_b_spec
    c: float = 1.0                  # atomic conversion gain, numerical normalization
    vartheta: float = 0.0           # reference arrival angle, rydberg_sim
    rsr_db: float = 10.0            # [TRACK D] paper Fig. 3 value, OUR convention
    master_seed: int = 20260827     # [TRACK D] distinct from Track B's 20250820

    @property
    def rsr_paper_equiv_db(self) -> float:
        """Our reference level expressed in the paper's MULTI-USER convention.

        SIGN CORRECTED 2026-08-28 (PROMPT 4 Part A). This previously returned
        ``rsr_db + 10log10(K)``, which is the opposite conversion::

            RSR_ours  = E|b|^2 / E|g_nk s_kp|^2      (ONE user)
            RSR_paper = E|b|^2 / E|H s_p|^2          (ALL K users)
            E|H s_p|^2 = K * E|g s|^2
            =>  RSR_paper = RSR_ours / K
            =>  RSR_paper_dB = RSR_ours_dB - 10 log10(K)

        Verified empirically on 300 realizations: measured RSR_ours 10.06 dB,
        RSR_paper 5.21 dB, difference 4.85 dB ~ 10log10(3) = 4.77 dB.

        So our 10 dB is 5.23 dB in the paper's terms, and the paper's 10 dB
        would be 14.77 dB in ours. The old value was the latter wearing the
        former's name, and it is written on every result row.
        """
        import math
        return float(self.rsr_db - 10.0 * math.log10(self.K))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
SeedRange = tuple[int, int]


@dataclass(frozen=True)
class DataConfig:
    """Dataset generation. Train/val/test seed ranges are disjoint by construction."""

    n_train: int = 20000            # paper Table I
    n_val: int = 2000               # [TRACK D]
    n_test: int = 2000              # [TRACK D]

    # Disjoint trial-index ranges. Asserted non-overlapping at construction.
    train_seed_range: SeedRange = (0, 1_000_000)
    val_seed_range: SeedRange = (1_000_000, 2_000_000)
    test_seed_range: SeedRange = (2_000_000, 3_000_000)

    pilot_mode: Literal["fixed_S", "random_S"] = "fixed_S"
    fixed_S_seed: int = 777_000_001         # only used when pilot_mode == "fixed_S"

    snr_mode: Literal["snr_range", "snr_fixed"] = "snr_range"
    # WIDENED from (0, 20) to (-10, 20) by PROMPT 4 A4. The D1 evaluation sweep
    # is D1_SNR_GRID_DB = (-10, -5, 0, 5, 10, 15, 20), so a [0,20] training
    # range would have forced the model to EXTRAPOLATE at the two lowest
    # evaluation points -- precisely the regime where phase retrieval is
    # hardest and where the paper claims its largest gains. Training must cover
    # evaluation with no extrapolation at either end.
    snr_range_db: tuple[float, float] = (-10.0, 20.0)
    snr_fixed_db: float = 5.0

    # RSR is FIXED, never sampled. Stated explicitly so no later reader assumes
    # a range was drawn (PROMPT 3 item 1). The value lives in
    # SystemConfig.rsr_db and is 10 dB in OUR single-user convention.
    #
    # Consequence, booked now: a model trained at fixed RSR is OFF-DISTRIBUTION
    # at any other reference level. The later Xiao-comparability experiment at
    # rsr_paper_equiv_dB therefore requires a RETRAINING, not merely a
    # re-evaluation of these checkpoints.
    rsr_train_mode: Literal["fixed", "range"] = "fixed"

    # For D2: P drawn uniformly from this set during training (see README sec. 6).
    p_train_choices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        # rsr_train_mode is declarative, and "range" has no implementation.
        # Reject it loudly rather than letting it sit as a silent no-op -- that
        # is exactly how `filter_init` came to be dead (PROMPT 4 A1).
        if self.rsr_train_mode != "fixed":
            raise NotImplementedError(
                f"rsr_train_mode={self.rsr_train_mode!r} is not implemented; "
                "Track D trains at a single fixed RSR (PROMPT 3 item 1). "
                "Comparing at another reference level requires a RETRAINING, "
                "not a re-evaluation."
            )
        rs = [self.train_seed_range, self.val_seed_range, self.test_seed_range]
        for a, b in ((0, 1), (0, 2), (1, 2)):
            lo1, hi1 = rs[a]
            lo2, hi2 = rs[b]
            if max(lo1, lo2) < min(hi1, hi2):
                raise ValueError(
                    f"seed ranges {rs[a]} and {rs[b]} overlap; train/val/test "
                    "must be disjoint (PROMPT 2 sec. 3)"
                )
        for name, n, rng in (
            ("n_train", self.n_train, self.train_seed_range),
            ("n_val", self.n_val, self.val_seed_range),
            ("n_test", self.n_test, self.test_seed_range),
        ):
            if n > rng[1] - rng[0]:
                raise ValueError(f"{name}={n} exceeds its seed range width {rng}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelConfig:
    """URformer architecture. Paper-faithful USER tokens only."""

    T_UR: int = 10                  # paper Table I
    d_model: int = 64               # paper Table I
    L_enc: int = 3                  # paper Table I
    n_heads: int = 4                # [TRACK D] paper does not state it
    ffn_mult: int = 4               # [TRACK D] standard Transformer ratio
    dropout: float = 0.0            # [TRACK D] off; dataset is synthetic and unlimited

    filter_hidden: int = 32         # [TRACK D] FilterNet width, paper says "compact"
    filter_input: Literal[
        "kappa", "log1p_kappa", "log1p_kappa_plus_logsigma2"
    ] = "log1p_kappa"
    # NOTE: until 2026-08-28 this field was declared but never read by
    # URformerLayer -- the warm-start machinery in filter_net.py existed and was
    # never called. It is now wired. The DEFAULT IS UNCHANGED ("random"), so
    # this fixes a dead field without altering any reported behaviour.
    filter_init: Literal["random", "emgs_warmstart"] = "random"
    filter_warmstart_cache: str = "reports/trackD_filternet_warmstart.pt"

    gate_init: Literal["near_gs", "near_emgs", "neutral"] = "near_gs"

    # Arm 2 of stage 1 ("URformer-filteronly"): FilterNet + gate + LS with the
    # Transformer residual module REMOVED ENTIRELY -- not zeroed, not disabled
    # at runtime, simply not constructed. ~980 parameters instead of 1,586,900.
    # This is the ablation that attributes any gain between "unrolling helped"
    # and "a 1.57M-parameter learned denoiser helped".
    use_transformer: bool = True

    # --- HS-URformer (PROMPT 6) -------------------------------------------
    # The Hankel projection sits INSIDE every unrolled layer, between the LS
    # step and the Transformer residual - that placement is what makes the
    # internal-vs-post-hoc distinction meaningful. Applied through a
    # STRAIGHT-THROUGH ESTIMATOR: forward is the exact projection, backward is
    # the identity, so no gradient passes through the ill-conditioned SVD while
    # the unrolled chain to earlier layers stays intact. Full detachment was
    # tried first and gate HK6 measured EXACTLY ZERO gradient everywhere but
    # the last Transformer - see urformer.URformerLayer.forward.
    use_hankel: bool = False
    hankel_rank: int = 7            # L_max; a system design assumption, NOT oracle
    hankel_mode: Literal["fixed", "adaptive", "oracle"] = "fixed"
    hankel_pencil: int | None = None        # None -> Track B default p = N//2
    hankel_iters: int = 1           # 1 == H^-1 . Pi_r . H exactly; >1 is Cadzow

    # --- gated Hankel (PROMPT 7) ------------------------------------------
    # G~ = beta_t * Project(G_lin) + (1 - beta_t) * G_lin, then the Transformer.
    #   "none"   -> unconditional projection (stage 3's H1)
    #   "scalar" -> G1: beta_t = sigmoid(g_t), one learned scalar per layer
    #   "snr"    -> G2: beta_t = sigmoid(MLP_t(log sigma^2)), conditioned on the
    #               noise level. sigma^2 is ALREADY an estimator input (kappa =
    #               2 Z |Y| / sigma^2), so this is not privileged information.
    # The (1 - beta_t) branch is a clean differentiable path, so a gated model
    # carries gradient regardless of how the projection itself is treated --
    # the HK6 severing failure cannot recur in this form.
    hankel_gate: Literal["none", "scalar", "snr"] = "none"
    hankel_gate_init: float = -2.0          # sigmoid(-2) = 0.119, the alpha convention
    hankel_gate_hidden: int = 16            # width of the per-layer gate MLP (G2)
    # MEASURED over 4000 training samples, not assumed:
    #   mean log sigma^2 = -0.0727, std = 1.998, range [-3.503, +3.401]
    # G2's gate input is (log sigma^2 - mean) / std, which lands in about
    # [-1.72, +1.74]. corr(log sigma^2, snr_db) = -1.000 exactly, so
    # conditioning on sigma^2 IS conditioning on SNR -- and sigma^2 is already
    # an estimator input via kappa, so no privileged information is added.
    log_sigma2_mean: float = -0.0727
    log_sigma2_std: float = 1.998

    # Untied weights per unrolled layer (PROMPT 2 sec. 5).
    tie_layers: bool = False

    deep_supervision: bool = False  # OFF in the reference run (PROMPT 2 sec. 7)

    @property
    def gate_init_value(self) -> float:
        return {"near_gs": -2.0, "near_emgs": 2.0, "neutral": 0.0}[self.gate_init]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32            # paper Table I
    lr: float = 1e-3                # paper Table I
    epochs: int = 50                # paper text, T_max = 50
    optimizer: str = "adam"
    scheduler: str = "cosine"
    weight_decay: float = 0.0
    grad_clip: float | None = 1.0   # [TRACK D] unrolled nets need it; recorded
    num_threads: int = 1            # [TRACK D] measured in sec. 15, see README
    seed: int = 20260827
    init: Literal["random", "spectral", "linearized_ls"] = "random"


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NumericConfig:
    dtype: Literal["float32", "float64"] = "float32"

    @property
    def eps(self) -> float:
        """Guard for |Y| and kappa. PROMPT 2 sec. 1."""
        return 1e-12 if self.dtype == "float64" else 1e-8

    @property
    def real_dtype(self):
        import torch
        return torch.float64 if self.dtype == "float64" else torch.float32

    @property
    def complex_dtype(self):
        import torch
        return torch.complex128 if self.dtype == "float64" else torch.complex64


# ---------------------------------------------------------------------------
# Classical baselines
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BaselineConfig:
    T_GS: int = 100                 # paper text: T_GS = 100 for classical baselines
    ridge: float = 0.0              # trackB_hankel_emgs/config.py:RIDGE


@dataclass(frozen=True)
class TrackDConfig:
    system: SystemConfig = field(default_factory=SystemConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    numeric: NumericConfig = field(default_factory=NumericConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["base_sha"] = TRACK_D_BASE_SHA
        d["audit_commit"] = TRACK_D_AUDIT_COMMIT
        d["rsr_paper_equiv_db"] = self.system.rsr_paper_equiv_db
        d["gate_init_value"] = self.model.gate_init_value
        d["eps"] = self.numeric.eps
        return d


# ---------------------------------------------------------------------------
# Experiment grids - scaffolding only, not executed in this phase.
# ---------------------------------------------------------------------------
D1_SNR_GRID_DB: tuple[float, ...] = (-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
D2_P_GRID: tuple[int, ...] = (6, 10, 15, 20, 30)
D3_N_GRID: tuple[int, ...] = (8, 16, 32)      # trackB_hankel_emgs/config.py:N_GRID

INITIALIZERS: tuple[str, ...] = ("random", "spectral", "linearized_ls")

__all__ = [
    "TRACK_D_BASE_SHA", "TRACK_D_AUDIT_COMMIT", "PAPER_TABLE_I",
    "PAPER_DIVERGENCES", "SystemConfig", "DataConfig", "ModelConfig",
    "TrainConfig", "NumericConfig", "BaselineConfig", "TrackDConfig",
    "D1_SNR_GRID_DB", "D2_P_GRID", "D3_N_GRID", "INITIALIZERS",
]

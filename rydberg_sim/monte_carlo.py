"""Monte Carlo execution harness (Step 14).

This module wraps Steps 1–13. It does **not** implement GD, PGD, neural
solvers, Track-C execution, or publication figure sweeps.

Common-random-number (CRN) policy
---------------------------------
For a given ``(trial_index, snr_db, rsr_db, configuration)`` every
algorithm being compared sees the **same** channel, pilots, reference,
noise, and observation. Algorithms consume a frozen trial object. They
must **not** regenerate G/S/B/W/Z internally. Execution order of
algorithms must not change the realization.

Operating-point policy (independent draws)
------------------------------------------
Each ``(trial_index, snr_db, rsr_db, experiment configuration)`` gets
its own deterministic world from a stable SeedSequence key. Iterating
SNR in reverse does **not** change the world attached to a given tuple.
Worlds are **not** reused across SNR points (no common-channel-across-SNR
policy unless a later experiment documents one).

Deterministic keying (no ``hash()``)
------------------------------------
    spawn_key = (trial_index, snr_key, rsr_key)
    snr_key   = round(snr_db * 1000) + 1_000_000
    rsr_key   = round(rsr_db * 1000) + 1_000_000

    SeedSequence(entropy=master_seed, spawn_key=spawn_key).spawn(6)

in the Step-1 stream order (channel, pilots, reference, noise, data,
solver). See :func:`rydberg_sim.rng.get_operating_point_rngs`.

Result key (resume / no duplicates)
-----------------------------------
    (config_fingerprint, experiment, track, trial, snr_key, rsr_key,
     algorithm)

``config_fingerprint`` is SHA-256 of a canonical JSON payload of the
**material** configuration (N, K, P, L_k, beta_k, c, master_seed,
vartheta, max_iter, ridge, qam_M, track, channel_model, …). It does
**not** include the SNR grid, trial count, or algorithm list, so a run
can be extended with more trials, SNR points, or algorithms. Changing
P, N, max_iter, etc. changes the fingerprint; the harness refuses to
append incompatible rows to an existing file.

Tracks
------
A — Cui detection: known ``A``, unknown QAM ``s``. Uses
  ``channel_model="cui_38901"`` (:mod:`rydberg_sim.channel_cui`).
B — ULA channel estimation: known ``S``, unknown ``G``. Uses
  ``channel_model="ula_geometric"`` (:mod:`rydberg_sim.channel`).
C — estimate then detect: schema reserved; **not executed**.

Long-form table
---------------
CSV, one scalar metric per row, with raw linear energies / bit counts
needed to re-aggregate. Optional diagnostics (GS objective history, …)
go to a sidecar JSON for selected trials only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from .baselines import linearised_closed_form_ls, zf_known_phase
from .calibration import (
    make_alpha_b,
    reference_user_beta,
    rsr_db_to_alpha_magnitude,
    snr_db_to_sigma2,
)
from .channel import generate_ula_channel
from .channel_cui import (
    CHANNEL_MODEL_CUI,
    CuiChannelParams,
    generate_cui_channel,
    generate_cui_reference,
)
from .confidence import (
    NmseUncertainty,
    WilsonInterval,
    nmse_ratio_standard_error,
    wilson_interval,
)
from .crlb import cui_crlb
from .config import SimulationConfig
from .forward import exact_forward, linearised_observation
from .gs import biased_gs, biased_gs_channel_rows, em_gs, em_gs_channel_rows
from .metrics import channel_nmse, detection_ber, detection_nmse, nmse_to_db
from .pilots import generate_gaussian_pilots
from .qam import generate_qam
from .reference import generate_reference_field
from .rng import (
    db_to_key,
    get_operating_point_rngs,
    operating_point_spawn_key,
)

TrackName = Literal["A", "B", "C"]

CHANNEL_MODEL_ULA: str = "ula_geometric"
CHANNEL_MODEL = CHANNEL_MODEL_ULA  # Track-B default; Track A must set cui_38901.

CHANNEL_ESTIMATORS: frozenset[str] = frozenset(
    {"biased_gs", "em_gs", "linearised_ls"}
)
DETECTION_ALGORITHMS: frozenset[str] = frozenset(
    {"genie_zf", "biased_gs", "em_gs", "cui_crlb"}
)
UNIMPLEMENTED_ALGORITHMS: frozenset[str] = frozenset(
    {"gd", "pgd", "neural", "neural_net", "learned", "cm_zf"}
)

RESULT_COLUMNS: tuple[str, ...] = (
    "experiment",
    "config_fingerprint",
    "track",
    "trial",
    "snr_db",
    "rsr_db",
    "N",
    "K",
    "P",
    "modulation",
    "algorithm",
    "metric",
    "value",
    "error_energy",
    "true_energy",
    "expected_symbol_energy",
    "bit_errors",
    "bit_count",
    "status",
    "error_type",
    "error_message",
    "master_seed",
    "sigma2",
    "alpha_b_abs",
    "max_iter",
)

MANIFEST_NAME = "run_manifest.json"
RESULTS_NAME = "results.csv"
DIAGNOSTICS_DIRNAME = "diagnostics"

OPERATING_POINT_POLICY = "independent_per_trial_snr_rsr"


class ConfigFingerprintError(ValueError):
    """Existing on-disk results do not match this experiment configuration."""


class TrackCNotImplementedError(NotImplementedError):
    """Track C (estimate G, then detect) is reserved for a later step."""


# ---------------------------------------------------------------------------
# Config fingerprint
# ---------------------------------------------------------------------------


def _json_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def fingerprint_payload(spec: "ExperimentSpec") -> dict[str, Any]:
    """Material configuration hashed into the experiment identity.

    Intentionally omitted: SNR/RSR grids, ``n_trials``, algorithm list,
    ``continue_on_error``, diagnostic flags. Those may grow a run without
    changing the world-generation / estimator hyperparameters.
    """
    cfg = spec.cfg
    payload: dict[str, Any] = {
        "channel_model": spec.channel_model,
        "track": spec.track,
        "N": int(cfg.N),
        "K": int(cfg.K),
        "master_seed": int(cfg.master_seed),
        "max_iter": int(spec.max_iter),
        "ridge": float(spec.ridge),
        "qam_M": int(spec.qam_M),
    }
    if spec.channel_model == CHANNEL_MODEL_ULA:
        payload.update(
            {
                "L_k": [int(x) for x in cfg.L_k],
                "beta_k": [float(x) for x in cfg.beta_k],
                "c": float(cfg.c),
                "P": int(spec.P),
                "vartheta": float(spec.vartheta),
                "beta_ref_user": int(spec.beta_ref_user),
                "phi_b": float(spec.phi_b),
            }
        )
    elif spec.channel_model == CHANNEL_MODEL_CUI:
        payload.update(spec.cui_params.as_fingerprint_dict())
        payload["P"] = int(spec.P)
    else:
        raise ValueError(f"unknown channel_model {spec.channel_model!r}")
    return payload


def config_fingerprint(spec: "ExperimentSpec") -> str:
    """SHA-256 hex digest of the canonical fingerprint JSON. Not ``hash()``."""
    blob = _json_dumps(fingerprint_payload(spec)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def result_key(
    *,
    config_fingerprint: str,
    experiment: str,
    track: str,
    trial: int,
    snr_db: float,
    rsr_db: float,
    algorithm: str,
) -> tuple[str, str, str, int, int, int, str]:
    """Unique completed-unit key. One algorithm at one operating point."""
    return (
        str(config_fingerprint),
        str(experiment),
        str(track),
        int(trial),
        db_to_key(snr_db, "snr_db"),
        db_to_key(rsr_db, "rsr_db"),
        str(algorithm),
    )


# ---------------------------------------------------------------------------
# Experiment spec
# ---------------------------------------------------------------------------


def _as_db_tuple(values: Sequence[float], name: str) -> tuple[float, ...]:
    out: list[float] = []
    for v in values:
        x = float(v)
        if not np.isfinite(x):
            raise ValueError(f"{name} entries must be finite, got {v!r}")
        db_to_key(x, name)  # validate millidB encoding early
        out.append(x)
    if not out:
        raise ValueError(f"{name} must be non-empty")
    return tuple(out)


@dataclass(frozen=True)
class AdaptiveBerPolicy:
    """Interface only: a later BER study may continue until a budget.

    Any field may be ``None`` (ignored). Step 14 does **not** run an
    adaptive Track-A/C study. Resume can reconstruct counts from the
    long table, so this policy can be applied later without rewriting
    the checkpoint format.

        min_errors  stop after at least this many bit errors
        max_bits    stop after this many bits
        max_trials  stop after this many trials
    """

    min_errors: int | None = None
    max_bits: int | None = None
    max_trials: int | None = None


def adaptive_ber_budget_reached(
    *,
    total_bit_errors: int,
    total_bit_count: int,
    n_trials: int,
    policy: AdaptiveBerPolicy,
) -> bool:
    """Return True if a later adaptive sampler may stop.

    ``min_errors`` is a *target*: sampling continues until that many
    errors are seen **or** a max-bits / max-trials cap is hit. A cap
    alone is also a stop. With no fields set, this never stops.
    """
    if policy.max_trials is not None and n_trials >= int(policy.max_trials):
        return True
    if policy.max_bits is not None and total_bit_count >= int(policy.max_bits):
        return True
    if policy.min_errors is not None and total_bit_errors >= int(policy.min_errors):
        return True
    return False


@dataclass(frozen=True)
class ExperimentSpec:
    """Frozen Monte Carlo experiment description.

    ``n_trials`` addresses trials ``0 .. n_trials-1``. ``algorithms`` is
    the comparison set; CRN still holds if the tuple is reordered.
    """

    experiment: str
    track: TrackName
    cfg: SimulationConfig
    P: int
    vartheta: float
    snr_db_grid: tuple[float, ...]
    rsr_db_grid: tuple[float, ...]
    n_trials: int
    algorithms: tuple[str, ...]
    max_iter: int = 5
    ridge: float = 0.0
    qam_M: int = 4
    beta_ref_user: int = 0
    phi_b: float = 0.0
    continue_on_error: bool = True
    store_diagnostics: bool = False
    diagnostic_trials: tuple[int, ...] = ()
    channel_model: str = ""
    cui_params: CuiChannelParams | None = None
    write_ber: bool = True

    def __post_init__(self) -> None:
        if not str(self.experiment):
            raise ValueError("experiment name must be non-empty")
        if self.track not in ("A", "B", "C"):
            raise ValueError(f"track must be 'A', 'B', or 'C', got {self.track!r}")
        model = str(self.channel_model) if self.channel_model else ""
        if not model:
            if self.track == "A":
                model = CHANNEL_MODEL_CUI
            elif self.track == "B":
                model = CHANNEL_MODEL_ULA
            else:
                model = ""
            object.__setattr__(self, "channel_model", model)
        if self.track == "A" and model != CHANNEL_MODEL_CUI:
            raise ValueError(
                f"Track A requires channel_model={CHANNEL_MODEL_CUI!r}, got {model!r}. "
                "Do not use the Track-B geometric ULA generator."
            )
        if self.track == "B" and model != CHANNEL_MODEL_ULA:
            raise ValueError(
                f"Track B requires channel_model={CHANNEL_MODEL_ULA!r}, got {model!r}."
            )
        if self.track == "A":
            cui = self.cui_params if self.cui_params is not None else CuiChannelParams()
            object.__setattr__(self, "cui_params", cui)
        elif self.cui_params is not None:
            raise ValueError("cui_params is only valid for Track A")
        if isinstance(self.P, (bool, np.bool_)) or int(self.P) != self.P:
            raise TypeError(f"P must be an integer, got {self.P!r}")
        object.__setattr__(self, "P", int(self.P))
        if self.P <= 0:
            raise ValueError(f"P must be > 0, got {self.P}")
        if isinstance(self.n_trials, (bool, np.bool_)) or int(self.n_trials) != self.n_trials:
            raise TypeError(f"n_trials must be an integer, got {self.n_trials!r}")
        object.__setattr__(self, "n_trials", int(self.n_trials))
        if self.n_trials <= 0:
            raise ValueError(f"n_trials must be > 0, got {self.n_trials}")
        if isinstance(self.max_iter, (bool, np.bool_)) or int(self.max_iter) != self.max_iter:
            raise TypeError(f"max_iter must be an integer, got {self.max_iter!r}")
        object.__setattr__(self, "max_iter", int(self.max_iter))
        if self.max_iter <= 0:
            raise ValueError(f"max_iter must be > 0, got {self.max_iter}")
        object.__setattr__(self, "snr_db_grid", _as_db_tuple(self.snr_db_grid, "snr_db_grid"))
        object.__setattr__(self, "rsr_db_grid", _as_db_tuple(self.rsr_db_grid, "rsr_db_grid"))
        algs = tuple(str(a) for a in self.algorithms)
        if not algs:
            raise ValueError("algorithms must be non-empty")
        object.__setattr__(self, "algorithms", algs)
        object.__setattr__(self, "diagnostic_trials", tuple(int(t) for t in self.diagnostic_trials))
        if self.track == "B" and self.P < 2 * int(self.cfg.K):
            raise ValueError(
                f"Track B requires P >= 2K, got P={self.P}, K={self.cfg.K}"
            )
        for alg in algs:
            if alg in UNIMPLEMENTED_ALGORITHMS:
                raise NotImplementedError(
                    f"algorithm {alg!r} is not part of Step 14 (infrastructure only)"
                )
            if self.track == "B" and alg not in CHANNEL_ESTIMATORS:
                raise ValueError(
                    f"Track B algorithm {alg!r} is unknown; "
                    f"supported: {sorted(CHANNEL_ESTIMATORS)}"
                )
            if self.track == "A" and alg not in DETECTION_ALGORITHMS:
                raise ValueError(
                    f"Track A algorithm {alg!r} is unknown; "
                    f"supported: {sorted(DETECTION_ALGORITHMS)}"
                )
            if self.track == "C":
                raise TrackCNotImplementedError(
                    "Track C is reserved: estimate G from pilots then detect "
                    "QAM using G_hat. Not executed in Step 14."
                )

    @property
    def trial_indices(self) -> tuple[int, ...]:
        return tuple(range(self.n_trials))

    @property
    def fingerprint(self) -> str:
        return config_fingerprint(self)

    @property
    def modulation_label(self) -> str:
        if self.track == "B":
            return "n/a"
        return f"{int(self.qam_M)}-QAM"


# ---------------------------------------------------------------------------
# Frozen trial worlds
# ---------------------------------------------------------------------------


def _freeze(arr: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    out = np.array(arr, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


def _freeze_tuple_arrays(items: Sequence[np.ndarray], dtype: np.dtype | type) -> tuple[np.ndarray, ...]:
    return tuple(_freeze(x, dtype) for x in items)


@dataclass(frozen=True, eq=False)
class ChannelEstimationTrial:
    """One immutable Track-B world. Algorithms must consume this object."""

    G: np.ndarray
    H: np.ndarray
    theta: tuple[np.ndarray, ...]
    psi: tuple[np.ndarray, ...]
    alpha: tuple[np.ndarray, ...]
    A_k: tuple[np.ndarray, ...]
    L_k: np.ndarray
    beta_k: np.ndarray
    S: np.ndarray
    B: np.ndarray
    W: np.ndarray
    Z: np.ndarray
    Y: np.ndarray
    Psi: np.ndarray
    E: np.ndarray
    sigma2: float
    alpha_b: complex
    snr_db: float
    rsr_db: float
    trial_index: int
    master_seed: int
    snr_key: int
    rsr_key: int
    spawn_key: tuple[int, int, int]
    vartheta: float
    cfg: SimulationConfig
    track: str = "B"


@dataclass(frozen=True, eq=False)
class DetectionTrial:
    """One immutable Track-A world: known ``A``, unknown QAM ``s``.

    Canonical observation (Cui eq. 22): ``z = |A^H s + b + w|`` with
    ``A ∈ C^{K × N}``. Solvers are called as ``biased_gs(M=A, ...)``
    with **no** channel-estimation conjugation adapter.
    """

    A: np.ndarray
    s: np.ndarray
    bits: np.ndarray
    b: np.ndarray
    w: np.ndarray
    z: np.ndarray
    theta: np.ndarray
    sigma2: float
    alpha_b: complex
    snr_db: float
    rsr_db: float
    trial_index: int
    master_seed: int
    snr_key: int
    rsr_key: int
    spawn_key: tuple[int, int, int]
    vartheta: float
    qam_M: int
    cfg: SimulationConfig
    track: str = "A"
    channel_model: str = CHANNEL_MODEL_CUI


def _alpha_b_for_spec(spec: ExperimentSpec, rsr_db: float) -> complex:
    beta_ref = reference_user_beta(spec.cfg.beta_k, spec.beta_ref_user)
    mag = rsr_db_to_alpha_magnitude(rsr_db, beta_ref)
    return make_alpha_b(mag, spec.phi_b)


def generate_channel_estimation_trial(
    spec: ExperimentSpec,
    trial_index: int,
    snr_db: float,
    rsr_db: float,
) -> ChannelEstimationTrial:
    """Build one Track-B world from the stable operating-point key.

    Generation order: channel → pilots → (deterministic B from RSR) →
    sigma2 → exact forward (noise) → linearised observation from the
    **same** Z. The ``data`` and ``solver`` streams are not consumed.
    """
    if spec.track != "B":
        raise ValueError(
            f"generate_channel_estimation_trial requires track='B', got {spec.track!r}"
        )
    cfg = spec.cfg
    spawn_key = operating_point_spawn_key(trial_index, snr_db, rsr_db)
    rngs = get_operating_point_rngs(cfg.master_seed, trial_index, snr_db, rsr_db)
    ch = generate_ula_channel(cfg, trial_index, rng=rngs.channel)
    pilots = generate_gaussian_pilots(K=cfg.K, P=spec.P, rng=rngs.pilots)
    alpha_b = _alpha_b_for_spec(spec, rsr_db)
    ref = generate_reference_field(
        N=cfg.N,
        P=spec.P,
        alpha_b=alpha_b,
        vartheta=spec.vartheta,
        c=cfg.c,
    )
    sigma2 = snr_db_to_sigma2(snr_db, cfg.beta_k, c=cfg.c)
    exact = exact_forward(ch.G, pilots.S, ref.B, sigma2, rng_noise=rngs.noise)
    lin = linearised_observation(exact)
    return ChannelEstimationTrial(
        G=_freeze(ch.G, np.complex128),
        H=_freeze(ch.H, np.complex128),
        theta=_freeze_tuple_arrays(ch.theta, np.float64),
        psi=_freeze_tuple_arrays(ch.psi, np.float64),
        alpha=_freeze_tuple_arrays(ch.alpha, np.complex128),
        A_k=_freeze_tuple_arrays(ch.A_k, np.complex128),
        L_k=_freeze(ch.L_k, np.int64),
        beta_k=_freeze(ch.beta_k, np.float64),
        S=_freeze(pilots.S, np.complex128),
        B=_freeze(ref.B, np.complex128),
        W=_freeze(exact.W, np.complex128),
        Z=_freeze(exact.Z, np.float64),
        Y=_freeze(lin.Y, np.float64),
        Psi=_freeze(lin.Psi, np.complex128),
        E=_freeze(exact.E, np.complex128),
        sigma2=float(sigma2),
        alpha_b=complex(alpha_b),
        snr_db=float(snr_db),
        rsr_db=float(rsr_db),
        trial_index=int(trial_index),
        master_seed=int(cfg.master_seed),
        snr_key=spawn_key[1],
        rsr_key=spawn_key[2],
        spawn_key=spawn_key,
        vartheta=float(spec.vartheta),
        cfg=cfg,
        track="B",
    )


def generate_detection_trial(
    spec: ExperimentSpec,
    trial_index: int,
    snr_db: float,
    rsr_db: float,
) -> DetectionTrial:
    """Build one Track-A world (known ``A``, unknown QAM ``s``).

    Uses :func:`generate_cui_channel` / :func:`generate_cui_reference`.
    Never calls the Track-B geometric ULA generator.
    """
    if spec.track != "A":
        raise ValueError(
            f"generate_detection_trial requires track='A', got {spec.track!r}"
        )
    if spec.channel_model != CHANNEL_MODEL_CUI:
        raise ValueError(
            f"Track A worlds require {CHANNEL_MODEL_CUI!r}, got {spec.channel_model!r}"
        )
    cfg = spec.cfg
    cui = spec.cui_params if spec.cui_params is not None else CuiChannelParams()
    spawn_key = operating_point_spawn_key(trial_index, snr_db, rsr_db)
    rngs = get_operating_point_rngs(cfg.master_seed, trial_index, snr_db, rsr_db)
    ch = generate_cui_channel(cfg.N, cfg.K, rngs.channel, params=cui)
    qam = generate_qam(rngs.data, cfg.K, spec.qam_M)
    b = generate_cui_reference(cfg.N, rngs.reference, rsr_db, params=cui)
    # Eq. 37 with row-normalized A and unit-energy QAM: E|a_n^H s|² = K.
    sigma2 = snr_db_to_sigma2(snr_db, np.ones(cfg.K, dtype=np.float64), c=1.0)
    if sigma2 == 0.0:
        w = _freeze(np.zeros(cfg.N, dtype=np.complex128), np.complex128)
    else:
        scale = np.sqrt(sigma2 / 2.0)
        real = rngs.noise.standard_normal(cfg.N)
        imag = rngs.noise.standard_normal(cfg.N)
        w = _freeze(scale * real + 1j * scale * imag, np.complex128)
    field = ch.A.conj().T @ qam.symbols + b + w
    z = _freeze(np.abs(field), np.float64)
    theta = _freeze(np.angle(field), np.float64)
    return DetectionTrial(
        A=_freeze(ch.A, np.complex128),
        s=_freeze(qam.symbols, np.complex128),
        bits=_freeze(qam.bits, np.uint8),
        b=_freeze(b, np.complex128),
        w=w,
        z=z,
        theta=theta,
        sigma2=float(sigma2),
        alpha_b=complex(np.sqrt(float(np.mean(np.abs(b) ** 2)))),
        snr_db=float(snr_db),
        rsr_db=float(rsr_db),
        trial_index=int(trial_index),
        master_seed=int(cfg.master_seed),
        snr_key=spawn_key[1],
        rsr_key=spawn_key[2],
        spawn_key=spawn_key,
        vartheta=float(cui.lo_azimuth_deg),
        qam_M=int(spec.qam_M),
        cfg=cfg,
        track="A",
        channel_model=CHANNEL_MODEL_CUI,
    )


def generate_track_c_trial(*args: object, **kwargs: object) -> None:
    """Track C is not executed in Step 14."""
    raise TrackCNotImplementedError(
        "Track C (estimate G from pilots, then detect QAM using G_hat) "
        "is not executed in Step 14. The result schema is already long-form "
        "and keyed by track so a later step can add it without rewriting "
        "the harness."
    )


def channel_trials_equal(
    a: ChannelEstimationTrial, b: ChannelEstimationTrial
) -> bool:
    """Exact array equality of the shared realization (CRN check)."""
    return (
        np.array_equal(a.G, b.G)
        and np.array_equal(a.H, b.H)
        and np.array_equal(a.S, b.S)
        and np.array_equal(a.B, b.B)
        and np.array_equal(a.W, b.W)
        and np.array_equal(a.Z, b.Z)
        and np.array_equal(a.Y, b.Y)
        and np.array_equal(a.Psi, b.Psi)
        and a.sigma2 == b.sigma2
        and a.alpha_b == b.alpha_b
        and a.trial_index == b.trial_index
        and a.snr_key == b.snr_key
        and a.rsr_key == b.rsr_key
        and a.master_seed == b.master_seed
    )


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------


def _fmt_float(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    x = float(value)  # type: ignore[arg-type]
    if not np.isfinite(x):
        if np.isnan(x):
            return ""
        return "inf" if x > 0 else "-inf"
    return format(x, ".17g")


def _fmt_optional_int(value: object) -> str:
    if value is None or value == "":
        return ""
    return str(int(value))  # type: ignore[arg-type]


def _format_row(row: Mapping[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for col in RESULT_COLUMNS:
        val = row.get(col, "")
        if col in {
            "value",
            "error_energy",
            "true_energy",
            "expected_symbol_energy",
            "snr_db",
            "rsr_db",
            "sigma2",
            "alpha_b_abs",
        }:
            out[col] = _fmt_float(val) if val != "" else ""
        elif col in {"bit_errors", "bit_count", "trial", "N", "K", "P", "master_seed", "max_iter"}:
            out[col] = _fmt_optional_int(val) if val != "" else ("" if val == "" else str(val))
        else:
            out[col] = "" if val is None else str(val)
    return out


def _parse_optional_float(text: str) -> float | None:
    if text == "":
        return None
    return float(text)


def _parse_optional_int(text: str) -> int | None:
    if text == "":
        return None
    return int(text)


def load_result_table(path: Path | str) -> list[dict[str, Any]]:
    """Load the long-form CSV into a list of parsed row dicts."""
    csv_path = Path(path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            rows.append(
                {
                    "experiment": raw["experiment"],
                    "config_fingerprint": raw["config_fingerprint"],
                    "track": raw["track"],
                    "trial": int(raw["trial"]),
                    "snr_db": float(raw["snr_db"]),
                    "rsr_db": float(raw["rsr_db"]),
                    "N": int(raw["N"]),
                    "K": int(raw["K"]),
                    "P": int(raw["P"]),
                    "modulation": raw["modulation"],
                    "algorithm": raw["algorithm"],
                    "metric": raw["metric"],
                    "value": _parse_optional_float(raw["value"]),
                    "error_energy": _parse_optional_float(raw["error_energy"]),
                    "true_energy": _parse_optional_float(raw["true_energy"]),
                    "expected_symbol_energy": _parse_optional_float(
                        raw["expected_symbol_energy"]
                    ),
                    "bit_errors": _parse_optional_int(raw["bit_errors"]),
                    "bit_count": _parse_optional_int(raw["bit_count"]),
                    "status": raw["status"],
                    "error_type": raw.get("error_type", ""),
                    "error_message": raw.get("error_message", ""),
                    "master_seed": int(raw["master_seed"]),
                    "sigma2": float(raw["sigma2"]),
                    "alpha_b_abs": float(raw["alpha_b_abs"]),
                    "max_iter": int(raw["max_iter"]),
                }
            )
    return rows


def sort_result_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Canonical order: trial, SNR key, RSR key, algorithm, metric."""

    def _key(row: Mapping[str, Any]) -> tuple:
        return (
            int(row["trial"]),
            db_to_key(float(row["snr_db"]), "snr_db"),
            db_to_key(float(row["rsr_db"]), "rsr_db"),
            str(row["algorithm"]),
            str(row["metric"]),
            str(row["status"]),
        )

    return [dict(r) for r in sorted(rows, key=_key)]


def _completed_algorithm_keys(rows: Sequence[Mapping[str, Any]]) -> set[tuple]:
    keys: set[tuple] = set()
    for row in rows:
        keys.add(
            result_key(
                config_fingerprint=str(row["config_fingerprint"]),
                experiment=str(row["experiment"]),
                track=str(row["track"]),
                trial=int(row["trial"]),
                snr_db=float(row["snr_db"]),
                rsr_db=float(row["rsr_db"]),
                algorithm=str(row["algorithm"]),
            )
        )
    return keys


def append_result_rows(csv_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Append rows and fsync. Creates the header if the file is empty."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = (not csv_path.exists()) or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="raise")
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow(_format_row(row))
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Algorithm evaluation
# ---------------------------------------------------------------------------


def _blank_metric_fields() -> dict[str, Any]:
    return {
        "value": None,
        "error_energy": None,
        "true_energy": None,
        "expected_symbol_energy": None,
        "bit_errors": None,
        "bit_count": None,
    }


def _base_row(
    spec: ExperimentSpec,
    *,
    trial_index: int,
    snr_db: float,
    rsr_db: float,
    algorithm: str,
    sigma2: float,
    alpha_b: complex,
) -> dict[str, Any]:
    cfg = spec.cfg
    return {
        "experiment": spec.experiment,
        "config_fingerprint": spec.fingerprint,
        "track": spec.track,
        "trial": int(trial_index),
        "snr_db": float(snr_db),
        "rsr_db": float(rsr_db),
        "N": int(cfg.N),
        "K": int(cfg.K),
        "P": int(spec.P),
        "modulation": spec.modulation_label,
        "algorithm": algorithm,
        "status": "ok",
        "error_type": "",
        "error_message": "",
        "master_seed": int(cfg.master_seed),
        "sigma2": float(sigma2),
        "alpha_b_abs": float(abs(alpha_b)),
        "max_iter": int(spec.max_iter),
        **_blank_metric_fields(),
    }


def _failure_row(
    spec: ExperimentSpec,
    *,
    trial_index: int,
    snr_db: float,
    rsr_db: float,
    algorithm: str,
    sigma2: float,
    alpha_b: complex,
    exc: BaseException,
) -> dict[str, Any]:
    row = _base_row(
        spec,
        trial_index=trial_index,
        snr_db=snr_db,
        rsr_db=rsr_db,
        algorithm=algorithm,
        sigma2=sigma2,
        alpha_b=alpha_b,
    )
    row.update(
        {
            "metric": "failure",
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    )
    return row


def _estimate_G(
    world: ChannelEstimationTrial, algorithm: str, spec: ExperimentSpec
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return ``G_hat`` and compact diagnostics. Does not redraw the world."""
    diagnostics: dict[str, Any] = {"phase_alignment_used": False}
    if algorithm == "biased_gs":
        result = biased_gs_channel_rows(
            world.S,
            world.Z,
            world.B,
            max_iter=spec.max_iter,
            ridge=spec.ridge,
        )
        diagnostics["init_error_energy"] = float(
            np.linalg.norm(result.G0 - world.G, ord="fro") ** 2
        )
        diagnostics["objective_history_row0"] = result.row_results[0].objective_history.tolist()
        return result.G_hat, diagnostics
    if algorithm == "em_gs":
        result = em_gs_channel_rows(
            world.S,
            world.Z,
            world.B,
            world.sigma2,
            max_iter=spec.max_iter,
            ridge=spec.ridge,
        )
        diagnostics["init_error_energy"] = float(
            np.linalg.norm(result.G0 - world.G, ord="fro") ** 2
        )
        diagnostics["objective_history_row0"] = result.row_results[0].objective_history.tolist()
        diagnostics["loglik_history_row0"] = result.row_results[0].loglik_history.tolist()
        return result.G_hat, diagnostics
    if algorithm == "linearised_ls":
        result = linearised_closed_form_ls(
            world.Y,
            world.S,
            world.Psi,
            observation_source="exact_magnitude",
            ridge=spec.ridge,
        )
        return result.G_hat, diagnostics
    raise ValueError(f"unknown Track B algorithm {algorithm!r}")


def evaluate_channel_algorithm(
    world: ChannelEstimationTrial,
    algorithm: str,
    spec: ExperimentSpec,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Run one estimator on a frozen world. Never regenerates G/S/B/W/Z."""
    G_hat, diagnostics = _estimate_G(world, algorithm, spec)
    nmse = channel_nmse(G_hat, world.G)
    row = _base_row(
        spec,
        trial_index=world.trial_index,
        snr_db=world.snr_db,
        rsr_db=world.rsr_db,
        algorithm=algorithm,
        sigma2=world.sigma2,
        alpha_b=world.alpha_b,
    )
    row.update(
        {
            "metric": "channel_nmse",
            "value": float(nmse.instantaneous_nmse),
            "error_energy": float(nmse.error_energy),
            "true_energy": float(nmse.true_energy),
        }
    )
    diagnostics["phase_aligned_error_energy"] = float(nmse.phase_aligned_error_energy)
    diagnostics["likely_phase_anchor_problem"] = bool(nmse.likely_phase_anchor_problem)
    store = spec.store_diagnostics and world.trial_index in spec.diagnostic_trials
    return [row], (diagnostics if store else None)


def evaluate_detection_algorithm(
    world: DetectionTrial,
    algorithm: str,
    spec: ExperimentSpec,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Run one Track-A method on a frozen world. ``M = A`` directly."""
    M = world.A
    if algorithm == "cui_crlb":
        crlb = cui_crlb(
            M, world.s, world.b, world.sigma2, expected_u_energy=float(world.s.size)
        )
        row = _base_row(
            spec,
            trial_index=world.trial_index,
            snr_db=world.snr_db,
            rsr_db=world.rsr_db,
            algorithm=algorithm,
            sigma2=world.sigma2,
            alpha_b=world.alpha_b,
        )
        row.update(
            {
                "metric": "detection_nmse",
                "value": float(crlb.normalized_crlb),
                "error_energy": float(np.trace(crlb.crlb).real),
                "expected_symbol_energy": float(crlb.expected_u_energy),
            }
        )
        return [row], None

    if algorithm == "biased_gs":
        s_hat = biased_gs(
            M, world.z, world.b, max_iter=spec.max_iter, ridge=spec.ridge
        ).u_hat
    elif algorithm == "em_gs":
        s_hat = em_gs(
            M,
            world.z,
            world.b,
            world.sigma2,
            max_iter=spec.max_iter,
            ridge=spec.ridge,
        ).u_hat
    elif algorithm == "genie_zf":
        s_hat = zf_known_phase(M, world.z, world.theta, world.b, ridge=spec.ridge)
    else:
        raise ValueError(f"unknown Track A algorithm {algorithm!r}")

    det = detection_nmse(s_hat, world.s)
    base = _base_row(
        spec,
        trial_index=world.trial_index,
        snr_db=world.snr_db,
        rsr_db=world.rsr_db,
        algorithm=algorithm,
        sigma2=world.sigma2,
        alpha_b=world.alpha_b,
    )
    nmse_row = dict(base)
    nmse_row.update(
        {
            "metric": "detection_nmse",
            "value": float(det.nmse_linear),
            "error_energy": float(det.error_energy),
            "expected_symbol_energy": float(det.expected_energy),
        }
    )
    rows = [nmse_row]
    if spec.write_ber:
        ber = detection_ber(s_hat, world.bits, spec.qam_M)
        ber_row = dict(base)
        ber_row.update(
            {
                "metric": "ber",
                "value": float(ber.ber),
                "bit_errors": int(ber.bit_errors),
                "bit_count": int(ber.bit_count),
            }
        )
        rows.append(ber_row)
    return rows, None


def _write_diagnostics(
    output_dir: Path,
    spec: ExperimentSpec,
    *,
    trial_index: int,
    snr_db: float,
    rsr_db: float,
    algorithm: str,
    payload: Mapping[str, Any],
) -> None:
    diag_dir = output_dir / DIAGNOSTICS_DIRNAME
    diag_dir.mkdir(parents=True, exist_ok=True)
    name = (
        f"trial{trial_index}_snr{db_to_key(snr_db, 'snr_db')}_"
        f"rsr{db_to_key(rsr_db, 'rsr_db')}_{algorithm}.json"
    )
    path = diag_dir / name
    body = {
        "experiment": spec.experiment,
        "config_fingerprint": spec.fingerprint,
        "trial": int(trial_index),
        "snr_db": float(snr_db),
        "rsr_db": float(rsr_db),
        "algorithm": algorithm,
        "diagnostics": dict(payload),
    }
    path.write_text(_json_dumps(body) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# One operating point
# ---------------------------------------------------------------------------


def _run_operating_point(
    spec: ExperimentSpec,
    trial_index: int,
    snr_db: float,
    rsr_db: float,
    algorithms: Sequence[str],
) -> tuple[list[dict[str, Any]], list[tuple[str, dict[str, Any]]]]:
    """Generate **one** world, then run the requested algorithms (CRN)."""
    rows: list[dict[str, Any]] = []
    diags: list[tuple[str, dict[str, Any]]] = []
    if spec.track == "C":
        raise TrackCNotImplementedError("Track C is not executed in Step 14.")
    if spec.track == "B":
        world: ChannelEstimationTrial | DetectionTrial = generate_channel_estimation_trial(
            spec, trial_index, snr_db, rsr_db
        )
        sigma2 = world.sigma2
        alpha_b = world.alpha_b
        evaluator = evaluate_channel_algorithm
    else:
        world = generate_detection_trial(spec, trial_index, snr_db, rsr_db)
        sigma2 = world.sigma2
        alpha_b = world.alpha_b
        evaluator = evaluate_detection_algorithm  # type: ignore[assignment]

    for algorithm in algorithms:
        try:
            alg_rows, diag = evaluator(world, algorithm, spec)  # type: ignore[arg-type]
            rows.extend(alg_rows)
            if diag is not None:
                diags.append((algorithm, diag))
        except Exception as exc:  # noqa: BLE001 — record and optionally continue
            rows.append(
                _failure_row(
                    spec,
                    trial_index=trial_index,
                    snr_db=snr_db,
                    rsr_db=rsr_db,
                    algorithm=algorithm,
                    sigma2=sigma2,
                    alpha_b=alpha_b,
                    exc=exc,
                )
            )
            if not spec.continue_on_error:
                raise
    return rows, diags


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AggregateRecord:
    """One grouped Monte Carlo summary (ratio-of-sums / pooled BER)."""

    experiment: str
    track: str
    snr_db: float
    rsr_db: float
    algorithm: str
    metric: str
    n_ok: int
    n_failed: int
    value_linear: float
    value_db: float | None
    se_linear: float | None
    wilson: WilsonInterval | None
    nmse: NmseUncertainty | None
    total_error_energy: float | None
    total_true_energy: float | None
    total_expected_symbol_energy: float | None
    total_bit_errors: int | None
    total_bit_count: int | None


def _group_key(row: Mapping[str, Any]) -> tuple:
    return (
        str(row["experiment"]),
        str(row["track"]),
        db_to_key(float(row["snr_db"]), "snr_db"),
        db_to_key(float(row["rsr_db"]), "rsr_db"),
        str(row["algorithm"]),
        float(row["snr_db"]),
        float(row["rsr_db"]),
    )


def aggregate_result_table(
    rows: Sequence[Mapping[str, Any]],
) -> list[AggregateRecord]:
    """Aggregate long-form rows with Step-13 conventions.

    Channel NMSE: ``sum(error_energy) / sum(true_energy)`` then ``10 log10``.
    Detection NMSE: ``sum(error_energy) / sum(expected_symbol_energy)``.
    BER: ``sum(bit_errors) / sum(bit_count)``.
    Failed rows are counted but excluded from the sums.
    """
    groups: dict[tuple, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_group_key(row), []).append(row)

    records: list[AggregateRecord] = []
    for key, group in sorted(groups.items(), key=lambda kv: kv[0][:5]):
        experiment, track, _sk, _rk, algorithm, snr_db, rsr_db = key
        n_failed = sum(1 for r in group if r["status"] != "ok")
        ok = [r for r in group if r["status"] == "ok"]

        ch = [r for r in ok if r["metric"] == "channel_nmse"]
        if ch:
            errors = np.asarray([float(r["error_energy"]) for r in ch], dtype=np.float64)
            energies = np.asarray([float(r["true_energy"]) for r in ch], dtype=np.float64)
            unc = nmse_ratio_standard_error(errors, energies)
            records.append(
                AggregateRecord(
                    experiment=experiment,
                    track=track,
                    snr_db=float(snr_db),
                    rsr_db=float(rsr_db),
                    algorithm=algorithm,
                    metric="channel_nmse",
                    n_ok=len(ch),
                    n_failed=n_failed,
                    value_linear=unc.nmse_linear,
                    value_db=nmse_to_db(unc.nmse_linear),
                    se_linear=unc.se_linear,
                    wilson=None,
                    nmse=unc,
                    total_error_energy=unc.total_error_energy,
                    total_true_energy=unc.total_true_energy,
                    total_expected_symbol_energy=None,
                    total_bit_errors=None,
                    total_bit_count=None,
                )
            )

        det = [r for r in ok if r["metric"] == "detection_nmse"]
        if det:
            errors = np.asarray([float(r["error_energy"]) for r in det], dtype=np.float64)
            energies = np.asarray(
                [float(r["expected_symbol_energy"]) for r in det], dtype=np.float64
            )
            unc = nmse_ratio_standard_error(errors, energies)
            records.append(
                AggregateRecord(
                    experiment=experiment,
                    track=track,
                    snr_db=float(snr_db),
                    rsr_db=float(rsr_db),
                    algorithm=algorithm,
                    metric="detection_nmse",
                    n_ok=len(det),
                    n_failed=n_failed,
                    value_linear=unc.nmse_linear,
                    value_db=nmse_to_db(unc.nmse_linear),
                    se_linear=unc.se_linear,
                    wilson=None,
                    nmse=unc,
                    total_error_energy=unc.total_error_energy,
                    total_true_energy=None,
                    total_expected_symbol_energy=unc.total_true_energy,
                    total_bit_errors=None,
                    total_bit_count=None,
                )
            )

        ber_rows = [r for r in ok if r["metric"] == "ber"]
        if ber_rows:
            k = int(sum(int(r["bit_errors"]) for r in ber_rows))
            n = int(sum(int(r["bit_count"]) for r in ber_rows))
            wil = wilson_interval(k, n)
            records.append(
                AggregateRecord(
                    experiment=experiment,
                    track=track,
                    snr_db=float(snr_db),
                    rsr_db=float(rsr_db),
                    algorithm=algorithm,
                    metric="ber",
                    n_ok=len(ber_rows),
                    n_failed=n_failed,
                    value_linear=wil.ber,
                    value_db=None,
                    se_linear=None,
                    wilson=wil,
                    nmse=None,
                    total_error_energy=None,
                    total_true_energy=None,
                    total_expected_symbol_energy=None,
                    total_bit_errors=k,
                    total_bit_count=n,
                )
            )
    return records


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _write_manifest(output_dir: Path, spec: ExperimentSpec) -> None:
    payload = {
        "experiment": spec.experiment,
        "track": spec.track,
        "config_fingerprint": spec.fingerprint,
        "fingerprint_payload": fingerprint_payload(spec),
        "operating_point_policy": OPERATING_POINT_POLICY,
        "seed_policy": (
            "SeedSequence(entropy=master_seed, "
            "spawn_key=(trial_index, snr_key, rsr_key)) with "
            "snr_key = round(snr_db*1000)+1e6; six spawned streams in "
            "Step-1 order. Python hash() is never used."
        ),
        "result_key": (
            "(config_fingerprint, experiment, track, trial, snr_key, rsr_key, "
            "algorithm)"
        ),
        "n_trials": spec.n_trials,
        "snr_db_grid": list(spec.snr_db_grid),
        "rsr_db_grid": list(spec.rsr_db_grid),
        "algorithms": list(spec.algorithms),
    }
    path = output_dir / MANIFEST_NAME
    path.write_text(_json_dumps(payload) + "\n", encoding="utf-8")


def _assert_compatible_store(csv_path: Path, spec: ExperimentSpec) -> None:
    existing = load_result_table(csv_path)
    if not existing:
        return
    fp = spec.fingerprint
    for row in existing:
        if row["config_fingerprint"] != fp:
            raise ConfigFingerprintError(
                "existing results have config_fingerprint "
                f"{row['config_fingerprint']!r}, this spec has {fp!r}. "
                "Refusing to append incompatible rows."
            )
        if row["experiment"] != spec.experiment:
            raise ConfigFingerprintError(
                f"existing experiment {row['experiment']!r} != {spec.experiment!r}"
            )
        if row["track"] != spec.track:
            raise ConfigFingerprintError(
                f"existing track {row['track']!r} != {spec.track!r}"
            )


def _jobs_for_spec(
    spec: ExperimentSpec,
    completed: set[tuple],
) -> list[tuple[int, float, float, tuple[str, ...]]]:
    jobs: list[tuple[int, float, float, tuple[str, ...]]] = []
    fp = spec.fingerprint
    for trial in spec.trial_indices:
        for snr_db in spec.snr_db_grid:
            for rsr_db in spec.rsr_db_grid:
                missing = tuple(
                    alg
                    for alg in spec.algorithms
                    if result_key(
                        config_fingerprint=fp,
                        experiment=spec.experiment,
                        track=spec.track,
                        trial=trial,
                        snr_db=snr_db,
                        rsr_db=rsr_db,
                        algorithm=alg,
                    )
                    not in completed
                )
                if missing:
                    jobs.append((trial, float(snr_db), float(rsr_db), missing))
    return jobs


def run_experiment(
    spec: ExperimentSpec,
    output_dir: Path | str,
    *,
    n_workers: int = 1,
    resume: bool = True,
) -> Path:
    """Run a Monte Carlo experiment, checkpointing the long table to CSV.

    ``n_workers=1`` is serial (default). ``n_workers>1`` uses a thread
    pool over ``(trial, SNR, RSR)`` jobs. Each worker derives RNGs only
    from the stable key. Serial and parallel runs with the same spec
    produce identical per-trial scalars after sorting.

    Resume: completed ``(fingerprint, experiment, track, trial, snr, rsr,
    algorithm)`` keys are skipped. Interrupted runs keep already-fsynced
    rows.
    """
    if spec.track == "C":
        raise TrackCNotImplementedError("Track C is not executed in Step 14.")
    if isinstance(n_workers, (bool, np.bool_)) or int(n_workers) != n_workers:
        raise TypeError(f"n_workers must be an integer, got {n_workers!r}")
    n_workers = int(n_workers)
    if n_workers <= 0:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / RESULTS_NAME
    _assert_compatible_store(csv_path, spec)
    _write_manifest(out, spec)

    existing = load_result_table(csv_path) if resume else []
    if not resume and csv_path.exists():
        csv_path.unlink()
        existing = []
    completed = _completed_algorithm_keys(existing) if resume else set()
    jobs = _jobs_for_spec(spec, completed)
    lock = threading.Lock()

    def _consume(
        trial: int,
        snr_db: float,
        rsr_db: float,
        algorithms: Sequence[str],
    ) -> None:
        rows, diags = _run_operating_point(spec, trial, snr_db, rsr_db, algorithms)
        with lock:
            append_result_rows(csv_path, rows)
            for algorithm, payload in diags:
                _write_diagnostics(
                    out,
                    spec,
                    trial_index=trial,
                    snr_db=snr_db,
                    rsr_db=rsr_db,
                    algorithm=algorithm,
                    payload=payload,
                )

    if n_workers == 1 or len(jobs) <= 1:
        for trial, snr_db, rsr_db, algs in jobs:
            _consume(trial, snr_db, rsr_db, algs)
        return csv_path

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(_consume, trial, snr_db, rsr_db, algs)
            for trial, snr_db, rsr_db, algs in jobs
        ]
        for fut in as_completed(futures):
            fut.result()
    return csv_path

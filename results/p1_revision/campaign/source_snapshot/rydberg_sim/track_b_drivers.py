"""Track-B B1/B2 drivers: channel NMSE vs SNR and vs pilot length.

B1: NMSE_G vs SNR at fixed pilot length.
B2: NMSE_G vs pilot length P at fixed SNR.

Every estimator on a given point sees the **same** realization
``{θ, α, G, S, B, W, Z}`` from :func:`generate_channel_estimation_trial`,
and ``θ``/``α`` are retained for later structural diagnostics.

The primary metric is the ratio of sums

    NMSE_G = Σ_trials ||Ĝ - G||_F²  /  Σ_trials ||G||_F²

reported as ``10 log10``. Per-trial dB values are never averaged.

Estimator labels
----------------
``biased_gs`` / ``em_gs``
    EXACT model — the row adapters on ``Z = |GS+B+W|``.
``structured_*``
    EXACT model plus a ULA structural projection (the Step-B4 prototype).
``linearised_ls``
    **LINEARIZED MODEL BASELINE** (Xu strong-reference). Retained only for
    comparison; it never substitutes for the exact-model estimators.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig
from .gs import biased_gs_channel_rows, em_gs_channel_rows
from .metrics import channel_nmse
from .monte_carlo import ExperimentSpec, generate_channel_estimation_trial
from .track_b_prototype import structured_exact_estimate

# --- Parameters from the ULA implementation plan, Part 5 -------------------
# Fig. B1: "Fixed: N = 8 (matches Xu's I = 8 for comparability), K = 3,
#           L_k ~ U{3,…,7}, RSR 10–12 dB. Two panels: P = 10 and P = 30."
# Fig. B2: "Sweep P from 2K to ~40, SNR = 5 dB."
TRACK_B_MASTER_SEED = 20250820
TRACK_B_N = 8
TRACK_B_K = 3
TRACK_B_L_MIN = 3
TRACK_B_L_MAX = 7
#: RSR = 12 dB. Inside the plan's stated 10–12 dB band, and the value Cui
#: uses throughout Figs. 5/7/8 with a physical justification (§VI-B: a LO
#: closer than a quarter of the user distance gives RSR > 10log10(4²) ≈ 12
#: dB). Deliberately *not* the 30 dB used in scratch testing, where
#: R(κ) → 1 makes EM-GS degenerate to GS and the two become inseparable.
TRACK_B_RSR_DB = 12.0

B1_SNR_DB = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
B1_P = (10, 30)            # the plan's two panels
B2_P = (6, 10, 14, 20, 30, 40)     # 2K = 6 up to ~40
B2_SNR_DB = 5.0

#: Estimators and whether they use the exact nonlinear model.
EXACT_MODEL = "EXACT"
LINEARIZED_MODEL = "LINEARIZED MODEL BASELINE"

ESTIMATORS: dict[str, str] = {
    "biased_gs": EXACT_MODEL,
    "em_gs": EXACT_MODEL,
    "structured_hankel": EXACT_MODEL,
    "structured_angular": EXACT_MODEL,
    "structured_esprit": EXACT_MODEL,
    "linearised_ls": LINEARIZED_MODEL,
}


@dataclass(frozen=True)
class TrackBPoint:
    sweep_key: str
    sweep_value: float
    estimator: str
    model: str
    nmse_db: float
    error_energy: float
    true_energy: float
    n_trials: int


def draw_L_k(
    trial_index: int,
    K: int = TRACK_B_K,
    *,
    master_seed: int = TRACK_B_MASTER_SEED,
    L_min: int = TRACK_B_L_MIN,
    L_max: int = TRACK_B_L_MAX,
) -> tuple[int, ...]:
    """``L_k ~ U{L_min,…,L_max}`` i.i.d. per user, per realization.

    The frozen system-model document lists ``L_k`` only as "user-dependent"
    and gives no distribution; the ULA implementation plan (Part 5, Fig. B1)
    is what specifies ``L_k ~ U{3,…,7}``. Drawn from a dedicated
    SeedSequence keyed by ``(master_seed, trial_index)`` so it is
    reproducible and independent of the channel/pilot/noise streams, and
    applied by building a per-trial config — the validated
    :func:`generate_ula_channel` is not modified.
    """
    ss = np.random.SeedSequence([int(master_seed), int(trial_index), 0x4C4B])
    rng = np.random.default_rng(ss)
    return tuple(int(v) for v in rng.integers(L_min, L_max + 1, size=int(K)))


def track_b_spec(
    *,
    P: int,
    n_trials: int,
    N: int = TRACK_B_N,
    K: int = TRACK_B_K,
    L: Sequence[int] = (TRACK_B_L_MIN,) * TRACK_B_K,
    master_seed: int = TRACK_B_MASTER_SEED,
    experiment: str = "track_b",
) -> ExperimentSpec:
    cfg = SimulationConfig.create(
        N=N, K=K, L=tuple(L), beta=1.0, master_seed=master_seed, c=1.0
    )
    return ExperimentSpec(
        experiment=experiment, track="B", cfg=cfg, P=int(P), vartheta=0.0,
        snr_db_grid=(0.0,), rsr_db_grid=(TRACK_B_RSR_DB,), n_trials=int(n_trials),
        algorithms=("biased_gs", "em_gs", "linearised_ls"),
        max_iter=50, qam_M=4, channel_model="ula_geometric", write_ber=False,
    )


def estimate(world, estimator: str, *, max_iter: int = 50, inner_iter: int = 1):
    """Run one estimator on one frozen world. Returns ``G_hat``."""
    if estimator == "biased_gs":
        return biased_gs_channel_rows(
            world.S, world.Z, world.B, max_iter=max_iter).G_hat
    if estimator == "em_gs":
        return em_gs_channel_rows(
            world.S, world.Z, world.B, world.sigma2, max_iter=max_iter).G_hat
    if estimator.startswith("structured_"):
        proj = estimator.split("_", 1)[1]
        return structured_exact_estimate(
            world.S, world.Z, world.B, world.sigma2, exact_step="em_gs",
            projection=proj, n_paths="auto", n_outer=1, max_iter=max_iter,
            inner_iter=inner_iter,
        ).G_hat
    if estimator == "linearised_ls":
        from .baselines import linearised_closed_form_ls
        return linearised_closed_form_ls(
            world.Y, world.S, world.Psi,
            observation_source="exact_magnitude").G_hat
    raise ValueError(f"unknown Track-B estimator {estimator!r}")


def track_b_world(trial_index: int, P: int, snr_db: float,
                  *, rsr_db: float = TRACK_B_RSR_DB,
                  N: int = TRACK_B_N, K: int = TRACK_B_K,
                  master_seed: int = TRACK_B_MASTER_SEED,
                  experiment: str = "track_b"):
    """One frozen Track-B world with ``L_k`` redrawn for this realization.

    All estimators at this ``(trial, P, SNR)`` receive this identical object,
    so common random numbers hold across algorithms by construction.
    """
    L = draw_L_k(trial_index, K, master_seed=master_seed)
    spec = track_b_spec(P=P, n_trials=trial_index + 1, N=N, K=K, L=L,
                        master_seed=master_seed, experiment=experiment)
    return generate_channel_estimation_trial(
        spec, int(trial_index), float(snr_db), float(rsr_db))


def _sweep(points, sweep_key, make_world, estimators, n_trials):
    out: list[TrackBPoint] = []
    for value in points:
        acc = {e: 0.0 for e in estimators}
        tot = 0.0
        for t in range(n_trials):
            world = make_world(value, t)
            tot += float(np.linalg.norm(world.G, ord="fro") ** 2)
            for e in estimators:
                G_hat = estimate(world, e)
                acc[e] += channel_nmse(G_hat, world.G).error_energy
        for e in estimators:
            out.append(TrackBPoint(
                sweep_key=sweep_key, sweep_value=float(value), estimator=e,
                model=ESTIMATORS[e], nmse_db=10 * np.log10(acc[e] / tot),
                error_energy=acc[e], true_energy=tot, n_trials=int(n_trials),
            ))
    return out


BASELINE_ESTIMATORS: tuple[str, ...] = ("biased_gs", "em_gs")


def run_b1(
    *, n_trials: int, snr_db=B1_SNR_DB, P: int = 10,
    estimators: Sequence[str] = BASELINE_ESTIMATORS,
) -> list[TrackBPoint]:
    """B1 — channel NMSE vs SNR at one pilot length (plan: P = 10 and 30)."""
    return _sweep(
        snr_db, "snr_db",
        lambda v, t: track_b_world(t, int(P), float(v), experiment="track_b_b1"),
        estimators, n_trials,
    )


def run_b2(
    *, n_trials: int, P_grid=B2_P, snr_db: float = B2_SNR_DB,
    estimators: Sequence[str] = BASELINE_ESTIMATORS,
) -> list[TrackBPoint]:
    """B2 — channel NMSE vs pilot length P (plan: 2K…~40 at SNR = 5 dB)."""
    return _sweep(
        P_grid, "P",
        lambda v, t: track_b_world(t, int(v), float(snr_db),
                                   experiment="track_b_b2"),
        estimators, n_trials,
    )


def format_table(points: Sequence[TrackBPoint]) -> str:
    keys = sorted({p.sweep_value for p in points})
    ests = list(dict.fromkeys(p.estimator for p in points))
    key = points[0].sweep_key if points else "x"
    head = f"{key:>6} | " + " ".join(f"{e:>18}" for e in ests)
    lines = [head, "-" * len(head)]
    for k in keys:
        row = {p.estimator: p.nmse_db for p in points if p.sweep_value == k}
        lines.append(f"{k:>6g} | " + " ".join(f"{row.get(e, float('nan')):18.2f}"
                                              for e in ests))
    return "\n".join(lines)


__all__ = [
    "B1_P", "B1_SNR_DB", "B2_P", "B2_SNR_DB", "ESTIMATORS", "EXACT_MODEL",
    "LINEARIZED_MODEL", "BASELINE_ESTIMATORS", "TrackBPoint", "draw_L_k",
    "estimate", "format_table", "run_b1", "run_b2", "track_b_spec",
    "track_b_world",
]

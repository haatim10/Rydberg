"""Cui Fig. 6 — detection NMSE vs RSR at fixed SNR (Track A).

Same frozen Track-A stack as Fig. 5: the Cui detection model
``z = |A^H s + b + w|`` with ``A`` known, ``s`` the unknown 16-QAM vector,
``b`` the known reference. Solvers are called as ``biased_gs(M=A, ...)``
and ``em_gs(M=A, ...)`` directly — never through the Track-B
channel-estimation conjugation adapter.

Nothing mathematical is defined here. The channel, calibration, solvers,
CRLB and metrics are all imported unchanged; this module only sweeps RSR
instead of SNR and adds the two Fig. 6-specific checks below.

Cui §VI-B: "As opposed to a uniform NMSE performance achieved by the ZF
with known phase, the NMSE of all PR solvers rapidly declines by 5 dB as
the RSR increases."

ZF flatness (the critical Fig. 6 sanity check)
----------------------------------------------
The genie ZF reconstructs ``z ⊙ e^{jθ} = A^H s + b + w`` exactly from the
true phase, then subtracts the known ``b``. Its error is therefore
``(A A^H)^{-1} A w``, which does not depend on ``b`` at all. The ZF curve
must be **statistically flat in RSR**; each RSR point is an independent
estimate of the same quantity, because the operating-point RNG key
includes ``rsr_db``. A significant slope means the experiment or the
calibration is wrong, not the solver.

:func:`zf_flatness` fits ``NMSE_dB = a + b * RSR_dB`` by weighted least
squares using each point's delta-method standard error, and reports the
slope with its standard error and t-statistic.

RSR calibration
---------------
:func:`measure_rsr_calibration` re-measures the achieved RSR from the
generated worlds using Cui's single-user definition, eq. 38:

    RSR = E(|b_n|^2) / E(|a_{n,k} s_k|^2)

evaluated as ``mean_n |b_n|^2 / mean_{n,k} |A[k,n]|^2`` with unit-energy
QAM, which is what :func:`rydberg_sim.channel_cui.generate_cui_reference`
targets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .channel_cui import CuiChannelParams
from .config import SimulationConfig
from .monte_carlo import (
    ExperimentSpec,
    aggregate_result_table,
    config_fingerprint,
    fingerprint_payload,
    generate_detection_trial,
    load_result_table,
    run_experiment,
)
from .track_a import FIG5_K, FIG5_MASTER_SEED, FIG5_N, FIG5_QAM, FIG5_T0, TRACK_A_RESULTS
from .track_a_fig5 import (
    FIG5_ALGORITHMS,
    FIG5_CONVERGENCE_ALGS,
    FIG5_CONVERGENCE_TOL_DB,
    OUTLIER_PERCENTILES,
    _agg_lookup,
    _json_dump,
    checkpoint_deltas_db,
    completed_trial_count,
    convergence_satisfied,
    enrich_aggregate_row,
    outlier_diagnostics,
    rows_with_trial_prefix,
)

FIG6_SNR_DB: float = 3.0
FIG6_RSR_DB: tuple[float, ...] = tuple(float(r) for r in range(0, 26))
FIG6_ALGORITHMS: tuple[str, ...] = FIG5_ALGORITHMS
FIG6_CHECKPOINTS: tuple[int, ...] = (250, 500, 1000, 2000)
FIG6_MIN_LARGE_CHECKPOINT: int = 250
FIG6_EXPERIMENT: str = "cui_fig6"
FIG6_DIRNAME: str = "fig6"
FIG6_GAP_RSR_DB: tuple[float, ...] = (0.0, 5.0, 10.0, 12.0, 15.0, 20.0, 25.0)
FIG6_CALIBRATION_RSR_DB: tuple[float, ...] = (0.0, 12.0, 25.0)
FIG6_CALIBRATION_TOL_DB: float = 0.1
FIG6_CALIBRATION_TRIALS: int = 200
# A ZF slope is called significant when |t| exceeds this. 2 sigma.
FIG6_ZF_SLOPE_T_MAX: float = 2.0

CONVERGENCE_CRITERION = (
    "For biased GS and EM-GS, the ratio-of-sums detection NMSE at every "
    f"integer-dB RSR in [0, 25] must change by less than "
    f"{FIG5_CONVERGENCE_TOL_DB} dB between the last two checkpoints with "
    f"n_trials >= {FIG6_MIN_LARGE_CHECKPOINT}. Ladder: "
    f"{list(FIG6_CHECKPOINTS)} trials/RSR. Stop as soon as the criterion "
    "holds; do not force the ceiling."
)


def default_fig6_dir() -> Path:
    return TRACK_A_RESULTS / FIG6_DIRNAME


def track_a_fig6_spec(
    *,
    n_trials: int,
    rsr_db_grid: Sequence[float] = FIG6_RSR_DB,
    experiment: str = FIG6_EXPERIMENT,
    master_seed: int = FIG5_MASTER_SEED,
    cui_params: CuiChannelParams | None = None,
) -> ExperimentSpec:
    """Cui Fig. 6: NMSE vs RSR at fixed SNR = 3 dB, 16-QAM, N x K = 36 x 3.

    Identical Track-A configuration to Fig. 5 except that RSR is swept and
    SNR is fixed. The config fingerprint therefore **matches** Fig. 5's by
    design — ``fingerprint_payload`` deliberately excludes the SNR/RSR
    grids so a run can be extended with more points. The two stores are
    kept apart by ``experiment`` (part of the result key, and checked by
    ``_assert_compatible_store``) and by living in separate directories.
    """
    cfg = SimulationConfig.create(
        N=FIG5_N, K=FIG5_K, L=1, beta=1.0, master_seed=master_seed, c=1.0
    )
    return ExperimentSpec(
        experiment=experiment,
        track="A",
        cfg=cfg,
        P=1,
        vartheta=0.0,
        snr_db_grid=(FIG6_SNR_DB,),
        rsr_db_grid=tuple(float(r) for r in rsr_db_grid),
        n_trials=n_trials,
        algorithms=FIG6_ALGORITHMS,
        max_iter=FIG5_T0,
        qam_M=FIG5_QAM,
        channel_model="cui_38901",
        cui_params=cui_params if cui_params is not None else CuiChannelParams(),
        write_ber=False,
    )


def aggregate_fig6_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Detection-NMSE aggregate records keyed by ``rsr_db``."""
    out: list[dict[str, Any]] = []
    for rec in aggregate_result_table(rows):
        if rec.metric != "detection_nmse":
            continue
        out.append(
            {
                "algorithm": rec.algorithm,
                "snr_db": rec.snr_db,
                "rsr_db": rec.rsr_db,
                "n_ok": rec.n_ok,
                "n_failed": rec.n_failed,
                "nmse_linear": rec.value_linear,
                "nmse_db": rec.value_db,
                "se_linear": rec.se_linear,
                "total_error_energy": rec.total_error_energy,
                "total_expected_symbol_energy": rec.total_expected_symbol_energy,
            }
        )
    out.sort(key=lambda d: (d["algorithm"], d["rsr_db"]))
    return out


def zf_flatness(agg: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Weighted straight-line fit of ZF NMSE-dB against RSR-dB.

    The genie ZF error is ``(A A^H)^{-1} A w``, independent of ``b``, so
    the true slope is exactly zero. Weights are ``1 / se_db^2`` from the
    delta-method standard error of each point.

    Returns the fitted slope in dB NMSE per dB RSR, its standard error,
    the t-statistic, and the range of the curve. ``significant_slope`` is
    True when ``|t| > FIG6_ZF_SLOPE_T_MAX``, which is the stop condition.
    """
    pts = [enrich_aggregate_row(r) for r in agg if str(r["algorithm"]) == "genie_zf"]
    pts.sort(key=lambda r: float(r["rsr_db"]))
    if len(pts) < 3:
        return {"n_points": len(pts), "fitted": False}
    x = np.asarray([float(p["rsr_db"]) for p in pts], dtype=np.float64)
    y = np.asarray([float(p["nmse_db"]) for p in pts], dtype=np.float64)
    se = np.asarray(
        [float(p["se_db"]) if p["se_db"] is not None else np.nan for p in pts],
        dtype=np.float64,
    )
    ok = np.isfinite(se) & (se > 0.0)
    w = np.where(ok, 1.0 / np.maximum(se, 1e-12) ** 2, 1.0)

    sw = float(np.sum(w))
    xm = float(np.sum(w * x) / sw)
    ym = float(np.sum(w * y) / sw)
    sxx = float(np.sum(w * (x - xm) ** 2))
    slope = float(np.sum(w * (x - xm) * (y - ym)) / sxx)
    intercept = ym - slope * xm
    slope_se = float(np.sqrt(1.0 / sxx)) if np.all(ok) else float("nan")
    resid = y - (intercept + slope * x)
    # Unweighted scatter, for a weight-free cross-check on the SE.
    dof = max(1, len(x) - 2)
    s2 = float(np.sum(w * resid**2) / dof)
    slope_se_scaled = float(np.sqrt(s2 / sxx))
    t_stat = slope / slope_se if np.isfinite(slope_se) and slope_se > 0 else float("nan")
    t_scaled = (
        slope / slope_se_scaled
        if np.isfinite(slope_se_scaled) and slope_se_scaled > 0
        else float("nan")
    )
    return {
        "n_points": len(pts),
        "fitted": True,
        "max_nmse_db": float(np.max(y)),
        "min_nmse_db": float(np.min(y)),
        "range_db": float(np.max(y) - np.min(y)),
        "mean_nmse_db": float(np.mean(y)),
        "mean_se_db": float(np.nanmean(se)),
        "slope_db_per_db": slope,
        "slope_se": slope_se,
        "slope_se_residual_scaled": slope_se_scaled,
        "t_stat": t_stat,
        "t_stat_residual_scaled": t_scaled,
        "t_threshold": FIG6_ZF_SLOPE_T_MAX,
        "significant_slope": bool(
            np.isfinite(t_scaled) and abs(t_scaled) > FIG6_ZF_SLOPE_T_MAX
        ),
        "intercept_db": intercept,
        "residual_rms_db": float(np.sqrt(np.mean(resid**2))),
        "note": (
            "Genie ZF error is (A A^H)^{-1} A w, independent of b, so the "
            "true slope is exactly 0. A significant slope means the "
            "experiment or calibration is wrong, not the solver."
        ),
    }


def measure_rsr_calibration(
    spec: ExperimentSpec,
    rsr_db_values: Sequence[float] = FIG6_CALIBRATION_RSR_DB,
    n_trials: int = FIG6_CALIBRATION_TRIALS,
) -> list[dict[str, Any]]:
    """Empirical RSR from generated worlds, Cui eq. 38 (single-user).

        RSR = mean_n |b_n|^2 / mean_{n,k} |A[k,n]|^2

    with unit-energy QAM. Uses the production world builder, so this
    measures what the sweep actually ran, not a re-derivation.
    """
    out: list[dict[str, Any]] = []
    for target in rsr_db_values:
        ref = np.empty(n_trials)
        usr = np.empty(n_trials)
        for t in range(n_trials):
            w = generate_detection_trial(spec, t, FIG6_SNR_DB, float(target))
            ref[t] = float(np.mean(np.abs(np.asarray(w.b)) ** 2))
            usr[t] = float(np.mean(np.abs(np.asarray(w.A)) ** 2))
        lin = float(np.mean(ref) / np.mean(usr))
        meas = float(10.0 * np.log10(lin))
        out.append(
            {
                "target_rsr_db": float(target),
                "measured_rsr_db": meas,
                "error_db": meas - float(target),
                "within_tol": bool(abs(meas - float(target)) <= FIG6_CALIBRATION_TOL_DB),
                "tol_db": FIG6_CALIBRATION_TOL_DB,
                "mean_reference_power": float(np.mean(ref)),
                "mean_single_user_power": float(np.mean(usr)),
                "n_trials": int(n_trials),
            }
        )
    return out


def rsr_improvement_db(
    agg: Sequence[Mapping[str, Any]], algorithm: str, lo: float, hi: float
) -> float | None:
    """``NMSE(lo) - NMSE(hi)`` in dB; positive means improvement with RSR."""
    a = _agg_lookup(agg, algorithm, lo, "rsr_db")
    b = _agg_lookup(agg, algorithm, hi, "rsr_db")
    if a is None or b is None:
        return None
    return float(a["nmse_db"]) - float(b["nmse_db"])


def em_gs_minus_gs_db(
    agg: Sequence[Mapping[str, Any]], rsr_db: float
) -> float | None:
    g = _agg_lookup(agg, "biased_gs", rsr_db, "rsr_db")
    e = _agg_lookup(agg, "em_gs", rsr_db, "rsr_db")
    if g is None or e is None:
        return None
    return float(e["nmse_db"]) - float(g["nmse_db"])


def crlb_crossing_fig6(agg: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Point-estimate CRLB crossing check against the delta-method CI."""
    per: list[dict[str, Any]] = []
    for alg in FIG6_CONVERGENCE_ALGS_LOCAL:
        for rsr in sorted({float(r["rsr_db"]) for r in agg}):
            a = _agg_lookup(agg, alg, rsr, "rsr_db")
            c = _agg_lookup(agg, "cui_crlb", rsr, "rsr_db")
            if a is None or c is None:
                continue
            ea = enrich_aggregate_row(a)
            diff = float(a["nmse_db"]) - float(c["nmse_db"])
            se = ea["se_db"] if ea["se_db"] is not None else 0.0
            per.append(
                {
                    "algorithm": alg,
                    "rsr_db": rsr,
                    "nmse_minus_crlb_db": diff,
                    "se_db": se,
                    "point_below": bool(diff < 0.0),
                    "statistically_below": bool(diff + 1.959963984540054 * se < 0.0),
                }
            )
    return {
        "any_point_estimate_below_crlb": any(p["point_below"] for p in per),
        "any_statistically_below_crlb": any(p["statistically_below"] for p in per),
        "n_statistically_below": sum(1 for p in per if p["statistically_below"]),
        "per_rsr": per,
    }


FIG6_CONVERGENCE_ALGS_LOCAL: tuple[str, ...] = FIG5_CONVERGENCE_ALGS


def delta_summary(deltas: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """max / median / p90 / count-over-tol for a checkpoint delta list."""
    if not deltas:
        return {"n_cells": 0}
    a = np.asarray([float(d["abs_delta_db"]) for d in deltas], dtype=np.float64)
    return {
        "n_cells": int(a.size),
        "max_abs_delta_db": float(a.max()),
        "median_abs_delta_db": float(np.median(a)),
        "p90_abs_delta_db": float(np.percentile(a, 90)),
        "n_over_tol": int((a >= FIG5_CONVERGENCE_TOL_DB).sum()),
        "tol_db": FIG5_CONVERGENCE_TOL_DB,
        "within_tol": int((a < FIG5_CONVERGENCE_TOL_DB).sum()),
    }

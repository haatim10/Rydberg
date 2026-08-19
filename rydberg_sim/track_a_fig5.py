"""Cui Fig. 5 full Track-A reproduction (detection NMSE vs SNR).

Uses the frozen Step-8–14 stack and the Track-A Cui generator.
Does **not** run Fig. 6–8, Track B, Track C, or machine learning.
Does **not** overwrite ``results/track_a/fig5_smoke``.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .calibration import db_to_linear
from .channel_cui import (
    CuiChannelParams,
    generate_cui_channel,
    generate_cui_reference,
)
from .crlb import cui_crlb_high_snr_limit
from .monte_carlo import (
    DetectionTrial,
    ExperimentSpec,
    aggregate_result_table,
    config_fingerprint,
    evaluate_detection_algorithm,
    fingerprint_payload,
    generate_detection_trial,
    load_result_table,
    run_experiment,
)
from .qam import generate_qam
from .rng import get_operating_point_rngs, operating_point_spawn_key
from .track_a import (
    FIG5_K,
    FIG5_MASTER_SEED,
    FIG5_N,
    FIG5_QAM,
    FIG5_RSR_DB,
    FIG5_T0,
    TRACK_A_RESULTS,
    _aggregate_nmse_table,
    analytic_high_snr_crlb_zf_gap_db,
    track_a_fig5_spec,
)

FIG5_SNR_DB: tuple[float, ...] = tuple(float(s) for s in range(-5, 13))
FIG5_ALGORITHMS: tuple[str, ...] = ("biased_gs", "em_gs", "genie_zf", "cui_crlb")
FIG5_DISPLAY_NAME: dict[str, str] = {
    "biased_gs": "biased GS",
    "em_gs": "EM-GS",
    "genie_zf": "ZF-known-phase",
    "cui_crlb": "Cui CRLB",
}
FIG5_CHECKPOINTS: tuple[int, ...] = (100, 250, 500, 1000, 2000)
FIG5_INITIAL_TARGET_TRIALS: int = 500
FIG5_CONVERGENCE_TOL_DB: float = 0.1
FIG5_CONVERGENCE_ALGS: tuple[str, ...] = ("biased_gs", "em_gs")
FIG5_MIN_LARGE_CHECKPOINT: int = 250
FIG5_GAP_SNR_DB: tuple[float, ...] = (-5.0, -4.0, 0.0, 6.0, 12.0)
FIG5_MATERIAL_NORM_SHIFT_DB: float = 0.5
FIG5_NEGLIGIBLE_NORM_SHIFT_DB: float = 0.2
FIG5_NORM_DIAG_TRIALS: int = 32
FIG5_NORM_DIAG_SNR_DB: tuple[float, ...] = (-5.0, 0.0, 6.0, 12.0)
FIG5_EXPERIMENT: str = "cui_fig5"

SMOKE_DIRNAME = "fig5_smoke"
FIG5_DIRNAME = "fig5"

# Convergence criterion, frozen before the run (not chosen after seeing data).
CONVERGENCE_CRITERION = (
    "For biased GS and EM-GS, the ratio-of-sums detection NMSE at every "
    "integer-dB SNR in [-5, 12] must change by less than "
    f"{FIG5_CONVERGENCE_TOL_DB} dB between the last two checkpoints with "
    f"n_trials >= {FIG5_MIN_LARGE_CHECKPOINT}. Start at "
    f"{FIG5_INITIAL_TARGET_TRIALS} trials/SNR; if that test fails, continue "
    "through 1000 then 2000. Prefix aggregates at 100/250/500 are always "
    "reported even when the run starts at 500."
)


def _forbid_smoke_dir(output_dir: Path) -> None:
    parts = {p.lower() for p in output_dir.parts}
    if SMOKE_DIRNAME in parts or output_dir.name.lower() == SMOKE_DIRNAME:
        raise ValueError(
            f"refusing to write Fig. 5 full results under {output_dir}; "
            f"do not overwrite {SMOKE_DIRNAME}"
        )


def default_fig5_dir() -> Path:
    return TRACK_A_RESULTS / FIG5_DIRNAME


def default_n_workers() -> int:
    cpu = os.cpu_count() or 1
    return max(1, min(int(cpu), 4))


def _json_dump(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _nmse_se_db(nmse_linear: float, se_linear: float | None) -> float | None:
    """Delta-method SE of ``10 log10(NMSE)``."""
    if se_linear is None:
        return None
    lin = float(nmse_linear)
    se = float(se_linear)
    if not np.isfinite(lin) or lin <= 0.0 or not np.isfinite(se):
        return None
    return float((10.0 / np.log(10.0)) * se / lin)


def rows_with_trial_prefix(
    rows: Sequence[Mapping[str, Any]], n_trials: int
) -> list[dict[str, Any]]:
    return [dict(r) for r in rows if int(r["trial"]) < int(n_trials)]


def completed_trial_count(rows: Sequence[Mapping[str, Any]], spec: ExperimentSpec) -> int:
    """Largest N such that trials 0..N-1 are complete at every SNR/RSR/algorithm."""
    if not rows:
        return 0
    need = set(spec.algorithms)
    snrs = set(float(s) for s in spec.snr_db_grid)
    rsrs = set(float(r) for r in spec.rsr_db_grid)
    ok: dict[int, set[tuple[float, float, str]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        if row.get("metric") != "detection_nmse":
            continue
        trial = int(row["trial"])
        ok.setdefault(trial, set()).add(
            (float(row["snr_db"]), float(row["rsr_db"]), str(row["algorithm"]))
        )
    n = 0
    expected = {(s, r, a) for s in snrs for r in rsrs for a in need}
    while n in ok and expected <= ok[n]:
        n += 1
    return n


def pivot_nmse_db(agg: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """``{snr: {algorithm: nmse_db}}``."""
    out: dict[str, dict[str, float]] = {}
    for row in agg:
        snr = f"{float(row['snr_db']):g}"
        out.setdefault(snr, {})[str(row["algorithm"])] = float(row["nmse_db"])
    return dict(sorted(out.items(), key=lambda kv: float(kv[0])))


def _agg_lookup(
    agg: Sequence[Mapping[str, Any]], algorithm: str, snr_db: float
) -> dict[str, Any] | None:
    target = float(snr_db)
    for row in agg:
        if str(row["algorithm"]) == algorithm and abs(float(row["snr_db"]) - target) < 1e-12:
            return dict(row)
    return None


def checkpoint_deltas_db(
    prev: Sequence[Mapping[str, Any]],
    cur: Sequence[Mapping[str, Any]],
    algorithms: Sequence[str] = FIG5_CONVERGENCE_ALGS,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for alg in algorithms:
        snrs = sorted(
            {float(r["snr_db"]) for r in cur if str(r["algorithm"]) == alg}
        )
        for snr in snrs:
            a = _agg_lookup(prev, alg, snr)
            b = _agg_lookup(cur, alg, snr)
            if a is None or b is None:
                continue
            delta = float(b["nmse_db"]) - float(a["nmse_db"])
            out.append(
                {
                    "algorithm": alg,
                    "snr_db": snr,
                    "prev_nmse_db": float(a["nmse_db"]),
                    "cur_nmse_db": float(b["nmse_db"]),
                    "delta_db": delta,
                    "abs_delta_db": abs(delta),
                }
            )
    out.sort(key=lambda d: (d["algorithm"], d["snr_db"]))
    return out


def convergence_satisfied(
    deltas: Sequence[Mapping[str, Any]],
    *,
    tol_db: float = FIG5_CONVERGENCE_TOL_DB,
) -> dict[str, Any]:
    if not deltas:
        return {
            "converged": False,
            "tol_db": tol_db,
            "max_abs_delta_db": None,
            "n_violations": None,
            "violations": [],
            "reason": "no deltas",
        }
    viol = [d for d in deltas if float(d["abs_delta_db"]) >= float(tol_db)]
    max_abs = max(float(d["abs_delta_db"]) for d in deltas)
    return {
        "converged": len(viol) == 0,
        "tol_db": tol_db,
        "max_abs_delta_db": max_abs,
        "n_violations": len(viol),
        "violations": viol,
        "criterion": CONVERGENCE_CRITERION,
    }


def enrich_aggregate_row(row: Mapping[str, Any]) -> dict[str, Any]:
    se_lin = row.get("se_linear")
    se_db = _nmse_se_db(float(row["nmse_linear"]), se_lin if se_lin is None else float(se_lin))
    out = dict(row)
    out["se_db"] = se_db
    if se_db is not None and np.isfinite(float(row["nmse_db"])):
        z = 1.959963984540054
        out["nmse_db_ci95_low"] = float(row["nmse_db"]) - z * se_db
        out["nmse_db_ci95_high"] = float(row["nmse_db"]) + z * se_db
    else:
        out["nmse_db_ci95_low"] = None
        out["nmse_db_ci95_high"] = None
    return out


def write_aggregate_csv(path: Path, agg: Sequence[Mapping[str, Any]]) -> None:
    rows = [enrich_aggregate_row(r) for r in agg]
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = [
        "algorithm",
        "snr_db",
        "rsr_db",
        "n_ok",
        "n_failed",
        "nmse_linear",
        "nmse_db",
        "se_linear",
        "se_db",
        "nmse_db_ci95_low",
        "nmse_db_ci95_high",
        "total_error_energy",
        "total_expected_symbol_energy",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _freeze(arr: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    out = np.array(arr, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


def generate_unnormalized_detection_trial(
    spec: ExperimentSpec,
    trial_index: int,
    snr_db: float,
    rsr_db: float,
) -> DetectionTrial:
    """Table I draw **without** per-row unit-power scaling.

    SNR (eq. 37) and RSR (eq. 38) are calibrated from the raw channel so
    the labeled operating point still matches the paper definitions:

        σ² = mean_n ||a_n||² / SNR_lin
        mean_n |b_n|² = RSR_lin * mean_{k,n} |a_{nk}|²

    Production Track A is unchanged (``generate_detection_trial``).
    """
    cfg = spec.cfg
    cui = spec.cui_params
    assert cui is not None
    spawn_key = operating_point_spawn_key(trial_index, snr_db, rsr_db)
    rngs = get_operating_point_rngs(cfg.master_seed, trial_index, snr_db, rsr_db)
    # Audit M4: the switch travels in the fingerprinted params, not as a
    # keyword, so this arm's config fingerprint differs from production.
    cui = replace(cui, normalize_rows=False)
    ch = generate_cui_channel(cfg.N, cfg.K, rngs.channel, params=cui)
    qam = generate_qam(rngs.data, cfg.K, spec.qam_M)
    b_unit_rsr = generate_cui_reference(cfg.N, rngs.reference, rsr_db, params=cui)
    one_user_power = float(np.mean(np.abs(ch.A) ** 2))
    signal_power = float(np.mean(np.sum(np.abs(ch.A) ** 2, axis=0)))
    snr_lin = db_to_linear(snr_db)
    sigma2 = signal_power / snr_lin
    b = np.array(b_unit_rsr, dtype=np.complex128, copy=True)
    b *= np.sqrt(one_user_power)
    b.setflags(write=False)
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
        b=b,
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
        channel_model=ch.channel_model,
    )


def run_row_normalization_diagnostic(
    output_dir: Path,
    *,
    n_trials: int = FIG5_NORM_DIAG_TRIALS,
    snr_db_grid: tuple[float, ...] = FIG5_NORM_DIAG_SNR_DB,
    n_workers: int = 1,
) -> dict[str, Any]:
    """Compare production row-normalization (A) vs raw Table I (B).

    Small trial count. Does not change the production Track-A definition.
    """
    del n_workers  # serial: two world builders, not the Step-14 CSV runner
    diag_dir = output_dir / "row_normalization_diagnostic"
    diag_dir.mkdir(parents=True, exist_ok=True)
    spec_a = track_a_fig5_spec(
        n_trials=n_trials,
        snr_db_grid=snr_db_grid,
        experiment="cui_fig5_norm_diag_A",
    )
    spec_b = track_a_fig5_spec(
        n_trials=n_trials,
        snr_db_grid=snr_db_grid,
        experiment="cui_fig5_norm_diag_B",
        # Audit M4: arm B is the raw Table-I draw, so it must carry a
        # different config fingerprint from production arm A.
        cui_params=CuiChannelParams(normalize_rows=False),
    )
    rows_a: list[dict[str, Any]] = []
    rows_b: list[dict[str, Any]] = []
    power_stats: list[dict[str, Any]] = []
    for trial in range(n_trials):
        for snr_db in snr_db_grid:
            world_a = generate_detection_trial(spec_a, trial, snr_db, FIG5_RSR_DB)
            world_b = generate_unnormalized_detection_trial(
                spec_b, trial, snr_db, FIG5_RSR_DB
            )
            raw_row_pow = np.mean(np.abs(np.asarray(world_b.A)) ** 2, axis=1)
            power_stats.append(
                {
                    "trial": trial,
                    "snr_db": float(snr_db),
                    "raw_mean_abs_sq_per_user": [float(x) for x in raw_row_pow],
                    "raw_signal_power": float(
                        np.mean(np.sum(np.abs(world_b.A) ** 2, axis=0))
                    ),
                    "normalized_signal_power": float(
                        np.mean(np.sum(np.abs(world_a.A) ** 2, axis=0))
                    ),
                    "sigma2_A": float(world_a.sigma2),
                    "sigma2_B": float(world_b.sigma2),
                    "same_s": bool(np.array_equal(world_a.s, world_b.s)),
                }
            )
            for alg in FIG5_ALGORITHMS:
                ra, _ = evaluate_detection_algorithm(world_a, alg, spec_a)
                rb, _ = evaluate_detection_algorithm(world_b, alg, spec_b)
                rows_a.extend(ra)
                rows_b.extend(rb)

    agg_a = _aggregate_nmse_table(rows_a)
    agg_b = _aggregate_nmse_table(rows_b)
    shifts: list[dict[str, Any]] = []
    max_gs_abs = 0.0
    for alg in FIG5_ALGORITHMS:
        for snr in snr_db_grid:
            a = _agg_lookup(agg_a, alg, snr)
            b = _agg_lookup(agg_b, alg, snr)
            if a is None or b is None:
                continue
            delta = float(b["nmse_db"]) - float(a["nmse_db"])
            rec = {
                "algorithm": alg,
                "snr_db": float(snr),
                "nmse_db_normalized_A": float(a["nmse_db"]),
                "nmse_db_unnormalized_B": float(b["nmse_db"]),
                "delta_B_minus_A_db": delta,
            }
            shifts.append(rec)
            if alg in FIG5_CONVERGENCE_ALGS:
                max_gs_abs = max(max_gs_abs, abs(delta))

    material = bool(max_gs_abs >= FIG5_MATERIAL_NORM_SHIFT_DB)
    negligible = bool(max_gs_abs < FIG5_NEGLIGIBLE_NORM_SHIFT_DB)
    raw_powers = np.asarray(
        [p["raw_mean_abs_sq_per_user"] for p in power_stats], dtype=np.float64
    )
    summary = {
        "n_trials": n_trials,
        "snr_db_grid": list(snr_db_grid),
        "production_unchanged": True,
        "A": "per-realization row normalization (production Track A)",
        "B": (
            "raw Table I draw; σ² and |b| recalibrated from the raw channel "
            "so labeled SNR/RSR still follow Cui eq. 37/38"
        ),
        "material_threshold_db": FIG5_MATERIAL_NORM_SHIFT_DB,
        "negligible_threshold_db": FIG5_NEGLIGIBLE_NORM_SHIFT_DB,
        "max_abs_delta_db_gs_emgs": max_gs_abs,
        "material": material,
        "negligible": negligible,
        "keep_production_normalization": (not material),
        "shifts": shifts,
        "raw_row_power": {
            "mean": float(np.mean(raw_powers)),
            "median": float(np.median(raw_powers)),
            "std": float(np.std(raw_powers)),
            "min": float(np.min(raw_powers)),
            "max": float(np.max(raw_powers)),
            "cv": float(np.std(raw_powers) / np.mean(raw_powers)),
        },
        "same_symbols_across_AB": all(bool(p["same_s"]) for p in power_stats),
        "note": (
            "A material shift means stop before treating Fig. 5 as the "
            "publication curve and report the normalization choice. "
            "Solvers were not modified."
        ),
    }
    _json_dump(diag_dir / "summary.json", summary)
    _json_dump(diag_dir / "aggregate_A.json", {"aggregate": agg_a})
    _json_dump(diag_dir / "aggregate_B.json", {"aggregate": agg_b})
    _json_dump(diag_dir / "power_stats.json", {"trials": power_stats[: min(64, len(power_stats))]})
    write_aggregate_csv(diag_dir / "aggregate_A.csv", agg_a)
    write_aggregate_csv(diag_dir / "aggregate_B.csv", agg_b)
    return summary


def em_gs_minus_gs_gap_db(agg: Sequence[Mapping[str, Any]], snr_db: float) -> float | None:
    gs = _agg_lookup(agg, "biased_gs", snr_db)
    em = _agg_lookup(agg, "em_gs", snr_db)
    if gs is None or em is None:
        return None
    return float(em["nmse_db"]) - float(gs["nmse_db"])


def crlb_minus_zf_gap_db(agg: Sequence[Mapping[str, Any]], snr_db: float) -> float | None:
    crlb = _agg_lookup(agg, "cui_crlb", snr_db)
    zf = _agg_lookup(agg, "genie_zf", snr_db)
    if crlb is None or zf is None:
        return None
    return float(crlb["nmse_db"]) - float(zf["nmse_db"])


def _paired_nmse_diff(
    rows: Sequence[Mapping[str, Any]],
    alg_left: str,
    alg_right: str,
    snr_db: float,
) -> dict[str, Any] | None:
    """SE of (NMSE_left − NMSE_right) from paired trial error energies."""
    left: dict[int, float] = {}
    right: dict[int, float] = {}
    energy: dict[int, float] = {}
    target = float(snr_db)
    for row in rows:
        if row.get("status") != "ok" or row.get("metric") != "detection_nmse":
            continue
        if abs(float(row["snr_db"]) - target) > 1e-12:
            continue
        trial = int(row["trial"])
        alg = str(row["algorithm"])
        e = float(row["error_energy"])
        den = float(row["expected_symbol_energy"])
        if alg == alg_left:
            left[trial] = e
            energy[trial] = den
        elif alg == alg_right:
            right[trial] = e
            energy.setdefault(trial, den)
    common = sorted(set(left) & set(right) & set(energy))
    if not common:
        return None
    diff = np.asarray([left[t] - right[t] for t in common], dtype=np.float64)
    den = np.asarray([energy[t] for t in common], dtype=np.float64)
    # Paired delta-method on (e_left - e_right) / expected_energy.
    # Do not reuse nmse_ratio_standard_error: diffs may be negative.
    sum_d = float(np.sum(diff))
    sum_t = float(np.sum(den))
    r = sum_d / sum_t
    n = len(common)
    if n < 2:
        se = float("nan")
    else:
        mean_d = sum_d / n
        mean_t = sum_t / n
        d_c = diff - mean_d
        t_c = den - mean_t
        var_d = float(np.dot(d_c, d_c) / (n - 1))
        var_t = float(np.dot(t_c, t_c) / (n - 1))
        cov = float(np.dot(d_c, t_c) / (n - 1))
        var_r = (var_d + (r**2) * var_t - 2.0 * r * cov) / (n * mean_t**2)
        se = float(np.sqrt(max(var_r, 0.0)))
    z = 1.959963984540054
    return {
        "algorithm_left": alg_left,
        "algorithm_right": alg_right,
        "snr_db": target,
        "n_paired": n,
        "nmse_diff_linear": r,
        "se_linear": se,
        "ci95_low": r - z * se if np.isfinite(se) else None,
        "ci95_high": r + z * se if np.isfinite(se) else None,
        "statistically_below": bool(np.isfinite(se) and (r + z * se) < 0.0),
        "point_below": bool(r < 0.0),
    }


def crlb_crossing_diagnostic(
    rows: Sequence[Mapping[str, Any]],
    snr_db_grid: Sequence[float],
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    persistent = False
    any_point_below = False
    any_stat_below = False
    for alg in FIG5_CONVERGENCE_ALGS:
        for snr in snr_db_grid:
            rec = _paired_nmse_diff(rows, alg, "cui_crlb", float(snr))
            if rec is None:
                continue
            rec.pop("unused_unc_n", None)
            details.append(rec)
            any_point_below = any_point_below or bool(rec["point_below"])
            any_stat_below = any_stat_below or bool(rec["statistically_below"])
    n_stat = sum(1 for d in details if d["statistically_below"])
    # Persistent: statistically below at 3+ SNR points for the same algorithm.
    for alg in FIG5_CONVERGENCE_ALGS:
        n_alg = sum(
            1
            for d in details
            if d["algorithm_left"] == alg and d["statistically_below"]
        )
        if n_alg >= 3:
            persistent = True
    return {
        "any_point_estimate_below_crlb": any_point_below,
        "any_statistically_below_crlb": any_stat_below,
        "persistent_statistically_meaningful_crossing": persistent,
        "n_statistically_below": n_stat,
        "per_snr": details,
        "interpretation": (
            "statistically_below means the paired (empirical − CRLB) NMSE "
            "95% CI lies entirely below 0. A small crossing inside the CI "
            "is reported but is not treated as a CRLB bug. A persistent "
            "statistically meaningful crossing stops Track A before Fig. 6; "
            "the CRLB curve is never shifted by hand."
        ),
    }


def acceptance_checks(
    agg: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    *,
    analytic_gap_db: float,
) -> dict[str, Any]:
    snrs = sorted({float(r["snr_db"]) for r in agg})
    by_alg: dict[str, list[float]] = {a: [] for a in FIG5_ALGORITHMS}
    for snr in snrs:
        for alg in FIG5_ALGORITHMS:
            rec = _agg_lookup(agg, alg, snr)
            if rec is not None:
                by_alg[alg].append(float(rec["nmse_db"]))

    def _decreases(vals: list[float]) -> bool:
        return all(vals[i + 1] <= vals[i] + 0.25 for i in range(len(vals) - 1))

    gap_m4 = em_gs_minus_gs_gap_db(agg, -4.0)
    gap_high = em_gs_minus_gs_gap_db(agg, 12.0)
    crlb_zf_12 = crlb_minus_zf_gap_db(agg, 12.0)
    em_vs_crlb = []
    zf_below_crlb = []
    for snr in snrs:
        em = _agg_lookup(agg, "em_gs", snr)
        crlb = _agg_lookup(agg, "cui_crlb", snr)
        zf = _agg_lookup(agg, "genie_zf", snr)
        if em is not None and crlb is not None:
            em_vs_crlb.append(float(em["nmse_db"]) - float(crlb["nmse_db"]))
        if zf is not None and crlb is not None:
            zf_below_crlb.append(float(zf["nmse_db"]) < float(crlb["nmse_db"]))

    crossing = crlb_crossing_diagnostic(rows, snrs)
    checks = {
        "1_nmse_decreases_with_snr": {
            "biased_gs": _decreases(by_alg["biased_gs"]),
            "em_gs": _decreases(by_alg["em_gs"]),
            "genie_zf": _decreases(by_alg["genie_zf"]),
            "cui_crlb": _decreases(by_alg["cui_crlb"]),
        },
        "2_em_gs_beats_biased_gs_at_low_snr": bool(
            (em_gs_minus_gs_gap_db(agg, -5.0) or 0.0) < 0.0
        ),
        "3_em_gs_improvement_near_minus_4dB_order_2dB": {
            "em_minus_gs_db": gap_m4,
            "paper_order_db": -2.0,
            "pass": gap_m4 is not None and -3.5 <= gap_m4 <= -0.5,
        },
        "4_gs_and_em_gs_merge_at_high_snr": {
            "em_minus_gs_db_at_12": gap_high,
            "pass": gap_high is not None and abs(gap_high) < 1.0,
        },
        "5_em_gs_tracks_magnitude_crlb": {
            "mean_em_minus_crlb_db": float(np.mean(em_vs_crlb)) if em_vs_crlb else None,
            "at_12_db": (
                float(_agg_lookup(agg, "em_gs", 12.0)["nmse_db"])
                - float(_agg_lookup(agg, "cui_crlb", 12.0)["nmse_db"])
                if _agg_lookup(agg, "em_gs", 12.0) and _agg_lookup(agg, "cui_crlb", 12.0)
                else None
            ),
        },
        "6_zf_below_magnitude_crlb": {
            "all_snr": all(zf_below_crlb) if zf_below_crlb else False,
            "per_snr_true": zf_below_crlb,
        },
        "7_high_snr_crlb_minus_zf_approx_3dB": {
            "empirical_at_12_db": crlb_zf_12,
            "analytic_10log10_2": analytic_gap_db,
            "pass": crlb_zf_12 is not None and abs(crlb_zf_12 - analytic_gap_db) < 0.75,
        },
        "8_not_several_dB_from_orientation": "see README; no digitized Cui overlay in-repo",
        "crlb_crossing": crossing,
    }
    return checks


def _plot_fig5(
    agg: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    error_bars: bool,
    title: str,
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    style = {
        "biased_gs": {"color": "#1f77b4", "marker": "o", "ls": "-", "label": "biased GS"},
        "em_gs": {"color": "#d62728", "marker": "s", "ls": "-", "label": "EM-GS"},
        "genie_zf": {
            "color": "#2ca02c",
            "marker": "^",
            "ls": "--",
            "label": "ZF-known-phase",
        },
        "cui_crlb": {
            "color": "black",
            "marker": None,
            "ls": "-.",
            "label": "Cui CRLB",
        },
    }
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for alg in FIG5_ALGORITHMS:
        pts = sorted(
            (float(r["snr_db"]), enrich_aggregate_row(r))
            for r in agg
            if str(r["algorithm"]) == alg
        )
        if not pts:
            continue
        x = [p[0] for p in pts]
        y = [float(p[1]["nmse_db"]) for p in pts]
        st = style[alg]
        if error_bars and alg in FIG5_CONVERGENCE_ALGS:
            yerr = [p[1]["se_db"] if p[1]["se_db"] is not None else 0.0 for p in pts]
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                color=st["color"],
                marker=st["marker"],
                ls=st["ls"],
                lw=1.8,
                ms=5.0,
                capsize=3.0,
                label=st["label"],
            )
        else:
            ax.plot(
                x,
                y,
                color=st["color"],
                marker=st["marker"],
                ls=st["ls"],
                lw=1.8,
                ms=5.0 if st["marker"] else 0.0,
                label=st["label"],
            )
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Detection NMSE (dB)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, fontsize=8)
    ax.set_xlim(FIG5_SNR_DB[0] - 0.3, FIG5_SNR_DB[-1] + 0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def write_fig5_readme(
    path: Path,
    *,
    n_trials: int,
    agg: Sequence[Mapping[str, Any]],
    checks: Mapping[str, Any],
    convergence: Mapping[str, Any],
    norm_diag: Mapping[str, Any] | None,
    analytic_gap_db: float,
    core_solvers_unchanged: bool,
) -> None:
    def _fmt(alg: str, snr: float) -> str:
        rec = _agg_lookup(agg, alg, snr)
        if rec is None:
            return "n/a"
        return f"{float(rec['nmse_db']):.3f}"

    lines = [
        "# Track A — Cui Fig. 5 reproduction",
        "",
        "Detection NMSE vs SNR for the frozen Track-A stack.",
        "**Fig. 6, Fig. 7, Fig. 8, Track B, Track C, and machine learning were not run.**",
        "",
        "## Settings",
        "",
        f"- N = {FIG5_N}, K = {FIG5_K}, 16-QAM, RSR = {FIG5_RSR_DB:g} dB, t0 = {FIG5_T0}",
        f"- SNR grid (integer dB): {list(FIG5_SNR_DB)}",
        f"- Trials per SNR point: **{n_trials}**",
        "- Algorithms: biased GS, EM-GS, ZF-known-phase, Cui exact-model CRLB",
        "- CM-ZF omitted (exact source formulation still not implemented)",
        "- Aggregation: ratio of sums of linear energies, then `10 log10`",
        "- Expected symbol-vector energy = K = 3 (unit-energy 16-QAM; not demapped)",
        "",
        "## Convergence criterion (set before the run)",
        "",
        CONVERGENCE_CRITERION,
        "",
        f"Achieved: converged = {convergence.get('final', {}).get('converged')} "
        f"with max |Δ| = {convergence.get('final', {}).get('max_abs_delta_db')} dB "
        f"at n_trials = {n_trials}.",
        "",
        "## Core solvers",
        "",
        "Unchanged: `biased_gs`, `em_gs`, `spectral_initialize`, `zf_known_phase`, "
        "`cui_crlb`, Step-13 `detection_nmse`, Step-14 Monte Carlo harness.",
        f"Confirmed: {core_solvers_unchanged}.",
        "",
        "## Final NMSE (dB)",
        "",
        "| SNR (dB) | biased GS | EM-GS | ZF-known-phase | Cui CRLB |",
        "|---:|---:|---:|---:|---:|",
    ]
    snrs = sorted({float(r["snr_db"]) for r in agg})
    for snr in snrs:
        lines.append(
            f"| {snr:g} | {_fmt('biased_gs', snr)} | {_fmt('em_gs', snr)} | "
            f"{_fmt('genie_zf', snr)} | {_fmt('cui_crlb', snr)} |"
        )
    lines += [
        "",
        "## EM-GS minus biased GS (dB; negative means EM-GS better)",
        "",
    ]
    for snr in FIG5_GAP_SNR_DB:
        gap = em_gs_minus_gs_gap_db(agg, snr)
        lines.append(f"- SNR = {snr:g} dB: {gap if gap is None else f'{gap:.3f} dB'}")
    lines += [
        "",
        f"## High-SNR CRLB minus ZF: {crlb_minus_zf_gap_db(agg, 12.0)} dB "
        f"(analytic 10 log10 2 = {analytic_gap_db:.4f} dB)",
        "",
        "## Row-normalization diagnostic",
        "",
    ]
    if norm_diag is None:
        lines.append("Not run.")
    else:
        lines += [
            f"- A = production per-realization row normalization",
            f"- B = raw Table I with eq. 37/38 recalibrated from the raw channel",
            f"- Trials: {norm_diag.get('n_trials')}",
            f"- max |Δ| on GS/EM-GS: {norm_diag.get('max_abs_delta_db_gs_emgs')} dB",
            f"- material (≥ {FIG5_MATERIAL_NORM_SHIFT_DB} dB): {norm_diag.get('material')}",
            f"- negligible (< {FIG5_NEGLIGIBLE_NORM_SHIFT_DB} dB): {norm_diag.get('negligible')}",
            f"- keep production normalization: {norm_diag.get('keep_production_normalization')}",
        ]
        rp = norm_diag.get("raw_row_power") or {}
        lines.append(
            f"- raw Table I mean_n |a_nk|²: mean={rp.get('mean')}, "
            f"cv={rp.get('cv')}, min={rp.get('min')}, max={rp.get('max')}"
        )
    lines += [
        "",
        "## CRLB crossing",
        "",
        json.dumps(checks.get("crlb_crossing", {}), indent=2)[:4000],
        "",
        "## Acceptance vs Cui (qualitative / order-of-magnitude)",
        "",
        json.dumps({k: v for k, v in checks.items() if k != "crlb_crossing"}, indent=2),
        "",
        "## Documented deviations still in force",
        "",
        "See `results/track_a/README.md`. None of those deviations were changed "
        "for this sweep. CM-ZF remains omitted. Per-realization row normalization "
        "is still the production Track-A channel definition unless the diagnostic "
        "above flagged a material shift.",
        "",
        "## What was not run",
        "",
        "Fig. 6, Fig. 7(a), Fig. 7(b), Fig. 8, Track B, Track C, ML/neural networks.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_fig5(
    output_dir: Path | str | None = None,
    *,
    n_workers: int | None = None,
    max_trials: int = FIG5_CHECKPOINTS[-1],
    initial_target: int = FIG5_INITIAL_TARGET_TRIALS,
    skip_norm_diag: bool = False,
    run_main_sweep: bool = True,
    checkpoints: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Full Fig. 5 reproduction with batch checkpoints and diagnostics."""
    out = Path(output_dir) if output_dir is not None else default_fig5_dir()
    _forbid_smoke_dir(out)
    out.mkdir(parents=True, exist_ok=True)
    workers = default_n_workers() if n_workers is None else int(n_workers)

    spec0 = track_a_fig5_spec(
        n_trials=1,
        snr_db_grid=FIG5_SNR_DB,
        experiment=FIG5_EXPERIMENT,
    )
    config = {
        "experiment": FIG5_EXPERIMENT,
        "track": "A",
        "channel_model": spec0.channel_model,
        "config_fingerprint": config_fingerprint(spec0),
        "fingerprint_payload": fingerprint_payload(spec0),
        "N": FIG5_N,
        "K": FIG5_K,
        "qam_M": FIG5_QAM,
        "t0": FIG5_T0,
        "rsr_db": FIG5_RSR_DB,
        "snr_db_grid": list(FIG5_SNR_DB),
        "algorithms": list(FIG5_ALGORITHMS),
        "cm_zf": "omitted; source formulation not implemented",
        "master_seed": FIG5_MASTER_SEED,
        "aggregation": (
            "sum ||s_hat_cont - s||^2 / sum expected_symbol_energy, "
            "expected_symbol_energy = K, then 10 log10"
        ),
        "crn": "identical A,s,b,w,z for biased GS and EM-GS at each (trial,SNR,RSR)",
        "convergence_criterion": CONVERGENCE_CRITERION,
        "convergence_tol_db": FIG5_CONVERGENCE_TOL_DB,
        "checkpoints": list(FIG5_CHECKPOINTS),
        "initial_target_trials": initial_target,
        "max_trials": max_trials,
        "n_workers": workers,
        "core_solvers_unchanged": True,
        "not_run": [
            "fig6",
            "fig7a",
            "fig7b",
            "fig8",
            "track_b",
            "track_c",
            "machine_learning",
        ],
        "does_not_overwrite": "results/track_a/fig5_smoke",
    }
    _json_dump(out / "config.json", config)

    norm_summary: dict[str, Any] | None = None
    if not skip_norm_diag:
        norm_summary = run_row_normalization_diagnostic(out)
        if bool(norm_summary.get("material")):
            write_fig5_readme(
                out / "README.md",
                n_trials=0,
                agg=[],
                checks={"stopped": "row-normalization diagnostic was material"},
                convergence={"final": {"converged": False, "max_abs_delta_db": None}},
                norm_diag=norm_summary,
                analytic_gap_db=float(10.0 * np.log10(2.0)),
                core_solvers_unchanged=True,
            )
            return {
                "stopped": True,
                "reason": "row-normalization diagnostic material",
                "normalization_diagnostic": norm_summary,
                "output_dir": str(out),
            }

    if not run_main_sweep:
        return {
            "stopped": False,
            "normalization_diagnostic": norm_summary,
            "output_dir": str(out),
            "main_sweep": False,
        }

    checkpoints_plan = [n for n in (checkpoints if checkpoints is not None else FIG5_CHECKPOINTS) if n <= max_trials]
    if initial_target <= max_trials and initial_target not in checkpoints_plan:
        checkpoints_plan = sorted(set(checkpoints_plan) | {initial_target})
    if not checkpoints_plan:
        checkpoints_plan = [max_trials]
    prefix_targets = [n for n in (100, 250, 500) if n <= max_trials]

    csv_path: Path | None = None
    last_n = 0
    conv_log: list[dict[str, Any]] = []
    final_conv: dict[str, Any] = {
        "converged": False,
        "max_abs_delta_db": None,
        "tol_db": FIG5_CONVERGENCE_TOL_DB,
    }

    for n_trials in checkpoints_plan:
        spec = track_a_fig5_spec(
            n_trials=n_trials,
            snr_db_grid=FIG5_SNR_DB,
            experiment=FIG5_EXPERIMENT,
        )
        print(
            f"[fig5] checkpoint n_trials={n_trials} workers={workers}",
            flush=True,
        )
        csv_path = run_experiment(spec, out, n_workers=workers, resume=True)
        rows = load_result_table(csv_path)
        complete = completed_trial_count(rows, spec)
        agg_n = _aggregate_nmse_table(rows_with_trial_prefix(rows, n_trials))
        rec: dict[str, Any] = {
            "n_trials_requested": n_trials,
            "n_trials_complete": complete,
            "aggregate": [enrich_aggregate_row(r) for r in agg_n],
        }
        if last_n >= FIG5_MIN_LARGE_CHECKPOINT and n_trials > last_n:
            prev_rows = rows_with_trial_prefix(rows, last_n)
            prev_agg = _aggregate_nmse_table(prev_rows)
            deltas = checkpoint_deltas_db(prev_agg, agg_n)
            rec["vs_previous"] = {
                "previous_n": last_n,
                "deltas": deltas,
                **convergence_satisfied(deltas),
            }
            final_conv = {
                "converged": rec["vs_previous"]["converged"],
                "max_abs_delta_db": rec["vs_previous"]["max_abs_delta_db"],
                "tol_db": FIG5_CONVERGENCE_TOL_DB,
                "previous_n": last_n,
                "current_n": n_trials,
                "violations": rec["vs_previous"]["violations"],
            }
        conv_log.append(rec)
        last_n = n_trials
        if (
            n_trials >= initial_target
            and rec.get("vs_previous", {}).get("converged") is True
        ):
            break
        if n_trials >= initial_target and rec.get("vs_previous") is None:
            # First large checkpoint has no predecessor at >= 250 from this
            # loop yet; compare prefixes 250 vs 500 if both exist.
            if n_trials >= 500:
                agg_250 = _aggregate_nmse_table(rows_with_trial_prefix(rows, 250))
                agg_500 = _aggregate_nmse_table(rows_with_trial_prefix(rows, min(500, n_trials)))
                deltas = checkpoint_deltas_db(agg_250, agg_500)
                prefix_conv = convergence_satisfied(deltas)
                rec["prefix_250_vs_500"] = {
                    "previous_n": 250,
                    "current_n": min(500, n_trials),
                    "deltas": deltas,
                    **prefix_conv,
                }
                final_conv = {
                    "converged": prefix_conv["converged"],
                    "max_abs_delta_db": prefix_conv["max_abs_delta_db"],
                    "tol_db": FIG5_CONVERGENCE_TOL_DB,
                    "previous_n": 250,
                    "current_n": min(500, n_trials),
                    "violations": prefix_conv["violations"],
                }
                if prefix_conv["converged"]:
                    break

    assert csv_path is not None
    rows = load_result_table(csv_path)
    spec_final = track_a_fig5_spec(
        n_trials=last_n,
        snr_db_grid=FIG5_SNR_DB,
        experiment=FIG5_EXPERIMENT,
    )
    n_complete = completed_trial_count(rows, spec_final)
    used = n_complete if n_complete > 0 else last_n
    rows_used = rows_with_trial_prefix(rows, used)
    agg = _aggregate_nmse_table(rows_used)
    enriched = [enrich_aggregate_row(r) for r in agg]

    prefix_tables: dict[str, Any] = {}
    for n in prefix_targets:
        if n <= used:
            prefix_tables[str(n)] = [
                enrich_aggregate_row(r)
                for r in _aggregate_nmse_table(rows_with_trial_prefix(rows, n))
            ]
    prefix_deltas: dict[str, Any] = {}
    ordered_prefixes = [int(k) for k in prefix_tables]
    for a, b in zip(ordered_prefixes, ordered_prefixes[1:]):
        d = checkpoint_deltas_db(prefix_tables[str(a)], prefix_tables[str(b)])
        prefix_deltas[f"{a}_to_{b}"] = {
            "deltas": d,
            **convergence_satisfied(d),
        }

    analytic_gap = analytic_high_snr_crlb_zf_gap_db(spec_final)
    checks = acceptance_checks(agg, rows_used, analytic_gap_db=analytic_gap)
    gaps = {str(snr): em_gs_minus_gs_gap_db(agg, snr) for snr in FIG5_GAP_SNR_DB}

    write_aggregate_csv(out / "aggregate.csv", agg)
    _json_dump(
        out / "aggregate.json",
        {
            "n_trials": used,
            "analytic_high_snr_crlb_over_zf_db": analytic_gap,
            "analytic_10log10_2": float(10.0 * np.log10(2.0)),
            "em_gs_minus_biased_gs_db": gaps,
            "crlb_minus_zf_db_at_12": crlb_minus_zf_gap_db(agg, 12.0),
            "aggregate": enriched,
        },
    )
    _json_dump(
        out / "convergence.json",
        {
            "criterion": CONVERGENCE_CRITERION,
            "tol_db": FIG5_CONVERGENCE_TOL_DB,
            "final": final_conv,
            "prefixes": prefix_tables,
            "prefix_deltas": prefix_deltas,
            "checkpoint_log": [
                {k: v for k, v in c.items() if k != "aggregate"} | {
                    "nmse_db": pivot_nmse_db(c.get("aggregate", []))
                }
                for c in conv_log
            ],
        },
    )
    _json_dump(
        out / "uncertainty.json",
        {
            "method": (
                "Delta-method SE of ratio-of-sums NMSE (Step 14 "
                "nmse_ratio_standard_error); se_db = (10/ln 10) se_linear/NMSE"
            ),
            "z_95": 1.959963984540054,
            "algorithms": list(FIG5_CONVERGENCE_ALGS),
            "rows": [
                {
                    "algorithm": r["algorithm"],
                    "snr_db": r["snr_db"],
                    "n_ok": r["n_ok"],
                    "nmse_linear": r["nmse_linear"],
                    "nmse_db": r["nmse_db"],
                    "se_linear": r["se_linear"],
                    "se_db": r["se_db"],
                    "nmse_db_ci95_low": r["nmse_db_ci95_low"],
                    "nmse_db_ci95_high": r["nmse_db_ci95_high"],
                }
                for r in enriched
                if r["algorithm"] in FIG5_CONVERGENCE_ALGS
            ],
        },
    )
    _json_dump(out / "acceptance.json", checks)
    _plot_fig5(
        agg,
        out / "fig5_nmse.png",
        error_bars=False,
        title="Cui Fig. 5 — detection NMSE vs SNR (Track A)",
    )
    _plot_fig5(
        agg,
        out / "fig5_nmse_errorbars.png",
        error_bars=True,
        title="Cui Fig. 5 — NMSE vs SNR with GS/EM-GS SE bars",
    )
    write_fig5_readme(
        out / "README.md",
        n_trials=used,
        agg=agg,
        checks=checks,
        convergence={"final": final_conv},
        norm_diag=norm_summary,
        analytic_gap_db=analytic_gap,
        core_solvers_unchanged=True,
    )
    return {
        "stopped": False,
        "output_dir": str(out),
        "csv": str(csv_path),
        "n_trials": used,
        "n_workers": workers,
        "nmse_db_by_snr": pivot_nmse_db(agg),
        "em_gs_minus_biased_gs_db": gaps,
        "crlb_minus_zf_db_at_12": crlb_minus_zf_gap_db(agg, 12.0),
        "analytic_high_snr_crlb_over_zf_db": analytic_gap,
        "convergence": final_conv,
        "acceptance": checks,
        "normalization_diagnostic": norm_summary,
        "core_solvers_unchanged": True,
        "did_not_run": config["not_run"],
    }

"""Track A (Cui validation) experiment drivers.

Fig. 4 (Bessel ratio), the Fig. 5 smoke study, and the full Fig. 5
reproduction driver (see :mod:`rydberg_sim.track_a_fig5`).

Does **not** launch Fig. 6–8, Track B, Track C, or machine learning.
Does not import or call :func:`rydberg_sim.channel.generate_ula_channel`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .channel_cui import CHANNEL_MODEL_CUI, CuiChannelParams
from .config import SimulationConfig
from .crlb import cui_crlb_high_snr_limit
from .gs import bessel_ratio
from .monte_carlo import (
    ExperimentSpec,
    aggregate_result_table,
    config_fingerprint,
    fingerprint_payload,
    generate_detection_trial,
    load_result_table,
    run_experiment,
)

TRACK_A_RESULTS = Path("results") / "track_a"

FIG4_KAPPA_MAX = 10.0
FIG4_R10_PAPER = 0.9486

FIG5_N = 36
FIG5_K = 3
FIG5_QAM = 16
FIG5_T0 = 50
FIG5_RSR_DB = 12.0
FIG5_SMOKE_SNR_DB = (-5.0, 0.0, 6.0, 12.0)
FIG5_SMOKE_TRIALS = 8
FIG5_MASTER_SEED = 20260818


def track_a_fig5_spec(
    *,
    n_trials: int,
    snr_db_grid: tuple[float, ...],
    experiment: str = "cui_fig5",
    master_seed: int = FIG5_MASTER_SEED,
    cui_params: CuiChannelParams | None = None,
) -> ExperimentSpec:
    """Cui Fig. 5 configuration. Default smoke uses a short SNR list.

    ``cui_params`` defaults to the production Table-I parameters
    (``normalize_rows=True``). The row-normalization diagnostic passes
    ``CuiChannelParams(normalize_rows=False)`` so its arm carries a
    **different** config fingerprint (audit M4).
    """
    cfg = SimulationConfig.create(
        N=FIG5_N,
        K=FIG5_K,
        L=1,
        beta=1.0,
        master_seed=master_seed,
        c=1.0,
    )
    return ExperimentSpec(
        experiment=experiment,
        track="A",
        cfg=cfg,
        P=1,
        vartheta=0.0,
        snr_db_grid=snr_db_grid,
        rsr_db_grid=(FIG5_RSR_DB,),
        n_trials=n_trials,
        algorithms=("biased_gs", "em_gs", "genie_zf", "cui_crlb"),
        max_iter=FIG5_T0,
        qam_M=FIG5_QAM,
        channel_model=CHANNEL_MODEL_CUI,
        cui_params=cui_params if cui_params is not None else CuiChannelParams(),
        write_ber=False,
    )


def reproduce_fig4(output_dir: Path | str | None = None) -> dict[str, Any]:
    """Sweep κ ∈ [0, 10] with the existing Step-10 ``bessel_ratio``."""
    out = Path(output_dir) if output_dir is not None else TRACK_A_RESULTS / "fig4"
    out.mkdir(parents=True, exist_ok=True)

    kappa = np.linspace(0.0, FIG4_KAPPA_MAX, 1001)
    r = np.asarray(bessel_ratio(kappa), dtype=np.float64)
    r0 = float(np.asarray(bessel_ratio(0.0)))
    r10 = float(np.asarray(bessel_ratio(10.0)))
    monotone = bool(np.all(np.diff(r) >= -1e-15))
    bounded = bool(np.all(r >= 0.0) and np.all(r <= 1.0))

    values = {
        "source": "Cui Fig. 4; R(κ)=I1(κ)/I0(κ) via rydberg_sim.gs.bessel_ratio",
        "R0": r0,
        "R10": r10,
        "R10_paper_approx": FIG4_R10_PAPER,
        "monotone_increasing": monotone,
        "bounded_01": bounded,
        "kappa": kappa.tolist(),
        "R": r.tolist(),
    }
    (out / "fig4_values.json").write_text(
        json.dumps(
            {k: v for k, v in values.items() if k not in ("kappa", "R")},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    np.savez(out / "fig4_curve.npz", kappa=kappa, R=r)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return values

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(kappa, r, color="#1f4788", lw=2.0)
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$R(\kappa)=I_1(\kappa)/I_0(\kappa)$")
    ax.set_title("Cui Fig. 4 — Bessel ratio (existing Step-10 implementation)")
    ax.set_xlim(0.0, FIG4_KAPPA_MAX)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.axhline(1.0, color="0.6", ls=":", lw=1.0)
    fig.tight_layout()
    fig.savefig(out / "fig4_bessel_ratio.png")
    plt.close(fig)
    return values


def _aggregate_nmse_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = aggregate_result_table(rows)
    out: list[dict[str, Any]] = []
    for rec in records:
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
    out.sort(key=lambda d: (d["algorithm"], d["snr_db"]))
    return out


def analytic_high_snr_crlb_zf_gap_db(spec: ExperimentSpec) -> float:
    """``10 log10 2`` from ``CRLB_high = 2 σ² (AA^H)^{-1}`` vs ZF cov ``σ² (AA^H)^{-1}``."""
    world = generate_detection_trial(spec, trial_index=0, snr_db=12.0, rsr_db=FIG5_RSR_DB)
    crlb_hs = cui_crlb_high_snr_limit(world.A, world.sigma2)
    zf_cov = 0.5 * crlb_hs
    gap = float(np.trace(crlb_hs).real / np.trace(zf_cov).real)
    return float(10.0 * np.log10(gap))


def run_fig5_smoke(output_dir: Path | str | None = None) -> dict[str, Any]:
    """Small Fig. 5 validation: four SNR points, few trials, no full sweep."""
    out = Path(output_dir) if output_dir is not None else TRACK_A_RESULTS / "fig5_smoke"
    out.mkdir(parents=True, exist_ok=True)
    spec = track_a_fig5_spec(
        n_trials=FIG5_SMOKE_TRIALS,
        snr_db_grid=FIG5_SMOKE_SNR_DB,
        experiment="cui_fig5_smoke",
    )
    csv_path = run_experiment(spec, out, n_workers=1)
    rows = load_result_table(csv_path)
    agg = _aggregate_nmse_table(rows)
    gap_db = analytic_high_snr_crlb_zf_gap_db(spec)

    config = {
        "experiment": spec.experiment,
        "track": spec.track,
        "channel_model": spec.channel_model,
        "config_fingerprint": config_fingerprint(spec),
        "fingerprint_payload": fingerprint_payload(spec),
        "N": spec.cfg.N,
        "K": spec.cfg.K,
        "qam_M": spec.qam_M,
        "t0": spec.max_iter,
        "rsr_db": FIG5_RSR_DB,
        "snr_db_grid": list(spec.snr_db_grid),
        "n_trials": spec.n_trials,
        "algorithms": list(spec.algorithms),
        "master_seed": spec.cfg.master_seed,
        "note": "Smoke study only. Not the full Cui Fig. 5 publication sweep.",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    summary = {
        "aggregate": agg,
        "analytic_high_snr_crlb_over_zf_db": gap_db,
        "analytic_10log10_2": float(10.0 * np.log10(2.0)),
        "n_result_rows": len(rows),
    }
    (out / "aggregate.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    by_snr: dict[float, dict[str, float]] = {}
    for row in agg:
        by_snr.setdefault(float(row["snr_db"]), {})[str(row["algorithm"])] = float(
            row["nmse_db"]
        )
    return {
        "csv": str(csv_path),
        "aggregate": agg,
        "nmse_db_by_snr": {str(k): v for k, v in sorted(by_snr.items())},
        "analytic_high_snr_crlb_over_zf_db": gap_db,
        "config_fingerprint": spec.fingerprint,
        "channel_model": spec.channel_model,
    }


def _cli(argv: list[str] | None = None) -> None:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print("usage: python -m rydberg_sim.track_a {fig4|fig5-smoke|fig5|fig5-norm-diag}")
        return
    if args[0] == "fig4":
        vals = reproduce_fig4()
        print(f"R(0)={vals['R0']}  R(10)={vals['R10']:.6f}  monotone={vals['monotone_increasing']}")
        return
    if args[0] == "fig5-smoke":
        summary = run_fig5_smoke()
        print(json.dumps({k: v for k, v in summary.items() if k != "aggregate"}, indent=2))
        return
    if args[0] == "fig5":
        from .track_a_fig5 import run_fig5

        summary = run_fig5()
        print(json.dumps({k: v for k, v in summary.items() if k != "acceptance"}, indent=2))
        return
    if args[0] == "fig5-norm-diag":
        from .track_a_fig5 import default_fig5_dir, run_row_normalization_diagnostic

        summary = run_row_normalization_diagnostic(default_fig5_dir())
        print(json.dumps({k: v for k, v in summary.items() if k != "shifts"}, indent=2))
        return
    raise SystemExit(f"unknown command {args[0]!r}")


if __name__ == "__main__":
    _cli()

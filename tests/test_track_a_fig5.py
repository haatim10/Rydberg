"""Fig. 5 driver tests. Do not run the full publication sweep here."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rydberg_sim.channel_cui import CuiChannelParams, generate_cui_channel
from rydberg_sim.monte_carlo import (
    config_fingerprint,
    fingerprint_payload,
    generate_detection_trial,
)
from rydberg_sim.rng import get_operating_point_rngs
from rydberg_sim.track_a import track_a_fig5_spec
from rydberg_sim.track_a_fig5 import (
    outlier_diagnostics,
    CONVERGENCE_CRITERION,
    FIG5_SNR_DB,
    _forbid_smoke_dir,
    checkpoint_deltas_db,
    completed_trial_count,
    convergence_satisfied,
    generate_unnormalized_detection_trial,
    rows_with_trial_prefix,
    run_fig5,
)


def test_fig5_snr_grid_is_integer_db() -> None:
    assert FIG5_SNR_DB[0] == -5.0
    assert FIG5_SNR_DB[-1] == 12.0
    assert FIG5_SNR_DB == tuple(float(s) for s in range(-5, 13))
    spec = track_a_fig5_spec(n_trials=1, snr_db_grid=FIG5_SNR_DB)
    assert spec.max_iter == 50
    assert spec.qam_M == 16
    assert spec.cfg.N == 36
    assert spec.cfg.K == 3
    assert "cm_zf" not in spec.algorithms
    assert spec.algorithms == ("biased_gs", "em_gs", "genie_zf", "cui_crlb")


def test_fig5_refuses_to_overwrite_smoke(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fig5_smoke"):
        _forbid_smoke_dir(tmp_path / "fig5_smoke")
    with pytest.raises(ValueError, match="fig5_smoke"):
        run_fig5(tmp_path / "results" / "track_a" / "fig5_smoke", skip_norm_diag=True)


def test_convergence_criterion_is_defined_a_priori() -> None:
    assert "0.1" in CONVERGENCE_CRITERION
    deltas = [
        {"algorithm": "biased_gs", "snr_db": 0.0, "abs_delta_db": 0.05, "delta_db": 0.05},
        {"algorithm": "em_gs", "snr_db": 0.0, "abs_delta_db": 0.09, "delta_db": -0.09},
    ]
    ok = convergence_satisfied(deltas, tol_db=0.1)
    assert ok["converged"] is True
    deltas[1]["abs_delta_db"] = 0.15
    bad = convergence_satisfied(deltas, tol_db=0.1)
    assert bad["converged"] is False
    assert bad["n_violations"] == 1


def test_checkpoint_deltas_and_prefix_filter() -> None:
    prev = [
        {"algorithm": "biased_gs", "snr_db": -5.0, "nmse_db": 1.0},
        {"algorithm": "em_gs", "snr_db": -5.0, "nmse_db": -1.0},
    ]
    cur = [
        {"algorithm": "biased_gs", "snr_db": -5.0, "nmse_db": 1.05},
        {"algorithm": "em_gs", "snr_db": -5.0, "nmse_db": -1.02},
    ]
    d = checkpoint_deltas_db(prev, cur)
    assert len(d) == 2
    assert d[0]["abs_delta_db"] == pytest.approx(0.05)
    rows = [
        {"trial": 0, "algorithm": "biased_gs"},
        {"trial": 99, "algorithm": "biased_gs"},
        {"trial": 100, "algorithm": "biased_gs"},
    ]
    assert [int(r["trial"]) for r in rows_with_trial_prefix(rows, 100)] == [0, 99]


def test_row_normalization_flag_is_scale_only() -> None:
    rng_a = get_operating_point_rngs(7, 3, -5.0, 12.0)
    rng_b = get_operating_point_rngs(7, 3, -5.0, 12.0)
    a = generate_cui_channel(8, 3, rng_a.channel, normalize_rows=True)
    b = generate_cui_channel(8, 3, rng_b.channel, normalize_rows=False)
    for k in range(3):
        scale = np.sqrt(float(np.mean(np.abs(b.A[k]) ** 2)))
        np.testing.assert_allclose(a.A[k] * scale, b.A[k], rtol=1e-12, atol=1e-12)
        assert float(np.mean(np.abs(a.A[k]) ** 2)) == pytest.approx(1.0, rel=1e-12)


def test_unnormalized_trial_recalibrates_snr_and_shares_symbols() -> None:
    spec = track_a_fig5_spec(n_trials=1, snr_db_grid=(-5.0,))
    wa = generate_detection_trial(spec, 0, -5.0, 12.0)
    wb = generate_unnormalized_detection_trial(spec, 0, -5.0, 12.0)
    np.testing.assert_array_equal(wa.s, wb.s)
    assert wa.sigma2 != pytest.approx(wb.sigma2)
    assert not np.allclose(wa.A, wb.A)
    for k in range(wa.A.shape[0]):
        assert float(np.mean(np.abs(wa.A[k]) ** 2)) == pytest.approx(1.0, rel=1e-12)
    raw_pow = float(np.mean(np.sum(np.abs(wb.A) ** 2, axis=0)))
    snr_lin = 10.0 ** (-5.0 / 10.0)
    assert wb.sigma2 == pytest.approx(raw_pow / snr_lin, rel=1e-12)


def test_completed_trial_count_requires_full_grid() -> None:
    spec = track_a_fig5_spec(n_trials=2, snr_db_grid=(-5.0, 0.0))
    rows = []
    for trial in (0, 1):
        for snr in (-5.0, 0.0):
            for alg in spec.algorithms:
                if trial == 1 and snr == 0.0 and alg == "cui_crlb":
                    continue
                rows.append(
                    {
                        "trial": trial,
                        "snr_db": snr,
                        "rsr_db": 12.0,
                        "algorithm": alg,
                        "metric": "detection_nmse",
                        "status": "ok",
                    }
                )
    assert completed_trial_count(rows, spec) == 1


def test_tiny_fig5_driver_writes_outputs_not_smoke(tmp_path: Path) -> None:
    out = tmp_path / "fig5"
    summary = run_fig5(
        out,
        n_workers=1,
        max_trials=1,
        initial_target=1,
        skip_norm_diag=True,
        checkpoints=(1,),
    )
    assert summary["stopped"] is False
    assert summary["n_trials"] == 1
    assert (out / "results.csv").is_file()
    assert (out / "aggregate.csv").is_file()
    assert (out / "aggregate.json").is_file()
    assert (out / "config.json").is_file()
    assert (out / "convergence.json").is_file()
    assert (out / "uncertainty.json").is_file()
    assert (out / "README.md").is_file()
    assert not (tmp_path / "fig5_smoke").exists()
    cfg_text = Path(out / "config.json").read_text(encoding="utf-8")
    assert "omitted" in cfg_text
    assert '"cm_zf"' not in Path(out / "aggregate.json").read_text(encoding="utf-8")


def test_tiny_norm_diag(tmp_path: Path) -> None:
    from rydberg_sim.track_a_fig5 import run_row_normalization_diagnostic

    summary = run_row_normalization_diagnostic(
        tmp_path, n_trials=2, snr_db_grid=(-5.0,)
    )
    assert summary["n_trials"] == 2
    assert summary["same_symbols_across_AB"] is True
    assert summary["production_unchanged"] is True
    assert (tmp_path / "row_normalization_diagnostic" / "summary.json").is_file()


# ---------------------------------------------------------------------------
# Row normalization is part of the experiment identity (audit M4)
# ---------------------------------------------------------------------------


def test_normalize_rows_is_in_the_config_fingerprint() -> None:
    """Normalized and unnormalized Track-A runs cannot share a fingerprint.

    Before this fix ``normalize_rows`` lived only as a keyword argument to
    ``generate_cui_channel``, outside ``CuiChannelParams``, so the most
    consequential Track-A modelling switch did not reach the fingerprint.
    Two runs differing only in normalization hashed identically and the
    resume/append guard would not have separated them.
    """
    prod = track_a_fig5_spec(n_trials=1, snr_db_grid=(0.0,))
    raw = track_a_fig5_spec(
        n_trials=1,
        snr_db_grid=(0.0,),
        cui_params=CuiChannelParams(normalize_rows=False),
    )
    assert prod.cui_params.normalize_rows is True
    assert raw.cui_params.normalize_rows is False
    assert "normalize_rows" in prod.cui_params.as_fingerprint_dict()
    assert fingerprint_payload(prod)["normalize_rows"] is True
    assert fingerprint_payload(raw)["normalize_rows"] is False
    assert config_fingerprint(prod) != config_fingerprint(raw)


def test_params_drive_normalization_without_the_keyword() -> None:
    """``params.normalize_rows`` alone selects the behaviour."""
    a = generate_cui_channel(
        8, 3, get_operating_point_rngs(7, 3, -5.0, 12.0).channel,
        params=CuiChannelParams(normalize_rows=True),
    )
    b = generate_cui_channel(
        8, 3, get_operating_point_rngs(7, 3, -5.0, 12.0).channel,
        params=CuiChannelParams(normalize_rows=False),
    )
    for k in range(3):
        assert float(np.mean(np.abs(a.A[k]) ** 2)) == pytest.approx(1.0, rel=1e-12)
        scale = np.sqrt(float(np.mean(np.abs(b.A[k]) ** 2)))
        np.testing.assert_allclose(a.A[k] * scale, b.A[k], rtol=1e-12, atol=1e-12)
    assert not np.allclose(a.A, b.A)


def test_unnormalized_diagnostic_trial_carries_the_raw_params() -> None:
    """The diagnostic arm's world really is built from unnormalized params."""
    spec = track_a_fig5_spec(
        n_trials=1,
        snr_db_grid=(-5.0,),
        experiment="norm_diag_B",
        cui_params=CuiChannelParams(normalize_rows=False),
    )
    world = generate_unnormalized_detection_trial(spec, 0, -5.0, 12.0)
    row_pow = np.mean(np.abs(np.asarray(world.A)) ** 2, axis=1)
    # Raw Table-I row power is ~200-250, nowhere near the normalized 1.0.
    assert np.all(row_pow > 10.0), row_pow


# ---------------------------------------------------------------------------
# Outlier / tail diagnostics (Fig. 5 section 8)
# ---------------------------------------------------------------------------


def _diag_row(alg: str, snr: float, trial: int, error: float, status: str = "ok"):
    return {
        "experiment": "diag", "config_fingerprint": "F", "track": "A",
        "trial": trial, "snr_db": snr, "rsr_db": 12.0, "N": 36, "K": 3, "P": 1,
        "modulation": "16-QAM", "algorithm": alg,
        "metric": "detection_nmse" if status == "ok" else "failure",
        "value": error / 3.0, "error_energy": error, "true_energy": None,
        "expected_symbol_energy": 3.0, "bit_errors": None, "bit_count": None,
        "status": status, "error_type": "", "error_message": "",
        "master_seed": 1, "sigma2": 3.0, "alpha_b_abs": 3.98, "max_iter": 50,
    }


def test_outlier_diagnostics_separates_aggregate_from_median() -> None:
    """One catastrophic trial must move the aggregate but not the median."""
    # Nine good trials at per-trial NMSE 0.1, one failure at 100.
    rows = [_diag_row("biased_gs", 0.0, t, 0.3) for t in range(9)]
    rows.append(_diag_row("biased_gs", 0.0, 9, 300.0))
    (rec,) = outlier_diagnostics(rows)

    assert rec["n_ok"] == 10
    assert rec["n_failed"] == 0
    # aggregate = sum(err)/sum(den) = (9*0.3 + 300)/30 = 10.09
    assert rec["nmse_linear"] == pytest.approx(302.7 / 30.0)
    # median per-trial NMSE is 0.1, untouched by the outlier
    assert rec["median_nmse_linear"] == pytest.approx(0.1)
    # ... so the aggregate sits ~20 dB above the median
    assert rec["aggregate_minus_median_db"] == pytest.approx(
        10.0 * np.log10((302.7 / 30.0) / 0.1)
    )
    # and the single worst trial carries ~99% of the error energy
    assert rec["top1pct_energy_share"] == pytest.approx(300.0 / 302.7)
    # one of ten trials is worse than the trivial s_hat = 0
    assert rec["worse_than_zero_rate"] == pytest.approx(0.1)
    assert rec["max_nmse_linear"] == pytest.approx(100.0)


def test_outlier_diagnostics_clean_case_has_no_tail_domination() -> None:
    """Identical trials: median == aggregate, energy share == the fraction."""
    rows = [_diag_row("em_gs", 6.0, t, 0.3) for t in range(100)]
    (rec,) = outlier_diagnostics(rows)
    assert rec["nmse_linear"] == pytest.approx(0.1)
    assert rec["median_nmse_linear"] == pytest.approx(0.1)
    assert rec["aggregate_minus_median_db"] == pytest.approx(0.0, abs=1e-12)
    assert rec["top1pct_energy_share"] == pytest.approx(0.01)
    assert rec["top5pct_energy_share"] == pytest.approx(0.05)
    assert rec["worse_than_zero_rate"] == 0.0
    for p in (10, 25, 50, 75, 90, 95, 99):
        assert rec[f"p{p}_nmse_linear"] == pytest.approx(0.1)


def test_outlier_diagnostics_counts_harness_failures() -> None:
    """status != 'ok' rows are counted as failures, not as NMSE samples."""
    rows = [_diag_row("biased_gs", 0.0, t, 0.3) for t in range(8)]
    rows += [_diag_row("biased_gs", 0.0, t, 0.0, status="failed") for t in (8, 9)]
    (rec,) = outlier_diagnostics(rows)
    assert rec["n_ok"] == 8
    assert rec["n_failed"] == 2
    assert rec["failure_rate"] == pytest.approx(0.2)
    assert rec["nmse_linear"] == pytest.approx(0.1)


def test_outlier_diagnostics_groups_by_algorithm_and_snr() -> None:
    rows = (
        [_diag_row("biased_gs", 0.0, t, 0.3) for t in range(4)]
        + [_diag_row("em_gs", 0.0, t, 0.6) for t in range(4)]
        + [_diag_row("biased_gs", 6.0, t, 0.15) for t in range(4)]
    )
    recs = outlier_diagnostics(rows)
    assert len(recs) == 3
    keyed = {(r["algorithm"], r["snr_db"]): r for r in recs}
    assert set(keyed) == {("biased_gs", 0.0), ("em_gs", 0.0), ("biased_gs", 6.0)}
    assert keyed[("biased_gs", 0.0)]["nmse_linear"] == pytest.approx(0.1)
    assert keyed[("em_gs", 0.0)]["nmse_linear"] == pytest.approx(0.2)
    assert keyed[("biased_gs", 6.0)]["nmse_linear"] == pytest.approx(0.05)

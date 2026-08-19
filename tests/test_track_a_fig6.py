"""Cui Fig. 6 driver tests (NMSE vs RSR at fixed SNR).

No mathematics is defined in ``track_a_fig6``; these cover the sweep
configuration, the RSR calibration measurement, the ZF-flatness fit, and
the reporting helpers.
"""

from __future__ import annotations

import numpy as np
import pytest

from rydberg_sim.channel_cui import CuiChannelParams
from rydberg_sim.monte_carlo import (
    config_fingerprint,
    generate_detection_trial,
)
from rydberg_sim.track_a import track_a_fig5_spec
from rydberg_sim.track_a_fig5 import FIG5_SNR_DB, checkpoint_deltas_db, outlier_diagnostics
from rydberg_sim.track_a_fig6 import (
    FIG6_CALIBRATION_TOL_DB,
    FIG6_RSR_DB,
    FIG6_SNR_DB,
    crlb_crossing_fig6,
    delta_summary,
    em_gs_minus_gs_db,
    measure_rsr_calibration,
    rsr_improvement_db,
    track_a_fig6_spec,
    zf_flatness,
)


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


def test_fig6_spec_sweeps_rsr_at_fixed_snr() -> None:
    spec = track_a_fig6_spec(n_trials=3)
    assert spec.snr_db_grid == (3.0,)
    assert spec.rsr_db_grid == FIG6_RSR_DB
    assert len(spec.rsr_db_grid) == 26
    assert spec.rsr_db_grid[0] == 0.0 and spec.rsr_db_grid[-1] == 25.0
    assert spec.track == "A"
    assert spec.channel_model == "cui_38901"
    assert spec.cfg.N == 36 and spec.cfg.K == 3
    assert spec.qam_M == 16
    assert spec.max_iter == 50
    assert spec.write_ber is False  # Fig. 6 is an NMSE figure
    assert set(spec.algorithms) == {"biased_gs", "em_gs", "genie_zf", "cui_crlb"}
    assert spec.cui_params.normalize_rows is True


def test_fig6_shares_fig5_fingerprint_but_not_experiment() -> None:
    """Same physical configuration, different sweep: same fingerprint by design.

    ``fingerprint_payload`` deliberately excludes the SNR/RSR grids so a run
    can be extended with more points. The two stores stay separate through
    ``experiment`` (part of the result key and checked on append) and
    through living in different directories.
    """
    s6 = track_a_fig6_spec(n_trials=1)
    s5 = track_a_fig5_spec(n_trials=1, snr_db_grid=FIG5_SNR_DB)
    assert config_fingerprint(s6) == config_fingerprint(s5)
    assert s6.experiment == "cui_fig6"
    assert s5.experiment == "cui_fig5"
    assert s6.experiment != s5.experiment


def test_fig6_rsr12_column_reproduces_fig5_snr3_column() -> None:
    """The operating-point RNG key is (trial, snr, rsr), so the shared point
    (SNR=3, RSR=12) is literally the same world in both figures."""
    s6 = track_a_fig6_spec(n_trials=2)
    s5 = track_a_fig5_spec(n_trials=2, snr_db_grid=FIG5_SNR_DB)
    for t in (0, 1):
        w6 = generate_detection_trial(s6, t, 3.0, 12.0)
        w5 = generate_detection_trial(s5, t, 3.0, 12.0)
        np.testing.assert_array_equal(w6.A, w5.A)
        np.testing.assert_array_equal(w6.s, w5.s)
        np.testing.assert_array_equal(w6.b, w5.b)
        np.testing.assert_array_equal(w6.w, w5.w)
        np.testing.assert_array_equal(w6.z, w5.z)
        assert w6.sigma2 == w5.sigma2


# ---------------------------------------------------------------------------
# RSR calibration
# ---------------------------------------------------------------------------


def test_rsr_calibration_hits_target() -> None:
    """Achieved RSR must match the target under Cui's single-user eq. 38."""
    spec = track_a_fig6_spec(n_trials=8)
    for rec in measure_rsr_calibration(spec, (0.0, 12.0, 25.0), n_trials=8):
        assert rec["within_tol"], rec
        assert abs(rec["error_db"]) <= FIG6_CALIBRATION_TOL_DB
        assert rec["mean_single_user_power"] == pytest.approx(1.0, rel=1e-9)


def test_rsr_calibration_scales_with_target() -> None:
    spec = track_a_fig6_spec(n_trials=4)
    recs = measure_rsr_calibration(spec, (0.0, 10.0), n_trials=4)
    lo, hi = recs[0]["mean_reference_power"], recs[1]["mean_reference_power"]
    assert hi / lo == pytest.approx(10.0, rel=1e-9)


# ---------------------------------------------------------------------------
# ZF flatness — the critical Fig. 6 sanity check
# ---------------------------------------------------------------------------


def _zf_rows(values_db, se_db=0.05):
    return [
        {
            "algorithm": "genie_zf",
            "rsr_db": float(r),
            "snr_db": 3.0,
            "nmse_db": float(v),
            "nmse_linear": 10 ** (v / 10.0),
            "se_linear": se_db * (10 ** (v / 10.0)) * np.log(10.0) / 10.0,
            "n_ok": 100,
            "n_failed": 0,
        }
        for r, v in enumerate(values_db)
    ]


def test_zf_flatness_detects_a_flat_curve() -> None:
    rng = np.random.default_rng(0)
    vals = -13.6 + 0.02 * rng.standard_normal(26)  # flat + small scatter
    res = zf_flatness(_zf_rows(vals))
    assert res["fitted"] is True
    assert res["n_points"] == 26
    assert abs(res["slope_db_per_db"]) < 0.01
    assert res["significant_slope"] is False
    assert res["range_db"] < 0.2


def test_zf_flatness_detects_a_real_slope() -> None:
    """A deliberate 0.1 dB/dB ramp must be flagged as significant."""
    vals = [-13.6 + 0.1 * r for r in range(26)]
    res = zf_flatness(_zf_rows(vals))
    assert res["slope_db_per_db"] == pytest.approx(0.1, rel=1e-6)
    assert res["significant_slope"] is True
    assert abs(res["t_stat_residual_scaled"]) > res["t_threshold"]
    assert res["range_db"] == pytest.approx(2.5, rel=1e-6)


def test_zf_flatness_needs_enough_points() -> None:
    assert zf_flatness(_zf_rows([-13.0, -13.1]))["fitted"] is False


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _agg(alg, pairs):
    return [
        {"algorithm": alg, "rsr_db": float(r), "snr_db": 3.0, "nmse_db": float(v),
         "nmse_linear": 10 ** (v / 10.0), "se_linear": 0.0, "n_ok": 10, "n_failed": 0}
        for r, v in pairs
    ]


def test_rsr_improvement_and_gap_helpers() -> None:
    agg = _agg("biased_gs", [(0, -4.0), (20, -9.0), (25, -9.5)]) + \
          _agg("em_gs", [(0, -5.0), (20, -9.4), (25, -9.8)])
    assert rsr_improvement_db(agg, "biased_gs", 0.0, 20.0) == pytest.approx(5.0)
    assert rsr_improvement_db(agg, "biased_gs", 0.0, 25.0) == pytest.approx(5.5)
    assert rsr_improvement_db(agg, "em_gs", 0.0, 20.0) == pytest.approx(4.4)
    # negative means EM-GS is better
    assert em_gs_minus_gs_db(agg, 0.0) == pytest.approx(-1.0)
    assert em_gs_minus_gs_db(agg, 20.0) == pytest.approx(-0.4)
    assert rsr_improvement_db(agg, "biased_gs", 0.0, 7.0) is None


def test_delta_summary_statistics() -> None:
    d = [{"abs_delta_db": v} for v in (0.01, 0.05, 0.09, 0.15, 0.30)]
    s = delta_summary(d)
    assert s["n_cells"] == 5
    assert s["max_abs_delta_db"] == pytest.approx(0.30)
    assert s["median_abs_delta_db"] == pytest.approx(0.09)
    assert s["n_over_tol"] == 2
    assert s["within_tol"] == 3
    assert delta_summary([])["n_cells"] == 0


def test_crlb_crossing_fig6_flags_only_significant_crossings() -> None:
    agg = (_agg("biased_gs", [(0, -4.0), (12, -9.0)])
           + _agg("em_gs", [(0, -4.5), (12, -9.2)])
           + _agg("cui_crlb", [(0, -5.0), (12, -9.1)]))
    res = crlb_crossing_fig6(agg)
    # em_gs at 12 dB is 0.1 dB BELOW the bound as a point estimate
    assert res["any_point_estimate_below_crlb"] is True
    # but se_linear = 0 here, so it registers as significant; check the record
    below = [p for p in res["per_rsr"] if p["point_below"]]
    assert {(p["algorithm"], p["rsr_db"]) for p in below} == {("em_gs", 12.0)}


# ---------------------------------------------------------------------------
# sweep_key generalization used by Fig. 6
# ---------------------------------------------------------------------------


def test_checkpoint_deltas_supports_rsr_sweep_key() -> None:
    prev = _agg("biased_gs", [(0, -4.0), (12, -9.0)])
    cur = _agg("biased_gs", [(0, -4.05), (12, -9.2)])
    d = checkpoint_deltas_db(prev, cur, ("biased_gs",), sweep_key="rsr_db")
    assert [x["rsr_db"] for x in d] == [0.0, 12.0]
    assert d[0]["delta_db"] == pytest.approx(-0.05)
    assert d[1]["abs_delta_db"] == pytest.approx(0.2)


def test_outlier_diagnostics_supports_rsr_sweep_key() -> None:
    rows = [
        {"algorithm": "em_gs", "rsr_db": r, "snr_db": 3.0, "trial": t,
         "metric": "detection_nmse", "status": "ok",
         "error_energy": 0.3, "expected_symbol_energy": 3.0}
        for r in (0.0, 12.0) for t in range(5)
    ]
    recs = outlier_diagnostics(rows, sweep_key="rsr_db")
    assert len(recs) == 2
    assert {r["rsr_db"] for r in recs} == {0.0, 12.0}
    assert all(r["nmse_linear"] == pytest.approx(0.1) for r in recs)

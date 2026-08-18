"""Fig. 5 driver tests. Do not run the full publication sweep here."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rydberg_sim.channel_cui import generate_cui_channel
from rydberg_sim.monte_carlo import generate_detection_trial
from rydberg_sim.rng import get_operating_point_rngs
from rydberg_sim.track_a import track_a_fig5_spec
from rydberg_sim.track_a_fig5 import (
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

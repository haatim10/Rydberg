"""Cui Fig. 7 / Fig. 8 BER driver tests.

No mathematics is defined in ``track_a_fig78``; these cover the sweep
configuration against the paper, the BER pooling rule, and the reporting
helpers.
"""

from __future__ import annotations

import numpy as np
import pytest

from rydberg_sim.metrics import detection_ber
from rydberg_sim.monte_carlo import (
    DETECTION_ALGORITHMS,
    config_fingerprint,
    evaluate_detection_algorithm,
    generate_detection_trial,
)
from rydberg_sim.qam import generate_qam
from rydberg_sim.track_a_fig78 import (
    FIG7_RSR_DB,
    FIG7_SNR_DB,
    FIG8_QAM,
    FIG8_QAM_CAPTION_CLAIM,
    FIG8_RSR_DB,
    FIG8_SNR_DB,
    aggregate_ber,
    ber_snr_gap_db,
    track_a_fig7a_spec,
    track_a_fig7b_spec,
    track_a_fig8_spec,
)


# ---------------------------------------------------------------------------
# Specs match the paper
# ---------------------------------------------------------------------------


def test_fig7a_matches_the_paper() -> None:
    """§VI-C: 'N x K = 36 x 3 with a 4-QAM modulator', RSR = 12 dB."""
    s = track_a_fig7a_spec(n_trials=3)
    assert (s.cfg.N, s.cfg.K) == (36, 3)
    assert s.qam_M == 4
    assert s.rsr_db_grid == (12.0,)
    assert s.snr_db_grid == FIG7_SNR_DB
    assert s.snr_db_grid[0] == -5.0 and s.snr_db_grid[-1] == 12.0
    assert s.max_iter == 50           # §VI-A: "t0 is set as 50"
    assert s.write_ber is True
    assert s.track == "A" and s.channel_model == "cui_38901"


def test_fig7b_matches_the_paper() -> None:
    """§VI-C: 'N x K = 100 x 6 with a 16-QAM modulator'."""
    s = track_a_fig7b_spec(n_trials=3)
    assert (s.cfg.N, s.cfg.K) == (100, 6)
    assert s.qam_M == 16
    assert s.rsr_db_grid == (12.0,)
    assert s.snr_db_grid == FIG7_SNR_DB


def test_fig7b_excludes_exhaustive_search() -> None:
    """Cui: 'the computation of exhaustive search method is prohibitive'."""
    algs = set(track_a_fig7b_spec(n_trials=1).algorithms)
    assert "exhaustive_ls" not in algs and "exhaustive_ml" not in algs
    assert algs == {"biased_gs", "em_gs", "genie_zf"}


def test_fig7a_includes_both_exhaustive_searches() -> None:
    algs = set(track_a_fig7a_spec(n_trials=1).algorithms)
    assert {"exhaustive_ls", "exhaustive_ml"} <= algs


def test_cm_zf_is_never_included() -> None:
    """CM-ZF is not specified well enough to implement; never invent it."""
    for build in (track_a_fig7a_spec, track_a_fig7b_spec, track_a_fig8_spec):
        assert not any("cm" in a.lower() for a in build(n_trials=1).algorithms)


def test_fig8_matches_the_body_text_not_the_caption() -> None:
    """Caption says 16-QAM, body text says 4-QAM; the BER levels back the text."""
    s = track_a_fig8_spec(n_trials=3)
    assert s.qam_M == FIG8_QAM == 4
    assert FIG8_QAM_CAPTION_CLAIM == 16
    assert s.snr_db_grid == (FIG8_SNR_DB,) == (3.0,)
    assert s.rsr_db_grid == FIG8_RSR_DB
    assert s.rsr_db_grid[0] == 0.0 and s.rsr_db_grid[-1] == 25.0
    assert (s.cfg.N, s.cfg.K) == (36, 3)


def test_fig8_caption_variant_is_reachable() -> None:
    s = track_a_fig8_spec(n_trials=1, qam_M=FIG8_QAM_CAPTION_CLAIM)
    assert s.qam_M == 16
    assert config_fingerprint(s) != config_fingerprint(track_a_fig8_spec(n_trials=1))


def test_each_figure_has_its_own_experiment_name() -> None:
    names = {
        track_a_fig7a_spec(n_trials=1).experiment,
        track_a_fig7b_spec(n_trials=1).experiment,
        track_a_fig8_spec(n_trials=1).experiment,
    }
    assert len(names) == 3


def test_fig5_fig6_stores_are_not_disturbed() -> None:
    """The BER specs must not collide with the canonical NMSE stores."""
    from rydberg_sim.track_a import track_a_fig5_spec
    from rydberg_sim.track_a_fig5 import FIG5_SNR_DB

    fig5 = track_a_fig5_spec(n_trials=1, snr_db_grid=FIG5_SNR_DB)
    assert fig5.write_ber is False
    assert config_fingerprint(fig5) == (
        "925f2ab8975505e255f5c37af74d115be38bc302991267f95f440e5f61ce4472"
    )
    # Fig. 7(a)/8 differ from Fig. 5 in N, K or qam_M, so they cannot share it
    for build in (track_a_fig7a_spec, track_a_fig7b_spec, track_a_fig8_spec):
        assert config_fingerprint(build(n_trials=1)) != config_fingerprint(fig5)


# ---------------------------------------------------------------------------
# Exhaustive search wiring
# ---------------------------------------------------------------------------


def test_exhaustive_algorithms_are_registered() -> None:
    assert {"exhaustive_ls", "exhaustive_ml"} <= DETECTION_ALGORITHMS


def test_exhaustive_output_lies_on_the_constellation() -> None:
    spec = track_a_fig7a_spec(n_trials=2, snr_db_grid=(3.0,))
    world = generate_detection_trial(spec, 0, 3.0, 12.0)
    from rydberg_sim.qam import QAMConstellation

    pts = QAMConstellation.create(4).points if hasattr(QAMConstellation, "create") else None
    for alg in ("exhaustive_ls", "exhaustive_ml"):
        rows, _ = evaluate_detection_algorithm(world, alg, spec)
        ber = [r for r in rows if r["metric"] == "ber"]
        assert ber and ber[0]["bit_count"] == spec.cfg.K * 2  # log2(4) = 2


def test_all_algorithms_see_the_same_world() -> None:
    """Common random numbers: one world, every algorithm."""
    spec = track_a_fig7a_spec(n_trials=2, snr_db_grid=(3.0,))
    a = generate_detection_trial(spec, 1, 3.0, 12.0)
    b = generate_detection_trial(spec, 1, 3.0, 12.0)
    for attr in ("A", "s", "b", "w", "z"):
        np.testing.assert_array_equal(getattr(a, attr), getattr(b, attr))


# ---------------------------------------------------------------------------
# BER aggregation
# ---------------------------------------------------------------------------


def _ber_rows(alg, pairs, sweep_key="snr_db"):
    return [
        {"algorithm": alg, sweep_key: float(x), "metric": "ber", "status": "ok",
         "bit_errors": int(e), "bit_count": int(n)}
        for x, e, n in pairs
    ]


def test_ber_is_a_ratio_of_sums() -> None:
    """Not the mean of per-trial BERs."""
    rows = _ber_rows("em_gs", [(0.0, 1, 10), (0.0, 9, 90)])
    agg = aggregate_ber(rows)
    assert len(agg) == 1
    assert agg[0]["bit_errors"] == 10 and agg[0]["bit_count"] == 100
    assert agg[0]["ber"] == pytest.approx(0.1)      # 10/100, not mean(0.1, 0.1)
    # a case where the two differ
    rows2 = _ber_rows("em_gs", [(0.0, 1, 10), (0.0, 0, 990)])
    assert aggregate_ber(rows2)[0]["ber"] == pytest.approx(1 / 1000)


def test_zero_ber_is_flagged_not_nan() -> None:
    agg = aggregate_ber(_ber_rows("em_gs", [(12.0, 0, 5000)]))
    assert agg[0]["ber"] == 0.0
    assert agg[0]["ber_is_zero"] is True
    assert agg[0]["ber_ci95_low"] == 0.0
    assert agg[0]["ber_ci95_high"] > 0.0        # Wilson upper bound is informative


def test_wilson_interval_brackets_the_estimate() -> None:
    agg = aggregate_ber(_ber_rows("em_gs", [(0.0, 50, 10000)]))
    r = agg[0]
    assert r["ber_ci95_low"] < r["ber"] < r["ber_ci95_high"]


def test_non_ber_rows_are_ignored() -> None:
    rows = _ber_rows("em_gs", [(0.0, 5, 100)])
    rows.append({"algorithm": "em_gs", "snr_db": 0.0, "metric": "detection_nmse",
                 "status": "ok", "bit_errors": 999, "bit_count": 999})
    assert aggregate_ber(rows)[0]["bit_errors"] == 5


def test_failed_rows_are_ignored() -> None:
    rows = _ber_rows("em_gs", [(0.0, 5, 100)])
    rows.append({"algorithm": "em_gs", "snr_db": 0.0, "metric": "ber",
                 "status": "failed", "bit_errors": 50, "bit_count": 100})
    assert aggregate_ber(rows)[0]["bit_errors"] == 5


def test_aggregate_supports_the_rsr_sweep_key() -> None:
    agg = aggregate_ber(_ber_rows("em_gs", [(0.0, 5, 100), (25.0, 1, 100)],
                                  sweep_key="rsr_db"), sweep_key="rsr_db")
    assert [r["rsr_db"] for r in agg] == [0.0, 25.0]


# ---------------------------------------------------------------------------
# SNR-gap helper (Cui reports 3~4 dB for Fig. 7(b))
# ---------------------------------------------------------------------------


def test_ber_snr_gap_recovers_a_known_offset() -> None:
    """Two curves offset by exactly 4 dB must report a 4 dB gap."""
    xs = np.arange(-5.0, 13.0)
    ber = 10.0 ** (-1.0 - 0.25 * (xs + 5.0))
    rows = []
    for x, b in zip(xs, ber):
        rows += _ber_rows("em_gs", [(x, int(b * 1e7), 10_000_000)])
    for x, b in zip(xs, ber):
        rows += _ber_rows("genie_zf", [(x - 4.0, int(b * 1e7), 10_000_000)])
    agg = aggregate_ber(rows)
    gap = ber_snr_gap_db(agg, "em_gs", "genie_zf", 1e-3)
    assert gap == pytest.approx(4.0, abs=0.15)


def test_ber_snr_gap_returns_none_when_uncrossed() -> None:
    agg = aggregate_ber(_ber_rows("em_gs", [(0.0, 10, 100), (1.0, 9, 100)]))
    assert ber_snr_gap_db(agg, "em_gs", "genie_zf", 1e-9) is None

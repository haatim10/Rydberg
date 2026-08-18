"""Step 14 confidence-interval helpers (Wilson BER, NMSE SE)."""

from __future__ import annotations

import numpy as np
import pytest

from rydberg_sim.confidence import (
    WILSON_Z_95,
    nmse_ratio_standard_error,
    rule_of_three,
    wilson_interval,
)


def test_wilson_known_small_example() -> None:
    """n=10, k=1, z=1.96 — hand-computed Wilson endpoints."""
    z = 1.96
    interval = wilson_interval(1, 10, z=z)
    p = 0.1
    z2 = z * z
    denom = 1.0 + z2 / 10.0
    center = (p + z2 / (2.0 * 10.0)) / denom
    rad = z * np.sqrt((p * (1.0 - p) + z2 / (4.0 * 10.0)) / 10.0) / denom
    assert interval.ber == pytest.approx(0.1)
    assert interval.low == pytest.approx(max(0.0, center - rad))
    assert interval.high == pytest.approx(min(1.0, center + rad))
    assert interval.rule_of_three is None
    assert 0.0 <= interval.low < interval.ber < interval.high <= 1.0


def test_wilson_default_z_is_95_percent() -> None:
    interval = wilson_interval(3, 20)
    assert interval.z == WILSON_Z_95


def test_wilson_zero_errors_not_treated_as_known_zero() -> None:
    interval = wilson_interval(0, 100)
    assert interval.ber == 0.0
    assert interval.low == 0.0
    assert interval.high > 0.0
    assert interval.rule_of_three == pytest.approx(0.03)
    assert interval.high < 1.0


def test_rule_of_three() -> None:
    assert rule_of_three(300) == pytest.approx(0.01)


def test_wilson_rejects_k_greater_than_n() -> None:
    with pytest.raises(ValueError, match="bit_errors"):
        wilson_interval(5, 4)


def test_nmse_ratio_standard_error_two_equal_energy_trials() -> None:
    """e=[1,3], t=[10,10] → R=0.2, SE=0.1 by the documented delta method."""
    unc = nmse_ratio_standard_error([1.0, 3.0], [10.0, 10.0])
    assert unc.n_trials == 2
    assert unc.nmse_linear == pytest.approx(0.2)
    assert unc.total_error_energy == pytest.approx(4.0)
    assert unc.total_true_energy == pytest.approx(20.0)
    assert unc.se_linear == pytest.approx(0.1)


def test_nmse_ratio_single_trial_se_is_nan() -> None:
    unc = nmse_ratio_standard_error([2.0], [8.0])
    assert unc.nmse_linear == pytest.approx(0.25)
    assert np.isnan(unc.se_linear)


def test_nmse_ratio_matches_ratio_of_sums_not_mean_of_ratios() -> None:
    errors = np.array([1.0, 1.0, 8.0])
    energies = np.array([1.0, 1.0, 100.0])
    unc = nmse_ratio_standard_error(errors, energies)
    assert unc.nmse_linear == pytest.approx(10.0 / 102.0)
    mean_of_ratios = float(np.mean(errors / energies))
    assert unc.nmse_linear != pytest.approx(mean_of_ratios)

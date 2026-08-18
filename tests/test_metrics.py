"""Step 13 acceptance tests: deterministic NMSE / BER metrics."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from rydberg_sim import (
    BerAccumulator,
    NmseAccumulator,
    build_qam_constellation,
    channel_nmse,
    decoded_bits,
    detection_ber,
    detection_nmse,
    expected_channel_frobenius_energy,
    generate_qam,
    nmse_to_db,
    project_to_qam,
    qam_to_bits,
)
from rydberg_sim.metrics import phase_align_channel_rows
from rydberg_sim.qam import nearest_qam_indices


def test_detection_nmse_exact_match_is_zero_and_minus_inf_db() -> None:
    s = np.array([1.0 + 0.0j, -1.0 / np.sqrt(2) + 1.0j / np.sqrt(2)])
    result = detection_nmse(s, s)
    assert result.error_energy == pytest.approx(0.0, abs=1e-15)
    assert result.nmse_linear == pytest.approx(0.0, abs=1e-15)
    assert result.nmse_db == -np.inf
    assert result.expected_energy == pytest.approx(2.0)


def test_detection_nmse_hand_calculation() -> None:
    s_true = np.array([1.0 + 0.0j, 0.0 + 0.0j])
    s_hat = np.array([1.0 + 0.0j, 1.0 + 0.0j])
    result = detection_nmse(s_hat, s_true)
    assert result.error_energy == pytest.approx(1.0)
    assert result.expected_energy == pytest.approx(2.0)
    assert result.realization_energy == pytest.approx(1.0)
    assert result.nmse_linear == pytest.approx(0.5)
    assert result.nmse_linear != pytest.approx(
        result.error_energy / result.realization_energy
    )


def test_detection_nmse_denominator_is_k_not_realization_norm() -> None:
    rng = np.random.default_rng(3)
    s_true = (rng.standard_normal(5) + 1j * rng.standard_normal(5)).astype(np.complex128)
    s_hat = s_true + 0.2
    result = detection_nmse(s_hat, s_true)
    assert result.expected_energy == pytest.approx(5.0)
    override = detection_nmse(s_hat, s_true, expected_symbol_energy=10.0)
    assert override.expected_energy == pytest.approx(10.0)
    assert override.nmse_linear == pytest.approx(result.error_energy / 10.0)


def test_nmse_db_is_10_log10_not_20() -> None:
    db = nmse_to_db(0.1)
    assert db == pytest.approx(10.0 * np.log10(0.1))
    assert db == pytest.approx(-10.0)
    assert db != pytest.approx(20.0 * np.log10(0.1))
    result = detection_nmse(np.array([1.0, 0.0]), np.array([0.0, 0.0]))
    assert result.nmse_db == pytest.approx(10.0 * np.log10(0.5))
    assert result.nmse_db != pytest.approx(20.0 * np.log10(0.5))


def test_detection_rejects_bad_shapes_and_nan() -> None:
    with pytest.raises(ValueError, match="1-D"):
        detection_nmse(np.ones((2, 2)), np.ones((2, 2)))
    with pytest.raises(ValueError, match="shapes"):
        detection_nmse(np.ones(3), np.ones(2))
    with pytest.raises(ValueError, match="finite"):
        detection_nmse(np.array([np.nan + 0j]), np.array([1.0 + 0j]))


def test_channel_nmse_exact_match_is_zero() -> None:
    G = np.array([[1.0 + 1.0j, 0.5], [-0.2j, 2.0]], dtype=np.complex128)
    result = channel_nmse(G, G)
    assert result.error_energy == pytest.approx(0.0, abs=1e-15)
    assert result.nmse_linear == pytest.approx(0.0, abs=1e-15)
    assert result.instantaneous_nmse == pytest.approx(0.0, abs=1e-15)
    assert result.phase_aligned_nmse_linear == pytest.approx(0.0, abs=1e-15)
    assert result.nmse_db == -np.inf
    assert result.likely_phase_anchor_problem is False


def test_channel_nmse_hand_frobenius() -> None:
    G = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    G_hat = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.complex128)
    result = channel_nmse(G_hat, G)
    assert result.error_energy == pytest.approx(1.0)
    assert result.true_energy == pytest.approx(2.0)
    assert result.instantaneous_nmse == pytest.approx(0.5)
    energy = expected_channel_frobenius_energy(2, (1.0, 1.0), c=1.0)
    result_exp = channel_nmse(G_hat, G, expected_channel_energy=energy)
    assert energy == pytest.approx(4.0)
    assert result_exp.nmse_linear == pytest.approx(1.0 / 4.0)
    assert result_exp.instantaneous_nmse == pytest.approx(0.5)


def test_expected_channel_energy_helper() -> None:
    assert expected_channel_frobenius_energy(4, (1.0, 1.0, 1.0), c=1.0) == pytest.approx(
        12.0
    )
    assert expected_channel_frobenius_energy(8, 1.0, c=1.0) == pytest.approx(8.0)
    assert expected_channel_frobenius_energy(3, (0.5, 1.5), c=2.0) == pytest.approx(
        4.0 * 3 * 2.0
    )


def test_channel_rejects_shape_nan() -> None:
    G = np.ones((2, 3), dtype=np.complex128)
    with pytest.raises(ValueError, match="must equal"):
        channel_nmse(np.ones((2, 2), dtype=np.complex128), G)
    with pytest.raises(ValueError, match="finite"):
        bad = G.copy()
        bad[0, 0] = np.inf
        channel_nmse(bad, G)


@pytest.mark.parametrize("M", [4, 16])
def test_qam_exact_points_demap_to_original_gray_bits(M: int) -> None:
    rng = np.random.default_rng(11 + M)
    draw = generate_qam(rng, 64, M)
    bits_exact = qam_to_bits(draw.symbols, M)
    np.testing.assert_array_equal(bits_exact, draw.bits)
    ber = detection_ber(draw.symbols, draw.bits, M)
    assert ber.bit_errors == 0
    assert ber.bit_count == 64 * int(np.log2(M))
    assert ber.ber == pytest.approx(0.0)
    np.testing.assert_array_equal(decoded_bits(draw.symbols, M), draw.bits)
    projected = project_to_qam(draw.symbols, M)
    np.testing.assert_allclose(projected, draw.symbols, rtol=0.0, atol=1e-15)


def test_ber_controlled_symbol_decision() -> None:
    const = build_qam_constellation(4)
    bits_true = const.bit_labels[[0]]
    s_true = const.points[0]
    d = np.abs(const.points - s_true)
    d[0] = np.inf
    idx_nn = int(np.argmin(d))
    hamming = int(np.sum(const.bit_labels[0] != const.bit_labels[idx_nn]))
    assert hamming == 1
    s_tilde = const.points[idx_nn]
    result = detection_ber(np.array([s_tilde]), bits_true, const)
    assert result.bit_errors == 1
    assert result.bit_count == 2
    assert result.ber == pytest.approx(0.5)
    assert result.symbol_errors == 1
    assert result.symbol_count == 1
    assert result.ser == pytest.approx(1.0)


def test_global_ber_is_not_mean_of_trial_bers() -> None:
    const = build_qam_constellation(4)
    bits1 = const.bit_labels[[0]]
    d = np.abs(const.points - const.points[0])
    d[0] = np.inf
    wrong = const.points[int(np.argmin(d))]
    r1 = detection_ber(np.array([wrong]), bits1, const)
    bits2 = const.bit_labels[[1, 2, 3]]
    r2 = detection_ber(const.points[[1, 2, 3]], bits2, const)
    acc = BerAccumulator()
    acc.add_result(r1)
    acc.add_result(r2)
    mean_ber = 0.5 * (r1.ber + r2.ber)
    global_ber = (r1.bit_errors + r2.bit_errors) / (r1.bit_count + r2.bit_count)
    print(
        f"\nBER accumulation: trial BERs={r1.ber}, {r2.ber}; "
        f"mean={mean_ber:.4f}; global={acc.ber:.4f} "
        f"({acc.total_bit_errors}/{acc.total_bit_count})"
    )
    assert acc.total_bit_errors == r1.bit_errors + r2.bit_errors
    assert acc.total_bit_count == r1.bit_count + r2.bit_count
    assert acc.ber == pytest.approx(global_ber)
    assert acc.ber != pytest.approx(mean_ber)
    assert acc.ber == pytest.approx(1.0 / 8.0)


def test_project_to_qam_is_shared_helper() -> None:
    import rydberg_sim.gs as gmod
    import rydberg_sim.qam as qmod

    assert gmod.project_to_qam is qmod.project_to_qam
    const = build_qam_constellation(16)
    noisy = const.points[3] + (0.01 + 0.02j)
    proj = project_to_qam(np.array([noisy]), const)
    np.testing.assert_allclose(proj[0], const.points[3], rtol=0.0, atol=1e-15)
    assert int(nearest_qam_indices(np.array([noisy]), const)[0]) == 3


def test_phase_alignment_collapses_row_phases() -> None:
    rng = np.random.default_rng(21)
    G = (rng.standard_normal((4, 3)) + 1j * rng.standard_normal((4, 3))).astype(
        np.complex128
    )
    phi = np.array([0.3, -1.1, 2.0, 0.7])
    G_hat = np.empty_like(G)
    for n, p in enumerate(phi):
        G_hat[n] = np.exp(1j * p) * G[n]
    G_hat_orig = G_hat.copy()
    result = channel_nmse(G_hat, G)
    assert result.nmse_linear > 0.1
    assert result.phase_aligned_nmse_linear == pytest.approx(0.0, abs=1e-12)
    assert result.likely_phase_anchor_problem is True
    np.testing.assert_allclose(G_hat, G_hat_orig, rtol=0.0, atol=0.0)
    aligned, _ = phase_align_channel_rows(G_hat, G)
    np.testing.assert_allclose(aligned, G, rtol=0.0, atol=1e-12)
    print(
        f"\nphase diagnostic: raw NMSE={result.nmse_linear:.4f}  "
        f"aligned={result.phase_aligned_nmse_linear:.3g}  "
        f"flag={result.likely_phase_anchor_problem}"
    )


def test_phase_align_zero_when_equal_and_does_not_mutate() -> None:
    G = np.array([[1.0 + 2.0j, -0.5], [0.3j, 4.0]], dtype=np.complex128)
    G_hat = G.copy()
    result = channel_nmse(G_hat, G)
    assert result.nmse_linear == pytest.approx(0.0, abs=1e-15)
    assert result.phase_aligned_nmse_linear == pytest.approx(0.0, abs=1e-15)
    np.testing.assert_array_equal(G_hat, G)


def test_nmse_accumulator_sums_linear_energies() -> None:
    acc = NmseAccumulator()
    acc.add(1.0, 4.0)
    acc.add(3.0, 4.0)
    assert acc.total_error_energy == pytest.approx(4.0)
    assert acc.total_true_energy == pytest.approx(8.0)
    assert acc.nmse_linear == pytest.approx(0.5)
    assert acc.nmse_db == pytest.approx(10.0 * np.log10(0.5))
    d1 = detection_nmse(np.array([1.0, 1.0]), np.array([0.0, 0.0]))
    d2 = detection_nmse(np.zeros(2), np.zeros(2) + 1)
    acc2 = NmseAccumulator()
    acc2.add_detection(d1)
    acc2.add_detection(d2)
    combined = (d1.error_energy + d2.error_energy) / (
        d1.expected_energy + d2.expected_energy
    )
    assert acc2.nmse_linear == pytest.approx(combined)


def test_step14_plus_not_implemented() -> None:
    import rydberg_sim.metrics as mmod

    assert hasattr(mmod, "detection_nmse")
    assert hasattr(mmod, "channel_nmse")
    assert hasattr(mmod, "detection_ber")
    assert not hasattr(mmod, "monte_carlo_harness")
    assert not hasattr(mmod, "run_trials")
    assert not hasattr(mmod, "confidence_interval")
    assert not hasattr(mmod, "plot_nmse")
    src = inspect.getsource(mmod)
    assert "Step 14+" in src
    assert "Never ``20 log10``" in src

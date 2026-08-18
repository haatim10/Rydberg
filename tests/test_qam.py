"""Step 4 Part B: Gray-mapped unit-energy square QAM."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from rydberg_sim import (
    bits_to_qam,
    build_qam_constellation,
    generate_qam,
    get_trial_rngs,
    qam_to_bits,
)

N_MC_SYMBOLS = 1_000_000


@pytest.mark.parametrize("M", [4, 16])
def test_constellation_size(M: int) -> None:
    const = build_qam_constellation(M)
    assert const.M == M
    assert const.points.shape == (M,)
    assert const.points.dtype == np.complex128
    unique = np.unique(np.round(const.points, decimals=12))
    assert unique.size == M


@pytest.mark.parametrize("M", [4, 16])
def test_exact_constellation_energy(M: int) -> None:
    const = build_qam_constellation(M)
    mean_energy = float(np.mean(np.abs(const.points) ** 2))
    np.testing.assert_allclose(mean_energy, 1.0, rtol=0.0, atol=1e-15)


@pytest.mark.parametrize("M", [4, 16])
def test_monte_carlo_symbol_energy(M: int) -> None:
    """mean(|x|^2) within 0.5% of 1 over 1e6 random symbols."""
    rng = np.random.default_rng(123)
    draw = generate_qam(rng, N_MC_SYMBOLS, M)
    mean_energy = float(np.mean(np.abs(draw.symbols) ** 2))
    np.testing.assert_allclose(mean_energy, 1.0, rtol=0.005, atol=0.0)


@pytest.mark.parametrize("M", [4, 16])
def test_bit_round_trip(M: int) -> None:
    rng = np.random.default_rng(7)
    draw = generate_qam(rng, 5000, M)
    symbols = bits_to_qam(draw.bits, M)
    np.testing.assert_array_equal(symbols, draw.symbols)
    bits_from_symbols = qam_to_bits(symbols, M)
    np.testing.assert_array_equal(bits_from_symbols, draw.bits)
    bits_from_indices = qam_to_bits(draw.indices, M)
    np.testing.assert_array_equal(bits_from_indices, draw.bits)


def test_4qam_gray_nearest_neighbors() -> None:
    const = build_qam_constellation(4)
    pts = const.points
    dist = np.abs(pts[:, None] - pts[None, :])
    np.fill_diagonal(dist, np.inf)
    nn_dist = dist.min()
    for i in range(4):
        neighbors = np.flatnonzero(np.abs(dist[i] - nn_dist) <= 1e-12)
        assert neighbors.size == 2  # square: two edge neighbors, not the diagonal
        for j in neighbors:
            hamming = int(np.sum(const.bit_labels[i] != const.bit_labels[j]))
            assert hamming == 1, (i, j, const.bit_labels[i], const.bit_labels[j])


def test_16qam_gray_axis_neighbors() -> None:
    const = build_qam_constellation(16)
    scale = const.scale
    unnorm = const.points / scale
    i_levels = np.real(unnorm)
    q_levels = np.imag(unnorm)
    axis = np.array([-3.0, -1.0, 1.0, 3.0])

    def hamming(i: int, j: int) -> int:
        return int(np.sum(const.bit_labels[i] != const.bit_labels[j]))

    # Horizontal neighbors: same Q, adjacent I.
    for q in axis:
        idx = np.flatnonzero(np.abs(q_levels - q) < 1e-12)
        order = idx[np.argsort(i_levels[idx])]
        for a, b in itertools.pairwise(order):
            assert hamming(int(a), int(b)) == 1

    # Vertical neighbors: same I, adjacent Q.
    for i in axis:
        idx = np.flatnonzero(np.abs(i_levels - i) < 1e-12)
        order = idx[np.argsort(q_levels[idx])]
        for a, b in itertools.pairwise(order):
            assert hamming(int(a), int(b)) == 1


def test_4qam_normalization() -> None:
    const = build_qam_constellation(4)
    expected_unnorm = np.array(
        [-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j], dtype=np.complex128
    )
    got = np.sort_complex(const.points * np.sqrt(2.0))
    want = np.sort_complex(expected_unnorm)
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(const.scale, 1.0 / np.sqrt(2.0), rtol=0.0, atol=1e-15)


def test_16qam_normalization() -> None:
    const = build_qam_constellation(16)
    levels = np.array([-3.0, -1.0, 1.0, 3.0])
    expected = np.array(
        [i + 1j * q for i in levels for q in levels], dtype=np.complex128
    )
    got = np.sort_complex(const.points * np.sqrt(10.0))
    want = np.sort_complex(expected)
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(const.scale, 1.0 / np.sqrt(10.0), rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(const.axis_levels, levels)


def test_documented_gray_axis_labels() -> None:
    const4 = build_qam_constellation(4)
    # I bit is MSB, Q bit is LSB; 0 -> -1, 1 -> +1.
    # index 0 = 00 -> -1-j, 1 = 01 -> -1+j, 2 = 10 -> +1-j, 3 = 11 -> +1+j
    expected4 = np.array([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j], dtype=np.complex128)
    np.testing.assert_allclose(
        const4.points * np.sqrt(2.0), expected4, rtol=0.0, atol=1e-15
    )

    const16 = build_qam_constellation(16)
    # I bits (MSB pair) / Q bits (LSB pair): 00=-3, 01=-1, 11=+1, 10=+3
    gray_to_level = {0b00: -3.0, 0b01: -1.0, 0b11: 1.0, 0b10: 3.0}
    for idx, point in enumerate(const16.points):
        i_bits = idx >> 2
        q_bits = idx & 0b11
        expected = (gray_to_level[i_bits] + 1j * gray_to_level[q_bits]) / np.sqrt(10.0)
        np.testing.assert_allclose(point, expected, rtol=0.0, atol=1e-15)


@pytest.mark.parametrize("M", [4, 16])
def test_qam_reproducibility(M: int) -> None:
    a = generate_qam(np.random.default_rng(99), 256, M)
    b = generate_qam(np.random.default_rng(99), 256, M)
    np.testing.assert_array_equal(a.symbols, b.symbols)
    np.testing.assert_array_equal(a.bits, b.bits)
    np.testing.assert_array_equal(a.indices, b.indices)


def test_qam_data_stream_does_not_change_pilots() -> None:
    """Using the data stream must not alter the pilots stream."""
    seed, trial = 5, 17
    pilots_before = get_trial_rngs(seed, trial).pilots.standard_normal(32)
    generate_qam(None, 20, 4, master_seed=seed, trial_index=trial)
    pilots_after = get_trial_rngs(seed, trial).pilots.standard_normal(32)
    np.testing.assert_array_equal(pilots_before, pilots_after)


def test_constellation_is_cached() -> None:
    a = build_qam_constellation(16)
    b = build_qam_constellation(16)
    assert a is b

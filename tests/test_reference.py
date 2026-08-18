"""Step 3 acceptance tests: known reference field."""

from __future__ import annotations

import numpy as np
import pytest

from rydberg_sim import (
    generate_reference_field,
    get_trial_rngs,
    spatial_frequency,
    steering_vector,
)


def _baseline(**kwargs):
    params = dict(
        N=8,
        P=5,
        alpha_b=1.7 * np.exp(1j * 0.4),
        vartheta=0.3,
        c=1.0,
    )
    params.update(kwargs)
    return generate_reference_field(**params)


def test_reference_shapes_and_dtype() -> None:
    ref = _baseline(N=8, P=5)
    assert ref.B.shape == (8, 5)
    assert ref.a_b.shape == (8,)
    assert ref.s_b.shape == (5,)
    assert ref.B.dtype == np.complex128
    assert ref.a_b.dtype == np.complex128
    assert ref.s_b.dtype == np.complex128
    assert isinstance(ref.alpha_b, complex)
    assert isinstance(ref.psi_b, float)
    assert isinstance(ref.vartheta, float)
    assert isinstance(ref.c, float)


def test_reference_is_nonzero() -> None:
    """B[n, p] != 0 for every n, p under valid baseline parameters."""
    ref = _baseline()
    assert np.all(np.abs(ref.B) > 0)


def test_constant_spatial_phase_progression() -> None:
    """B[n, p] / B[n-1, p] = exp(-j psi_b) via the complex ratio."""
    ref = _baseline(N=16, P=4, vartheta=0.7, alpha_b=0.5 + 0.2j)
    expected_ratio = np.exp(-1j * ref.psi_b)
    ratios = ref.B[1:, :] / ref.B[:-1, :]
    np.testing.assert_allclose(ratios, expected_ratio, rtol=0.0, atol=1e-12)


def test_baseline_columns_are_identical() -> None:
    """Baseline s_b[p] = 1 ⇒ every column of B equals column 0."""
    ref = _baseline(N=6, P=7)
    np.testing.assert_array_equal(ref.s_b, np.ones(7, dtype=np.complex128))
    for p in range(ref.B.shape[1]):
        np.testing.assert_array_equal(ref.B[:, p], ref.B[:, 0])


def test_baseline_magnitude() -> None:
    """|B[n, p]| = |c alpha_b| when |a_b[n]| = 1 and |s_b[p]| = 1."""
    ref = _baseline(c=0.8, alpha_b=1.2 * np.exp(-1j * 0.9))
    expected = np.abs(ref.c * ref.alpha_b)
    np.testing.assert_allclose(np.abs(ref.B), expected, rtol=0.0, atol=1e-12)


def test_known_analytical_example() -> None:
    """Deterministic example: vartheta = π/6 ⇒ psi_b = π/2."""
    N, P = 4, 3
    c = 1.0
    alpha_b = 2.0 * np.exp(1j * 0.3)
    vartheta = np.pi / 6.0
    ref = generate_reference_field(
        N=N, P=P, c=c, alpha_b=alpha_b, vartheta=vartheta
    )

    psi_b = np.pi * np.sin(np.pi / 6.0)
    np.testing.assert_allclose(ref.psi_b, psi_b, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(ref.psi_b, 0.5 * np.pi, rtol=0.0, atol=1e-15)

    n = np.arange(N, dtype=np.float64)
    a_expected = np.exp(-1j * n * psi_b).astype(np.complex128)
    np.testing.assert_allclose(ref.a_b, a_expected, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(
        ref.a_b,
        np.array([1.0, -1j, -1.0, 1j], dtype=np.complex128),
        rtol=0.0,
        atol=1e-12,
    )

    s_expected = np.ones(P, dtype=np.complex128)
    B_expected = (c * alpha_b) * np.outer(a_expected, s_expected)
    np.testing.assert_allclose(ref.B, B_expected, rtol=0.0, atol=1e-15)


def test_varying_known_s_b_api() -> None:
    """API accepts an explicit known s_b; this is not the baseline."""
    P = 4
    s_b = np.array([1.0, 1j, -1.0, -1j], dtype=np.complex128)
    ref = generate_reference_field(
        N=5,
        P=P,
        alpha_b=0.4 + 0.3j,
        vartheta=-0.2,
        c=1.25,
        s_b=s_b,
    )
    np.testing.assert_array_equal(ref.s_b, s_b)
    expected = (ref.c * ref.alpha_b) * np.outer(ref.a_b, s_b)
    np.testing.assert_allclose(ref.B, expected, rtol=0.0, atol=1e-15)
    # Columns must differ for this non-constant s_b (API check only).
    assert not np.allclose(ref.B[:, 0], ref.B[:, 1])


def test_reuses_ula_steering_vector() -> None:
    ref = _baseline(N=10, vartheta=0.55)
    np.testing.assert_array_equal(ref.a_b, steering_vector(ref.vartheta, 10))
    np.testing.assert_allclose(
        ref.psi_b, float(spatial_frequency(ref.vartheta)), rtol=0.0, atol=0.0
    )


def test_does_not_consume_reference_rng_stream() -> None:
    """Baseline reference is deterministic and must not touch trial RNGs."""
    before = np.random.get_state()
    a = get_trial_rngs(99, 3)
    generate_reference_field(
        N=4, P=3, alpha_b=1.0 + 0.0j, vartheta=0.1, c=1.0
    )
    b = get_trial_rngs(99, 3)
    np.testing.assert_array_equal(
        a.reference.standard_normal(16), b.reference.standard_normal(16)
    )
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])


@pytest.mark.parametrize("N", [0, -1])
def test_rejects_non_positive_N(N: int) -> None:
    with pytest.raises(ValueError, match="N must be > 0"):
        generate_reference_field(N=N, P=2, alpha_b=1.0, vartheta=0.0)


@pytest.mark.parametrize("P", [0, -3])
def test_rejects_non_positive_P(P: int) -> None:
    with pytest.raises(ValueError, match="P must be > 0"):
        generate_reference_field(N=2, P=P, alpha_b=1.0, vartheta=0.0)


def test_rejects_zero_alpha_b() -> None:
    with pytest.raises(ValueError, match="alpha_b must be nonzero"):
        generate_reference_field(N=2, P=2, alpha_b=0.0, vartheta=0.0)
    with pytest.raises(ValueError, match="alpha_b must be nonzero"):
        generate_reference_field(N=2, P=2, alpha_b=0.0 + 0.0j, vartheta=0.0)


def test_rejects_non_finite_alpha_b() -> None:
    with pytest.raises(ValueError, match="alpha_b must be finite"):
        generate_reference_field(N=2, P=2, alpha_b=np.inf, vartheta=0.0)
    with pytest.raises(ValueError, match="alpha_b must be finite"):
        generate_reference_field(N=2, P=2, alpha_b=1.0 + 1j * np.nan, vartheta=0.0)


def test_rejects_non_finite_vartheta() -> None:
    with pytest.raises(ValueError, match="vartheta must be finite"):
        generate_reference_field(N=2, P=2, alpha_b=1.0, vartheta=np.nan)
    with pytest.raises(ValueError, match="vartheta must be finite"):
        generate_reference_field(N=2, P=2, alpha_b=1.0, vartheta=np.inf)


def test_rejects_wrong_s_b_shape() -> None:
    with pytest.raises(ValueError, match="s_b must have shape"):
        generate_reference_field(
            N=2, P=3, alpha_b=1.0, vartheta=0.0, s_b=np.ones(4)
        )
    with pytest.raises(ValueError, match="s_b must have shape"):
        generate_reference_field(
            N=2, P=3, alpha_b=1.0, vartheta=0.0, s_b=np.ones((3, 1))
        )

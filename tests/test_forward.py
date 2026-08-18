"""Step 5 acceptance tests: exact and linearised forward models."""

from __future__ import annotations

import numpy as np
import pytest

from rydberg_sim import (
    SimulationConfig,
    exact_forward,
    generate_gaussian_pilots,
    generate_reference_field,
    generate_ula_channel,
    get_trial_rngs,
    linearised_observation,
    make_alpha_b,
    rsr_db_to_alpha_magnitude,
)

MASTER_SEED = 20260818
N, K, P = 8, 3, 8

# Noise-power Monte Carlo: |W|^2 has mean sigma2 and variance sigma2^2
# (exponential). 4000 draws of an 8x8 matrix → 256_000 samples.
N_MC = 4000
N_NOISE_SAMPLES = N_MC * N * P
SIGMA2 = 0.4
POWER_TOL = 5.0 * SIGMA2 / np.sqrt(N_NOISE_SAMPLES)


def _gsb(trial: int = 0):
    cfg = SimulationConfig.create(
        N=N, K=K, L=3, beta=1.0, master_seed=MASTER_SEED, c=1.0
    )
    ch = generate_ula_channel(cfg, trial)
    pilots = generate_gaussian_pilots(
        K=K, P=P, master_seed=MASTER_SEED, trial_index=trial
    )
    ref = generate_reference_field(
        N=N, P=P, alpha_b=1.2 * np.exp(1j * 0.4), vartheta=0.3, c=1.0
    )
    return ch.G, pilots.S, ref.B


def _tiny_hand_example():
    """Small complex example chosen so a +j Psi sign error is obvious."""
    G = np.array([[1 + 2j, 0.5], [-1j, 2 - 1j]], dtype=np.complex128)
    S = np.array([[0.7 - 0.2j, 1j], [1.0, -0.5 + 0.5j]], dtype=np.complex128)
    B = np.array([[3 + 4j, 2 - 1j], [-1 + 2j, 0.5 + 0.5j]], dtype=np.complex128)
    return G, S, B


def test_exact_shapes_and_dtypes() -> None:
    G, S, B = _gsb()
    obs = exact_forward(G, S, B, SIGMA2, master_seed=MASTER_SEED, trial_index=0)
    assert obs.signal.shape == (N, P)
    assert obs.B.shape == (N, P)
    assert obs.W.shape == (N, P)
    assert obs.E.shape == (N, P)
    assert obs.Z.shape == (N, P)
    assert obs.G.dtype == np.complex128
    assert obs.S.dtype == np.complex128
    assert obs.B.dtype == np.complex128
    assert obs.W.dtype == np.complex128
    assert obs.E.dtype == np.complex128
    assert obs.Z.dtype == np.float64
    assert np.all(obs.Z >= 0.0)
    # Amplitude, not power: Z == |E| and (for this draw) not |E|^2.
    np.testing.assert_allclose(obs.Z, np.abs(obs.E), rtol=0.0, atol=1e-15)
    assert not np.allclose(obs.Z, np.abs(obs.E) ** 2)


def test_noiseless_analytical_identity() -> None:
    G, S, B = _gsb()
    obs = exact_forward(G, S, B, 0.0)
    np.testing.assert_array_equal(obs.W, np.zeros_like(obs.W))
    np.testing.assert_allclose(obs.E, G @ S + B, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(obs.signal, G @ S, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(obs.Z, np.abs(G @ S + B), rtol=0.0, atol=1e-15)


def test_noise_power() -> None:
    """mean(|W|^2) ≈ sigma2 and Re/Im variances ≈ sigma2/2.

    |W|^2 is exponential with mean sigma2, so the sample-mean std is
    sigma2 / sqrt(N_NOISE_SAMPLES). The tolerance is 5σ.
    """
    G = np.ones((N, K), dtype=np.complex128)
    S = np.ones((K, P), dtype=np.complex128)
    B = np.ones((N, P), dtype=np.complex128)
    power = 0.0
    re2 = 0.0
    im2 = 0.0
    count = 0
    for t in range(N_MC):
        W = exact_forward(
            G, S, B, SIGMA2, master_seed=MASTER_SEED, trial_index=t
        ).W
        power += float(np.sum(np.abs(W) ** 2))
        re2 += float(np.sum(np.square(W.real)))
        im2 += float(np.sum(np.square(W.imag)))
        count += W.size
    assert count == N_NOISE_SAMPLES
    np.testing.assert_allclose(power / count, SIGMA2, rtol=0.0, atol=POWER_TOL)
    var_tol = 5.0 * (SIGMA2 / 2.0) * np.sqrt(2.0 / count)  # Gaussian var std ~ σ²√(2/n)
    np.testing.assert_allclose(re2 / count, SIGMA2 / 2.0, rtol=0.0, atol=max(var_tol, 1e-3))
    np.testing.assert_allclose(im2 / count, SIGMA2 / 2.0, rtol=0.0, atol=max(var_tol, 1e-3))


def test_noise_mean_and_circularity() -> None:
    G = np.ones((N, K), dtype=np.complex128)
    S = np.ones((K, P), dtype=np.complex128)
    B = np.ones((N, P), dtype=np.complex128)
    re_acc = 0.0
    im_acc = 0.0
    cross = 0.0
    count = 0
    for t in range(N_MC):
        W = exact_forward(
            G, S, B, SIGMA2, master_seed=MASTER_SEED, trial_index=t
        ).W
        re_acc += float(W.real.sum())
        im_acc += float(W.imag.sum())
        cross += float(np.sum(W.real * W.imag))
        count += W.size
    mean_tol = 5.0 * np.sqrt((SIGMA2 / 2.0) / count)
    np.testing.assert_allclose(re_acc / count, 0.0, atol=mean_tol, rtol=0.0)
    np.testing.assert_allclose(im_acc / count, 0.0, atol=mean_tol, rtol=0.0)
    cov = cross / count
    assert abs(cov) < 5.0 * (SIGMA2 / 2.0) / np.sqrt(count)


def test_noise_reproducibility() -> None:
    G, S, B = _gsb()
    a = exact_forward(G, S, B, SIGMA2, master_seed=MASTER_SEED, trial_index=137)
    b = exact_forward(G, S, B, SIGMA2, master_seed=MASTER_SEED, trial_index=137)
    np.testing.assert_array_equal(a.W, b.W)
    np.testing.assert_array_equal(a.Z, b.Z)
    injected = exact_forward(
        G, S, B, SIGMA2, get_trial_rngs(MASTER_SEED, 137).noise
    )
    np.testing.assert_array_equal(a.W, injected.W)
    other = exact_forward(G, S, B, SIGMA2, master_seed=MASTER_SEED, trial_index=138)
    assert not np.array_equal(a.W, other.W)


def test_Y_is_Z_minus_abs_B() -> None:
    G, S, B = _gsb()
    exact = exact_forward(G, S, B, SIGMA2, master_seed=MASTER_SEED, trial_index=3)
    lin = linearised_observation(exact)
    np.testing.assert_allclose(lin.Y, exact.Z - np.abs(exact.B), rtol=0.0, atol=1e-15)


def test_Psi_sign_convention() -> None:
    """Psi = exp(-1j * angle(B)), not exp(+1j * angle(B))."""
    G, S, B = _tiny_hand_example()
    exact = exact_forward(G, S, B, 0.0)
    lin = linearised_observation(exact)
    expected = np.exp(-1j * np.angle(B))
    wrong_sign = np.exp(+1j * np.angle(B))
    np.testing.assert_allclose(lin.Psi, expected, rtol=0.0, atol=1e-15)
    assert not np.allclose(lin.Psi, wrong_sign)
    np.testing.assert_allclose(np.abs(lin.Psi), 1.0, rtol=0.0, atol=1e-15)


def test_linear_signal_hand_example() -> None:
    G, S, B = _tiny_hand_example()
    exact = exact_forward(G, S, B, 0.0)
    lin = linearised_observation(exact)
    signal = G @ S
    psi = np.exp(-1j * np.angle(B))
    expected = np.real(psi * signal)
    np.testing.assert_allclose(lin.Y_linear_signal, expected, rtol=0.0, atol=1e-15)
    # Conjugation / swapped-real-imag mistakes must fail this check.
    conjugated = np.real(np.conj(psi) * signal)
    assert not np.allclose(lin.Y_linear_signal, conjugated)


def test_strong_reference_approximation_improves_with_amplitude() -> None:
    """Relative linearisation error falls as official RSR increases.

    ``B`` is generated from Step-6 ``|alpha_b|(RSR)``, including the
    RSR = 30 dB strong-reference point. ``sigma2 = 0`` isolates
    Taylor error from noise. This is an acceptance test, not a figure.
    """
    G, S, _ = _gsb(trial=4)
    rsr_dbs = (0.0, 10.0, 20.0, 30.0)
    errors = []
    for rsr_db in rsr_dbs:
        alpha_b = make_alpha_b(rsr_db_to_alpha_magnitude(rsr_db, beta_ref=1.0))
        B = generate_reference_field(
            N=N, P=P, alpha_b=alpha_b, vartheta=0.3, c=1.0
        ).B
        exact = exact_forward(G, S, B, 0.0)
        lin = linearised_observation(exact)
        errors.append(lin.relative_frobenius_error)
    for weaker, stronger in zip(errors, errors[1:]):
        assert stronger < weaker, (rsr_dbs, errors)
    assert rsr_dbs[-1] == 30.0
    assert errors[-1] < 0.05, errors


def test_noiseless_taylor_limit() -> None:
    """With W = 0, Z - |B| → Re{Psi ⊙ GS} as |B| ≫ |GS|."""
    G, S, B0 = _gsb(trial=5)
    signal = G @ S
    scale0 = float(np.mean(np.abs(signal)) / np.mean(np.abs(B0)))
    B_unit = scale0 * B0
    errors = []
    for amp in (1.0, 10.0, 100.0, 1000.0):
        exact = exact_forward(G, S, amp * B_unit, 0.0)
        lin = linearised_observation(exact)
        errors.append(lin.relative_frobenius_error)
        np.testing.assert_array_equal(exact.W, 0)
    for weaker, stronger in zip(errors, errors[1:]):
        assert stronger < weaker, errors
    assert errors[-1] < 1e-3


def test_linearised_does_not_add_extra_noise() -> None:
    """Y is exactly Z - |B|; no second Nbar draw."""
    G, S, B = _gsb()
    exact = exact_forward(G, S, B, 0.0)
    lin = linearised_observation(exact)
    np.testing.assert_array_equal(lin.Y, exact.Z - np.abs(B))


def test_sigma2_zero_does_not_require_rng() -> None:
    G, S, B = _tiny_hand_example()
    before = np.random.get_state()
    exact_forward(G, S, B, 0.0)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])


def test_rejects_incompatible_shapes() -> None:
    G = np.ones((4, 2), dtype=np.complex128)
    S = np.ones((3, 5), dtype=np.complex128)
    B = np.ones((4, 5), dtype=np.complex128)
    with pytest.raises(ValueError, match="incompatible G and S"):
        exact_forward(G, S, B, 0.0)
    S_ok = np.ones((2, 5), dtype=np.complex128)
    B_bad = np.ones((4, 4), dtype=np.complex128)
    with pytest.raises(ValueError, match="incompatible B"):
        exact_forward(G, S_ok, B_bad, 0.0)


def test_rejects_negative_or_nonfinite_sigma2() -> None:
    G, S, B = _tiny_hand_example()
    with pytest.raises(ValueError, match="sigma2 must be >= 0"):
        exact_forward(G, S, B, -0.1)
    with pytest.raises(ValueError, match="sigma2 must be finite"):
        exact_forward(G, S, B, np.nan)


def test_rejects_nonfinite_matrices() -> None:
    G, S, B = _tiny_hand_example()
    G_bad = G.copy()
    G_bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="G must be finite"):
        exact_forward(G_bad, S, B, 0.0)

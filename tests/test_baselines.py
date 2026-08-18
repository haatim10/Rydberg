"""Step 7 acceptance tests: debugging / reference baselines."""

from __future__ import annotations

import numpy as np
import pytest

from rydberg_sim import (
    SimulationConfig,
    build_qam_constellation,
    exact_forward,
    generate_gaussian_pilots,
    generate_reference_field,
    generate_ula_channel,
    linearised_observation,
    make_alpha_b,
    rsr_db_to_alpha_magnitude,
)
from rydberg_sim.baselines import (
    DEFAULT_MAX_CANDIDATES,
    FUTURE_GD_VS_CLOSED_FORM_TEST,
    ExhaustiveSearchTooLargeError,
    cm_zf,
    enumerate_qam_symbol_vectors,
    exhaustive_magnitude_ls,
    exhaustive_magnitude_ml,
    exhaustive_search_complexity_gate,
    linearised_closed_form_ls,
    linearised_design_matrix,
    log_bessel_i0,
    make_ideal_linear_y,
    pack_gtilde,
    qam_candidate_count,
    reconstruct_complex_observation,
    rician_log_likelihood,
    theoretical_linearised_ls_gtilde_covariance,
    theoretical_zf_error_covariance,
    true_complex_observation,
    unpack_gtilde,
    zf_known_phase,
    zf_known_phase_from_truth,
)
from rydberg_sim.qam import QAMConstellation

MASTER_SEED = 20260818


def _full_rank_M(rng: np.random.Generator, D: int, Q: int) -> np.ndarray:
    M = (rng.standard_normal((D, Q)) + 1j * rng.standard_normal((D, Q))).astype(
        np.complex128
    )
    s = np.linalg.svd(M, compute_uv=False)
    assert s[-1] / s[0] > 1e-6
    return M


# ---------------------------------------------------------------------------
# Part A — ZF known phase
# ---------------------------------------------------------------------------


def test_zf_exact_reconstruction() -> None:
    """z * exp(1j theta) equals lambda_noisy to floating-point precision."""
    rng = np.random.default_rng(1)
    D, Q = 3, 8
    M = _full_rank_M(rng, D, Q)
    u_true = (rng.standard_normal(D) + 1j * rng.standard_normal(D)).astype(np.complex128)
    b = (rng.standard_normal(Q) + 1j * rng.standard_normal(Q)).astype(np.complex128)
    w = 0.3 * (rng.standard_normal(Q) + 1j * rng.standard_normal(Q)).astype(np.complex128)
    lam = true_complex_observation(M, u_true, b, w)
    z = np.abs(lam)
    theta = np.angle(lam)
    recon = reconstruct_complex_observation(z, theta)
    np.testing.assert_allclose(recon, lam, rtol=0.0, atol=1e-15)
    # Polar reconstruction identity: |λ| exp(j angle(λ)) == λ.
    np.testing.assert_allclose(z * np.exp(1j * theta), lam, rtol=0.0, atol=1e-15)


def test_zf_noiseless_recovery() -> None:
    """w = 0 and full-row-rank M ⇒ u_hat ≈ u_true to near machine precision."""
    rng = np.random.default_rng(2)
    D, Q = 4, 10
    M = _full_rank_M(rng, D, Q)
    u_true = (rng.standard_normal(D) + 1j * rng.standard_normal(D)).astype(np.complex128)
    b = (rng.standard_normal(Q) + 1j * rng.standard_normal(Q)).astype(np.complex128)
    w = np.zeros(Q, dtype=np.complex128)
    u_hat = zf_known_phase_from_truth(M, u_true, b, w, ridge=0.0)
    np.testing.assert_allclose(u_hat, u_true, rtol=0.0, atol=1e-12)


def test_zf_noiseless_recovery_via_explicit_phase() -> None:
    rng = np.random.default_rng(3)
    D, Q = 2, 6
    M = _full_rank_M(rng, D, Q)
    u_true = np.array([1.0 + 2.0j, -0.5 + 0.25j], dtype=np.complex128)
    b = np.array([0.3 - 0.1j, 1.0, -1j, 0.2 + 0.4j, 0.0, 0.7 - 0.8j], dtype=np.complex128)
    lam = M.conj().T @ u_true + b
    u_hat = zf_known_phase(M, np.abs(lam), np.angle(lam), b, ridge=0.0)
    np.testing.assert_allclose(u_hat, u_true, rtol=0.0, atol=1e-12)


def test_zf_does_not_invert_and_default_ridge_is_zero() -> None:
    """ridge=0 is exact ZF; a silent 1e-12 Tikhonov must not be applied."""
    rng = np.random.default_rng(4)
    M = _full_rank_M(rng, 3, 7)
    u_true = (rng.standard_normal(3) + 1j * rng.standard_normal(3)).astype(np.complex128)
    b = (rng.standard_normal(7) + 1j * rng.standard_normal(7)).astype(np.complex128)
    w = np.zeros(7, dtype=np.complex128)
    u0 = zf_known_phase_from_truth(M, u_true, b, w)
    u_explicit = zf_known_phase_from_truth(M, u_true, b, w, ridge=0.0)
    np.testing.assert_array_equal(u0, u_explicit)
    # A large ridge moves the estimate; default must not match that.
    u_ridged = zf_known_phase_from_truth(M, u_true, b, w, ridge=1.0)
    assert not np.allclose(u_ridged, u0, rtol=1e-6, atol=1e-6)


def test_zf_error_covariance_trace() -> None:
    """Empirical Cov(u_hat - u) approaches sigma2 (M M^H)^{-1}."""
    rng = np.random.default_rng(5)
    D, Q = 3, 12
    M = _full_rank_M(rng, D, Q)
    u_true = (rng.standard_normal(D) + 1j * rng.standard_normal(D)).astype(np.complex128)
    b = (rng.standard_normal(Q) + 1j * rng.standard_normal(Q)).astype(np.complex128)
    sigma2 = 0.4
    n_mc = 5000
    scale = np.sqrt(sigma2 / 2.0)
    errors = np.empty((n_mc, D), dtype=np.complex128)
    for t in range(n_mc):
        w = scale * (rng.standard_normal(Q) + 1j * rng.standard_normal(Q))
        u_hat = zf_known_phase_from_truth(M, u_true, b, w, ridge=0.0)
        errors[t] = u_hat - u_true
    emp_cov = (errors.conj().T @ errors) / n_mc
    th_cov = theoretical_zf_error_covariance(M, sigma2, ridge=0.0)
    tr_emp = float(np.trace(emp_cov).real)
    tr_th = float(np.trace(th_cov).real)
    rel = abs(tr_emp - tr_th) / tr_th
    assert rel < 0.08, (tr_emp, tr_th, rel)
    # Hermitian positive-definite check on the theory matrix.
    np.testing.assert_allclose(th_cov, th_cov.conj().T, rtol=0.0, atol=1e-12)


def test_zf_rejects_bad_dimensions() -> None:
    M = np.eye(3, 5, dtype=np.complex128)
    z = np.ones(5)
    theta = np.zeros(5)
    b = np.zeros(5, dtype=np.complex128)
    with pytest.raises(ValueError, match="length 5"):
        zf_known_phase(M, z[:4], theta, b)
    with pytest.raises(ValueError, match="M must be a 2-D"):
        zf_known_phase(np.ones(3), z, theta, b)


# ---------------------------------------------------------------------------
# Part B — linearised closed-form LS
# ---------------------------------------------------------------------------


def _tiny_linear_example() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic complex G, S, Psi chosen so a +Im sign error is obvious."""
    G = np.array([[1.0 + 2.0j, -0.5 + 0.25j], [0.3 - 1.1j, 2.0 + 0.4j]], dtype=np.complex128)
    S = np.array(
        [[0.7 - 0.2j, 1.0j, -1.2 + 0.3j], [1.0, -0.5 + 0.5j, 0.4 - 0.8j]],
        dtype=np.complex128,
    )
    # Unit-modulus Psi, not all-real, not all-ones.
    angles = np.array([[0.4, -1.1, 2.0], [1.3, 0.2, -2.5]], dtype=np.float64)
    Psi = np.exp(-1j * angles).astype(np.complex128)
    return G, S, Psi


def test_linearised_ls_dimensions() -> None:
    G, S, Psi = _tiny_linear_example()
    N, K = G.shape
    P = S.shape[1]
    Y = np.real(Psi * (G @ S))
    result = linearised_closed_form_ls(
        Y, S, Psi, observation_source="ideal_linear", ridge=0.0
    )
    assert result.Phi.shape == (N, P, 2 * K)
    for n in range(N):
        assert result.Phi[n].shape == (P, 2 * K)
        assert result.gtilde_hat[n].shape == (2 * K,)
    assert result.G_hat.shape == (N, K)
    assert result.observation_source == "ideal_linear"


def test_linearised_ls_noiseless_ideal_model() -> None:
    """Y = Re{Psi ⊙ GS}, no noise, full-column-rank Phi ⇒ G_hat ≈ G."""
    rng = np.random.default_rng(6)
    N, K, P = 4, 3, 8
    G = (rng.standard_normal((N, K)) + 1j * rng.standard_normal((N, K))).astype(
        np.complex128
    )
    S = (rng.standard_normal((K, P)) + 1j * rng.standard_normal((K, P))).astype(
        np.complex128
    )
    Psi = np.exp(-1j * rng.uniform(-np.pi, np.pi, size=(N, P)))
    Y = np.real(Psi * (G @ S))
    result = linearised_closed_form_ls(Y, S, Psi, observation_source="ideal_linear")
    for n in range(N):
        svals = np.linalg.svd(result.Phi[n], compute_uv=False)
        assert svals[-1] / svals[0] > 1e-8
    np.testing.assert_allclose(result.G_hat, G, rtol=0.0, atol=1e-12)


def test_linearised_complex_sign_check() -> None:
    """Phi_n @ gtilde_n == Re{Psi[n,:] * (G @ S)[n,:]}; catches +Im vs -Im."""
    G, S, Psi = _tiny_linear_example()
    GS = G @ S
    N, K = G.shape
    for n in range(N):
        phi_n = linearised_design_matrix(Psi[n], S)
        gtilde = pack_gtilde(G[n])
        lhs = phi_n @ gtilde
        rhs = np.real(Psi[n] * GS[n])
        np.testing.assert_allclose(lhs, rhs, rtol=0.0, atol=1e-15)
        # The common bug +Im(...) in the second block fails this example.
        U = Psi[n][:, None] * S.T
        phi_wrong = np.concatenate([U.real, U.imag], axis=1)
        lhs_wrong = phi_wrong @ gtilde
        assert not np.allclose(lhs_wrong, rhs, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(unpack_gtilde(pack_gtilde(G[0])), G[0], rtol=0.0, atol=0.0)


def test_linearised_ls_requires_observation_source() -> None:
    G, S, Psi = _tiny_linear_example()
    Y = np.real(Psi * (G @ S))
    with pytest.raises(TypeError):
        linearised_closed_form_ls(Y, S, Psi)  # type: ignore[misc]
    with pytest.raises(ValueError, match="observation_source"):
        linearised_closed_form_ls(Y, S, Psi, observation_source="something_else")  # type: ignore[arg-type]


def test_linearised_ls_high_rsr_exact_magnitude() -> None:
    """Exact Z = |GS+B+W| and Y = Z - |B| with Step-6 RSR calibration.

    The linearised LS estimate improves as RSR grows, consistent with the
    strong-reference approximation. Exact equality is not expected at finite
    RSR because Y comes from the nonlinear model.
    """
    N, K, P = 8, 3, 8
    cfg = SimulationConfig.create(
        N=N, K=K, L=3, beta=1.0, master_seed=MASTER_SEED, c=1.0
    )
    trial = 11
    G = generate_ula_channel(cfg, trial).G
    S = generate_gaussian_pilots(K=K, P=P, master_seed=MASTER_SEED, trial_index=trial).S
    rsr_dbs = (0.0, 10.0, 20.0, 30.0)
    nmses = []
    for rsr_db in rsr_dbs:
        alpha_b = make_alpha_b(rsr_db_to_alpha_magnitude(rsr_db, beta_ref=1.0))
        B = generate_reference_field(N=N, P=P, alpha_b=alpha_b, vartheta=0.3, c=1.0).B
        exact = exact_forward(G, S, B, 0.0)
        lin = linearised_observation(exact)
        result = linearised_closed_form_ls(
            lin.Y, S, lin.Psi, observation_source="exact_magnitude"
        )
        assert result.observation_source == "exact_magnitude"
        nmse = float(np.sum(np.abs(result.G_hat - G) ** 2) / np.sum(np.abs(G) ** 2))
        nmses.append(nmse)
    for weaker, stronger in zip(nmses, nmses[1:]):
        assert stronger < weaker, (rsr_dbs, nmses)
    assert rsr_dbs[-1] == 30.0
    assert nmses[-1] < 0.05, nmses
    # Finite-RSR exact-magnitude Y is not the ideal linear model.
    alpha_b = make_alpha_b(rsr_db_to_alpha_magnitude(30.0, beta_ref=1.0))
    B = generate_reference_field(N=N, P=P, alpha_b=alpha_b, vartheta=0.3, c=1.0).B
    exact = exact_forward(G, S, B, 0.0)
    lin = linearised_observation(exact)
    ideal = linearised_closed_form_ls(
        lin.Y_linear_signal, S, lin.Psi, observation_source="ideal_linear"
    )
    np.testing.assert_allclose(ideal.G_hat, G, rtol=0.0, atol=1e-10)
    exact_ls = linearised_closed_form_ls(
        lin.Y, S, lin.Psi, observation_source="exact_magnitude"
    )
    assert not np.allclose(exact_ls.G_hat, G, rtol=0.0, atol=1e-10)


def test_linearised_ls_ideal_noise_covariance() -> None:
    """Empirical Cov vs (sigma2/2) (Phi^T Phi)^{-1} on the ideal linear model."""
    rng = np.random.default_rng(7)
    N, K, P = 2, 2, 8
    G = (rng.standard_normal((N, K)) + 1j * rng.standard_normal((N, K))).astype(
        np.complex128
    )
    S = (rng.standard_normal((K, P)) + 1j * rng.standard_normal((K, P))).astype(
        np.complex128
    )
    Psi = np.exp(-1j * rng.uniform(-np.pi, np.pi, size=(N, P)))
    sigma2 = 0.5
    n_mc = 4000
    n = 0
    phi0 = linearised_design_matrix(Psi[n], S)
    gtilde_true = pack_gtilde(G[n])
    errors = np.empty((n_mc, 2 * K), dtype=np.float64)
    for t in range(n_mc):
        Y = make_ideal_linear_y(G, S, Psi, sigma2=sigma2, rng=rng)
        result = linearised_closed_form_ls(Y, S, Psi, observation_source="ideal_linear")
        errors[t] = result.gtilde_hat[n] - gtilde_true
        np.testing.assert_allclose(result.Phi[n], phi0, rtol=0.0, atol=0.0)
    emp_cov = (errors.T @ errors) / n_mc
    th_cov = theoretical_linearised_ls_gtilde_covariance(phi0, sigma2, ridge=0.0)
    tr_emp = float(np.trace(emp_cov))
    tr_th = float(np.trace(th_cov))
    rel = abs(tr_emp - tr_th) / tr_th
    assert rel < 0.10, (tr_emp, tr_th, rel)


def test_future_gd_comparison_is_documented_not_implemented() -> None:
    """Do not implement GD in Step 7 just to satisfy a later 1e-6 test."""
    assert "GD" in FUTURE_GD_VS_CLOSED_FORM_TEST
    assert "1e-6" in FUTURE_GD_VS_CLOSED_FORM_TEST
    import rydberg_sim.baselines as bl

    assert not hasattr(bl, "xu_gd")
    assert not hasattr(bl, "gradient_descent")
    assert not hasattr(bl, "biased_gs")
    assert not hasattr(bl, "em_gs")
    assert not hasattr(bl, "spectral_init")


# ---------------------------------------------------------------------------
# Part C — CM-ZF left unimplemented
# ---------------------------------------------------------------------------


def test_cm_zf_is_unimplemented() -> None:
    with pytest.raises(NotImplementedError, match="CM-ZF"):
        cm_zf()
    with pytest.raises(NotImplementedError, match="not specified"):
        cm_zf(np.ones((2, 2)), np.ones((2, 3)))


# ---------------------------------------------------------------------------
# Part D — exhaustive QAM search
# ---------------------------------------------------------------------------


def test_exhaustive_candidate_count_4qam_d3() -> None:
    assert qam_candidate_count(4, 3) == 64
    n = exhaustive_search_complexity_gate(4, 3)
    assert n == 64
    const = build_qam_constellation(4)
    cands = enumerate_qam_symbol_vectors(const, 3)
    assert cands.shape == (64, 3)


def test_exhaustive_ls_noiseless_recovers_true_qam_vector() -> None:
    rng = np.random.default_rng(8)
    D, Q = 3, 8
    M = _full_rank_M(rng, D, Q)
    const = build_qam_constellation(4)
    # Pick a true vector from the Step-4 alphabet.
    idx = rng.integers(0, 4, size=D)
    u_true = const.points[idx]
    b = (rng.standard_normal(Q) + 1j * rng.standard_normal(Q)).astype(np.complex128)
    lam = M.conj().T @ u_true + b
    z = np.abs(lam)
    result = exhaustive_magnitude_ls(M, b, z, const, max_candidates=DEFAULT_MAX_CANDIDATES)
    assert result.num_candidates == 64
    assert result.criterion == "ls"
    assert result.unique_minimizer
    np.testing.assert_allclose(result.u_hat, u_true, rtol=0.0, atol=1e-12)
    np.testing.assert_array_equal(result.u_hat, u_true)


def test_exhaustive_complexity_gate_blocks_16qam_d6() -> None:
    n = qam_candidate_count(16, 6)
    assert n == 16_777_216
    with pytest.raises(ExhaustiveSearchTooLargeError, match="16_777_216|16777216"):
        exhaustive_search_complexity_gate(16, 6)
    with pytest.raises(ExhaustiveSearchTooLargeError):
        exhaustive_search_complexity_gate(16, 6, max_candidates=DEFAULT_MAX_CANDIDATES)
    with pytest.raises(ExhaustiveSearchTooLargeError):
        enumerate_qam_symbol_vectors(16, 6)
    M = np.eye(6, 8, dtype=np.complex128)
    b = np.zeros(8, dtype=np.complex128)
    z = np.ones(8)
    with pytest.raises(ExhaustiveSearchTooLargeError):
        exhaustive_magnitude_ls(M, b, z, 16)


def test_exhaustive_uses_step4_unit_energy_gray_qam() -> None:
    const = build_qam_constellation(4)
    cands = enumerate_qam_symbol_vectors(4, 2, max_candidates=64)
    assert cands.shape == (16, 2)
    # Every entry is a Step-4 constellation point; no second alphabet.
    for val in np.unique(cands):
        assert np.any(np.isclose(const.points, val, rtol=0.0, atol=1e-15))
    # Mean energy of the alphabet (not of the Cartesian product) is 1.
    np.testing.assert_allclose(np.mean(np.abs(const.points) ** 2), 1.0, rtol=0.0, atol=1e-15)
    result = exhaustive_magnitude_ls(
        np.eye(2, 4, dtype=np.complex128),
        np.zeros(4, dtype=np.complex128),
        np.ones(4),
        4,
    )
    assert result.constellation is build_qam_constellation(4)
    np.testing.assert_array_equal(result.constellation.points, const.points)
    np.testing.assert_array_equal(result.constellation.bit_labels, const.bit_labels)
    # A homemade constellation object that is not the cache is rejected.
    fake = QAMConstellation(
        M=4,
        bits_per_symbol=2,
        points=const.points.copy(),
        bit_labels=const.bit_labels.copy(),
        scale=const.scale,
        axis_levels=const.axis_levels.copy(),
    )
    with pytest.raises(ValueError, match="Step-4"):
        enumerate_qam_symbol_vectors(fake, 2)


def test_exhaustive_ml_log_likelihood_matches_rician_formula() -> None:
    """ML uses stable log I0, not a product of Rician probabilities."""
    z = np.array([0.5, 1.2, 2.0])
    lam = np.array([0.3 + 0.1j, -1.0, 0.4 - 0.7j], dtype=np.complex128)
    sigma2 = 0.8
    ell = rician_log_likelihood(z, lam, sigma2)
    abs_lam = np.abs(lam)
    arg = 2.0 * z * abs_lam / sigma2
    expected = float(np.sum(-(abs_lam**2) / sigma2 + log_bessel_i0(arg)))
    np.testing.assert_allclose(ell, expected, rtol=0.0, atol=1e-12)
    # log I0(0) = 0.
    np.testing.assert_allclose(log_bessel_i0(0.0), 0.0, rtol=0.0, atol=1e-15)


def test_exhaustive_ml_recovers_true_vector_at_high_snr() -> None:
    rng = np.random.default_rng(9)
    D, Q = 3, 8
    M = _full_rank_M(rng, D, Q)
    const = build_qam_constellation(4)
    u_true = const.points[rng.integers(0, 4, size=D)]
    b = (rng.standard_normal(Q) + 1j * rng.standard_normal(Q)).astype(np.complex128)
    sigma2 = 1e-4
    scale = np.sqrt(sigma2 / 2.0)
    w = scale * (rng.standard_normal(Q) + 1j * rng.standard_normal(Q))
    z = np.abs(M.conj().T @ u_true + b + w)
    result = exhaustive_magnitude_ml(M, b, z, const, sigma2=sigma2)
    assert result.criterion == "ml"
    assert result.num_candidates == 64
    np.testing.assert_allclose(result.u_hat, u_true, rtol=0.0, atol=1e-12)
    with pytest.raises(ValueError, match="sigma2 must be > 0"):
        exhaustive_magnitude_ml(M, b, z, const, sigma2=0.0)


def test_exhaustive_ml_is_not_silently_the_ls_metric() -> None:
    """LS and ML share the candidate set but use different metrics."""
    rng = np.random.default_rng(10)
    D, Q = 2, 5
    M = _full_rank_M(rng, D, Q)
    const = build_qam_constellation(4)
    u_true = const.points[rng.integers(0, 4, size=D)]
    b = (rng.standard_normal(Q) + 1j * rng.standard_normal(Q)).astype(np.complex128)
    sigma2 = 0.6
    z = np.abs(M.conj().T @ u_true + b)
    ls = exhaustive_magnitude_ls(M, b, z, const)
    ml = exhaustive_magnitude_ml(M, b, z, const, sigma2=sigma2)
    assert ls.criterion != ml.criterion
    # Metrics live on different scales; they must not be bitwise-equal numbers
    # that would indicate a copy-paste of J_LS into the ML path.
    assert ls.metric != pytest.approx(ml.metric)

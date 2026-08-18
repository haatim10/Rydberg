"""Step 12 acceptance tests: linearised channel CRLB from the real Gaussian model."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from rydberg_sim import (
    expected_channel_frobenius_energy,
    linearised_channel_crlb,
    linearised_closed_form_ls,
    linearised_row_crlb,
    linearised_row_fisher,
    reference_phase_matrix,
)
from rydberg_sim.baselines import (
    linearised_design_matrix,
    make_ideal_linear_y,
    pack_gtilde,
    theoretical_linearised_ls_gtilde_covariance,
)
from rydberg_sim.forward import exact_forward, linearised_observation
from rydberg_sim.linearised_crlb import RankDeficientPhiError

MASTER_SEED = 20260818


def _full_rank_complex(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    return (
        rng.standard_normal((rows, cols)) + 1j * rng.standard_normal((rows, cols))
    ).astype(np.complex128)


def _conditioned_phi(rng: np.random.Generator, P: int, K: int) -> np.ndarray:
    S = _full_rank_complex(rng, K, P)
    psi = np.exp(-1j * rng.uniform(-np.pi, np.pi, size=P))
    phi = linearised_design_matrix(psi, S)
    s = np.linalg.svd(phi, compute_uv=False)
    assert s[-1] / s[0] > 1e-6
    return phi


# ---------------------------------------------------------------------------
# Shared Phi / Psi conventions
# ---------------------------------------------------------------------------


def test_reuses_step7_design_matrix_and_minus_im() -> None:
    rng = np.random.default_rng(4)
    K, P, N = 3, 8, 2
    S = _full_rank_complex(rng, K, P)
    B = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(N, P))) * (
        0.4 + rng.random((N, P))
    )
    result = linearised_channel_crlb(S, B, 0.3, beta_k=1.0)
    Psi = reference_phase_matrix(B)
    np.testing.assert_allclose(result.Psi, Psi, rtol=0.0, atol=0.0)
    for n in range(N):
        phi_step7 = linearised_design_matrix(Psi[n], S)
        np.testing.assert_allclose(result.Phi[n], phi_step7, rtol=0.0, atol=0.0)
        G_row = rng.standard_normal(K) + 1j * rng.standard_normal(K)
        lhs = result.Phi[n] @ pack_gtilde(G_row)
        rhs = np.real(Psi[n] * (G_row @ S))
        np.testing.assert_allclose(lhs, rhs, rtol=0.0, atol=1e-14)


def test_psi_matches_forward_linearisation() -> None:
    rng = np.random.default_rng(2)
    N, K, P = 3, 2, 6
    G = _full_rank_complex(rng, N, K)
    S = _full_rank_complex(rng, K, P)
    B = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(N, P)))
    exact = exact_forward(G, S, B, 0.0)
    lin = linearised_observation(exact)
    np.testing.assert_allclose(reference_phase_matrix(B), lin.Psi, rtol=0.0, atol=0.0)


def test_row_crlb_matches_step7_theoretical_covariance() -> None:
    rng = np.random.default_rng(9)
    phi = _conditioned_phi(rng, P=10, K=3)
    sigma2 = 0.35
    row = linearised_row_crlb(phi, sigma2)
    step7 = theoretical_linearised_ls_gtilde_covariance(phi, sigma2, ridge=0.0)
    np.testing.assert_allclose(row.crlb, step7, rtol=0.0, atol=1e-12)


def test_equations_are_two_over_sigma2_not_xu_copy() -> None:
    rng = np.random.default_rng(1)
    phi = _conditioned_phi(rng, P=8, K=2)
    sigma2 = 0.5
    F = linearised_row_fisher(phi, sigma2)
    gram = phi.T @ phi
    np.testing.assert_allclose(F, (2.0 / sigma2) * gram, rtol=0.0, atol=1e-12)
    row = linearised_row_crlb(phi, sigma2)
    np.testing.assert_allclose(
        row.crlb, (sigma2 / 2.0) * np.linalg.solve(gram, np.eye(4)), atol=1e-12
    )
    np.testing.assert_allclose(F @ row.crlb, np.eye(4), atol=1e-11)
    src = inspect.getsource(linearised_row_crlb)
    assert "No prefactor is copied from Xu" in src
    assert "sigma2_val / 2.0" in src


# ---------------------------------------------------------------------------
# Rank / P >= 2K
# ---------------------------------------------------------------------------


def test_full_rank_p_ge_2k_finite_rank_deficient_raises() -> None:
    rng = np.random.default_rng(5)
    phi_ok = _conditioned_phi(rng, P=6, K=3)
    row = linearised_row_crlb(phi_ok, 0.2)
    assert np.all(np.isfinite(row.crlb))
    assert row.crlb.shape == (6, 6)
    phi_short = rng.standard_normal((5, 6))
    with pytest.raises(RankDeficientPhiError, match="P="):
        linearised_row_crlb(phi_short, 0.2)
    phi_wide_but_dup = np.concatenate([phi_ok[:3], phi_ok[:3]], axis=0)
    with pytest.raises(RankDeficientPhiError):
        linearised_row_crlb(phi_wide_but_dup, 0.2)


def test_no_silent_ridge_or_explicit_inverse() -> None:
    import rydberg_sim.linearised_crlb as lmod

    src = inspect.getsource(lmod)
    assert "np.linalg.inv" not in src
    assert "pinv" not in src
    assert "ridge=" not in inspect.getsource(linearised_row_crlb)
    assert "ridge =" not in inspect.getsource(linearised_row_crlb)


# ---------------------------------------------------------------------------
# B / Psi dependence
# ---------------------------------------------------------------------------


def test_crlb_depends_on_b_phase_not_magnitude() -> None:
    rng = np.random.default_rng(8)
    K, P, N = 2, 8, 3
    S = _full_rank_complex(rng, K, P)
    angles = rng.uniform(-np.pi, np.pi, size=(N, P))
    B = np.exp(1j * angles)
    a = linearised_channel_crlb(S, B, 0.4, beta_k=1.0)
    b = linearised_channel_crlb(S, 7.5 * B, 0.4, beta_k=1.0)
    np.testing.assert_allclose(a.crlb, b.crlb, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(a.Phi, b.Phi, rtol=0.0, atol=1e-14)
    B_other = np.exp(1j * (angles + 0.9))
    c = linearised_channel_crlb(S, B_other, 0.4, beta_k=1.0)
    assert not np.allclose(a.Phi, c.Phi)


def test_expected_energy_is_c2_n_sum_beta_not_trial_norm() -> None:
    energy = expected_channel_frobenius_energy(4, (1.0, 1.0, 1.0), c=1.0)
    assert energy == pytest.approx(12.0)
    energy_c = expected_channel_frobenius_energy(4, 2.0, c=0.5)
    assert energy_c == pytest.approx(2.0)
    rng = np.random.default_rng(3)
    S = _full_rank_complex(rng, 2, 8)
    B = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(5, 8)))
    result = linearised_channel_crlb(S, B, 0.1, beta_k=(1.0, 1.0), c=1.0)
    assert result.expected_channel_energy == pytest.approx(5 * 2.0)
    np.testing.assert_allclose(
        result.normalized_crlb,
        result.mse_bound / result.expected_channel_energy,
        rtol=0.0,
        atol=0.0,
    )


# ---------------------------------------------------------------------------
# Scaling, nested pilots, Kronecker sanity
# ---------------------------------------------------------------------------


def test_crlb_scales_exactly_with_sigma2() -> None:
    rng = np.random.default_rng(12)
    phi = _conditioned_phi(rng, P=9, K=2)
    s1, s2 = 0.2, 0.05
    r1 = linearised_row_crlb(phi, s1)
    r2 = linearised_row_crlb(phi, s2)
    np.testing.assert_allclose(r2.crlb / r1.crlb, s2 / s1, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(r2.trace / r1.trace, s2 / s1, rtol=0.0, atol=1e-12)
    print(f"\nSNR scaling: tr2/tr1={r2.trace / r1.trace:.12f} sigma2_2/sigma2_1={s2 / s1}")


def test_nested_pilots_do_not_reduce_information() -> None:
    rng = np.random.default_rng(15)
    K, P1, P2 = 2, 5, 9
    S = _full_rank_complex(rng, K, P2)
    B = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(1, P2)))
    sigma2 = 0.3
    row1 = linearised_channel_crlb(S[:, :P1], B[:, :P1], sigma2, beta_k=1.0)
    row2 = linearised_channel_crlb(S, B, sigma2, beta_k=1.0)
    F1, F2 = row1.F[0], row2.F[0]
    evals = np.linalg.eigvalsh(F2 - F1)
    print(
        f"\nnested pilots P={P1}->{P2}: min eig(F2-F1)={evals[0]:.6g}  "
        f"tr(CRLB1)={row1.mse_bound:.6g} tr(CRLB2)={row2.mse_bound:.6g}"
    )
    assert evals[0] >= -1e-10
    assert row2.mse_bound <= row1.mse_bound + 1e-12


def test_identical_phi_kronecker_sanity() -> None:
    rng = np.random.default_rng(6)
    K, P, N = 2, 8, 4
    S = _full_rank_complex(rng, K, P)
    psi_row = np.exp(-1j * rng.uniform(-np.pi, np.pi, size=P))
    B = np.broadcast_to(psi_row, (N, P)).copy()
    result = linearised_channel_crlb(S, B, 0.25, beta_k=1.0)
    for n in range(1, N):
        np.testing.assert_allclose(result.crlb[n], result.crlb[0], atol=1e-12)
        np.testing.assert_allclose(result.Phi[n], result.Phi[0], atol=1e-12)
    block = np.kron(np.eye(N), result.crlb[0])
    stacked = np.zeros((N * 4, N * 4))
    for n in range(N):
        stacked[n * 4 : (n + 1) * 4, n * 4 : (n + 1) * 4] = result.crlb[n]
    np.testing.assert_allclose(stacked, block, atol=1e-12)


# ---------------------------------------------------------------------------
# Ideal linear-Gaussian efficiency (most important)
# ---------------------------------------------------------------------------


def test_ideal_ls_covariance_and_mse_land_on_crlb() -> None:
    rng = np.random.default_rng(MASTER_SEED)
    K, P = 2, 8
    S = _full_rank_complex(rng, K, P)
    Psi = np.exp(-1j * rng.uniform(-np.pi, np.pi, size=(1, P)))
    G = _full_rank_complex(rng, 1, K)
    phi = linearised_design_matrix(Psi[0], S)
    sigma2 = 0.4
    n_mc = 12_000
    gtilde = pack_gtilde(G[0])
    row = linearised_row_crlb(phi, sigma2)
    errors = np.empty((n_mc, 2 * K), dtype=np.float64)
    for t in range(n_mc):
        Y = make_ideal_linear_y(G, S, Psi, sigma2=sigma2, rng=rng)
        est = linearised_closed_form_ls(Y, S, Psi, observation_source="ideal_linear")
        errors[t] = est.gtilde_hat[0] - gtilde
        np.testing.assert_allclose(est.Phi[0], phi, rtol=0.0, atol=1e-14)
    emp_cov = (errors.T @ errors) / n_mc
    emp_mse = float(np.mean(np.sum(errors**2, axis=1)))
    rel_f = float(
        np.linalg.norm(emp_cov - row.crlb, ord="fro") / np.linalg.norm(row.crlb, ord="fro")
    )
    rel_mse = abs(emp_mse - row.trace) / row.trace
    gap_db = 10.0 * np.log10(emp_mse / row.trace)
    print(
        f"\nideal LS vs CRLB (n_mc={n_mc}): "
        f"||Cov-CRLB||_F/||CRLB||={rel_f:.4f}  "
        f"MSE={emp_mse:.6g} tr(CRLB)={row.trace:.6g} rel={rel_mse:.4f}  "
        f"10 log10(MSE/tr)={gap_db:.4f} dB"
    )
    assert rel_f < 0.12
    assert rel_mse < 0.08
    assert abs(gap_db) < 0.4
    assert abs(gap_db - 3.0103) > 2.0
    assert abs(gap_db + 3.0103) > 2.0


def test_whole_channel_mse_and_normalized_nmse() -> None:
    rng = np.random.default_rng(MASTER_SEED + 1)
    N, K, P = 4, 2, 8
    G = _full_rank_complex(rng, N, K)
    S = _full_rank_complex(rng, K, P)
    B = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(N, P)))
    Psi = reference_phase_matrix(B)
    sigma2 = 0.25
    beta_k = (1.0, 1.0)
    c = 1.0
    bound = linearised_channel_crlb(S, B, sigma2, beta_k=beta_k, c=c)
    n_mc = 6_000
    mse = np.empty(n_mc, dtype=np.float64)
    for t in range(n_mc):
        Y = make_ideal_linear_y(G, S, Psi, sigma2=sigma2, rng=rng)
        est = linearised_closed_form_ls(Y, S, Psi, observation_source="ideal_linear")
        mse[t] = float(np.linalg.norm(est.G_hat - G, ord="fro") ** 2)
    emp_mse = float(np.mean(mse))
    energy = expected_channel_frobenius_energy(N, beta_k, c=c)
    emp_nmse = emp_mse / energy
    rel = abs(emp_mse - bound.mse_bound) / bound.mse_bound
    gap_db = 10.0 * np.log10(emp_mse / bound.mse_bound)
    print(
        f"\nwhole-channel n_mc={n_mc}: emp MSE={emp_mse:.6g}  "
        f"Σ tr(CRLB)={bound.mse_bound:.6g} rel={rel:.4f}  "
        f"emp NMSE={emp_nmse:.6g}  norm CRLB={bound.normalized_crlb:.6g}  "
        f"10 log10(emp/bound)={gap_db:.4f} dB  E||G||_F^2={energy}"
    )
    assert energy == pytest.approx(N * K)
    assert rel < 0.08
    np.testing.assert_allclose(emp_nmse, bound.normalized_crlb, rtol=0.08)
    assert abs(gap_db) < 0.4


def test_crlb_independent_of_true_g() -> None:
    rng = np.random.default_rng(10)
    S = _full_rank_complex(rng, 2, 8)
    B = np.exp(1j * rng.uniform(-np.pi, np.pi, size=(3, 8)))
    a = linearised_channel_crlb(S, B, 0.2, beta_k=1.0)
    assert a.Phi.shape[0] == 3
    assert a.crlb.shape == (3, 4, 4)


def test_step11_cui_crlb_still_separate() -> None:
    from rydberg_sim.crlb import cui_crlb, fisher_beta, high_snr_fisher_beta_limit

    beta = float(np.asarray(fisher_beta(1.0, 1e-4)))
    ratio = beta / high_snr_fisher_beta_limit(1e-4)
    assert ratio == pytest.approx(1.0, rel=2e-4)
    rng = np.random.default_rng(1)
    M = rng.standard_normal((2, 10)) + 1j * rng.standard_normal((2, 10))
    u = rng.standard_normal(2) + 1j * rng.standard_normal(2)
    b = np.zeros(10, dtype=np.complex128)
    cui = cui_crlb(M, u, b, 1e-4)
    assert cui.crlb.shape == (2, 2)


def test_step13_plus_not_implemented() -> None:
    import rydberg_sim.baselines as bmod
    import rydberg_sim.crlb as cmod
    import rydberg_sim.gs as gmod
    import rydberg_sim.linearised_crlb as lmod

    assert hasattr(lmod, "linearised_channel_crlb")
    assert hasattr(lmod, "linearised_row_crlb")
    assert not hasattr(lmod, "xu_crlb")
    assert not hasattr(lmod, "xu_gd")
    assert not hasattr(lmod, "monte_carlo_harness")
    assert not hasattr(lmod, "metrics")
    assert not hasattr(gmod, "xu_gd")
    assert not hasattr(bmod, "xu_crlb")
    assert not hasattr(cmod, "xu_crlb")
    src = inspect.getsource(lmod)
    assert "No prefactor is copied from Xu" in src
    assert "Step 13+" in src

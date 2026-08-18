"""Step 11 acceptance tests: Cui CRLB for the magnitude-only Rician model."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from rydberg_sim import (
    SimulationConfig,
    cui_crlb,
    cui_crlb_high_snr_limit,
    cui_fisher_information,
    fisher_beta,
    fisher_expectation_z2_r2,
    generate_gaussian_pilots,
    generate_reference_field,
    generate_ula_channel,
    high_snr_fisher_beta_limit,
    make_alpha_b,
    rician_amplitude_pdf,
    rsr_db_to_alpha_magnitude,
)
from rydberg_sim.baselines import log_bessel_i0
from rydberg_sim.crlb import (
    QUAD_TAIL_SIGMAS,
    SingularFisherError,
    rician_fisher_scalar,
)
from rydberg_sim.gs import bessel_ratio

MASTER_SEED = 20260818
DB10_LOG10_2 = 10.0 * np.log10(2.0)


def _full_rank_M(rng: np.random.Generator, D: int, Q: int) -> np.ndarray:
    M = (rng.standard_normal((D, Q)) + 1j * rng.standard_normal((D, Q))).astype(
        np.complex128
    )
    s = np.linalg.svd(M, compute_uv=False)
    assert s[-1] / s[0] > 1e-6
    return M


def _rel_frob(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, ord="fro") / np.linalg.norm(b, ord="fro"))


# ---------------------------------------------------------------------------
# Density, Bessel reuse, quadrature sanity
# ---------------------------------------------------------------------------


def test_rician_pdf_matches_step7_i0e_identity() -> None:
    z = np.linspace(0.0, 4.0, 81)
    abs_lam = 1.25
    sigma2 = 0.4
    p = rician_amplitude_pdf(z, abs_lam, sigma2)
    # Reconstruct from log I0 = log(i0e) + |x|  (Step 7).
    pos = z > 0
    kappa = 2.0 * z[pos] * abs_lam / sigma2
    log_p = (
        np.log(2.0 * z[pos] / sigma2)
        - (z[pos] ** 2 + abs_lam**2) / sigma2
        + log_bessel_i0(kappa)
    )
    np.testing.assert_allclose(p[pos], np.exp(log_p), rtol=1e-10, atol=1e-12)
    assert float(np.asarray(rician_amplitude_pdf(0.0, abs_lam, sigma2))) == 0.0
    mass = rician_fisher_scalar(abs_lam, sigma2).pdf_mass
    assert mass == pytest.approx(1.0, abs=1e-10)


def test_reuses_step10_bessel_ratio_not_a_second_copy() -> None:
    import rydberg_sim.crlb as cmod

    assert cmod.bessel_ratio is bessel_ratio
    src = inspect.getsource(cmod)
    assert "def bessel_ratio" not in src
    assert "i1(" not in src


def test_fisher_beta_finite_and_nonnegative_typical() -> None:
    for abs_lam in (0.25, 0.5, 1.0, 2.0):
        for sigma2 in (1.0, 0.2, 0.05):
            term = rician_fisher_scalar(abs_lam, sigma2)
            assert np.isfinite(term.expectation_z2_r2)
            assert np.isfinite(term.beta)
            assert term.expectation_z2_r2 > 0.0
            assert term.beta > 0.0
            assert term.pdf_mass == pytest.approx(1.0, abs=1e-9)
            print(
                f"|λ|={abs_lam:g} σ²={sigma2:g}  E={term.expectation_z2_r2:.6f}  "
                f"β={term.beta:.6f}  mass={term.pdf_mass:.12f}"
            )


def test_quadrature_versus_monte_carlo_expectation() -> None:
    rng = np.random.default_rng(MASTER_SEED)
    n_mc = 120_000
    settings = ((1.0, 0.5), (2.0, 0.2), (0.5, 1.0), (0.0, 0.4))
    for abs_lam, sigma2 in settings:
        eq = float(np.asarray(fisher_expectation_z2_r2(abs_lam, sigma2)))
        scale = np.sqrt(sigma2 / 2.0)
        w = scale * (rng.standard_normal(n_mc) + 1j * rng.standard_normal(n_mc))
        z = np.abs(abs_lam + w)
        kappa = (2.0 * z * abs_lam) / sigma2
        r = np.asarray(bessel_ratio(kappa), dtype=np.float64)
        samples = z * z * r * r
        mc = float(np.mean(samples))
        se = float(np.std(samples, ddof=1) / np.sqrt(n_mc))
        print(
            f"MC |λ|={abs_lam:g} σ²={sigma2:g}: quad={eq:.6f} mc={mc:.6f} se={se:.4g} "
            f"|quad-mc|={abs(eq - mc):.4g}"
        )
        assert abs(eq - mc) < max(5.0 * se, 1e-10)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_small_lambda_finite_no_nan() -> None:
    for abs_lam in (0.0, 1e-16, 1e-12, 1e-8):
        for sigma2 in (1.0, 1e-2, 1e-6):
            term = rician_fisher_scalar(abs_lam, sigma2)
            assert np.isfinite(term.beta)
            assert np.isfinite(term.expectation_z2_r2)
            assert np.isfinite(term.pdf_mass)
            assert term.beta >= 0.0
    zero = rician_fisher_scalar(0.0, 0.3)
    assert zero.expectation_z2_r2 == pytest.approx(0.0, abs=1e-15)
    assert zero.beta == pytest.approx(0.0, abs=1e-15)
    print(f"\n|λ|=0: E={zero.expectation_z2_r2} β={zero.beta}")


def test_large_lambda_high_snr_stable_positive() -> None:
    abs_lam = 10.0
    sigma2 = 1e-8
    term = rician_fisher_scalar(abs_lam, sigma2)
    limit = high_snr_fisher_beta_limit(sigma2)
    ratio = term.beta / limit
    print(
        f"\nlarge |λ|={abs_lam} σ²={sigma2:g}: β={term.beta:.8g}  "
        f"1/(2σ²)={limit:.8g}  ratio={ratio:.10f}  mass={term.pdf_mass:.16f}"
    )
    assert np.isfinite(term.beta) and np.isfinite(term.expectation_z2_r2)
    assert term.beta > 0.0
    assert ratio == pytest.approx(1.0, rel=1e-5)


def test_beta_not_clipped_to_limit() -> None:
    """At moderate SNR β is *below* the high-SNR limit; do not force it."""
    sigma2 = 1.0
    beta = float(np.asarray(fisher_beta(0.5, sigma2)))
    limit = high_snr_fisher_beta_limit(sigma2)
    assert beta < 0.5 * limit
    assert beta > 0.0


# ---------------------------------------------------------------------------
# High-SNR β → 1/(2 σ²)   (factor-of-two convention check)
# ---------------------------------------------------------------------------


def test_high_snr_beta_approaches_one_over_two_sigma2() -> None:
    rows = []
    for abs_lam in (0.5, 1.0, 2.0):
        ratios = []
        for sigma2 in (1e-2, 1e-4, 1e-6):
            beta = float(np.asarray(fisher_beta(abs_lam, sigma2)))
            limit = high_snr_fisher_beta_limit(sigma2)
            ratio = beta / limit
            ratios.append(ratio)
            rows.append((abs_lam, sigma2, beta, limit, ratio))
            print(
                f"|λ|={abs_lam:g} σ²={sigma2:.0e}  β={beta:.8g}  "
                f"1/(2σ²)={limit:.8g}  β/(1/(2σ²))={ratio:.10f}"
            )
        assert ratios[0] > 0.98
        assert ratios[1] == pytest.approx(1.0, rel=2e-4)
        assert ratios[2] == pytest.approx(1.0, rel=2e-5)
        assert ratios[0] < ratios[1] <= ratios[2] + 1e-8
    # A factor-2 convention bug would land near 0.5 or 2.0.
    for _, _, _, _, ratio in rows:
        assert 0.97 < ratio < 1.03


def test_sigma2_squared_is_sigma_to_the_fourth() -> None:
    """β uses 1/σ⁴ = 1/(sigma2)**2, not 1/sigma2."""
    abs_lam, sigma2 = 1.0, 1e-4
    beta = float(np.asarray(fisher_beta(abs_lam, sigma2)))
    eq = float(np.asarray(fisher_expectation_z2_r2(abs_lam, sigma2)))
    reconstructed = (eq - abs_lam**2) / (sigma2**2)
    # Difference-integrand β vs post-subtraction; they must agree closely.
    np.testing.assert_allclose(beta, reconstructed, rtol=1e-8, atol=1e-6)
    wrong_over_sigma2 = (eq - abs_lam**2) / sigma2
    assert not np.isclose(beta, wrong_over_sigma2, rtol=0.1)


# ---------------------------------------------------------------------------
# Fisher matrix / CRLB
# ---------------------------------------------------------------------------


def test_fisher_is_hermitian_pd_for_full_rank_M() -> None:
    rng = np.random.default_rng(7)
    D, Q = 3, 24
    M = _full_rank_M(rng, D, Q)
    u = (rng.standard_normal(D) + 1j * rng.standard_normal(D)).astype(np.complex128)
    u /= np.linalg.norm(u)
    b = 0.8 * np.exp(1j * rng.uniform(-np.pi, np.pi, size=Q))
    fim = cui_fisher_information(M, u, b, 0.15)
    F = fim.F
    np.testing.assert_allclose(F, F.conj().T, rtol=0.0, atol=1e-14)
    evals = np.linalg.eigh(F)[0]
    assert evals[0] > 0.0
    assert fim.beta.shape == (Q,)
    assert np.all(fim.beta > 0.0)
    assert np.all(np.isfinite(fim.beta))


def test_canonical_api_has_no_channel_or_qam_special_case() -> None:
    for fn in (cui_fisher_information, cui_crlb):
        params = inspect.signature(fn).parameters
        assert "S" not in params
        assert "G" not in params
        assert "constellation" not in params
        assert list(params)[:4] == ["M", "u", "b", "sigma2"]


def test_crlb_default_energy_is_D_not_empirical_norm() -> None:
    rng = np.random.default_rng(3)
    D, Q = 4, 20
    M = _full_rank_M(rng, D, Q)
    u = 3.0 * np.ones(D, dtype=np.complex128)  # ||u||² = 36, not D
    b = np.zeros(Q, dtype=np.complex128)
    result = cui_crlb(M, u, b, 0.2)
    assert result.expected_u_energy == float(D)
    np.testing.assert_allclose(
        result.normalized_crlb,
        np.trace(result.crlb).real / D,
        rtol=0.0,
        atol=0.0,
    )
    other = cui_crlb(M, u, b, 0.2, expected_u_energy=2.5)
    np.testing.assert_allclose(
        other.normalized_crlb,
        np.trace(other.crlb).real / 2.5,
        rtol=0.0,
        atol=0.0,
    )


def test_singular_fisher_raises_no_ridge() -> None:
    rng = np.random.default_rng(1)
    # Rank-1 dictionary cannot identify D=2.
    M = rng.standard_normal((2, 1)) + 1j * rng.standard_normal((2, 1))
    u = np.array([1.0 + 0j, 0.0])
    b = np.array([0.3 + 0.1j])
    with pytest.raises(SingularFisherError):
        cui_crlb(M, u, b, 0.5)


def test_high_snr_crlb_matrix_limit() -> None:
    rng = np.random.default_rng(195)
    D, Q = 3, 32
    M = _full_rank_M(rng, D, Q)
    u = (rng.standard_normal(D) + 1j * rng.standard_normal(D)).astype(np.complex128)
    u /= np.linalg.norm(u)
    b = 1.2 * np.exp(1j * rng.uniform(-np.pi, np.pi, size=Q))
    sigma2 = 1e-6
    result = cui_crlb(M, u, b, sigma2)
    crlb_limit = cui_crlb_high_snr_limit(M, sigma2)
    F_limit = (1.0 / (2.0 * sigma2)) * (M @ M.conj().T)
    F_limit = 0.5 * (F_limit + F_limit.conj().T)
    rel_f = _rel_frob(result.F, F_limit)
    rel_c = _rel_frob(result.crlb, crlb_limit)
    mean_ratio = float(np.mean(result.beta / high_snr_fisher_beta_limit(sigma2)))
    print(
        f"\nhigh-SNR CRLB σ²={sigma2:g}: "
        f"||F-F_lim||_F/||F_lim||={rel_f:.6g}  "
        f"||C-C_lim||_F/||C_lim||={rel_c:.6g}  "
        f"mean β/(1/(2σ²))={mean_ratio:.8f}"
    )
    assert rel_f < 1e-4
    assert rel_c < 1e-4
    assert mean_ratio == pytest.approx(1.0, rel=1e-4)


def test_high_snr_crlb_vs_zf_is_3db() -> None:
    rng = np.random.default_rng(11)
    D, Q = 3, 28
    M = _full_rank_M(rng, D, Q)
    u = (rng.standard_normal(D) + 1j * rng.standard_normal(D)).astype(np.complex128)
    u /= np.linalg.norm(u)
    b = np.exp(1j * rng.uniform(-np.pi, np.pi, size=Q))
    sigma2 = 1e-6
    result = cui_crlb(M, u, b, sigma2)
    gram = M @ M.conj().T
    cov_zf = sigma2 * np.linalg.solve(gram, np.eye(D, dtype=np.complex128))
    gap_db = 10.0 * np.log10(
        float(np.trace(result.crlb).real / np.trace(cov_zf).real)
    )
    print(
        f"\nCRLB vs ZF-known-phase: gap={gap_db:.6f} dB  "
        f"(10 log10 2 = {DB10_LOG10_2:.6f} dB)"
    )
    assert gap_db == pytest.approx(DB10_LOG10_2, abs=0.02)
    # 0 dB or 6 dB would mean a real-vs-complex Fisher factor bug.
    assert abs(gap_db - 0.0) > 1.0
    assert abs(gap_db - 6.0206) > 1.0


def test_zf_below_crlb_is_documented() -> None:
    import rydberg_sim.baselines as bmod
    import rydberg_sim.crlb as cmod

    assert "below Cui's CRLB" in bmod.zf_known_phase.__doc__
    assert "allowed" in cmod.__doc__.lower()
    assert "3.0103" in cmod.__doc__


# ---------------------------------------------------------------------------
# Channel-row adapter identity (not Xu CRLB)
# ---------------------------------------------------------------------------


def test_channel_row_lambda_identity_canonical_fisher_runs() -> None:
    """Canonical λ magnitudes match the physical row; this is not Xu CRLB."""
    N, K, P = 4, 2, 8
    cfg = SimulationConfig.create(
        N=N, K=K, L=2, beta=1.0, master_seed=MASTER_SEED, c=1.0
    )
    G = generate_ula_channel(cfg, 3).G
    S = generate_gaussian_pilots(K=K, P=P, master_seed=MASTER_SEED, trial_index=3).S
    B = generate_reference_field(
        N=N,
        P=P,
        alpha_b=make_alpha_b(rsr_db_to_alpha_magnitude(0.0, beta_ref=1.0)),
        vartheta=0.2,
        c=1.0,
    ).B
    n = 1
    np.testing.assert_allclose(
        np.abs(S.conj().T @ np.conjugate(G[n]) + np.conjugate(B[n])),
        np.abs(S.T @ G[n] + B[n]),
        rtol=0.0,
        atol=1e-14,
    )
    fim = cui_fisher_information(S, np.conjugate(G[n]), np.conjugate(B[n]), 0.25)
    assert fim.F.shape == (K, K)
    np.testing.assert_allclose(fim.F, fim.F.conj().T, atol=1e-14)
    assert np.linalg.eigh(fim.F)[0][0] > 0.0


def test_step12_plus_not_implemented() -> None:
    import rydberg_sim.baselines as bmod
    import rydberg_sim.crlb as cmod
    import rydberg_sim.gs as gmod
    import rydberg_sim.spectral as smod

    assert hasattr(cmod, "cui_crlb")
    assert hasattr(cmod, "fisher_beta")
    assert not hasattr(cmod, "xu_crlb")
    assert not hasattr(cmod, "xu_fisher")
    assert not hasattr(cmod, "xu_gd")
    assert not hasattr(gmod, "crlb")
    assert not hasattr(gmod, "xu_gd")
    assert not hasattr(smod, "cui_crlb")
    assert not hasattr(bmod, "cui_crlb")
    assert not hasattr(bmod, "xu_crlb")
    src = inspect.getsource(cmod)
    assert "xu_crlb" not in src.lower() or "do not import or copy xu" in src.lower()


def test_requires_positive_sigma2() -> None:
    with pytest.raises(ValueError, match="sigma2"):
        fisher_beta(1.0, 0.0)
    with pytest.raises(ValueError, match="sigma2"):
        fisher_beta(1.0, -0.2)


def test_quadrature_window_uses_sqrt_sigma2() -> None:
    """σ = sqrt(σ²); do not add 8*sigma2 to |λ|."""
    abs_lam, sigma2 = 1.0, 0.04
    term = rician_fisher_scalar(abs_lam, sigma2)
    sigma = np.sqrt(sigma2)
    assert term.z_high == pytest.approx(abs_lam + QUAD_TAIL_SIGMAS * sigma)
    assert term.z_high != pytest.approx(abs_lam + QUAD_TAIL_SIGMAS * sigma2)

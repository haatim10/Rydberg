"""Step 8 acceptance tests: canonical Cui spectral initialization."""

from __future__ import annotations

import numpy as np
import pytest

from rydberg_sim import (
    SimulationConfig,
    exact_forward,
    generate_gaussian_pilots,
    generate_reference_field,
    generate_ula_channel,
    make_alpha_b,
    rsr_db_to_alpha_magnitude,
    snr_db_to_sigma2,
    spectral_initialize,
    spectral_initialize_channel_rows,
)
from rydberg_sim.spectral import (
    FUTURE_GS_SPECTRAL_VS_RANDOM_TEST,
    build_augmented_dictionary,
    scale_and_anchor,
    spectral_matrix_from_columns,
)

MASTER_SEED = 20260818


def _full_rank_M(rng: np.random.Generator, D: int, Q: int) -> np.ndarray:
    M = (rng.standard_normal((D, Q)) + 1j * rng.standard_normal((D, Q))).astype(
        np.complex128
    )
    s = np.linalg.svd(M, compute_uv=False)
    assert s[-1] / s[0] > 1e-6
    return M


def _canonical_draw(rng: np.random.Generator, D: int = 3, Q: int = 8):
    M = _full_rank_M(rng, D, Q)
    u = (rng.standard_normal(D) + 1j * rng.standard_normal(D)).astype(np.complex128)
    b = (rng.standard_normal(Q) + 1j * rng.standard_normal(Q)).astype(np.complex128)
    return M, u, b


# ---------------------------------------------------------------------------
# Acceptance tests 1–6
# ---------------------------------------------------------------------------


def test_augmentation_identity() -> None:
    """MbarH @ [u; 1] == M^H u + b to floating-point precision."""
    rng = np.random.default_rng(1)
    M, u, b = _canonical_draw(rng, D=3, Q=8)
    MbarH, Mbar = build_augmented_dictionary(M, b)
    D, Q = M.shape
    assert MbarH.shape == (Q, D + 1)
    assert Mbar.shape == (D + 1, Q)
    np.testing.assert_allclose(Mbar, MbarH.conj().T, rtol=0.0, atol=0.0)

    ubar_true = np.concatenate([u, np.array([1.0 + 0.0j])])
    lhs = MbarH @ ubar_true
    rhs = M.conj().T @ u + b
    np.testing.assert_allclose(lhs, rhs, rtol=0.0, atol=1e-15)


def test_m_spec_direct_vs_vectorized() -> None:
    """Explicit Σ_q z_q outer(mbar_q, conj(mbar_q)) matches production."""
    rng = np.random.default_rng(2)
    M, u, b = _canonical_draw(rng, D=4, Q=10)
    w = 0.2 * (rng.standard_normal(10) + 1j * rng.standard_normal(10))
    z = np.abs(M.conj().T @ u + b + w)
    result = spectral_initialize(M, z, b)
    looped = spectral_matrix_from_columns(result.Mbar, z)
    looped_h = 0.5 * (looped + looped.conj().T)
    np.testing.assert_allclose(result.M_spec, looped_h, rtol=0.0, atol=1e-12)
    # Using z^2 instead of z must not match (guards the common bug).
    looped_sq = spectral_matrix_from_columns(result.Mbar, z**2)
    assert not np.allclose(
        result.M_spec, 0.5 * (looped_sq + looped_sq.conj().T), rtol=1e-6
    )


def test_m_spec_hermitian_and_real_eigenvalues() -> None:
    rng = np.random.default_rng(3)
    M, u, b = _canonical_draw(rng)
    z = np.abs(M.conj().T @ u + b)
    result = spectral_initialize(M, z, b)
    np.testing.assert_allclose(
        result.M_spec, result.M_spec.conj().T, rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        result.eigenvalues, np.real(result.eigenvalues), rtol=0.0, atol=1e-14
    )
    imag_evals = np.linalg.eigvals(result.M_spec).imag
    np.testing.assert_allclose(imag_evals, 0.0, rtol=0.0, atol=1e-12)


def test_principal_eigenvector() -> None:
    rng = np.random.default_rng(4)
    M, u, b = _canonical_draw(rng, D=5, Q=12)
    z = np.abs(M.conj().T @ u + b)
    result = spectral_initialize(M, z, b)
    assert result.v.shape == (M.shape[0] + 1,)
    assert result.principal_eigenvalue == pytest.approx(result.eigenvalues[-1])
    assert result.eigenvalues[-1] == pytest.approx(np.max(result.eigenvalues))
    np.testing.assert_allclose(
        result.M_spec @ result.v,
        result.principal_eigenvalue * result.v,
        rtol=1e-10,
        atol=1e-10,
    )


def test_rbar_formula() -> None:
    rng = np.random.default_rng(5)
    M, u, b = _canonical_draw(rng)
    z = np.abs(M.conj().T @ u + b)
    result = spectral_initialize(M, z, b)
    proj = np.abs(result.MbarH @ result.v)
    expected = float(np.sum(proj * z) / np.sum(proj**2))
    np.testing.assert_allclose(result.rbar, expected, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(result.projection, proj, rtol=0.0, atol=1e-15)


def test_phase_anchor_and_eigensolver_invariance() -> None:
    rng = np.random.default_rng(6)
    M, u, b = _canonical_draw(rng, D=3, Q=9)
    z = np.abs(M.conj().T @ u + b)
    result = spectral_initialize(M, z, b)
    last = result.ubar0_anchored[-1]
    np.testing.assert_allclose(last.imag, 0.0, rtol=0.0, atol=1e-15)
    assert last.real >= -1e-15
    np.testing.assert_allclose(
        result.u0, result.ubar0_anchored[:-1], rtol=0.0, atol=0.0
    )

    phi = 1.7
    v2 = result.v * np.exp(1j * phi)
    _, _, _, anchored2, u0_2 = scale_and_anchor(result.MbarH, v2, z)
    np.testing.assert_allclose(u0_2, result.u0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(anchored2[-1].imag, 0.0, rtol=0.0, atol=1e-15)
    assert anchored2[-1].real >= -1e-15


def test_rejects_bad_shapes_and_nonfinite() -> None:
    M = np.eye(3, 5, dtype=np.complex128)
    z = np.ones(5)
    b = np.zeros(5, dtype=np.complex128)
    with pytest.raises(ValueError, match="length 5"):
        spectral_initialize(M, z[:4], b)
    with pytest.raises(ValueError, match="M must be a 2-D"):
        spectral_initialize(np.ones(3), z, b)
    z_nan = z.copy()
    z_nan[0] = np.nan
    with pytest.raises(ValueError, match="z must be finite"):
        spectral_initialize(M, z_nan, b)
    with pytest.raises(ValueError, match="nonnegative"):
        spectral_initialize(M, -z, b)


def test_dimensions_one_case() -> None:
    """Recorded production shapes for the report: D=3, Q=8."""
    rng = np.random.default_rng(7)
    D, Q = 3, 8
    M, u, b = _canonical_draw(rng, D=D, Q=Q)
    z = np.abs(M.conj().T @ u + b)
    result = spectral_initialize(M, z, b)
    assert M.shape == (D, Q)
    assert z.shape == (Q,)
    assert b.shape == (Q,)
    assert result.MbarH.shape == (Q, D + 1)
    assert result.Mbar.shape == (D + 1, Q)
    assert result.M_spec.shape == (D + 1, D + 1)
    assert result.v.shape == (D + 1,)
    assert result.u0.shape == (D,)
    assert result.projection.shape == (Q,)
    assert result.ubar0.shape == (D + 1,)
    assert result.ubar0_anchored.shape == (D + 1,)


# ---------------------------------------------------------------------------
# Acceptance test 7 — high SNR / high RSR sanity
# ---------------------------------------------------------------------------


def test_high_snr_high_rsr_relative_error() -> None:
    """||u0 - u_true|| / ||u_true|| < 0.5 (weak sanity criterion).

    Spectral init recovers the direction of ``ubar = [u; 1]``. That only
    stays well-conditioned when the known coefficient 1 is comparable to
    ``||u||`` (QAM-like unknown) *and* the reference is strong but not so
    large that every column of ``Mbar`` collapses onto the last axis.
    Extreme RSR (``|b| >> |M^H u|``) makes ``u0 ≈ 0``; that is expected
    for this initializer and is not a conjugate/``z`` vs ``z^2`` bug.
    """
    rng = np.random.default_rng(195)
    D, Q = 3, 128
    M = _full_rank_M(rng, D, Q)
    u_true = (rng.standard_normal(D) + 1j * rng.standard_normal(D)).astype(
        np.complex128
    )
    u_true = u_true / np.linalg.norm(u_true)
    signal = M.conj().T @ u_true
    # Strong nonzero reference: |b| on the same order as |M^H u| (RSR ~ 0 dB).
    bmag = 1.5
    b = bmag * np.exp(1j * rng.uniform(-np.pi, np.pi, size=Q))
    snr_lin = 10.0 ** (40.0 / 10.0)
    sigma2 = float(np.mean(np.abs(signal) ** 2) / snr_lin)
    scale = np.sqrt(sigma2 / 2.0)
    w = scale * (rng.standard_normal(Q) + 1j * rng.standard_normal(Q))
    z = np.abs(signal + b + w)
    result = spectral_initialize(M, z, b)
    rel = float(np.linalg.norm(result.u0 - u_true) / np.linalg.norm(u_true))
    rsr_db = 10.0 * np.log10(float(np.mean(np.abs(b) ** 2) / np.mean(np.abs(signal) ** 2)))
    print(
        f"\ncanonical high-SNR sanity: rel={rel:.6g}  RSR={rsr_db:.2f} dB  "
        f"SNR=40 dB  D={D} Q={Q}"
    )
    assert rel < 0.5, rel


# ---------------------------------------------------------------------------
# Channel-estimation adapter
# ---------------------------------------------------------------------------


def test_channel_adapter_high_snr_high_rsr() -> None:
    """M=S, u=conj(g_n), b_solver=conj(b_n); g0=conj(u0).

    Physical row: ``z_n = |S^T g_n + b_n + w_n|``. The conjugation belongs
    only in this adapter. Canonical ``spectral_initialize`` is unchanged.

    The well-scaled canonical sanity instance is remapped through the
    adapter so the 0.5 criterion tests the conjugation, not a new draw.
    """
    rng = np.random.default_rng(195)
    D, Q = 3, 128
    M = _full_rank_M(rng, D, Q)
    u_true = (rng.standard_normal(D) + 1j * rng.standard_normal(D)).astype(
        np.complex128
    )
    u_true = u_true / np.linalg.norm(u_true)
    signal_can = M.conj().T @ u_true
    b_can = 1.5 * np.exp(1j * rng.uniform(-np.pi, np.pi, size=Q))
    snr_lin = 10.0 ** (40.0 / 10.0)
    sigma2 = float(np.mean(np.abs(signal_can) ** 2) / snr_lin)
    scale = np.sqrt(sigma2 / 2.0)
    w_can = scale * (rng.standard_normal(Q) + 1j * rng.standard_normal(Q))
    z = np.abs(signal_can + b_can + w_can)

    # Inverse of the adapter map: S = M, g = conj(u), b_phys = conj(b_solver).
    S = M
    g_true = np.conjugate(u_true)
    b_phys = np.conjugate(b_can)
    # Physical identity: |S.T g + b_phys| == |M^H u + b_can|.
    np.testing.assert_allclose(
        np.abs(S.T @ g_true + b_phys),
        np.abs(M.conj().T @ u_true + b_can),
        rtol=0.0,
        atol=1e-14,
    )
    Z = z.reshape(1, -1)
    B = b_phys.reshape(1, -1)
    adapted = spectral_initialize_channel_rows(S, Z, B)
    g0 = adapted.G0[0]
    rel = float(np.linalg.norm(g0 - g_true) / np.linalg.norm(g_true))
    print(f"\nchannel-adapter high-SNR relative error (one row): {rel:.6g}")
    assert rel < 0.5, rel
    np.testing.assert_allclose(
        g0, np.conjugate(adapted.row_results[0].u0), rtol=0.0, atol=0.0
    )
    direct = spectral_initialize(S, z, np.conjugate(b_phys))
    np.testing.assert_allclose(np.conjugate(direct.u0), g0, rtol=0.0, atol=1e-15)


def test_channel_adapter_distinct_m_spec_across_elements() -> None:
    """M=S is shared; z_n and b_n differ, so each n has its own M_spec."""
    rng = np.random.default_rng(21)
    N, K, P = 4, 3, 16
    S = _full_rank_M(rng, K, P)
    G = (rng.standard_normal((N, K)) + 1j * rng.standard_normal((N, K))).astype(
        np.complex128
    )
    B = (rng.standard_normal((N, P)) + 1j * rng.standard_normal((N, P))).astype(
        np.complex128
    )
    Z = np.abs(G @ S + B)
    adapted = spectral_initialize_channel_rows(S, Z, B)
    specs = [row.M_spec for row in adapted.row_results]
    assert specs[0].shape == (K + 1, K + 1)
    n_distinct = sum(
        1
        for i in range(1, N)
        if not np.allclose(specs[i], specs[0], rtol=1e-12, atol=1e-12)
    )
    assert n_distinct == N - 1
    # Wrapper is a loop of canonical calls, not one cached init.
    for n in range(N):
        row = spectral_initialize(S, Z[n], np.conjugate(B[n]))
        np.testing.assert_allclose(adapted.G0[n], np.conjugate(row.u0), rtol=0.0, atol=0.0)


def test_channel_adapter_ula_high_rsr_reports_error() -> None:
    """ULA + Step-6 RSR=30 dB: adapter still builds N distinct M_spec.

    Accuracy is not asserted here. At this RSR the columns of ``Mbar``
    align with the reference axis, so spectral init alone is a weak
    ``G`` estimate (GS in Step 9 is what refines it).
    """
    N, K, P = 8, 3, 12
    cfg = SimulationConfig.create(
        N=N, K=K, L=3, beta=1.0, master_seed=MASTER_SEED, c=1.0
    )
    trial = 4
    G = generate_ula_channel(cfg, trial).G
    S = generate_gaussian_pilots(K=K, P=P, master_seed=MASTER_SEED, trial_index=trial).S
    alpha_b = make_alpha_b(rsr_db_to_alpha_magnitude(30.0, beta_ref=1.0))
    B = generate_reference_field(N=N, P=P, alpha_b=alpha_b, vartheta=0.3, c=1.0).B
    sigma2 = snr_db_to_sigma2(40.0, cfg.beta_k, c=cfg.c)
    exact = exact_forward(
        G, S, B, sigma2, master_seed=MASTER_SEED, trial_index=trial
    )
    adapted = spectral_initialize_channel_rows(S, exact.Z, exact.B)
    rel = float(
        np.linalg.norm(adapted.G0 - G, ord="fro") / np.linalg.norm(G, ord="fro")
    )
    print(f"\nULA adapter RSR=30 dB SNR=40 dB relative Frobenius error: {rel:.6g}")
    assert adapted.G0.shape == (N, K)
    assert not np.allclose(
        adapted.row_results[0].M_spec,
        adapted.row_results[1].M_spec,
        rtol=1e-12,
        atol=1e-12,
    )


def test_canonical_initializer_has_no_channel_special_case() -> None:
    """spectral_initialize does not take S, G, or a model= flag."""
    import inspect

    from rydberg_sim.spectral import spectral_initialize as fn

    params = inspect.signature(fn).parameters
    assert "S" not in params
    assert "G" not in params
    assert "model" not in params
    assert list(params)[:3] == ["M", "z", "b"]


def test_future_gs_comparison_is_documented_not_implemented() -> None:
    assert "SNR = -5 dB" in FUTURE_GS_SPECTRAL_VS_RANDOM_TEST
    assert "Step 9" in FUTURE_GS_SPECTRAL_VS_RANDOM_TEST
    import rydberg_sim.spectral as sp

    assert not hasattr(sp, "biased_gs")
    assert not hasattr(sp, "em_gs")
    assert not hasattr(sp, "bessel_ratio")
    assert not hasattr(sp, "xu_gd")
    import rydberg_sim.baselines as bl

    assert not hasattr(bl, "biased_gs")
    assert not hasattr(bl, "spectral_initialize")

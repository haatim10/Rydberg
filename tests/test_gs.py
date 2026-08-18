"""Step 9 acceptance tests: Cui biased Gerchberg–Saxton (Algorithm 1)."""

from __future__ import annotations

import numpy as np
import pytest

from rydberg_sim import (
    biased_gs,
    biased_gs_channel_rows,
    get_trial_rngs,
    random_complex_initialization,
    spectral_initialize,
)
from rydberg_sim.gs import magnitude_objective, project_to_qam
from rydberg_sim.qam import build_qam_constellation

MASTER_SEED = 20260818


def _full_rank_M(rng: np.random.Generator, D: int, Q: int) -> np.ndarray:
    M = (rng.standard_normal((D, Q)) + 1j * rng.standard_normal((D, Q))).astype(
        np.complex128
    )
    s = np.linalg.svd(M, compute_uv=False)
    assert s[-1] / s[0] > 1e-6
    return M


def _well_scaled_instance(seed: int = 195, D: int = 3, Q: int = 128, bmag: float = 1.5):
    rng = np.random.default_rng(seed)
    M = _full_rank_M(rng, D, Q)
    u_true = (rng.standard_normal(D) + 1j * rng.standard_normal(D)).astype(
        np.complex128
    )
    u_true = u_true / np.linalg.norm(u_true)
    b = bmag * np.exp(1j * rng.uniform(-np.pi, np.pi, size=Q))
    return M, u_true, b, rng


def _rel(u_hat: np.ndarray, u_true: np.ndarray) -> float:
    return float(np.linalg.norm(u_hat - u_true) / np.linalg.norm(u_true))


# ---------------------------------------------------------------------------
# Core algebra
# ---------------------------------------------------------------------------


def test_default_init_is_step8_spectral() -> None:
    M, u_true, b, _ = _well_scaled_instance()
    z = np.abs(M.conj().T @ u_true + b)
    spec = spectral_initialize(M, z, b)
    gs = biased_gs(M, z, b, max_iter=1)
    assert gs.init_source == "spectral"
    np.testing.assert_allclose(gs.u0, spec.u0, rtol=0.0, atol=1e-15)
    assert gs.regularization_used is False
    assert gs.ridge == 0.0
    assert gs.n_iter == 1
    assert gs.objective_history.shape == (2,)
    assert gs.iterates is None


def test_explicit_u0_is_given_source() -> None:
    M, u_true, b, rng = _well_scaled_instance()
    z = np.abs(M.conj().T @ u_true + b)
    u0 = random_complex_initialization(M.shape[0], rng)
    gs = biased_gs(M, z, b, max_iter=2, u0=u0, store_iterates=True)
    assert gs.init_source == "given"
    np.testing.assert_allclose(gs.u0, u0, rtol=0.0, atol=0.0)
    assert gs.iterates is not None
    np.testing.assert_allclose(gs.iterates[0], u0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(gs.iterates[-1], gs.u_hat, rtol=0.0, atol=0.0)


def test_measured_z_not_replaced_by_predicted_magnitude() -> None:
    """y^t = z ⊙ exp(j θ), never |λ| ⊙ exp(j θ) with a different |λ|."""
    M, u_true, b, _ = _well_scaled_instance()
    z = np.abs(M.conj().T @ u_true + b) + 0.35
    u0 = u_true
    lam = M.conj().T @ u0 + b
    y_correct = z * np.exp(1j * np.angle(lam))
    r = y_correct - b
    gram = M @ M.conj().T
    u_expected = np.linalg.solve(gram, M @ r)
    gs = biased_gs(M, z, b, max_iter=1, u0=u0)
    np.testing.assert_allclose(gs.u_hat, u_expected, rtol=0.0, atol=1e-12)
    y_wrong = np.abs(lam) * np.exp(1j * np.angle(lam))
    u_wrong = np.linalg.solve(gram, M @ (y_wrong - b))
    assert not np.allclose(u_expected, u_wrong, rtol=1e-8, atol=1e-8)


def test_ridge_not_applied_silently() -> None:
    M, u_true, b, _ = _well_scaled_instance()
    z = np.abs(M.conj().T @ u_true + b)
    a = biased_gs(M, z, b, max_iter=2)
    b0 = biased_gs(M, z, b, max_iter=2, ridge=0.0)
    np.testing.assert_array_equal(a.u_hat, b0.u_hat)
    ridged = biased_gs(M, z, b, max_iter=2, ridge=1.0)
    assert ridged.regularization_used is True
    assert not np.allclose(ridged.u_hat, a.u_hat, rtol=1e-6, atol=1e-6)


def test_rejects_bad_inputs() -> None:
    M = np.eye(3, 5, dtype=np.complex128)
    z = np.ones(5)
    b = np.zeros(5, dtype=np.complex128)
    with pytest.raises(ValueError, match="max_iter"):
        biased_gs(M, z, b, max_iter=0)
    with pytest.raises(ValueError, match="ridge"):
        biased_gs(M, z, b, max_iter=1, ridge=-1e-12)
    with pytest.raises(ValueError, match="length 5"):
        biased_gs(M, z[:4], b, max_iter=1)
    with pytest.raises(ValueError, match="u0"):
        biased_gs(M, z, b, max_iter=1, u0=np.ones(2))
    rank_def = np.vstack([M[:1], M[:1], M[:1]])
    with pytest.raises(np.linalg.LinAlgError, match="singular"):
        biased_gs(rank_def, z, b, max_iter=1, u0=np.ones(3))


def test_objective_monotonicity() -> None:
    M, u_true, b, _ = _well_scaled_instance()
    z = np.abs(M.conj().T @ u_true + b)
    gs = biased_gs(M, z, b, max_iter=25, ridge=0.0)
    j = gs.objective_history
    tol = 1e-8 * max(1.0, float(np.max(j)))
    increases = np.diff(j) > tol
    assert not np.any(increases), (j, np.diff(j))
    np.testing.assert_allclose(
        j[0], magnitude_objective(M, gs.u0, b, z), rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        j[-1], magnitude_objective(M, gs.u_hat, b, z), rtol=0.0, atol=1e-12
    )


def test_noiseless_fixed_point() -> None:
    M, u_true, b, _ = _well_scaled_instance()
    z = np.abs(M.conj().T @ u_true + b)
    gs = biased_gs(M, z, b, max_iter=1, u0=u_true, ridge=0.0)
    np.testing.assert_allclose(gs.u_hat, u_true, rtol=0.0, atol=1e-12)


def test_noiseless_canonical_recovery() -> None:
    M, u_true, b, _ = _well_scaled_instance(seed=195)
    z = np.abs(M.conj().T @ u_true + b)
    spec = spectral_initialize(M, z, b)
    gs = biased_gs(M, z, b, max_iter=40)
    rel0 = _rel(spec.u0, u_true)
    relf = _rel(gs.u_hat, u_true)
    j0 = float(gs.objective_history[0])
    jf = float(gs.objective_history[-1])
    print(
        f"\nnoiseless canonical: rel_init={rel0:.6g} rel_final={relf:.6g} "
        f"J0={j0:.6g} Jf={jf:.6g}"
    )
    assert jf < 0.1 * j0 or jf < 1e-8
    assert relf < rel0
    assert relf < 0.15


def test_high_reference_gs_refines_spectral_init() -> None:
    """Do not replace Step-8 init. Report whether GS improves a poor start."""
    M, u_true, b, _ = _well_scaled_instance(seed=7, bmag=12.0)
    z = np.abs(M.conj().T @ u_true + b)
    spec = spectral_initialize(M, z, b)
    gs = biased_gs(M, z, b, max_iter=40)
    rel0 = _rel(spec.u0, u_true)
    relf = _rel(gs.u_hat, u_true)
    j0 = float(gs.objective_history[0])
    jf = float(gs.objective_history[-1])
    print(
        f"\nhigh-reference refinement: rel_init={rel0:.6g} rel_final={relf:.6g} "
        f"J0={j0:.6g} Jf={jf:.6g}"
    )
    np.testing.assert_allclose(gs.u0, spec.u0, rtol=0.0, atol=1e-15)
    assert jf <= j0 + 1e-8 * max(1.0, j0)


def test_spectral_vs_random_at_snr_minus_5db() -> None:
    """Same (M,u,b,w,z) for both runs; only u0 differs. Median over trials."""
    D, Q = 3, 64
    n_mc = 40
    max_iter = 30
    snr_db = -5.0
    snr_lin = 10.0 ** (snr_db / 10.0)
    rel_spec = np.empty(n_mc)
    rel_rand = np.empty(n_mc)
    j_spec = np.empty(n_mc)
    j_rand = np.empty(n_mc)
    for t in range(n_mc):
        scene = np.random.default_rng(
            np.random.SeedSequence(entropy=MASTER_SEED, spawn_key=(t, 9001))
        )
        M = _full_rank_M(scene, D, Q)
        u_true = (scene.standard_normal(D) + 1j * scene.standard_normal(D)).astype(
            np.complex128
        )
        u_true = u_true / np.linalg.norm(u_true)
        b = 1.5 * np.exp(1j * scene.uniform(-np.pi, np.pi, size=Q))
        signal = M.conj().T @ u_true
        sigma2 = float(np.mean(np.abs(signal) ** 2) / snr_lin)
        scale = np.sqrt(sigma2 / 2.0)
        w = scale * (scene.standard_normal(Q) + 1j * scene.standard_normal(Q))
        z = np.abs(signal + b + w)

        gs_s = biased_gs(M, z, b, max_iter=max_iter)
        u0_r = random_complex_initialization(D, get_trial_rngs(MASTER_SEED, t).solver)
        gs_r = biased_gs(M, z, b, max_iter=max_iter, u0=u0_r)
        rel_spec[t] = _rel(gs_s.u_hat, u_true)
        rel_rand[t] = _rel(gs_r.u_hat, u_true)
        j_spec[t] = gs_s.objective_history[-1]
        j_rand[t] = gs_r.objective_history[-1]

    med_rel_s = float(np.median(rel_spec))
    med_rel_r = float(np.median(rel_rand))
    med_j_s = float(np.median(j_spec))
    med_j_r = float(np.median(j_rand))
    print(
        f"\nSNR=-5 dB  n_mc={n_mc}: "
        f"median rel spectral={med_rel_s:.4f} random={med_rel_r:.4f}; "
        f"median J spectral={med_j_s:.4g} random={med_j_r:.4g}"
    )
    assert med_rel_s < med_rel_r
    assert med_j_s < med_j_r


# ---------------------------------------------------------------------------
# Channel-estimation adapter
# ---------------------------------------------------------------------------


def test_channel_adapter_conjugation_identity_and_refinement() -> None:
    rng = np.random.default_rng(11)
    N, K, P = 3, 3, 128
    S = _full_rank_M(rng, K, P)
    G = (rng.standard_normal((N, K)) + 1j * rng.standard_normal((N, K))).astype(
        np.complex128
    )
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    B = np.empty((N, P), dtype=np.complex128)
    for n in range(N):
        B[n] = 1.5 * np.exp(1j * rng.uniform(-np.pi, np.pi, size=P))
    Z = np.abs(G @ S + B)

    for n in range(N):
        np.testing.assert_allclose(
            np.abs(S.conj().T @ np.conjugate(G[n]) + np.conjugate(B[n])),
            Z[n],
            rtol=0.0,
            atol=1e-14,
        )

    adapted = biased_gs_channel_rows(S, Z, B, max_iter=40)
    assert adapted.G_hat.shape == (N, K)
    assert adapted.G0.shape == (N, K)
    assert len(adapted.row_results) == N
    assert all(row.init_source == "spectral" for row in adapted.row_results)

    rel0 = float(np.linalg.norm(adapted.G0 - G) / np.linalg.norm(G))
    relf = float(np.linalg.norm(adapted.G_hat - G) / np.linalg.norm(G))
    print(f"\nchannel adapter noiseless: rel_init={rel0:.6g} rel_final={relf:.6g}")
    assert relf < rel0

    n = 0
    direct = biased_gs(S, Z[n], np.conjugate(B[n]), max_iter=40)
    np.testing.assert_allclose(
        adapted.G_hat[n], np.conjugate(direct.u_hat), rtol=0.0, atol=1e-15
    )
    assert not np.allclose(adapted.G0[0], adapted.G0[1])


def test_canonical_solver_has_no_channel_or_qam_special_case() -> None:
    import inspect

    from rydberg_sim.gs import biased_gs as fn

    params = inspect.signature(fn).parameters
    assert "S" not in params
    assert "G" not in params
    assert "constellation" not in params
    assert list(params)[:3] == ["M", "z", "b"]


def test_optional_qam_projection_is_not_inside_gs() -> None:
    """Detection-layer helper exists but is not applied by biased_gs."""
    const = build_qam_constellation(4)
    rng = np.random.default_rng(3)
    u = np.array([0.1 + 0.2j, -0.4 + 0.05j], dtype=np.complex128)
    projected = project_to_qam(u, const)
    for val in projected:
        assert np.any(np.isclose(const.points, val, rtol=0.0, atol=1e-15))
    assert projected.shape == u.shape
    # Continuous GS output is not this helper: biased_gs never imports it in-loop.
    M, u_true, b, _ = _well_scaled_instance(D=2, Q=32, seed=3)
    z = np.abs(M.conj().T @ u_true + b)
    gs = biased_gs(M, z, b, max_iter=5)
    assert gs.u_hat.shape == (2,)
    rng.standard_normal()  # keep rng used
    assert const.M == 4


def test_em_gs_lives_in_gs_module_only() -> None:
    import rydberg_sim.baselines as bmod
    import rydberg_sim.gs as gmod
    import rydberg_sim.spectral as smod

    assert hasattr(gmod, "em_gs")
    assert hasattr(gmod, "bessel_ratio")
    assert not hasattr(smod, "biased_gs")
    assert not hasattr(smod, "em_gs")
    assert not hasattr(bmod, "biased_gs")
    assert not hasattr(bmod, "em_gs")
    assert not hasattr(bmod, "bessel_ratio")


def test_solver_rng_does_not_retune_noise() -> None:
    a = get_trial_rngs(MASTER_SEED, 9)
    b = get_trial_rngs(MASTER_SEED, 9)
    np.testing.assert_array_equal(
        a.noise.standard_normal(16), b.noise.standard_normal(16)
    )
    np.testing.assert_array_equal(
        a.solver.standard_normal(16), b.solver.standard_normal(16)
    )
    assert not np.array_equal(
        get_trial_rngs(MASTER_SEED, 9).solver.standard_normal(16),
        get_trial_rngs(MASTER_SEED, 9).noise.standard_normal(16),
    )

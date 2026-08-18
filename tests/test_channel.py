"""Step 2 acceptance tests: geometric ULA channel generator."""

from __future__ import annotations

import numpy as np

from rydberg_sim import (
    RANK_SV_REL_TOL,
    SimulationConfig,
    generate_ula_channel,
    get_trial_rngs,
    is_full_column_rank,
    steering_vector,
)

# Monte Carlo sample size required by the power/energy tests.
N_MC = 10_000

# Conservative 5-sigma relative tolerance for Monte Carlo means.
#
# Per-element power: h_{n,k} | angles is CN(0, beta_k), so |h_{n,k}|^2 is
# exponential with mean beta_k and variance beta_k^2. The sample-mean
# standard deviation is beta_k / sqrt(N_MC) = 0.01 beta_k. Five standard
# deviations is 0.05 beta_k.
#
# Total energy: the most variable case is L_k = 1, where
# ||h_k||_2^2 = N |alpha|^2 and |alpha|^2 is exponential with mean beta_k,
# so Var(||h||^2) = (N beta_k)^2 and the sample-mean std is
# N beta_k / sqrt(N_MC). The same 5-sigma relative bound 0.05 applies.
#
# L_k > 1 only concentrates further, so this bound is conservative.
MC_REL_TOL = 5.0 / np.sqrt(N_MC)


def _cfg(**kwargs) -> SimulationConfig:
    params = dict(N=8, K=2, L=3, beta=1.0, master_seed=20260818)
    params.update(kwargs)
    if "L_k" in kwargs:
        params.pop("L", None)
    if "beta_k" in kwargs:
        params.pop("beta", None)
    return SimulationConfig.create(**params)


def test_steering_vector_norm() -> None:
    """||a(theta)||_2^2 = N to floating-point precision.

    Every entry of a(theta) has magnitude 1, so the squared Euclidean
    norm equals N up to rounding of complex exponentials.
    """
    N = 32
    rng = np.random.default_rng(12345)
    thetas = rng.uniform(-0.5 * np.pi, 0.5 * np.pi, size=500)
    thetas = np.concatenate(
        [thetas, np.array([0.0, -0.5 * np.pi, 0.5 * np.pi - 1e-15])]
    )
    for theta in thetas:
        a = steering_vector(float(theta), N)
        assert a.shape == (N,)
        assert a.dtype == np.complex128
        np.testing.assert_allclose(np.abs(a), 1.0, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(
            np.linalg.norm(a) ** 2, N, rtol=0.0, atol=1e-12
        )


def test_per_element_channel_power_normalization() -> None:
    """E[|h_{n,k}|^2] -> beta_k for representative receive elements.

    See ``MC_REL_TOL`` for the 5-sigma / sqrt(N_MC) justification.
    Distinct per-user beta_k values are used so a swapped-user bug
    cannot pass.
    """
    beta = (0.7, 2.3)
    cfg = _cfg(N=8, K=2, L=4, beta=beta, master_seed=11)
    N, K = cfg.N, cfg.K
    elements = (0, N // 2, N - 1)
    acc = np.zeros((len(elements), K), dtype=np.float64)

    for t in range(N_MC):
        ch = generate_ula_channel(cfg, t)
        for i, n in enumerate(elements):
            acc[i] += np.abs(ch.H[n, :]) ** 2

    empirical = acc / N_MC
    expected = np.broadcast_to(np.asarray(beta, dtype=np.float64), empirical.shape)
    # Relative bound: |mean - beta| <= MC_REL_TOL * beta
    np.testing.assert_allclose(empirical, expected, rtol=MC_REL_TOL, atol=0.0)


def test_total_channel_energy_array_scaling() -> None:
    """E[||h_k||_2^2] -> N * beta_k, confirming linear scaling in N.

    See ``MC_REL_TOL`` for the 5-sigma / sqrt(N_MC) justification.
    """
    beta = (0.7, 2.3)
    cfg = _cfg(N=8, K=2, L=4, beta=beta, master_seed=11)
    acc = np.zeros(cfg.K, dtype=np.float64)

    for t in range(N_MC):
        ch = generate_ula_channel(cfg, t)
        acc += np.sum(np.abs(ch.H) ** 2, axis=0)

    empirical = acc / N_MC
    expected = cfg.N * np.asarray(beta, dtype=np.float64)
    np.testing.assert_allclose(empirical, expected, rtol=MC_REL_TOL, atol=0.0)


def test_steering_matrix_rank() -> None:
    """rank(A_k) = L_k under the documented relative singular-value test."""
    cfg = _cfg(N=16, K=3, L_k=(1, 5, 16), beta=1.0, master_seed=99)
    for t in range(200):
        ch = generate_ula_channel(cfg, t)
        for k, A in enumerate(ch.A_k):
            Lk = int(ch.L_k[k])
            assert A.shape == (cfg.N, Lk)
            assert is_full_column_rank(A, rel_tol=RANK_SV_REL_TOL)
            # Cross-check: numerical rank with the same relative floor.
            sv = np.linalg.svd(A, compute_uv=False)
            rank = int(np.sum(sv >= RANK_SV_REL_TOL * sv[0]))
            assert rank == Lk


def test_channel_shapes_and_dtypes() -> None:
    cfg = _cfg(N=12, K=3, L_k=(2, 5, 7), beta_k=(0.5, 1.0, 1.5), c=1.0)
    ch = generate_ula_channel(cfg, trial_index=0)

    assert ch.G.shape == (cfg.N, cfg.K)
    assert ch.H.shape == (cfg.N, cfg.K)
    assert ch.G.dtype == np.complex128
    assert ch.H.dtype == np.complex128
    assert ch.L_k.shape == (cfg.K,)
    assert ch.L_k.dtype == np.int64
    assert ch.beta_k.shape == (cfg.K,)
    assert ch.beta_k.dtype == np.float64
    np.testing.assert_array_equal(ch.L_k, np.array(cfg.L_k))
    np.testing.assert_allclose(ch.beta_k, np.array(cfg.beta_k))

    for k in range(cfg.K):
        Lk = cfg.L_k[k]
        assert ch.A_k[k].shape == (cfg.N, Lk)
        assert ch.A_k[k].dtype == np.complex128
        assert ch.theta[k].shape == (Lk,)
        assert ch.theta[k].dtype == np.float64
        assert ch.psi[k].shape == (Lk,)
        assert ch.psi[k].dtype == np.float64
        assert ch.alpha[k].dtype == np.complex128
        assert ch.alpha[k].shape == (Lk,)
        np.testing.assert_allclose(ch.psi[k], np.pi * np.sin(ch.theta[k]))
        np.testing.assert_array_equal(ch.H[:, k], ch.A_k[k] @ ch.alpha[k])


def test_G_equals_H_when_c_is_one() -> None:
    cfg = _cfg(c=1.0)
    ch = generate_ula_channel(cfg, trial_index=5)
    np.testing.assert_array_equal(ch.G, ch.H)


def test_G_scales_H_by_c() -> None:
    cfg = _cfg(c=0.37)
    ch = generate_ula_channel(cfg, trial_index=5)
    np.testing.assert_array_equal(ch.G, cfg.c * ch.H)


def test_generate_uses_channel_substream() -> None:
    """Default generation matches an explicit channel-stream Generator."""
    cfg = _cfg()
    from_index = generate_ula_channel(cfg, trial_index=3)
    injected = generate_ula_channel(
        cfg, trial_index=3, rng=get_trial_rngs(cfg.master_seed, 3).channel
    )
    np.testing.assert_array_equal(from_index.H, injected.H)
    for k in range(cfg.K):
        np.testing.assert_array_equal(from_index.theta[k], injected.theta[k])
        np.testing.assert_array_equal(from_index.alpha[k], injected.alpha[k])


def test_channel_reproducible_from_trial_index() -> None:
    cfg = _cfg()
    direct = generate_ula_channel(cfg, 137)
    for t in range(137):
        generate_ula_channel(cfg, t)
    after = generate_ula_channel(cfg, 137)
    np.testing.assert_array_equal(direct.H, after.H)
    for k in range(cfg.K):
        np.testing.assert_array_equal(direct.theta[k], after.theta[k])
        np.testing.assert_array_equal(direct.alpha[k], after.alpha[k])

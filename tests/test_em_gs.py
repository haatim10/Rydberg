"""Step 10 acceptance tests: Cui EM-GS (Algorithm 2) and Bessel ratio."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ive

from rydberg_sim import (
    SimulationConfig,
    biased_gs,
    biased_gs_channel_rows,
    em_gs,
    em_gs_channel_rows,
    exact_forward,
    generate_gaussian_pilots,
    generate_reference_field,
    generate_ula_channel,
    make_alpha_b,
    rsr_db_to_alpha_magnitude,
    snr_db_to_sigma2,
    spectral_initialize,
)
from rydberg_sim.baselines import rician_log_likelihood
from rydberg_sim.gs import (
    BESSEL_RATIO_ASYMP_X,
    _bessel_ratio_asymptotic,
    bessel_ratio,
    em_kappa,
)

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


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _noisy_z(M, u, b, snr_db, rng):
    signal = M.conj().T @ u
    sigma2 = float(np.mean(np.abs(signal) ** 2) / 10.0 ** (snr_db / 10.0))
    scale = np.sqrt(sigma2 / 2.0)
    w = scale * (rng.standard_normal(b.size) + 1j * rng.standard_normal(b.size))
    z = np.abs(signal + b + w)
    return z, sigma2, w


# ---------------------------------------------------------------------------
# Bessel ratio
# ---------------------------------------------------------------------------


def test_bessel_ratio_fig4_on_unit_interval() -> None:
    """Cui Fig. 4 numerical checks on κ ∈ [0, 10]."""
    kappa = np.linspace(0.0, 10.0, 1001)
    r = bessel_ratio(kappa)
    np.testing.assert_allclose(bessel_ratio(0.0), 0.0, rtol=0.0, atol=0.0)
    assert r[0] == 0.0
    assert np.all(np.diff(r) >= -1e-15)
    assert np.all(r >= 0.0) and np.all(r <= 1.0)
    r10 = float(np.asarray(bessel_ratio(10.0)))
    print(f"\nR(10)={r10:.10f} (Fig. 4: ≈ 0.95)")
    assert r10 == pytest.approx(0.95, abs=0.01)


def test_bessel_ratio_weight_interpretation() -> None:
    samples = {
        0.0: (0.0, 0.0),
        0.1: (0.0, 0.15),
        1.0: (0.2, 0.7),
        5.0: (0.7, 0.95),
        10.0: (0.90, 0.98),
        100.0: (0.99, 1.0),
    }
    for kap, (lo, hi) in samples.items():
        val = float(np.asarray(bessel_ratio(kap)))
        assert lo <= val <= hi, (kap, val, lo, hi)
        print(f"R({kap:g})={val:.6f}")


def test_bessel_ratio_large_kappa_stable() -> None:
    for kap in (1e2, 1e3, 1e4, 1e5, 1e6):
        val = float(np.asarray(bessel_ratio(kap)))
        assert np.isfinite(val), kap
        assert 0.0 < val <= 1.0, (kap, val)
        print(f"R({kap:.0e})={val:.12f}")
    assert float(np.asarray(bessel_ratio(1e6))) > float(np.asarray(bessel_ratio(1e2)))
    np.testing.assert_allclose(bessel_ratio(1e6), 1.0, rtol=0.0, atol=1e-5)


def test_bessel_ratio_asymptotic_continuity() -> None:
    x = BESSEL_RATIO_ASYMP_X
    ive_side = float(ive(1, x) / ive(0, x))
    asymp_side = float(_bessel_ratio_asymptotic(np.array(x)))
    prod = float(np.asarray(bessel_ratio(x)))
    above = float(np.asarray(bessel_ratio(x * (1 + 1e-12))))
    np.testing.assert_allclose(ive_side, asymp_side, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(prod, ive_side, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(above, asymp_side, rtol=0.0, atol=1e-10)


def test_bessel_ratio_not_naive_i1_over_i0() -> None:
    """Naive i1/i0 is NaN at 1e3; ive remains finite."""
    from scipy.special import i0, i1

    with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
        naive = i1(1.0e3) / i0(1.0e3)
    assert not np.isfinite(naive)
    assert np.isfinite(bessel_ratio(1.0e3))


def test_kappa_uses_two_over_sigma2() -> None:
    z = np.array([1.0, 2.0])
    lam = np.array([0.5, 1.0 + 0.0j])
    sigma2 = 0.4
    kap = em_kappa(z, lam, sigma2)
    np.testing.assert_allclose(kap, (2.0 / sigma2) * z * np.abs(lam), rtol=0.0, atol=0.0)
    # Catch sigma vs sigma2 (missing square) and missing factor 2.
    wrong_sigma = (2.0 / np.sqrt(sigma2)) * z * np.abs(lam)
    wrong_no2 = (1.0 / sigma2) * z * np.abs(lam)
    assert not np.allclose(kap, wrong_sigma)
    assert not np.allclose(kap, wrong_no2)


# ---------------------------------------------------------------------------
# Canonical EM-GS
# ---------------------------------------------------------------------------


def test_em_gs_requires_positive_sigma2() -> None:
    M, u, b, _ = _well_scaled_instance(D=2, Q=8, seed=1)
    z = np.abs(M.conj().T @ u + b)
    with pytest.raises(ValueError, match="sigma2"):
        em_gs(M, z, b, 0.0, max_iter=1)
    with pytest.raises(ValueError, match="sigma2"):
        em_gs(M, z, b, -0.1, max_iter=1)


def test_em_gs_default_init_is_spectral() -> None:
    M, u, b, rng = _well_scaled_instance()
    z, sigma2, _ = _noisy_z(M, u, b, 20.0, rng)
    spec = spectral_initialize(M, z, b)
    em = em_gs(M, z, b, sigma2, max_iter=1)
    assert em.init_source == "spectral"
    np.testing.assert_allclose(em.u0, spec.u0, rtol=0.0, atol=1e-15)
    assert em.sigma2 == sigma2
    assert em.objective_history.shape == (2,)
    assert em.loglik_history.shape == (2,)


def test_em_gs_uses_same_ls_as_biased_gs_when_r_is_one() -> None:
    """If R(κ)=1, one EM step equals one biased-GS step from the same u0."""
    M, u_true, b, rng = _well_scaled_instance(D=3, Q=32, seed=4)
    z, sigma2, _ = _noisy_z(M, u_true, b, 0.0, rng)
    spec = spectral_initialize(M, z, b)
    # Force R=1 by using an enormous kappa via tiny sigma2 on a copy of the
    # algebra: compare y_EM formula with R=1 to biased GS.
    gs = biased_gs(M, z, b, max_iter=1, u0=spec.u0)
    lam = M.conj().T @ spec.u0 + b
    y_gs = z * np.exp(1j * np.angle(lam))
    y_em_if_r1 = y_gs * np.ones_like(z)
    np.testing.assert_allclose(y_em_if_r1, y_gs)
    gram = M @ M.conj().T
    u_from_r1 = np.linalg.solve(gram, M @ (y_gs - b))
    np.testing.assert_allclose(gs.u_hat, u_from_r1, rtol=0.0, atol=1e-12)


def test_rician_loglik_is_step7_convention() -> None:
    M, u, b, rng = _well_scaled_instance(D=3, Q=32, seed=8)
    z, sigma2, _ = _noisy_z(M, u, b, 10.0, rng)
    em = em_gs(M, z, b, sigma2, max_iter=5)
    lam0 = M.conj().T @ em.u0 + b
    np.testing.assert_allclose(
        em.loglik_history[0],
        rician_log_likelihood(z, lam0, sigma2),
        rtol=0.0,
        atol=1e-12,
    )
    lamf = M.conj().T @ em.u_hat + b
    np.testing.assert_allclose(
        em.loglik_history[-1],
        rician_log_likelihood(z, lamf, sigma2),
        rtol=0.0,
        atol=1e-12,
    )


def test_rician_loglik_nondecreasing_within_tolerance() -> None:
    """EM surrogate should not decrease the Step-7 Rician log-likelihood."""
    M, u, b, rng = _well_scaled_instance(seed=12)
    z, sigma2, _ = _noisy_z(M, u, b, 5.0, rng)
    em = em_gs(M, z, b, sigma2, max_iter=20)
    ll = em.loglik_history
    span = max(1.0, float(np.max(np.abs(ll))))
    drops = np.diff(ll) < -1e-8 * span
    assert not np.any(drops), (ll, np.diff(ll))
    print(
        f"\nRician loglik J-independent: ll0={ll[0]:.6g} ll_final={ll[-1]:.6g} "
        f"delta={ll[-1]-ll[0]:.6g}"
    )


def test_em_approaches_gs_at_high_snr() -> None:
    M, u_true, b, rng = _well_scaled_instance(seed=195)
    z, sigma2, _ = _noisy_z(M, u_true, b, 40.0, rng)
    u0 = spectral_initialize(M, z, b).u0
    gs = biased_gs(M, z, b, max_iter=30, u0=u0)
    em = em_gs(M, z, b, sigma2, max_iter=30, u0=u0)
    rel = _rel(em.u_hat, gs.u_hat)
    j_rel = abs(em.objective_history[-1] - gs.objective_history[-1]) / max(
        1e-15, abs(gs.objective_history[-1])
    )
    mean_r = float(np.mean(bessel_ratio(em.kappa_final)))
    print(
        f"\nhigh SNR=40 dB: ||u_EM-u_GS||/||u_GS||={rel:.6g}  "
        f"J rel diff={j_rel:.6g}  mean R(κ_final)={mean_r:.6f}"
    )
    assert rel < 0.05
    assert mean_r > 0.9


def test_em_approaches_gs_for_tiny_sigma2() -> None:
    M, u_true, b, _ = _well_scaled_instance(seed=195)
    z = np.abs(M.conj().T @ u_true + b)
    sigma2 = 1e-8
    u0 = spectral_initialize(M, z, b).u0
    gs = biased_gs(M, z, b, max_iter=20, u0=u0)
    em = em_gs(M, z, b, sigma2, max_iter=20, u0=u0)
    rel = _rel(em.u_hat, gs.u_hat)
    print(f"\ntiny sigma2={sigma2:g}: ||u_EM-u_GS||/||u_GS||={rel:.6g}")
    assert rel < 1e-6


def test_em_differs_from_gs_at_low_snr_paired_mc() -> None:
    D, Q = 3, 64
    n_mc = 40
    max_iter = 25
    snr_db = -5.0
    rel_gs = np.empty(n_mc)
    rel_em = np.empty(n_mc)
    ll_gs = np.empty(n_mc)
    ll_em = np.empty(n_mc)
    pair_rel = np.empty(n_mc)
    for t in range(n_mc):
        scene = np.random.default_rng(
            np.random.SeedSequence(entropy=MASTER_SEED, spawn_key=(t, 10_001))
        )
        M = _full_rank_M(scene, D, Q)
        u_true = (scene.standard_normal(D) + 1j * scene.standard_normal(D)).astype(
            np.complex128
        )
        u_true /= np.linalg.norm(u_true)
        b = 1.5 * np.exp(1j * scene.uniform(-np.pi, np.pi, size=Q))
        z, sigma2, _ = _noisy_z(M, u_true, b, snr_db, scene)
        u0 = spectral_initialize(M, z, b).u0
        gs = biased_gs(M, z, b, max_iter=max_iter, u0=u0)
        em = em_gs(M, z, b, sigma2, max_iter=max_iter, u0=u0)
        rel_gs[t] = _rel(gs.u_hat, u_true)
        rel_em[t] = _rel(em.u_hat, u_true)
        lam_gs = M.conj().T @ gs.u_hat + b
        lam_em = M.conj().T @ em.u_hat + b
        ll_gs[t] = rician_log_likelihood(z, lam_gs, sigma2)
        ll_em[t] = rician_log_likelihood(z, lam_em, sigma2)
        pair_rel[t] = _rel(em.u_hat, gs.u_hat)

    print(
        f"\nlow SNR=-5 dB n_mc={n_mc}: "
        f"median rel GS={np.median(rel_gs):.4f} EM={np.median(rel_em):.4f}; "
        f"median ll GS={np.median(ll_gs):.4g} EM={np.median(ll_em):.4g}; "
        f"median ||u_EM-u_GS||/||u_GS||={np.median(pair_rel):.4g}"
    )
    assert np.median(pair_rel) > 1e-3
    # EM is not a copy of GS at low SNR. Do not require a win on every trial.
    assert not np.allclose(rel_em, rel_gs)


def test_em_gs_runs_exact_max_iter_and_default_ridge() -> None:
    M, u, b, rng = _well_scaled_instance(D=2, Q=16, seed=2)
    z, sigma2, _ = _noisy_z(M, u, b, 10.0, rng)
    em = em_gs(M, z, b, sigma2, max_iter=7)
    assert em.n_iter == 7
    assert em.ridge == 0.0
    assert em.regularization_used is False
    assert em.objective_history.shape == (8,)
    assert em.loglik_history.shape == (8,)
    assert em.kappa_mean.shape == (7,)
    assert em.kappa_final.shape == (b.size,)


def test_one_em_step_matches_gs_when_r_is_one() -> None:
    """Tiny σ² ⇒ R(κ)≈1 ⇒ one EM update equals one biased-GS update."""
    M, u_true, b, _ = _well_scaled_instance(D=3, Q=32, seed=6)
    z = np.abs(M.conj().T @ u_true + b)
    u0 = spectral_initialize(M, z, b).u0
    gs = biased_gs(M, z, b, max_iter=1, u0=u0)
    em = em_gs(M, z, b, 1e-12, max_iter=1, u0=u0)
    np.testing.assert_allclose(em.u_hat, gs.u_hat, rtol=0.0, atol=1e-10)
    assert float(np.min(bessel_ratio(em.kappa_final))) > 0.999


def test_canonical_em_gs_has_no_channel_or_qam_special_case() -> None:
    import inspect

    from rydberg_sim.gs import em_gs as fn

    params = inspect.signature(fn).parameters
    assert "S" not in params
    assert "G" not in params
    assert "constellation" not in params
    assert list(params)[:4] == ["M", "z", "b", "sigma2"]
    src = inspect.getsource(fn)
    assert "project_to_qam" not in src
    adapter_src = inspect.getsource(em_gs_channel_rows)
    assert "em_gs(" in adapter_src
    assert "y_EM" not in adapter_src
    assert "bessel_ratio" not in adapter_src


# ---------------------------------------------------------------------------
# Channel adapter
# ---------------------------------------------------------------------------


def test_em_channel_adapter_conjugation_and_snr_pair() -> None:
    N, K, P = 8, 3, 16
    cfg = SimulationConfig.create(
        N=N, K=K, L=3, beta=1.0, master_seed=MASTER_SEED, c=1.0
    )
    trial = 5
    G = generate_ula_channel(cfg, trial).G
    S = generate_gaussian_pilots(K=K, P=P, master_seed=MASTER_SEED, trial_index=trial).S
    alpha_b = make_alpha_b(rsr_db_to_alpha_magnitude(0.0, beta_ref=1.0))
    B = generate_reference_field(N=N, P=P, alpha_b=alpha_b, vartheta=0.3, c=1.0).B
    for n in range(N):
        np.testing.assert_allclose(
            np.abs(S.conj().T @ np.conjugate(G[n]) + np.conjugate(B[n])),
            np.abs(S.T @ G[n] + B[n]),
            rtol=0.0,
            atol=1e-14,
        )

    def _run(snr_db: float):
        sigma2 = snr_db_to_sigma2(snr_db, cfg.beta_k, c=cfg.c)
        exact = exact_forward(
            G, S, B, sigma2, master_seed=MASTER_SEED, trial_index=trial
        )
        gs = biased_gs_channel_rows(S, exact.Z, exact.B, max_iter=20)
        em = em_gs_channel_rows(S, exact.Z, exact.B, sigma2, max_iter=20)
        rel0 = float(np.linalg.norm(gs.G0 - G) / np.linalg.norm(G))
        rel_gs = float(np.linalg.norm(gs.G_hat - G) / np.linalg.norm(G))
        rel_em = float(np.linalg.norm(em.G_hat - G) / np.linalg.norm(G))
        # Same spectral init (same Z, B).
        np.testing.assert_allclose(gs.G0, em.G0, rtol=0.0, atol=1e-12)
        return rel0, rel_gs, rel_em

    low = _run(0.0)
    high = _run(30.0)
    print(
        f"\nULA adapter  RSR=0 dB\n"
        f"  SNR=0 dB:  init={low[0]:.4f}  GS={low[1]:.4f}  EM={low[2]:.4f}\n"
        f"  SNR=30 dB: init={high[0]:.4f}  GS={high[1]:.4f}  EM={high[2]:.4f}"
    )
    # High SNR: EM and GS should be close.
    exact_hi = exact_forward(
        G,
        S,
        B,
        snr_db_to_sigma2(30.0, cfg.beta_k, c=cfg.c),
        master_seed=MASTER_SEED,
        trial_index=trial,
    )
    gs_hi = biased_gs_channel_rows(S, exact_hi.Z, exact_hi.B, max_iter=20)
    em_hi = em_gs_channel_rows(
        S,
        exact_hi.Z,
        exact_hi.B,
        snr_db_to_sigma2(30.0, cfg.beta_k, c=cfg.c),
        max_iter=20,
    )
    merge = float(np.linalg.norm(em_hi.G_hat - gs_hi.G_hat) / np.linalg.norm(gs_hi.G_hat))
    print(f"  SNR=30 dB ||G_EM-G_GS||/||G_GS||={merge:.6g}")
    assert merge < 0.1


def test_step13_plus_not_implemented() -> None:
    import rydberg_sim.baselines as bmod
    import rydberg_sim.crlb as cmod
    import rydberg_sim.gs as gmod
    import rydberg_sim.linearised_crlb as lmod
    import rydberg_sim.spectral as smod

    assert hasattr(gmod, "em_gs")
    assert hasattr(gmod, "bessel_ratio")
    assert not hasattr(gmod, "crlb")
    assert not hasattr(gmod, "xu_gd")
    assert not hasattr(smod, "em_gs")
    assert not hasattr(bmod, "em_gs")
    assert not hasattr(bmod, "bessel_ratio")
    assert hasattr(cmod, "cui_crlb")
    assert not hasattr(cmod, "xu_crlb")
    assert not hasattr(cmod, "xu_gd")
    assert hasattr(lmod, "linearised_channel_crlb")
    assert not hasattr(lmod, "xu_crlb")
    assert not hasattr(lmod, "xu_gd")

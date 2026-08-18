"""Step 6 acceptance tests: SNR/RSR power calibration."""

from __future__ import annotations

import numpy as np
import pytest

from rydberg_sim import (
    SimulationConfig,
    exact_forward,
    generate_gaussian_pilots,
    generate_reference_field,
    generate_ula_channel,
    linearised_observation,
    make_alpha_b,
    measure_rsr,
    measure_snr,
    reference_user_beta,
    rsr_db_to_alpha_magnitude,
    snr_db_to_sigma2,
)
from rydberg_sim.calibration import db_to_linear, linear_to_db

MASTER_SEED = 20260818
N_MC = 10_000
SNR_DB_SWEEP = (0.0, 10.0, 20.0)
RSR_DB_SWEEP = (0.0, 10.0, 20.0, 30.0)
DB_TOL = 0.1  # implementation-plan acceptance: 0.1 dB


def test_snr_simple_cases() -> None:
    """c=1, beta_k=1: sigma2 = K / SNR_lin."""
    np.testing.assert_allclose(
        snr_db_to_sigma2(0.0, (1.0, 1.0, 1.0), c=1.0), 3.0, rtol=0.0, atol=1e-15
    )
    np.testing.assert_allclose(
        snr_db_to_sigma2(10.0, (1.0, 1.0, 1.0), c=1.0), 0.3, rtol=0.0, atol=1e-15
    )


def test_snr_uses_sum_of_beta_not_single_user() -> None:
    sigma2 = snr_db_to_sigma2(10.0, (0.5, 1.0, 2.0), c=1.0)
    np.testing.assert_allclose(sigma2, 3.5 / 10.0, rtol=0.0, atol=1e-15)


def test_snr_includes_c_squared() -> None:
    sigma2 = snr_db_to_sigma2(0.0, (1.0,), c=2.0)
    np.testing.assert_allclose(sigma2, 4.0, rtol=0.0, atol=1e-15)


def test_snr_k_scaling_doubles_sigma2() -> None:
    """Fixed total SNR: doubling K doubles sigma2. Per-user SNR is not held fixed."""
    snr_db = 10.0
    s3 = snr_db_to_sigma2(snr_db, (1.0,) * 3)
    s6 = snr_db_to_sigma2(snr_db, (1.0,) * 6)
    np.testing.assert_allclose(s3, 3.0 / 10.0, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(s6, 6.0 / 10.0, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(s6, 2.0 * s3, rtol=0.0, atol=1e-15)


def test_rsr_simple_cases() -> None:
    """beta_ref=1, E|s_b|^2=1: |alpha_b| = sqrt(RSR_lin)."""
    cases = {
        0.0: 1.0,
        10.0: np.sqrt(10.0),
        20.0: 10.0,
        30.0: np.sqrt(1000.0),
    }
    for rsr_db, mag in cases.items():
        got = rsr_db_to_alpha_magnitude(rsr_db, beta_ref=1.0, e_s_b_sq=1.0)
        np.testing.assert_allclose(got, mag, rtol=0.0, atol=1e-15)


def test_rsr_does_not_gain_factor_K() -> None:
    """|alpha_b| is sqrt(RSR_lin) for both K=3 and K=6; never sqrt(K RSR) or sqrt(RSR/K)."""
    rsr_db = 10.0
    mag = rsr_db_to_alpha_magnitude(rsr_db, beta_ref=1.0)
    np.testing.assert_allclose(mag, np.sqrt(10.0), rtol=0.0, atol=1e-15)
    assert mag != pytest.approx(np.sqrt(3.0 * 10.0))
    assert mag != pytest.approx(np.sqrt(6.0 * 10.0))
    assert mag != pytest.approx(np.sqrt(10.0 / 3.0))
    assert mag != pytest.approx(np.sqrt(10.0 / 6.0))


def test_rsr_uses_explicit_beta_ref_not_mean() -> None:
    betas = (0.5, 1.0, 2.0)
    mag_ref2 = rsr_db_to_alpha_magnitude(10.0, beta_ref=reference_user_beta(betas, 2))
    mag_if_mean = rsr_db_to_alpha_magnitude(10.0, beta_ref=float(np.mean(betas)))
    np.testing.assert_allclose(mag_ref2, np.sqrt(10.0 * 2.0), rtol=0.0, atol=1e-15)
    assert mag_ref2 != pytest.approx(mag_if_mean)


def test_alpha_b_phase_does_not_change_magnitude() -> None:
    mag = rsr_db_to_alpha_magnitude(20.0, beta_ref=1.0)
    a0 = make_alpha_b(mag, phi_b=0.0)
    a1 = make_alpha_b(mag, phi_b=1.3)
    np.testing.assert_allclose(abs(a0), mag, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(abs(a1), mag, rtol=0.0, atol=1e-15)
    np.testing.assert_allclose(abs(a0), abs(a1), rtol=0.0, atol=1e-15)
    assert a0.imag == pytest.approx(0.0)
    assert a1 != pytest.approx(a0)


def test_phase_does_not_change_measured_rsr() -> None:
    cfg = SimulationConfig.create(N=8, K=3, L=3, beta=1.0, master_seed=1, c=1.0)
    ch = generate_ula_channel(cfg, 0)
    pilots = generate_gaussian_pilots(K=3, P=8, master_seed=1, trial_index=0)
    mag = rsr_db_to_alpha_magnitude(10.0, beta_ref=1.0)
    b0 = generate_reference_field(
        N=8, P=8, alpha_b=make_alpha_b(mag, 0.0), vartheta=0.2, c=1.0
    ).B
    b1 = generate_reference_field(
        N=8, P=8, alpha_b=make_alpha_b(mag, 2.1), vartheta=0.2, c=1.0
    ).B
    r0 = measure_rsr(b0, ch.G, pilots.S, user_index=0)
    r1 = measure_rsr(b1, ch.G, pilots.S, user_index=0)
    np.testing.assert_allclose(r0.rsr_lin, r1.rsr_lin, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(b0), np.abs(b1), rtol=0.0, atol=1e-12)


def test_measure_rsr_uses_single_user_not_full_gs() -> None:
    G = np.array([[1.0, 10.0], [1.0, 10.0]], dtype=np.complex128)
    S = np.ones((2, 3), dtype=np.complex128)
    B = np.ones((2, 3), dtype=np.complex128)
    rsr_user0 = measure_rsr(B, G, S, user_index=0)
    rsr_user1 = measure_rsr(B, G, S, user_index=1)
    assert rsr_user0.rsr_lin != pytest.approx(rsr_user1.rsr_lin)
    np.testing.assert_allclose(rsr_user0.rsr_lin / rsr_user1.rsr_lin, 100.0, rtol=1e-12)


def _run_empirical_calibration(K: int) -> dict[str, dict[float, float]]:
    """10_000-trial linear-power accumulation for one K."""
    P = 2 * K
    N = 8
    cfg = SimulationConfig.create(
        N=N, K=K, L=3, beta=1.0, master_seed=MASTER_SEED, c=1.0
    )
    betas = cfg.beta_k
    beta_ref = reference_user_beta(betas, 0)
    user_index = 0

    sum_sig = 0.0
    sum_user = 0.0
    n_sig = 0
    sum_noise = {snr: 0.0 for snr in SNR_DB_SWEEP}
    n_noise = {snr: 0 for snr in SNR_DB_SWEEP}
    sum_ref = {rsr: 0.0 for rsr in RSR_DB_SWEEP}
    n_ref = {rsr: 0 for rsr in RSR_DB_SWEEP}

    dummy_B = generate_reference_field(
        N=N,
        P=P,
        alpha_b=make_alpha_b(rsr_db_to_alpha_magnitude(0.0, beta_ref)),
        vartheta=0.3,
        c=cfg.c,
    ).B

    for t in range(N_MC):
        G = generate_ula_channel(cfg, t).G
        S = generate_gaussian_pilots(
            K=K, P=P, master_seed=MASTER_SEED, trial_index=t
        ).S
        gs = G @ S
        sum_sig += float(np.sum(np.abs(gs) ** 2))
        user = G[:, user_index : user_index + 1] @ S[user_index : user_index + 1, :]
        sum_user += float(np.sum(np.abs(user) ** 2))
        n_sig += gs.size

        for snr_db in SNR_DB_SWEEP:
            sigma2 = snr_db_to_sigma2(snr_db, betas, c=cfg.c)
            W = exact_forward(
                G,
                S,
                dummy_B,
                sigma2,
                master_seed=MASTER_SEED,
                trial_index=t,
            ).W
            sum_noise[snr_db] += float(np.sum(np.abs(W) ** 2))
            n_noise[snr_db] += W.size

        for rsr_db in RSR_DB_SWEEP:
            alpha_b = make_alpha_b(rsr_db_to_alpha_magnitude(rsr_db, beta_ref))
            B = generate_reference_field(
                N=N, P=P, alpha_b=alpha_b, vartheta=0.3, c=cfg.c
            ).B
            sum_ref[rsr_db] += float(np.sum(np.abs(B) ** 2))
            n_ref[rsr_db] += B.size

    snr_meas = {}
    for snr_db in SNR_DB_SWEEP:
        assert n_sig == n_noise[snr_db]
        snr_meas[snr_db] = linear_to_db(sum_sig / sum_noise[snr_db])
    rsr_meas = {}
    for rsr_db in RSR_DB_SWEEP:
        assert n_sig == n_ref[rsr_db]
        rsr_meas[rsr_db] = linear_to_db(sum_ref[rsr_db] / sum_user)
    return {"snr_db": snr_meas, "rsr_db": rsr_meas}


@pytest.mark.parametrize("K", [3, 6])
def test_empirical_snr_rsr_within_0p1_db(K: int) -> None:
    """Plan requirement: 10_000 trials, 0.1 dB on aggregated linear powers."""
    measured = _run_empirical_calibration(K)
    print(f"\nK={K} empirical calibration ({N_MC} trials, linear-power aggregate)")
    for target, got in measured["snr_db"].items():
        print(f"  SNR  target={target:5.1f} dB  measured={got:8.4f} dB  err={got-target:+.4f} dB")
        assert abs(got - target) <= DB_TOL, (
            f"K={K} SNR target {target} dB, measured {got:.4f} dB "
            f"(err {got - target:+.4f} dB, tol {DB_TOL})"
        )
    for target, got in measured["rsr_db"].items():
        print(f"  RSR  target={target:5.1f} dB  measured={got:8.4f} dB  err={got-target:+.4f} dB")
        assert abs(got - target) <= DB_TOL, (
            f"K={K} RSR target {target} dB, measured {got:.4f} dB "
            f"(err {got - target:+.4f} dB, tol {DB_TOL})"
        )


def test_calibrated_rsr_linearisation_includes_30_db() -> None:
    """Step-5 linearisation vs official RSR, including 30 dB.

    Replaces the earlier raw-|B| amplitude sweep for the strong-reference
    check. This is an acceptance test, not a publication figure. sigma2
    comes from Step-6 SNR calibration at 30 dB so noise does not hide
    the Taylor trend.
    """
    K, P, N = 3, 8, 8
    cfg = SimulationConfig.create(
        N=N, K=K, L=3, beta=1.0, master_seed=MASTER_SEED, c=1.0
    )
    trial = 9
    G = generate_ula_channel(cfg, trial).G
    S = generate_gaussian_pilots(
        K=K, P=P, master_seed=MASTER_SEED, trial_index=trial
    ).S
    sigma2 = 0.0  # isolate Taylor linearisation from noise
    rsr_dbs = (0.0, 10.0, 20.0, 30.0)
    errors = []
    for rsr_db in rsr_dbs:
        alpha_b = make_alpha_b(rsr_db_to_alpha_magnitude(rsr_db, beta_ref=1.0))
        B = generate_reference_field(
            N=N, P=P, alpha_b=alpha_b, vartheta=0.3, c=1.0
        ).B
        exact = exact_forward(
            G, S, B, sigma2, master_seed=MASTER_SEED, trial_index=trial
        )
        lin = linearised_observation(exact)
        errors.append(lin.relative_frobenius_error)
    for weaker, stronger in zip(errors, errors[1:]):
        assert stronger < weaker, (rsr_dbs, errors)
    assert errors[-1] < 0.05, errors
    assert rsr_dbs[-1] == 30.0


def test_db_round_trip() -> None:
    for x in (0.0, 10.0, -3.0, 30.0):
        np.testing.assert_allclose(linear_to_db(db_to_linear(x)), x, rtol=0.0, atol=1e-12)


def test_measure_snr_matches_formula_on_one_draw() -> None:
    G = np.ones((2, 2), dtype=np.complex128)
    S = np.eye(2, dtype=np.complex128)
    W = np.ones((2, 2), dtype=np.complex128) * 0.5
    m = measure_snr(G, S, W)
    np.testing.assert_allclose(m.signal_power, mean_abs_sq_local(G @ S))
    np.testing.assert_allclose(m.snr_lin, m.signal_power / m.noise_power)


def mean_abs_sq_local(arr: np.ndarray) -> float:
    return float(np.mean(np.abs(arr) ** 2))

"""Track-B end-to-end guard rails (audit fix H2).

The adversarial audit showed that the Track-B conjugation adapter could be
broken -- ``b`` passed unconjugated, or the output conjugation dropped -- and
the entire test suite would still pass. Every existing adapter test asserted
either a structural property (shapes, ``status == "ok"``) or a relative
statement that survives the break (``final beats init``, ``EM merges with
GS``). The low/high-SNR numbers were computed and printed, never asserted.

This module closes that hole with **numerical** assertions on the estimate
itself:

A. :func:`test_noiseless_moderate_reference_recovers_G`
   Noiseless, moderate reference: relative Frobenius error below a tight
   threshold for both biased GS and EM-GS.

B. :func:`test_channel_nmse_improves_with_snr`
   Paired trials at low and high SNR: the ratio-of-sums channel NMSE must
   drop by close to the ideal 10 dB per 10 dB of SNR.

C. :func:`test_broken_adapter_is_caught_by_these_thresholds`
   A local, deliberately wrong adapter (``b`` instead of ``conj(b)``, no
   output conjugation) is run through the *same* metric and thresholds, and
   asserted to fail them by a wide margin. This is a negative regression
   test: it proves the thresholds above have teeth. The shipped code is not
   modified.

D. :func:`test_canonical_row_identity`
   The mapping itself, per row:
   ``|S^H conj(g_n) + conj(b_n)| == |S^T g_n + b_n|``.

Physical model (SystemModel.pdf Sections 7-8)::

    Z = |G S + B + W|

Canonical adapter (implementation plan Part 1)::

    M        = S
    u        = conj(g_n)          g_n = n-th row of G, as a column
    b_solver = conj(B[n, :])
    g_hat_n  = conj(u_hat)

The whole expression inside the magnitude is conjugated, so ``b`` is
conjugated too. Magnitudes are invariant under conjugation, which is exactly
why getting this wrong is silent -- and why these tests exist.
"""

from __future__ import annotations

import numpy as np
import pytest

from rydberg_sim import (
    SimulationConfig,
    biased_gs_channel_rows,
    channel_nmse,
    em_gs_channel_rows,
    exact_forward,
    generate_gaussian_pilots,
    generate_reference_field,
    generate_ula_channel,
    get_operating_point_rngs,
    make_alpha_b,
    nmse_to_db,
    rsr_db_to_alpha_magnitude,
    snr_db_to_sigma2,
)
from rydberg_sim.gs import biased_gs, em_gs

# Deterministic Track-B instance. P = 30 >> 2K = 6, so the per-element
# problem is well over-determined and both solvers are known to converge.
N = 8
K = 3
P = 30
L = 4
MASTER_SEED = 20260819
VARTHETA = 0.3
T0 = 50

# --- thresholds -------------------------------------------------------------
# A: noiseless, RSR = 12 dB. Cui's t0 = 50 is a production iteration budget,
# not a convergence budget: at t0 = 50 the noiseless residual is still
# 4.2e-5 and falling. Run this test to convergence instead. Measured on this
# instance at t0 = 200 (and unchanged at 500 and 2000, i.e. the fixed point):
#
#     biased GS  8.1e-16   EM-GS  5.3e-14
#
# 1e-10 leaves four orders of headroom over EM-GS for BLAS/platform drift
# while staying ~10 orders tighter than anything a broken adapter reaches
# (a broken adapter sits at ~1.3-1.7; see test C).
NOISELESS_ITERS = 200
NOISELESS_REL_FRO_MAX = 1e-10

# B: NMSE slope. Ideal is exactly -10 dB per +10 dB SNR. Measured on this
# instance over 12 paired trials: -10.4 to -10.7 dB per step. The band is
# wide enough for Monte-Carlo wobble and narrow enough that a floored
# estimator (slope ~0 dB) fails immediately.
SNR_LOW_DB = 0.0
SNR_HIGH_DB = 30.0
N_TRIALS = 12
RSR_DB = 12.0
MIN_TOTAL_GAIN_DB = 25.0  # over a 30 dB SNR span; ideal is 30 dB
MAX_TOTAL_GAIN_DB = 35.0

# C: what a broken adapter actually reaches. Measured: +3.5 to +4.3 dB NMSE
# at every SNR, i.e. worse than the trivial estimate G_hat = 0.
BROKEN_ADAPTER_MIN_NMSE_DB = 0.0
BROKEN_ADAPTER_MAX_TOTAL_GAIN_DB = 5.0


def _world(snr_db: float, rsr_db: float, trial: int):
    """One frozen Track-B world: G, S, B, W, Z, sigma2."""
    cfg = SimulationConfig.create(
        N=N, K=K, L=L, beta=1.0, master_seed=MASTER_SEED, c=1.0
    )
    rngs = get_operating_point_rngs(cfg.master_seed, trial, snr_db, rsr_db)
    ch = generate_ula_channel(cfg, trial, rng=rngs.channel)
    pilots = generate_gaussian_pilots(K=K, P=P, rng=rngs.pilots)
    alpha_b = make_alpha_b(rsr_db_to_alpha_magnitude(rsr_db, beta_ref=1.0))
    ref = generate_reference_field(
        N=N, P=P, alpha_b=alpha_b, vartheta=VARTHETA, c=cfg.c
    )
    sigma2 = snr_db_to_sigma2(snr_db, cfg.beta_k, c=cfg.c)
    exact = exact_forward(ch.G, pilots.S, ref.B, sigma2, rng_noise=rngs.noise)
    return ch.G, pilots.S, ref.B, exact.Z, sigma2


def _rel_fro(G_hat: np.ndarray, G: np.ndarray) -> float:
    return float(
        np.linalg.norm(G_hat - G, ord="fro") / np.linalg.norm(G, ord="fro")
    )


# ---------------------------------------------------------------------------
# The broken adapters (test C only). NEVER imported by production code.
# ---------------------------------------------------------------------------


def _broken_gs_unconjugated_b(S, Z, B, *, max_iter):
    """Wrong adapter: passes ``B[n]`` where ``conj(B[n])`` is required."""
    G_hat = np.empty((Z.shape[0], S.shape[0]), dtype=np.complex128)
    for n in range(Z.shape[0]):
        G_hat[n] = np.conjugate(biased_gs(S, Z[n], B[n], max_iter=max_iter).u_hat)
    return G_hat


def _broken_gs_no_output_conjugation(S, Z, B, *, max_iter):
    """Wrong adapter: correct ``conj(b)``, but forgets ``g_hat = conj(u_hat)``."""
    G_hat = np.empty((Z.shape[0], S.shape[0]), dtype=np.complex128)
    for n in range(Z.shape[0]):
        G_hat[n] = biased_gs(
            S, Z[n], np.conjugate(B[n]), max_iter=max_iter
        ).u_hat
    return G_hat


# ---------------------------------------------------------------------------
# D. Canonical identity
# ---------------------------------------------------------------------------


def test_canonical_row_identity() -> None:
    """``|S^H conj(g_n) + conj(b_n)| == |S^T g_n + b_n|`` for every row.

    This is the mapping the whole Track-B stack rests on. If it is ever
    changed, this fails first and loudest.
    """
    G, S, B, _Z, _s2 = _world(SNR_HIGH_DB, RSR_DB, trial=0)

    for n in range(N):
        g_n = G[n, :]
        canonical = np.abs(S.conj().T @ np.conjugate(g_n) + np.conjugate(B[n]))
        physical = np.abs(S.T @ g_n + B[n])
        np.testing.assert_allclose(canonical, physical, rtol=0.0, atol=1e-13)

        # The signal term alone is not enough: the WHOLE expression is
        # conjugated. S^H conj(g) == conj(S^T g) == conj((GS)[n, :]).
        np.testing.assert_allclose(
            S.conj().T @ np.conjugate(g_n),
            np.conjugate(S.T @ g_n),
            rtol=0.0,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            np.conjugate(S.T @ g_n),
            np.conjugate((G @ S)[n, :]),
            rtol=0.0,
            atol=1e-13,
        )

    # And leaving b unconjugated is NOT a small perturbation.
    worst = max(
        float(
            np.max(
                np.abs(
                    np.abs(S.conj().T @ np.conjugate(G[n]) + B[n])
                    - np.abs(S.T @ G[n] + B[n])
                )
            )
        )
        for n in range(N)
    )
    assert worst > 1e-3, worst


# ---------------------------------------------------------------------------
# A. Noiseless recovery
# ---------------------------------------------------------------------------


def test_noiseless_moderate_reference_recovers_G() -> None:
    """Noiseless, RSR = 12 dB: both solvers recover G to high accuracy."""
    cfg = SimulationConfig.create(
        N=N, K=K, L=L, beta=1.0, master_seed=MASTER_SEED, c=1.0
    )
    rngs = get_operating_point_rngs(cfg.master_seed, 0, 0.0, RSR_DB)
    G = generate_ula_channel(cfg, 0, rng=rngs.channel).G
    S = generate_gaussian_pilots(K=K, P=P, rng=rngs.pilots).S
    alpha_b = make_alpha_b(rsr_db_to_alpha_magnitude(RSR_DB, beta_ref=1.0))
    B = generate_reference_field(
        N=N, P=P, alpha_b=alpha_b, vartheta=VARTHETA, c=cfg.c
    ).B
    # sigma2 = 0 exactly: no noise stream is consumed.
    Z = exact_forward(G, S, B, 0.0).Z

    gs = biased_gs_channel_rows(S, Z, B, max_iter=NOISELESS_ITERS)
    rel_gs = _rel_fro(gs.G_hat, G)

    # EM-GS needs sigma2 > 0 (kappa contains 1/sigma2). Use a tiny value:
    # R(kappa) -> 1 and EM-GS must reduce to biased GS.
    em = em_gs_channel_rows(S, Z, B, 1e-12, max_iter=NOISELESS_ITERS)
    rel_em = _rel_fro(em.G_hat, G)

    print(f"\nnoiseless RSR={RSR_DB} dB: GS rel_fro={rel_gs:.3e} "
          f"EM-GS rel_fro={rel_em:.3e}")
    assert rel_gs < NOISELESS_REL_FRO_MAX, rel_gs
    assert rel_em < NOISELESS_REL_FRO_MAX, rel_em

    # Shapes and dtype, so a transposed estimate cannot slip through.
    assert gs.G_hat.shape == (N, K)
    assert em.G_hat.shape == (N, K)
    assert gs.G_hat.dtype == np.complex128


# ---------------------------------------------------------------------------
# B. SNR trend
# ---------------------------------------------------------------------------


def _paired_nmse_db(estimator, *, snr_db: float) -> float:
    """Ratio-of-sums channel NMSE in dB over ``N_TRIALS`` paired worlds."""
    err = 0.0
    energy = 0.0
    for trial in range(N_TRIALS):
        G, S, B, Z, sigma2 = _world(snr_db, RSR_DB, trial)
        G_hat = estimator(S, Z, B, sigma2)
        result = channel_nmse(G_hat, G)
        err += result.error_energy
        energy += result.true_energy
    return nmse_to_db(err / energy)


def _shipped_gs(S, Z, B, sigma2):
    return biased_gs_channel_rows(S, Z, B, max_iter=T0).G_hat


def _shipped_em(S, Z, B, sigma2):
    return em_gs_channel_rows(S, Z, B, sigma2, max_iter=T0).G_hat


@pytest.mark.parametrize(
    "name,estimator", [("biased_gs", _shipped_gs), ("em_gs", _shipped_em)]
)
def test_channel_nmse_improves_with_snr(name: str, estimator) -> None:
    """NMSE must fall by ~10 dB per 10 dB of SNR, not sit on a floor.

    A broken adapter plateaus at roughly +4 dB independent of SNR (see
    :func:`test_broken_adapter_is_caught_by_these_thresholds`), so a total
    gain of at least 25 dB over a 30 dB span cannot be faked.
    """
    low = _paired_nmse_db(estimator, snr_db=SNR_LOW_DB)
    high = _paired_nmse_db(estimator, snr_db=SNR_HIGH_DB)
    gain = low - high
    print(f"\n{name}: NMSE {low:.3f} dB @ {SNR_LOW_DB} dB SNR -> "
          f"{high:.3f} dB @ {SNR_HIGH_DB} dB SNR (gain {gain:.3f} dB)")

    assert high < low, (low, high)
    assert MIN_TOTAL_GAIN_DB <= gain <= MAX_TOTAL_GAIN_DB, gain
    # Absolute sanity: a working estimator is well below 0 dB at high SNR.
    assert high < -25.0, high


# ---------------------------------------------------------------------------
# C. Negative regression: the thresholds above catch a broken adapter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,broken",
    [
        ("unconjugated_b", _broken_gs_unconjugated_b),
        ("no_output_conjugation", _broken_gs_no_output_conjugation),
    ],
)
def test_broken_adapter_is_caught_by_these_thresholds(name: str, broken) -> None:
    """Prove the suite would catch the exact failure the audit demonstrated.

    This runs a deliberately wrong adapter through the *same* metric and the
    *same* thresholds used by tests A and B, and asserts it fails all of them.
    Production code is untouched -- the broken adapters are local to this
    module.
    """
    # Fails threshold A (noiseless recovery).
    cfg = SimulationConfig.create(
        N=N, K=K, L=L, beta=1.0, master_seed=MASTER_SEED, c=1.0
    )
    rngs = get_operating_point_rngs(cfg.master_seed, 0, 0.0, RSR_DB)
    G = generate_ula_channel(cfg, 0, rng=rngs.channel).G
    S = generate_gaussian_pilots(K=K, P=P, rng=rngs.pilots).S
    alpha_b = make_alpha_b(rsr_db_to_alpha_magnitude(RSR_DB, beta_ref=1.0))
    B = generate_reference_field(
        N=N, P=P, alpha_b=alpha_b, vartheta=VARTHETA, c=cfg.c
    ).B
    Z = exact_forward(G, S, B, 0.0).Z
    rel_broken = _rel_fro(broken(S, Z, B, max_iter=NOISELESS_ITERS), G)
    print(f"\nbroken[{name}] noiseless rel_fro={rel_broken:.4f} "
          f"(threshold A is < {NOISELESS_REL_FRO_MAX})")
    assert rel_broken > NOISELESS_REL_FRO_MAX
    # Not marginal: orders of magnitude, not a tolerance shave.
    assert rel_broken > 1e6 * NOISELESS_REL_FRO_MAX

    # Fails threshold B (SNR trend): the NMSE floors instead of tracking SNR.
    def _broken_estimator(S_, Z_, B_, sigma2_):
        return broken(S_, Z_, B_, max_iter=T0)

    low = _paired_nmse_db(_broken_estimator, snr_db=SNR_LOW_DB)
    high = _paired_nmse_db(_broken_estimator, snr_db=SNR_HIGH_DB)
    gain = low - high
    print(f"broken[{name}]: NMSE {low:.3f} dB -> {high:.3f} dB "
          f"(gain {gain:.3f} dB, threshold B needs >= {MIN_TOTAL_GAIN_DB})")

    assert gain < MIN_TOTAL_GAIN_DB, gain
    assert gain < BROKEN_ADAPTER_MAX_TOTAL_GAIN_DB, gain
    # It plateaus above 0 dB -- worse than the trivial estimate G_hat = 0.
    assert high > BROKEN_ADAPTER_MIN_NMSE_DB, high

    # And it fails the high-SNR absolute check from test B.
    assert not (high < -25.0)


def test_shipped_adapter_passes_where_broken_adapter_fails() -> None:
    """Side-by-side at one operating point, so the contrast is explicit."""
    G, S, B, Z, sigma2 = _world(SNR_HIGH_DB, RSR_DB, trial=0)
    shipped = _rel_fro(biased_gs_channel_rows(S, Z, B, max_iter=T0).G_hat, G)
    wrong_b = _rel_fro(_broken_gs_unconjugated_b(S, Z, B, max_iter=T0), G)
    wrong_out = _rel_fro(_broken_gs_no_output_conjugation(S, Z, B, max_iter=T0), G)
    print(f"\nSNR={SNR_HIGH_DB} dB rel_fro: shipped={shipped:.5f} "
          f"unconjugated_b={wrong_b:.5f} no_output_conj={wrong_out:.5f}")
    assert shipped < 0.05
    assert wrong_b > 20.0 * shipped
    assert wrong_out > 20.0 * shipped


# ---------------------------------------------------------------------------
# The three shipped adapters must stay in agreement with the canonical solver
# ---------------------------------------------------------------------------


def test_adapters_equal_explicit_canonical_calls() -> None:
    """``*_channel_rows`` is exactly a loop of canonical calls, nothing more."""
    G, S, B, Z, sigma2 = _world(10.0, RSR_DB, trial=1)
    gs = biased_gs_channel_rows(S, Z, B, max_iter=5)
    em = em_gs_channel_rows(S, Z, B, sigma2, max_iter=5)
    for n in range(N):
        direct_gs = biased_gs(S, Z[n], np.conjugate(B[n]), max_iter=5)
        np.testing.assert_allclose(
            gs.G_hat[n], np.conjugate(direct_gs.u_hat), rtol=0.0, atol=0.0
        )
        direct_em = em_gs(S, Z[n], np.conjugate(B[n]), sigma2, max_iter=5)
        np.testing.assert_allclose(
            em.G_hat[n], np.conjugate(direct_em.u_hat), rtol=0.0, atol=0.0
        )

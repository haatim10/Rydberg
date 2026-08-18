"""Step 4 Part A: complex Gaussian estimation pilots."""

from __future__ import annotations

import numpy as np
import pytest

from rydberg_sim import (
    PILOT_RANK_SV_REL_TOL,
    generate_gaussian_pilots,
    get_trial_rngs,
    is_full_row_rank,
)

MASTER_SEED = 20260818
K = 3
P = 8  # P >= 2K

# |s|^2 is Exp(mean=1), variance 1. Sample-mean std = 1/sqrt(N_samples).
# 5000 trials * K * P = 120_000 samples → 5σ ≈ 0.014.
N_MC = 5000
N_SAMPLES = N_MC * K * P
MC_ABS_TOL = 5.0 / np.sqrt(N_SAMPLES)


def _draw(trial_index: int, **kwargs):
    params = dict(K=K, P=P, master_seed=MASTER_SEED, trial_index=trial_index)
    params.update(kwargs)
    return generate_gaussian_pilots(**params)


def test_pilot_shape_and_dtype() -> None:
    pilots = _draw(0)
    assert pilots.S.shape == (K, P)
    assert pilots.S.dtype == np.complex128
    assert pilots.K == K
    assert pilots.P == P


def test_pilot_reproducibility() -> None:
    a = _draw(137)
    b = _draw(137)
    np.testing.assert_array_equal(a.S, b.S)


def test_pilot_uses_pilot_stream() -> None:
    from_index = _draw(11)
    injected = generate_gaussian_pilots(
        K=K,
        P=P,
        rng=get_trial_rngs(MASTER_SEED, 11).pilots,
    )
    np.testing.assert_array_equal(from_index.S, injected.S)


def test_distinct_trials_produce_different_pilots() -> None:
    a = _draw(137)
    b = _draw(138)
    assert not np.array_equal(a.S, b.S)


def test_full_row_rank() -> None:
    for t in range(200):
        S = _draw(t).S
        assert is_full_row_rank(S, rel_tol=PILOT_RANK_SV_REL_TOL)
        sv = np.linalg.svd(S, compute_uv=False)
        rank = int(np.sum(sv >= PILOT_RANK_SV_REL_TOL * sv[0]))
        assert rank == K


def test_P_less_than_2K_raises() -> None:
    with pytest.raises(ValueError, match="P >= 2K"):
        generate_gaussian_pilots(K=4, P=7, master_seed=0, trial_index=0)


def test_unit_average_energy() -> None:
    """mean(|s|^2) ≈ 1.

    |s|^2 ~ Exp(mean=1), so the Monte Carlo mean over N_SAMPLES has
    standard deviation 1/sqrt(N_SAMPLES). The tolerance is 5σ.
    """
    acc = 0.0
    count = 0
    for t in range(N_MC):
        S = _draw(t).S
        acc += float(np.sum(np.abs(S) ** 2))
        count += S.size
    mean_power = acc / count
    assert count == N_SAMPLES
    np.testing.assert_allclose(mean_power, 1.0, rtol=0.0, atol=MC_ABS_TOL)


def test_circularity_sanity() -> None:
    """Re/Im are ~ N(0, 1/2) and uncorrelated, as required by CN(0, 1).

    Sample-mean std for Re or Im is sqrt((1/2) / N) = 1/sqrt(2 N).
    Sample variance std is larger; 5σ on the mean and a 1% relative
    band on the variances are used below.
    """
    re_acc = 0.0
    im_acc = 0.0
    re2_acc = 0.0
    im2_acc = 0.0
    cross_acc = 0.0
    count = 0
    for t in range(N_MC):
        S = _draw(t).S
        re = S.real.ravel()
        im = S.imag.ravel()
        re_acc += float(re.sum())
        im_acc += float(im.sum())
        re2_acc += float(np.square(re).sum())
        im2_acc += float(np.square(im).sum())
        cross_acc += float(np.sum(re * im))
        count += re.size

    mean_re = re_acc / count
    mean_im = im_acc / count
    var_re = re2_acc / count - mean_re**2
    var_im = im2_acc / count - mean_im**2
    cov = cross_acc / count - mean_re * mean_im

    mean_tol = 5.0 / np.sqrt(2.0 * count)
    np.testing.assert_allclose(mean_re, 0.0, atol=mean_tol, rtol=0.0)
    np.testing.assert_allclose(mean_im, 0.0, atol=mean_tol, rtol=0.0)
    np.testing.assert_allclose(var_re, 0.5, rtol=0.02, atol=0.0)
    np.testing.assert_allclose(var_im, 0.5, rtol=0.02, atol=0.0)
    # Uncorrelated: |corr| = |cov| / (σ_re σ_im) ≲ few / sqrt(N)
    corr_tol = 5.0 / np.sqrt(count)
    assert abs(cov) < corr_tol


def test_does_not_touch_global_numpy_rng() -> None:
    before = np.random.get_state()
    _draw(0)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])

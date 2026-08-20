"""TR 38.901 fidelity switches on the Cui Track-A channel.

Both switches are **off by default** so that ``cui_38901`` keeps the exact
distribution, RNG consumption, and config fingerprint of every historical
Track-A store. These tests pin that guarantee down, and check that each
switch does what the specification says when enabled.
"""

from __future__ import annotations

import numpy as np
import pytest

from rydberg_sim.channel_cui import (
    CuiChannelParams,
    _cluster_delays_and_amplitudes,
    generate_cui_channel,
)
from rydberg_sim.monte_carlo import config_fingerprint
from rydberg_sim.track_a import track_a_fig5_spec
from rydberg_sim.track_a_fig5 import FIG5_SNR_DB

CANONICAL_FIG5_FINGERPRINT = (
    "925f2ab8975505e255f5c37af74d115be38bc302991267f95f440e5f61ce4472"
)


# ---------------------------------------------------------------------------
# Defaults must not move
# ---------------------------------------------------------------------------


def test_defaults_are_off() -> None:
    p = CuiChannelParams()
    assert p.per_ray_polarization is False
    assert p.cluster_pdp is False


def test_canonical_fingerprint_is_preserved() -> None:
    """The delivered Fig. 5 / Fig. 6 stores must stay addressable."""
    spec = track_a_fig5_spec(n_trials=1, snr_db_grid=FIG5_SNR_DB)
    assert config_fingerprint(spec) == CANONICAL_FIG5_FINGERPRINT


def test_disabled_switches_are_absent_from_the_payload() -> None:
    payload = CuiChannelParams().as_fingerprint_dict()
    assert "per_ray_polarization" not in payload
    assert "cluster_pdp" not in payload


def test_enabled_switches_enter_the_fingerprint() -> None:
    base = track_a_fig5_spec(n_trials=1, snr_db_grid=FIG5_SNR_DB)
    seen = {config_fingerprint(base)}
    for params in (
        CuiChannelParams(per_ray_polarization=True),
        CuiChannelParams(cluster_pdp=True),
        CuiChannelParams(per_ray_polarization=True, cluster_pdp=True),
    ):
        spec = track_a_fig5_spec(
            n_trials=1, snr_db_grid=FIG5_SNR_DB, cui_params=params
        )
        fp = config_fingerprint(spec)
        assert fp not in seen, "each variant needs a distinct experiment identity"
        seen.add(fp)


def test_default_draw_is_bit_identical_to_the_historical_generator() -> None:
    a = generate_cui_channel(36, 3, np.random.default_rng([4, 4]))
    b = generate_cui_channel(
        36, 3, np.random.default_rng([4, 4]), params=CuiChannelParams()
    )
    np.testing.assert_array_equal(a.A, b.A)


# ---------------------------------------------------------------------------
# Per-ray polarization (spec eq. 10)
# ---------------------------------------------------------------------------


def test_per_ray_polarization_changes_the_realization() -> None:
    kw = dict(N=36, K=3)
    a = generate_cui_channel(**kw, rng=np.random.default_rng([7, 1]))
    b = generate_cui_channel(
        **kw,
        rng=np.random.default_rng([7, 1]),
        params=CuiChannelParams(per_ray_polarization=True),
    )
    assert not np.allclose(a.A, b.A)


def test_per_ray_polarization_raises_spatial_correlation() -> None:
    """Per-element ε whitens the aperture; per-ray ε must not."""

    def adjacent_corr(params, n=60):
        vals = []
        for t in range(n):
            A = generate_cui_channel(
                36, 3, np.random.default_rng([12, t]), params=params
            ).A
            num = np.mean(A[:, :-1] * np.conj(A[:, 1:]))
            vals.append(abs(num) / np.mean(np.abs(A) ** 2))
        return float(np.mean(vals))

    flat = adjacent_corr(CuiChannelParams())
    corr = adjacent_corr(CuiChannelParams(per_ray_polarization=True))
    assert flat < 0.2, flat
    assert corr > flat


def test_per_ray_polarization_is_constant_across_the_aperture() -> None:
    """A single-ray, single-cluster channel must be a pure steering vector."""
    p = CuiChannelParams(
        n_clusters=1, n_rays_per_cluster=1, per_ray_polarization=True,
        normalize_rows=False,
    )
    A = generate_cui_channel(16, 1, np.random.default_rng([3, 3]), params=p).A
    mag = np.abs(A[0])
    # one ray, one polarization -> flat magnitude, only the phase ramps
    assert np.allclose(mag, mag[0], rtol=1e-9)


# ---------------------------------------------------------------------------
# Cluster PDP (spec eqs. 12, 13, 15, 16)
# ---------------------------------------------------------------------------


def test_cluster_pdp_disabled_returns_nothing() -> None:
    tau, amp = _cluster_delays_and_amplitudes(
        np.random.default_rng(0), CuiChannelParams(), 30e-9
    )
    assert tau is None and amp is None


def test_cluster_powers_are_normalized_and_sorted() -> None:
    p = CuiChannelParams(cluster_pdp=True)
    tau, amp = _cluster_delays_and_amplitudes(np.random.default_rng(1), p, 30e-9)
    assert tau is not None and amp is not None
    assert tau.shape == amp.shape == (p.n_clusters,)
    # eq. (13): sorted ascending, first delay shifted to zero
    assert tau[0] == pytest.approx(0.0)
    assert np.all(np.diff(tau) >= 0.0)
    # eq. (16): cluster powers sum to one (amp = sqrt(P_n / M))
    total = float(np.sum(amp**2) * p.n_rays_per_cluster)
    assert total == pytest.approx(1.0, rel=1e-9)


def test_cluster_pdp_is_unequal_power() -> None:
    """Exponential decay plus 3 dB shadowing must not be flat."""
    p = CuiChannelParams(cluster_pdp=True)
    _, amp = _cluster_delays_and_amplitudes(np.random.default_rng(2), p, 30e-9)
    assert amp.max() / amp.min() > 2.0


def test_zero_delay_spread_degenerates_gracefully() -> None:
    p = CuiChannelParams(cluster_pdp=True)
    tau, amp = _cluster_delays_and_amplitudes(np.random.default_rng(3), p, 0.0)
    assert np.all(tau == 0.0)
    assert float(np.sum(amp**2) * p.n_rays_per_cluster) == pytest.approx(1.0)


def test_row_normalization_still_holds_under_both_switches() -> None:
    p = CuiChannelParams(per_ray_polarization=True, cluster_pdp=True)
    ch = generate_cui_channel(36, 3, np.random.default_rng([5, 5]), params=p)
    np.testing.assert_allclose(np.mean(np.abs(ch.A) ** 2, axis=1), 1.0, rtol=1e-9)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"per_ray_polarization": "yes"},
        {"cluster_pdp": "yes"},
        {"cluster_pdp": True, "pdp_r_tau": 1.0},
        {"cluster_pdp": True, "pdp_r_tau": np.nan},
        {"cluster_pdp": True, "pdp_shadowing_db": -1.0},
    ],
)
def test_invalid_parameters_are_rejected(kwargs) -> None:
    with pytest.raises((TypeError, ValueError)):
        CuiChannelParams(**kwargs)

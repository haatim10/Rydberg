"""Track-A Cui channel generator tests. Must not use channel_ula."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from rydberg_sim.channel import generate_ula_channel
from rydberg_sim.channel_cui import (
    CHANNEL_MODEL_CUI,
    CuiChannelParams,
    generate_cui_channel,
    generate_cui_reference,
)
from rydberg_sim.config import SimulationConfig
from rydberg_sim.monte_carlo import ExperimentSpec, config_fingerprint, generate_detection_trial
from rydberg_sim.rng import get_operating_point_rngs


def test_channel_cui_does_not_import_ula_generator() -> None:
    import rydberg_sim.channel_cui as cmod

    src = inspect.getsource(cmod)
    assert "from .channel import" not in src
    assert "from rydberg_sim.channel" not in src
    assert "generate_ula_channel" not in generate_cui_channel.__code__.co_names
    assert cmod.CHANNEL_MODEL_CUI == "cui_38901"


def test_cui_channel_dimensions_finite_and_normalized() -> None:
    rng = get_operating_point_rngs(7, 3, 0.0, 12.0)
    ch = generate_cui_channel(8, 3, rng.channel)
    assert ch.A.shape == (3, 8)
    assert ch.A.dtype == np.complex128
    assert np.all(np.isfinite(ch.A))
    assert ch.channel_model == CHANNEL_MODEL_CUI
    for k in range(3):
        assert float(np.mean(np.abs(ch.A[k]) ** 2)) == pytest.approx(1.0, rel=1e-12)
    assert ch.A.flags.writeable is False
    assert len(ch.theta_deg) == 3
    assert ch.theta_deg[0].shape == (23 * 20,)


def test_cui_channel_reproducible_and_trials_differ() -> None:
    a = generate_cui_channel(6, 2, get_operating_point_rngs(11, 5, 0.0, 12.0).channel)
    b = generate_cui_channel(6, 2, get_operating_point_rngs(11, 5, 0.0, 12.0).channel)
    c = generate_cui_channel(6, 2, get_operating_point_rngs(11, 6, 0.0, 12.0).channel)
    np.testing.assert_array_equal(a.A, b.A)
    assert not np.array_equal(a.A, c.A)


def test_cui_channel_distinct_from_ula() -> None:
    cfg = SimulationConfig.create(N=8, K=2, L=2, beta=1.0, master_seed=11)
    ula = generate_ula_channel(cfg, 5, rng=get_operating_point_rngs(11, 5, 0.0, 12.0).channel)
    cui = generate_cui_channel(8, 2, get_operating_point_rngs(11, 5, 0.0, 12.0).channel)
    assert ula.G.shape == (8, 2)
    assert cui.A.shape == (2, 8)
    assert ula.G.shape != cui.A.shape
    assert not np.allclose(ula.G, cui.A.T)


def test_cui_reference_power_matches_rsr() -> None:
    rng = get_operating_point_rngs(3, 1, 0.0, 12.0)
    b = generate_cui_reference(16, rng.reference, rsr_db=12.0)
    assert b.shape == (16,)
    assert np.all(np.isfinite(b))
    assert float(np.mean(np.abs(b) ** 2)) == pytest.approx(10.0 ** 1.2, rel=1e-12)


def test_track_a_world_uses_cui_not_ula() -> None:
    spec = ExperimentSpec(
        experiment="iso",
        track="A",
        cfg=SimulationConfig.create(N=6, K=2, L=1, beta=1.0, master_seed=9),
        P=1,
        vartheta=0.0,
        snr_db_grid=(0.0,),
        rsr_db_grid=(12.0,),
        n_trials=1,
        algorithms=("genie_zf",),
        channel_model="cui_38901",
    )
    world = generate_detection_trial(spec, 0, 0.0, 12.0)
    assert world.channel_model == "cui_38901"
    assert world.A.shape == (2, 6)
    assert "generate_cui_channel" in generate_detection_trial.__code__.co_names
    assert "generate_ula_channel" not in generate_detection_trial.__code__.co_names


def test_all_algorithms_see_the_same_world() -> None:
    spec = ExperimentSpec(
        experiment="crn",
        track="A",
        cfg=SimulationConfig.create(N=8, K=2, L=1, beta=1.0, master_seed=4),
        P=1,
        vartheta=0.0,
        snr_db_grid=(5.0,),
        rsr_db_grid=(12.0,),
        n_trials=1,
        algorithms=("biased_gs", "em_gs", "genie_zf"),
        max_iter=3,
        channel_model="cui_38901",
        write_ber=False,
    )
    w1 = generate_detection_trial(spec, 1, 5.0, 12.0)
    w2 = generate_detection_trial(spec, 1, 5.0, 12.0)
    np.testing.assert_array_equal(w1.A, w2.A)
    np.testing.assert_array_equal(w1.s, w2.s)
    np.testing.assert_array_equal(w1.b, w2.b)
    np.testing.assert_array_equal(w1.w, w2.w)
    np.testing.assert_array_equal(w1.z, w2.z)
    from rydberg_sim.monte_carlo import evaluate_detection_algorithm

    rows_a, _ = evaluate_detection_algorithm(w1, "biased_gs", spec)
    rows_b, _ = evaluate_detection_algorithm(w1, "em_gs", spec)
    np.testing.assert_array_equal(w1.z, w2.z)
    assert rows_a[0]["algorithm"] != rows_b[0]["algorithm"]
    assert rows_a[0]["sigma2"] == rows_b[0]["sigma2"]


def test_track_a_b_fingerprints_differ() -> None:
    cfg = SimulationConfig.create(N=8, K=2, L=2, beta=1.0, master_seed=1)
    spec_a = ExperimentSpec(
        experiment="fp",
        track="A",
        cfg=cfg,
        P=8,
        vartheta=0.3,
        snr_db_grid=(0.0,),
        rsr_db_grid=(12.0,),
        n_trials=1,
        algorithms=("genie_zf",),
        channel_model="cui_38901",
    )
    spec_b = ExperimentSpec(
        experiment="fp",
        track="B",
        cfg=cfg,
        P=8,
        vartheta=0.3,
        snr_db_grid=(0.0,),
        rsr_db_grid=(12.0,),
        n_trials=1,
        algorithms=("linearised_ls",),
        channel_model="ula_geometric",
    )
    assert spec_a.channel_model == "cui_38901"
    assert spec_b.channel_model == "ula_geometric"
    assert config_fingerprint(spec_a) != config_fingerprint(spec_b)


def test_wrong_channel_model_is_rejected() -> None:
    cfg = SimulationConfig.create(N=8, K=2, L=2, beta=1.0, master_seed=1)
    with pytest.raises(ValueError, match="cui_38901"):
        ExperimentSpec(
            experiment="bad",
            track="A",
            cfg=cfg,
            P=1,
            vartheta=0.0,
            snr_db_grid=(0.0,),
            rsr_db_grid=(12.0,),
            n_trials=1,
            algorithms=("genie_zf",),
            channel_model="ula_geometric",
        )
    with pytest.raises(ValueError, match="ula_geometric"):
        ExperimentSpec(
            experiment="bad",
            track="B",
            cfg=cfg,
            P=8,
            vartheta=0.0,
            snr_db_grid=(0.0,),
            rsr_db_grid=(12.0,),
            n_trials=1,
            algorithms=("linearised_ls",),
            channel_model="cui_38901",
        )


def test_cm_zf_still_unavailable() -> None:
    cfg = SimulationConfig.create(N=8, K=2, L=1, beta=1.0, master_seed=1)
    with pytest.raises(NotImplementedError, match="cm_zf"):
        ExperimentSpec(
            experiment="cm",
            track="A",
            cfg=cfg,
            P=1,
            vartheta=0.0,
            snr_db_grid=(0.0,),
            rsr_db_grid=(12.0,),
            n_trials=1,
            algorithms=("cm_zf",),
            channel_model="cui_38901",
        )


def test_cui_params_table_i_defaults() -> None:
    p = CuiChannelParams()
    assert p.n_clusters == 23
    assert p.n_rays_per_cluster == 20
    assert p.carrier_hz == 5.0e9
    assert p.channel_model == "cui_38901"

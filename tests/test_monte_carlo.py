"""Step 14 acceptance tests: Monte Carlo harness, CRN, resume, aggregation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from rydberg_sim import (
    ConfigFingerprintError,
    ExperimentSpec,
    SimulationConfig,
    TrackCNotImplementedError,
    adaptive_ber_budget_reached,
    AdaptiveBerPolicy,
    aggregate_result_table,
    channel_trials_equal,
    config_fingerprint,
    generate_channel_estimation_trial,
    generate_detection_trial,
    generate_track_c_trial,
    get_operating_point_rngs,
    load_result_table,
    run_experiment,
    sort_result_rows,
)
from rydberg_sim.monte_carlo import (
    RESULT_COLUMNS,
    evaluate_channel_algorithm,
    result_key,
)
from rydberg_sim.rng import db_to_key, operating_point_spawn_key


def _cfg(*, P_unused: int = 8, master_seed: int = 20260818) -> SimulationConfig:
    return SimulationConfig.create(
        N=4,
        K=2,
        L=2,
        beta=1.0,
        master_seed=master_seed,
        c=1.0,
    )


def _spec(
    tmp_name: str = "step14",
    *,
    P: int = 8,
    n_trials: int = 2,
    snr_db_grid: tuple[float, ...] = (0.0,),
    rsr_db_grid: tuple[float, ...] = (20.0,),
    algorithms: tuple[str, ...] = ("linearised_ls",),
    max_iter: int = 2,
    track: str = "B",
    master_seed: int = 20260818,
    store_diagnostics: bool = False,
    diagnostic_trials: tuple[int, ...] = (),
) -> ExperimentSpec:
    return ExperimentSpec(
        experiment=tmp_name,
        track=track,  # type: ignore[arg-type]
        cfg=_cfg(master_seed=master_seed),
        P=P,
        vartheta=0.3,
        snr_db_grid=snr_db_grid,
        rsr_db_grid=rsr_db_grid,
        n_trials=n_trials,
        algorithms=algorithms,
        max_iter=max_iter,
        ridge=0.0,
        qam_M=4,
        store_diagnostics=store_diagnostics,
        diagnostic_trials=diagnostic_trials,
    )


def _row_identity(rows: list[dict]) -> list[tuple]:
    sorted_rows = sort_result_rows(rows)
    identity = []
    for row in sorted_rows:
        identity.append(
            tuple((col, row[col]) for col in RESULT_COLUMNS)
        )
    return identity


def test_operating_point_key_is_stable_and_avoids_hash() -> None:
    assert db_to_key(-5.0, "snr_db") == -5000 + 1_000_000
    assert operating_point_spawn_key(137, -5.0, 20.0) == (
        137,
        db_to_key(-5.0),
        db_to_key(20.0),
    )
    # Distinct from the Step-1 trial-only key (trial_index,).
    rng_a = get_operating_point_rngs(7, 137, -5.0, 20.0)
    rng_b = get_operating_point_rngs(7, 137, -5.0, 20.0)
    np.testing.assert_array_equal(
        rng_a.channel.standard_normal(8),
        rng_b.channel.standard_normal(8),
    )


def test_operating_point_loop_order_does_not_change_streams() -> None:
    draws = {}
    for snr in (5.0, 0.0, -5.0):
        rngs = get_operating_point_rngs(11, 3, snr, 10.0)
        draws[snr] = rngs.noise.standard_normal(16)
    draws_fwd = {}
    for snr in (-5.0, 0.0, 5.0):
        rngs = get_operating_point_rngs(11, 3, snr, 10.0)
        draws_fwd[snr] = rngs.noise.standard_normal(16)
    for snr in draws:
        np.testing.assert_array_equal(draws[snr], draws_fwd[snr])


def test_independent_draws_per_snr_not_shared_channel() -> None:
    spec = _spec()
    a = generate_channel_estimation_trial(spec, 4, -5.0, 20.0)
    b = generate_channel_estimation_trial(spec, 4, 0.0, 20.0)
    assert not np.array_equal(a.G, b.G)
    assert not np.array_equal(a.Z, b.Z)


def test_one_trial_one_world_is_immutable_and_reproducible() -> None:
    spec = _spec()
    a = generate_channel_estimation_trial(spec, 7, 0.0, 20.0)
    b = generate_channel_estimation_trial(spec, 7, 0.0, 20.0)
    assert channel_trials_equal(a, b)
    assert a.G.flags.writeable is False
    assert a.S.flags.writeable is False
    assert a.B.flags.writeable is False
    assert a.W.flags.writeable is False
    assert a.Z.flags.writeable is False
    with pytest.raises(ValueError):
        a.G[0, 0] = 0.0


def test_crn_algorithms_consume_identical_realization() -> None:
    spec = _spec(algorithms=("biased_gs", "em_gs"), max_iter=2)
    world = generate_channel_estimation_trial(spec, 1, 5.0, 20.0)
    z_before = world.Z.copy()
    g_before = world.G.copy()
    s_before = world.S.copy()
    rows_gs, _ = evaluate_channel_algorithm(world, "biased_gs", spec)
    rows_em, _ = evaluate_channel_algorithm(world, "em_gs", spec)
    np.testing.assert_array_equal(world.Z, z_before)
    np.testing.assert_array_equal(world.G, g_before)
    np.testing.assert_array_equal(world.S, s_before)
    np.testing.assert_array_equal(world.B, world.B)
    assert rows_gs[0]["algorithm"] == "biased_gs"
    assert rows_em[0]["algorithm"] == "em_gs"
    # Same world metadata; only the algorithm (and therefore the metric) differs.
    assert rows_gs[0]["trial"] == rows_em[0]["trial"]
    assert rows_gs[0]["sigma2"] == rows_em[0]["sigma2"]
    world2 = generate_channel_estimation_trial(spec, 1, 5.0, 20.0)
    assert channel_trials_equal(world, world2)


def test_serial_vs_parallel_trial_137_worlds_identical() -> None:
    spec = _spec()

    def _make() -> object:
        return generate_channel_estimation_trial(spec, 137, 0.0, 20.0)

    serial = _make()
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(_make)
        fut_b = pool.submit(_make)
        par_a = fut_a.result()
        par_b = fut_b.result()
    assert channel_trials_equal(serial, par_a)
    assert channel_trials_equal(serial, par_b)
    np.testing.assert_array_equal(serial.G, par_a.G)
    np.testing.assert_array_equal(serial.S, par_a.S)
    np.testing.assert_array_equal(serial.B, par_a.B)
    np.testing.assert_array_equal(serial.W, par_a.W)
    np.testing.assert_array_equal(serial.Z, par_a.Z)


def test_track_a_world_generator_is_reproducible() -> None:
    spec = _spec(track="A", algorithms=("genie_zf",), P=8)
    a = generate_detection_trial(spec, 2, 5.0, 15.0)
    b = generate_detection_trial(spec, 2, 5.0, 15.0)
    np.testing.assert_array_equal(a.G, b.G)
    np.testing.assert_array_equal(a.s, b.s)
    np.testing.assert_array_equal(a.z, b.z)
    np.testing.assert_array_equal(a.w, b.w)
    assert a.G.flags.writeable is False


def test_track_c_not_executed() -> None:
    with pytest.raises(TrackCNotImplementedError):
        generate_track_c_trial()
    with pytest.raises(TrackCNotImplementedError):
        _spec(track="C", algorithms=("biased_gs",))


def test_exact_reproducibility_of_long_table(tmp_path: Path) -> None:
    spec = _spec(
        "repro",
        n_trials=2,
        snr_db_grid=(0.0, 5.0),
        algorithms=("linearised_ls", "biased_gs"),
        max_iter=2,
    )
    p1 = run_experiment(spec, tmp_path / "a", n_workers=1)
    p2 = run_experiment(spec, tmp_path / "b", n_workers=1)
    assert _row_identity(load_result_table(p1)) == _row_identity(load_result_table(p2))


def test_loop_order_independence_of_snr_grid(tmp_path: Path) -> None:
    fwd = _spec(
        "loop",
        n_trials=2,
        snr_db_grid=(-5.0, 0.0, 5.0),
        algorithms=("linearised_ls",),
    )
    rev = _spec(
        "loop",
        n_trials=2,
        snr_db_grid=(5.0, 0.0, -5.0),
        algorithms=("linearised_ls",),
    )
    p1 = run_experiment(fwd, tmp_path / "fwd")
    p2 = run_experiment(rev, tmp_path / "rev")
    assert _row_identity(load_result_table(p1)) == _row_identity(load_result_table(p2))


def test_algorithm_order_independence(tmp_path: Path) -> None:
    a = _spec(
        "algord",
        n_trials=2,
        snr_db_grid=(0.0,),
        algorithms=("biased_gs", "em_gs"),
        max_iter=2,
    )
    b = _spec(
        "algord",
        n_trials=2,
        snr_db_grid=(0.0,),
        algorithms=("em_gs", "biased_gs"),
        max_iter=2,
    )
    p1 = run_experiment(a, tmp_path / "gs_first")
    p2 = run_experiment(b, tmp_path / "em_first")
    assert _row_identity(load_result_table(p1)) == _row_identity(load_result_table(p2))


def test_serial_vs_parallel_result_table(tmp_path: Path) -> None:
    spec = _spec(
        "par",
        n_trials=4,
        snr_db_grid=(-5.0, 5.0),
        algorithms=("linearised_ls", "biased_gs"),
        max_iter=2,
    )
    p_serial = run_experiment(spec, tmp_path / "serial", n_workers=1)
    p_par = run_experiment(spec, tmp_path / "parallel", n_workers=3)
    assert _row_identity(load_result_table(p_serial)) == _row_identity(
        load_result_table(p_par)
    )


def test_resume_does_not_duplicate_and_matches_fresh_run(tmp_path: Path) -> None:
    spec10 = _spec(
        "resume",
        n_trials=10,
        snr_db_grid=(0.0,),
        algorithms=("linearised_ls",),
    )
    spec20 = _spec(
        "resume",
        n_trials=20,
        snr_db_grid=(0.0,),
        algorithms=("linearised_ls",),
    )
    out = tmp_path / "resume_run"
    run_experiment(spec10, out)
    rows10 = load_result_table(out / "results.csv")
    assert {r["trial"] for r in rows10} == set(range(10))
    assert len(rows10) == 10

    run_experiment(spec20, out)
    rows20 = load_result_table(out / "results.csv")
    trials = [r["trial"] for r in rows20]
    assert len(trials) == 20
    assert trials.count(0) == 1
    assert set(trials) == set(range(20))

    fresh = run_experiment(spec20, tmp_path / "fresh20")
    assert _row_identity(rows20) == _row_identity(load_result_table(fresh))

    agg_resume = aggregate_result_table(rows20)
    agg_fresh = aggregate_result_table(load_result_table(fresh))
    assert len(agg_resume) == 1
    assert agg_resume[0].value_linear == pytest.approx(agg_fresh[0].value_linear)
    assert agg_resume[0].n_ok == 20


def test_aggregation_ratio_of_sums_not_mean_of_ratios() -> None:
    def _row(
        trial: int,
        *,
        metric: str,
        error: float | None = None,
        true_e: float | None = None,
        exp_e: float | None = None,
        bit_errors: int | None = None,
        bit_count: int | None = None,
    ) -> dict:
        return {
            "experiment": "fake",
            "config_fingerprint": "abc",
            "track": "B",
            "trial": trial,
            "snr_db": 0.0,
            "rsr_db": 20.0,
            "N": 4,
            "K": 2,
            "P": 8,
            "modulation": "n/a",
            "algorithm": "toy",
            "metric": metric,
            "value": 0.0,
            "error_energy": error,
            "true_energy": true_e,
            "expected_symbol_energy": exp_e,
            "bit_errors": bit_errors,
            "bit_count": bit_count,
            "status": "ok",
            "error_type": "",
            "error_message": "",
            "master_seed": 1,
            "sigma2": 1.0,
            "alpha_b_abs": 1.0,
            "max_iter": 1,
        }

    channel_rows = [
        _row(0, metric="channel_nmse", error=1.0, true_e=1.0),
        _row(1, metric="channel_nmse", error=1.0, true_e=1.0),
        _row(2, metric="channel_nmse", error=8.0, true_e=100.0),
    ]
    ch = aggregate_result_table(channel_rows)[0]
    assert ch.metric == "channel_nmse"
    assert ch.value_linear == pytest.approx(10.0 / 102.0)
    mean_ratios = (1.0 + 1.0 + 0.08) / 3.0
    assert ch.value_linear != pytest.approx(mean_ratios)
    assert ch.value_db == pytest.approx(10.0 * np.log10(10.0 / 102.0))

    det_rows = [
        _row(0, metric="detection_nmse", error=1.0, exp_e=2.0),
        _row(1, metric="detection_nmse", error=3.0, exp_e=2.0),
    ]
    det = aggregate_result_table(det_rows)[0]
    assert det.value_linear == pytest.approx(4.0 / 4.0)

    ber_rows = [
        _row(0, metric="ber", bit_errors=1, bit_count=10),
        _row(1, metric="ber", bit_errors=0, bit_count=1000),
    ]
    ber = aggregate_result_table(ber_rows)[0]
    assert ber.value_linear == pytest.approx(1.0 / 1010.0)
    mean_ber = (0.1 + 0.0) / 2.0
    assert ber.value_linear != pytest.approx(mean_ber)
    assert ber.total_bit_errors == 1
    assert ber.total_bit_count == 1010
    assert ber.wilson is not None
    assert ber.wilson.high > ber.value_linear


def test_config_fingerprint_changes_with_P() -> None:
    a = _spec(P=20, n_trials=1)
    b = _spec(P=30, n_trials=1)
    assert config_fingerprint(a) != config_fingerprint(b)
    # n_trials / SNR grid / algorithm list are not material.
    c = _spec(P=20, n_trials=99, snr_db_grid=(0.0, 5.0), algorithms=("biased_gs", "linearised_ls"))
    assert config_fingerprint(a) == config_fingerprint(c)


def test_fingerprint_mismatch_refuses_incompatible_append(tmp_path: Path) -> None:
    a = _spec("fp", P=20, n_trials=1, algorithms=("linearised_ls",))
    b = _spec("fp", P=30, n_trials=1, algorithms=("linearised_ls",))
    out = tmp_path / "fp_run"
    run_experiment(a, out)
    with pytest.raises(ConfigFingerprintError, match="config_fingerprint"):
        run_experiment(b, out)


def test_failed_trial_is_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = _spec("fail", n_trials=2, algorithms=("linearised_ls",))

    from rydberg_sim import monte_carlo as mc

    def _boom(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("injected numerical failure")

    monkeypatch.setattr(mc, "linearised_closed_form_ls", _boom)
    path = run_experiment(spec, tmp_path / "fail_run")
    rows = load_result_table(path)
    assert len(rows) == 2
    assert all(r["status"] == "failed" for r in rows)
    assert all(r["error_type"] == "RuntimeError" for r in rows)
    assert "injected numerical failure" in rows[0]["error_message"]


def test_adaptive_ber_interface_only() -> None:
    policy = AdaptiveBerPolicy(min_errors=50, max_bits=10_000, max_trials=1000)
    assert not adaptive_ber_budget_reached(
        total_bit_errors=10, total_bit_count=100, n_trials=5, policy=policy
    )
    assert adaptive_ber_budget_reached(
        total_bit_errors=50, total_bit_count=200, n_trials=8, policy=policy
    )
    assert adaptive_ber_budget_reached(
        total_bit_errors=1, total_bit_count=10_000, n_trials=8, policy=policy
    )
    assert adaptive_ber_budget_reached(
        total_bit_errors=1, total_bit_count=10, n_trials=1000, policy=policy
    )


def test_result_key_includes_configuration_identity() -> None:
    spec = _spec()
    k1 = result_key(
        config_fingerprint=spec.fingerprint,
        experiment=spec.experiment,
        track=spec.track,
        trial=3,
        snr_db=0.0,
        rsr_db=20.0,
        algorithm="linearised_ls",
    )
    k2 = result_key(
        config_fingerprint=spec.fingerprint,
        experiment=spec.experiment,
        track=spec.track,
        trial=3,
        snr_db=5.0,
        rsr_db=20.0,
        algorithm="linearised_ls",
    )
    assert k1 != k2
    assert k1[0] == spec.fingerprint


def test_tiny_track_b_integration(tmp_path: Path) -> None:
    spec = _spec(
        "tinyB",
        n_trials=2,
        snr_db_grid=(-5.0, 5.0),
        rsr_db_grid=(20.0,),
        algorithms=("biased_gs", "em_gs", "linearised_ls"),
        max_iter=3,
        store_diagnostics=True,
        diagnostic_trials=(0,),
    )
    path = run_experiment(spec, tmp_path / "tinyB")
    rows = load_result_table(path)
    assert {r["algorithm"] for r in rows} == {"biased_gs", "em_gs", "linearised_ls"}
    assert {r["trial"] for r in rows} == {0, 1}
    assert all(r["status"] == "ok" for r in rows)
    assert all(r["metric"] == "channel_nmse" for r in rows)
    assert all(r["error_energy"] is not None and r["error_energy"] >= 0.0 for r in rows)
    assert all(r["true_energy"] is not None and r["true_energy"] > 0.0 for r in rows)
    agg = aggregate_result_table(rows)
    assert len(agg) == 6  # 2 SNR × 3 algorithms
    diag_dir = tmp_path / "tinyB" / "diagnostics"
    assert diag_dir.is_dir()
    assert list(diag_dir.glob("*.json"))


def test_unimplemented_algorithms_rejected() -> None:
    with pytest.raises(NotImplementedError, match="gd"):
        _spec(algorithms=("gd",))

"""Step 1 acceptance tests: per-trial RNG policy."""

from __future__ import annotations

import itertools

import numpy as np

from rydberg_sim import get_trial_rngs


MASTER_SEED = 20260818
N_DRAW = 64


def _draw_all(master_seed: int, trial_index: int) -> dict[str, np.ndarray]:
    rngs = get_trial_rngs(master_seed, trial_index)
    return {
        "channel": rngs.channel.standard_normal(N_DRAW),
        "pilots": rngs.pilots.standard_normal(N_DRAW),
        "reference": rngs.reference.standard_normal(N_DRAW),
        "noise": rngs.noise.standard_normal(N_DRAW),
        "data": rngs.data.standard_normal(N_DRAW),
        "solver": rngs.solver.standard_normal(N_DRAW),
    }


def test_isolated_reproducibility() -> None:
    """Creating RNGs for trial 137 twice yields bit-identical sequences."""
    a = _draw_all(MASTER_SEED, 137)
    b = _draw_all(MASTER_SEED, 137)
    for name in a:
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)


def test_execution_order_independence() -> None:
    """Trial 137 alone matches trial 137 after accessing trials 0..136."""
    direct = _draw_all(MASTER_SEED, 137)

    after_prefix = None
    for t in range(137):
        _draw_all(MASTER_SEED, t)
    after_prefix = _draw_all(MASTER_SEED, 137)

    for name in direct:
        np.testing.assert_array_equal(
            direct[name], after_prefix[name], err_msg=name
        )


def test_distinct_trials() -> None:
    """Trial 137 and trial 138 must not share a channel sequence."""
    a = get_trial_rngs(MASTER_SEED, 137).channel.standard_normal(N_DRAW)
    b = get_trial_rngs(MASTER_SEED, 138).channel.standard_normal(N_DRAW)
    assert not np.array_equal(a, b)


def test_independent_component_streams() -> None:
    """Within one trial, component streams are deterministic but distinct."""
    first = _draw_all(MASTER_SEED, 42)
    second = _draw_all(MASTER_SEED, 42)
    names = ("channel", "pilots", "reference", "noise", "data", "solver")
    for name in names:
        np.testing.assert_array_equal(first[name], second[name], err_msg=name)
    for left, right in itertools.combinations(names, 2):
        assert not np.array_equal(first[left], first[right]), (left, right)


def test_data_stream_does_not_retune_legacy_streams() -> None:
    """Appending the data child must not change spawn children 0..3."""
    entropy, trial = MASTER_SEED, 137
    four = np.random.SeedSequence(entropy=entropy, spawn_key=(trial,)).spawn(4)
    five = np.random.SeedSequence(entropy=entropy, spawn_key=(trial,)).spawn(5)
    for i in range(4):
        a = np.random.default_rng(four[i]).standard_normal(N_DRAW)
        b = np.random.default_rng(five[i]).standard_normal(N_DRAW)
        np.testing.assert_array_equal(a, b, err_msg=f"child {i}")


def test_solver_stream_does_not_retune_legacy_streams() -> None:
    """Appending the solver child must not change spawn children 0..4."""
    entropy, trial = MASTER_SEED, 137
    five = np.random.SeedSequence(entropy=entropy, spawn_key=(trial,)).spawn(5)
    six = np.random.SeedSequence(entropy=entropy, spawn_key=(trial,)).spawn(6)
    for i in range(5):
        a = np.random.default_rng(five[i]).standard_normal(N_DRAW)
        b = np.random.default_rng(six[i]).standard_normal(N_DRAW)
        np.testing.assert_array_equal(a, b, err_msg=f"child {i}")


def test_does_not_touch_global_numpy_rng() -> None:
    """The policy must not use np.random.seed / the global RandomState."""
    before = np.random.get_state()
    _draw_all(MASTER_SEED, 0)
    _draw_all(MASTER_SEED, 1)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]

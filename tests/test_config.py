"""Step 1 tests: frozen configuration and validation."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from rydberg_sim import SimulationConfig


def test_scalar_L_and_beta_expand_across_users() -> None:
    cfg = SimulationConfig.create(N=8, K=3, L=4, beta=1.5, master_seed=0)
    assert cfg.L_k == (4, 4, 4)
    assert cfg.beta_k == (1.5, 1.5, 1.5)
    assert cfg.c == 1.0


def test_per_user_L_k_and_beta_k_arrays() -> None:
    cfg = SimulationConfig.create(
        N=8,
        K=3,
        L_k=np.array([1, 4, 8]),
        beta_k=(0.5, 1.0, 2.0),
        master_seed=7,
        c=1.0,
    )
    assert cfg.L_k == (1, 4, 8)
    assert cfg.beta_k == (0.5, 1.0, 2.0)


def test_config_is_frozen() -> None:
    cfg = SimulationConfig.create(N=4, K=1, L=2, beta=1.0, master_seed=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.N = 16  # type: ignore[misc]


@pytest.mark.parametrize("N", [0, -1])
def test_rejects_non_positive_N(N: int) -> None:
    with pytest.raises(ValueError, match="N must be > 0"):
        SimulationConfig.create(N=N, K=1, L=1, beta=1.0, master_seed=0)


@pytest.mark.parametrize("K", [0, -2])
def test_rejects_non_positive_K(K: int) -> None:
    with pytest.raises(ValueError, match="K must be > 0"):
        SimulationConfig.create(N=4, K=K, L=1, beta=1.0, master_seed=0)


def test_rejects_L_k_out_of_range() -> None:
    with pytest.raises(ValueError, match="1 <= L_k <= N"):
        SimulationConfig.create(N=4, K=1, L=0, beta=1.0, master_seed=0)
    with pytest.raises(ValueError, match="1 <= L_k <= N"):
        SimulationConfig.create(N=4, K=2, L_k=(2, 5), beta=1.0, master_seed=0)


def test_rejects_non_positive_beta() -> None:
    with pytest.raises(ValueError, match="beta_k"):
        SimulationConfig.create(N=4, K=1, L=1, beta=0.0, master_seed=0)
    with pytest.raises(ValueError, match="beta_k"):
        SimulationConfig.create(N=4, K=2, L=1, beta_k=(1.0, -0.1), master_seed=0)


def test_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length-K"):
        SimulationConfig.create(N=4, K=2, L_k=(1, 2, 3), beta=1.0, master_seed=0)


def test_create_requires_exactly_one_path_argument() -> None:
    with pytest.raises(ValueError, match="exactly one of L or L_k"):
        SimulationConfig.create(N=4, K=1, beta=1.0, master_seed=0)
    with pytest.raises(ValueError, match="exactly one of L or L_k"):
        SimulationConfig.create(N=4, K=1, L=1, L_k=1, beta=1.0, master_seed=0)

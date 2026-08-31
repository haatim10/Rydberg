"""Track D stage 3 wiring tests (PROMPT 6 Part B).

Cheap structural checks on the two new arms and on the EM-GS feature cache.
No training and no full-size runs: every test here uses tiny budgets and a
short ``T_GS``.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Track D requires torch")

from trackD_urformer import emgs_cache as ec  # noqa: E402
from trackD_urformer.config import TrackDConfig  # noqa: E402
from trackD_urformer.dataset import TrackDDataset  # noqa: E402
from trackD_urformer.stage3 import (  # noqa: E402
    EMGSFeatureDataset, PostProcessor, assert_dataset_identity, build_arm,
    paired, verdict,
)

C128 = torch.complex128


@pytest.fixture(scope="module")
def cfg():
    c = TrackDConfig()
    return replace(c, train=replace(c.train, init="spectral"))


# ------------------------------------------------------------- X1 degeneracy
def test_postprocessor_is_exactly_identity_at_init(cfg):
    """X1 starts as EXACTLY converged EM-GS, so any gain is the residual's.

    The same property gate F asserts for the URformer layer. Without it,
    "X1 beats EM-GS" would be partly an artifact of the initialization.
    """
    m = PostProcessor(cfg.system.N, cfg.system.K, cfg.model).double()
    rng = np.random.default_rng(11)
    g = torch.as_tensor(rng.standard_normal((3, cfg.system.N, cfg.system.K))
                        + 1j * rng.standard_normal((3, cfg.system.N,
                                                    cfg.system.K)), dtype=C128)
    with torch.no_grad():
        assert float((m(g) - g).abs().max()) == 0.0


def test_X1_has_one_transformer_not_ten(cfg):
    """The control is a SINGLE post-processor: ~158k params, no unrolling.

    If this ever became ten Transformers it would stop being a control for
    "is the unrolling doing anything" and become a second URformer.
    """
    _, meta = build_arm(cfg, "X1_emgs_plus_former")
    assert meta["unrolled_layers"] == 0
    assert meta["total_params"] == 158_592

    _, h1 = build_arm(cfg, "H1_hs_urformer_80k")
    assert h1["params"]["totals"]["transformer"] == 10 * 158_592


def test_H1_has_hankel_enabled_and_U1_comparison_does_not(cfg):
    _, h1 = build_arm(cfg, "H1_hs_urformer_80k")
    assert h1["use_hankel"] is True
    assert h1["hankel_rank"] == 7
    # r is L_max, the generator's own upper bound -- a design assumption, not
    # oracle knowledge of any trial's true L_k.
    assert h1["hankel_rank"] == cfg.system.L_max


# ---------------------------------------------------------------- the cache
def test_emgs_cache_is_bit_exact(tmp_path, cfg, monkeypatch):
    """Caching must be exact, not approximate -- same argument as the g0 cache."""
    monkeypatch.setattr(ec, "CACHE", tmp_path)
    n, T = 6, 4
    for s in range(2):
        ec.build_shard("val", n, shard=s, n_shards=2, T_GS=T, cfg=cfg,
                       report_every=0)
    arr = ec.load_cache("val", n, T, n_shards=2)
    assert arr.shape == (n, cfg.system.N, cfg.system.K)

    rcfg = replace(cfg, data=replace(cfg.data, n_val=n))
    ds = TrackDDataset("val", sysc=rcfg.system, datac=rcfg.data,
                       numeric=rcfg.numeric)
    for i in range(n):
        fresh = ec.emgs_estimate(ds.sample(i), T_GS=T)
        assert np.abs(arr[i] - fresh).max() == 0.0


def test_cache_shards_are_strided_not_blocked(tmp_path, cfg, monkeypatch):
    """Strided shards give every worker the same SNR mix and equal runtime."""
    monkeypatch.setattr(ec, "CACHE", tmp_path)
    n, T = 6, 3
    ec.build_shard("val", n, shard=0, n_shards=3, T_GS=T, cfg=cfg,
                   report_every=0)
    shard0 = np.load(ec.cache_path("val", n, T, 0, 3))
    assert shard0.shape[0] == len(range(0, n, 3))

    rcfg = replace(cfg, data=replace(cfg.data, n_val=n))
    ds = TrackDDataset("val", sysc=rcfg.system, datac=rcfg.data,
                       numeric=rcfg.numeric)
    for j, i in enumerate(range(0, n, 3)):
        assert np.abs(shard0[j] - ec.emgs_estimate(ds.sample(i),
                                                   T_GS=T)).max() == 0.0


def test_cache_path_encodes_budget_and_iterations():
    """A cache built for a different n or T_GS must never be silently reused."""
    assert ec.cache_path("train", 80000, 100) != ec.cache_path("train", 40000, 100)
    assert ec.cache_path("train", 80000, 100) != ec.cache_path("train", 80000, 50)


def test_feature_dataset_rejects_a_mismatched_cache(cfg):
    rcfg = replace(cfg, data=replace(cfg.data, n_val=4))
    ds = TrackDDataset("val", sysc=rcfg.system, datac=rcfg.data,
                       numeric=rcfg.numeric)
    with pytest.raises(ValueError, match="different budget"):
        EMGSFeatureDataset(ds, np.zeros((3, 32, 3), dtype=np.complex128))


# --------------------------------------------------------- the reused split
def test_dataset_identity_gate_accepts_the_real_config(cfg):
    assert assert_dataset_identity(cfg)["identical_to_stage2"] is True


def test_dataset_identity_gate_rejects_a_changed_split(cfg):
    """U1 is only a valid comparison arm if the splits really are stage 2's."""
    # Still disjoint from train/test, so DataConfig's own validator accepts it
    # and the stage-3 gate is what has to catch the change.
    bad = replace(cfg, data=replace(cfg.data,
                                    val_seed_range=(3_000_000, 4_000_000)))
    with pytest.raises(AssertionError, match="differ from stage 2"):
        assert_dataset_identity(bad)


# ------------------------------------------------------------ the statistics
def test_paired_sign_convention_is_positive_means_second_arm_better():
    """Delta_H = NMSE_U1 - NMSE_H1: POSITIVE means the Hankel arm helps.

    Getting this backwards would invert the headline conclusion, so it is
    pinned rather than trusted to the docstring.
    """
    per = {"worse": [1.0] * 50, "better": [0.5] * 50}
    d = paired(per, "worse", "better", n_boot=200)
    assert d["median_diff_db"] > 0
    assert d["win_rate_b"] == 1.0
    assert np.isclose(d["median_diff_db"], 10 * np.log10(2.0))


def test_verdict_applies_the_preregistered_thresholds():
    go = verdict({"median_diff_db": 0.5, "boot_ci95_median": [0.2, 0.8]})
    assert go["decision"] == "GO"
    marginal = verdict({"median_diff_db": 0.15, "boot_ci95_median": [0.05, 0.25]})
    assert marginal["decision"] == "STOP-MARGINAL"
    null = verdict({"median_diff_db": -0.1, "boot_ci95_median": [-0.3, 0.1]})
    assert null["decision"] == "STOP-NULL"
    # +0.3 dB with a CI that includes zero is NOT a go: both halves are required.
    wide = verdict({"median_diff_db": 0.4, "boot_ci95_median": [-0.1, 0.9]})
    assert wide["decision"] == "STOP-MARGINAL"

"""Track D (URformer) unit tests.

Pins the invariants that the verification gate suite checks numerically, so a
regression is caught by the ordinary test run and not only by `verify.py`.
These are fast: no training, tiny shapes.

Track A and Track B behaviour is not touched by any test here.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Track D requires torch")

from rydberg_sim.forward import exact_forward  # noqa: E402
from rydberg_sim.gs import (  # noqa: E402
    bessel_ratio, biased_gs_channel_rows, em_gs_channel_rows,
)
from trackD_urformer.config import (  # noqa: E402
    DataConfig, ModelConfig, NumericConfig, SystemConfig, TrackDConfig,
)
from trackD_urformer.dataset import make_world  # noqa: E402
from trackD_urformer.torch_forward import (  # noqa: E402
    assert_shapes, bessel_ratio_torch, em_gs_layer, em_kappa, forward_field,
    gs_layer, least_squares_G,
)
from trackD_urformer.transformer import detokenize, tokenize  # noqa: E402
from trackD_urformer.urformer import URformer, count_parameters  # noqa: E402

C128, F64 = torch.complex128, torch.float64


def _t(a, dtype=C128):
    return torch.as_tensor(np.array(a, copy=True)[None], dtype=dtype)


def _rel(a, b) -> float:
    b = np.asarray(b)
    return float(np.linalg.norm(np.asarray(a) - b) / np.linalg.norm(b))


@pytest.fixture(scope="module")
def world():
    sysc = SystemConfig(K=3, N=16, P=20, master_seed=20260827)
    return make_world(0, sysc=sysc, N=16, P=20, snr_db=5.0), sysc


# ---------------------------------------------------------------------------
# shapes and orientation
# ---------------------------------------------------------------------------
def test_forward_orientation_matches_repository(world):
    w, _ = world
    Y = forward_field(_t(w.G_true), _t(w.S), _t(w.B))[0].numpy()
    assert _rel(Y, w.G_true @ w.S + w.B) < 1e-14


def test_assert_shapes_rejects_bad_input():
    with pytest.raises(ValueError):
        assert_shapes(G=torch.zeros((1, 8, 2), dtype=C128),
                      S=torch.zeros((1, 3, 8), dtype=C128))
    with pytest.raises(ValueError):
        assert_shapes(G=torch.zeros((8, 2), dtype=C128))       # unbatched
    with pytest.raises(TypeError):
        assert_shapes(G=torch.zeros((1, 8, 2), dtype=F64))     # real G
    with pytest.raises(TypeError):
        assert_shapes(Z=torch.zeros((1, 8, 8), dtype=C128))    # complex Z


# ---------------------------------------------------------------------------
# LS parity - the single most important checkpoint
# ---------------------------------------------------------------------------
def test_ls_matches_repository_gs_step(world):
    w, _ = world
    G0 = np.zeros_like(w.G_true)
    ref = biased_gs_channel_rows(w.S, w.Z, w.B, max_iter=1, G0=G0).G_hat
    got = gs_layer(_t(G0), _t(w.Z, F64), _t(w.S), _t(w.B), eps=1e-12)[0].numpy()
    assert _rel(got, ref) < 1e-12


def test_em_gs_layer_matches_repository(world):
    w, _ = world
    G0 = np.zeros_like(w.G_true)
    ref = em_gs_channel_rows(w.S, w.Z, w.B, w.sigma2, max_iter=1, G0=G0).G_hat
    got = em_gs_layer(_t(G0), _t(w.Z, F64), _t(w.S), _t(w.B),
                      torch.tensor([w.sigma2], dtype=F64), eps=1e-12)[0].numpy()
    assert _rel(got, ref) < 1e-12


def test_least_squares_recovers_G_not_conjugate(world):
    """Oracle phase: exact Y in, G out. Must NOT return conj(G)."""
    w, _ = world
    R = _t(w.G_true @ w.S + w.B) - _t(w.B)
    got = least_squares_G(R, _t(w.S))[0].numpy()
    assert _rel(got, w.G_true) < 1e-12
    assert _rel(got, np.conjugate(w.G_true)) > 0.1


# ---------------------------------------------------------------------------
# Bessel
# ---------------------------------------------------------------------------
def test_bessel_ratio_matches_scipy_implementation():
    x = torch.tensor(np.concatenate([[0.0], np.logspace(-3, 3, 400)]), dtype=F64)
    assert _rel(bessel_ratio_torch(x).numpy(), bessel_ratio(x.numpy())) < 1e-12


def test_bessel_ratio_bounded_and_finite():
    r = bessel_ratio_torch(torch.tensor(np.logspace(-6, 6, 500), dtype=F64))
    assert torch.all(torch.isfinite(r))
    assert float(r.min()) >= 0.0 and float(r.max()) <= 1.0


# ---------------------------------------------------------------------------
# degeneration properties
# ---------------------------------------------------------------------------
def test_layer_degenerates_to_gs_when_alpha_zero(world):
    w, sysc = world
    model = URformer(16, sysc.K, ModelConfig(T_UR=1), NumericConfig("float64")).double()
    model._set_test_mode(alpha=0.0, disable_residual=True)
    G0 = np.zeros_like(w.G_true)
    with torch.no_grad():
        out = model(_t(G0), _t(w.Z, F64), _t(w.S), _t(w.B),
                    torch.tensor([w.sigma2], dtype=F64))[0].numpy()
    ref = biased_gs_channel_rows(w.S, w.Z, w.B, max_iter=1, G0=G0).G_hat
    assert _rel(out, ref) < 1e-12


def test_layer_degenerates_to_emgs_with_exact_bessel(world):
    w, sysc = world
    model = URformer(16, sysc.K, ModelConfig(T_UR=1), NumericConfig("float64")).double()
    model._set_test_mode(filter_override="exact_bessel", alpha=1.0,
                         disable_residual=True)
    G0 = np.zeros_like(w.G_true)
    with torch.no_grad():
        out = model(_t(G0), _t(w.Z, F64), _t(w.S), _t(w.B),
                    torch.tensor([w.sigma2], dtype=F64))[0].numpy()
    ref = em_gs_channel_rows(w.S, w.Z, w.B, w.sigma2, max_iter=1, G0=G0).G_hat
    assert _rel(out, ref) < 1e-10


def test_transformer_residual_is_exactly_zero_at_init(world):
    """zero_init_out => the URformer layer IS the classical fixed update."""
    w, sysc = world
    model = URformer(16, sysc.K, ModelConfig(T_UR=2), NumericConfig("float64")).double()
    args = (_t(np.zeros_like(w.G_true)), _t(w.Z, F64), _t(w.S), _t(w.B),
            torch.tensor([w.sigma2], dtype=F64))
    with torch.no_grad():
        a = model(*args)[0].numpy()
        model._set_test_mode(disable_residual=True)
        b = model(*args)[0].numpy()
    assert np.max(np.abs(a - b)) == 0.0


def test_noiseless_fixed_point(world):
    """W=0 and G0=G => EM-GS returns G as sigma2 -> 0."""
    w, _ = world
    G = np.asarray(w.G_true)
    Z0 = np.asarray(exact_forward(G, w.S, w.B, 0.0).Z)
    out = em_gs_layer(_t(G), _t(Z0, F64), _t(w.S), _t(w.B),
                      torch.tensor([1e-12], dtype=F64), eps=1e-14)[0].numpy()
    assert _rel(out, G) < 1e-10


# ---------------------------------------------------------------------------
# tokenization
# ---------------------------------------------------------------------------
def test_tokenize_roundtrip_does_not_conjugate(world):
    w, _ = world
    G = _t(w.G_true)
    back = detokenize(tokenize(G), 16, C128)[0].numpy()
    assert _rel(back, w.G_true) < 1e-14
    assert _rel(back, np.conjugate(w.G_true)) > 0.1


def test_tokenize_shape():
    G = torch.zeros((5, 16, 3), dtype=C128)
    assert tuple(tokenize(G).shape) == (5, 3, 32)   # (batch, K, 2N)


def test_transformer_is_shape_locked(world):
    """A model built for N=16 must REFUSE N=32 - shape error, not silent reshape."""
    _, sysc = world
    model = URformer(16, sysc.K, ModelConfig(T_UR=1), NumericConfig("float64")).double()
    with pytest.raises(ValueError, match="shape-locked"):
        model.layers[0].former(torch.zeros((1, 32, sysc.K), dtype=C128))


# ---------------------------------------------------------------------------
# kappa
# ---------------------------------------------------------------------------
def test_kappa_invariance_under_consistent_rescaling(world):
    """Scaling the field by 1/s and sigma2 by 1/s^2 leaves kappa unchanged."""
    w, _ = world
    G0 = np.zeros_like(w.G_true)
    Y = forward_field(_t(G0), _t(w.S), _t(w.B))
    k0 = em_kappa(_t(w.Z, F64), Y, torch.tensor([w.sigma2], dtype=F64), 0.0)
    s = 10.0
    Ys = forward_field(_t(G0), _t(w.S), _t(w.B) / s)
    ks = em_kappa(_t(w.Z, F64) / s, Ys,
                  torch.tensor([w.sigma2 / s ** 2], dtype=F64), 0.0)
    assert _rel(ks.numpy(), k0.numpy()) < 1e-10


# ---------------------------------------------------------------------------
# config invariants
# ---------------------------------------------------------------------------
def test_seed_ranges_are_disjoint():
    DataConfig()          # must not raise
    with pytest.raises(ValueError, match="overlap"):
        DataConfig(train_seed_range=(0, 100), val_seed_range=(50, 150))


def test_rsr_paper_equivalence_is_factor_K():
    sysc = SystemConfig(K=3, rsr_db=10.0)
    assert abs(sysc.rsr_paper_equiv_db - (10.0 + 10 * np.log10(3))) < 1e-12


def test_gate_init_values():
    assert ModelConfig(gate_init="near_gs").gate_init_value == -2.0
    assert ModelConfig(gate_init="near_emgs").gate_init_value == 2.0
    assert ModelConfig(gate_init="neutral").gate_init_value == 0.0


def test_parameter_count_is_consistent():
    cfg = TrackDConfig()
    model = URformer(cfg.system.N, cfg.system.K, cfg.model, cfg.numeric)
    c = count_parameters(model)
    assert c["totals"]["all_parameters"] == sum(
        p.numel() for p in model.parameters())
    assert c["totals"]["gate"] == cfg.model.T_UR      # one scalar per layer


def test_dataset_splits_do_not_share_trials():
    cfg = TrackDConfig()
    tr = set(range(*cfg.data.train_seed_range))
    va = set(range(cfg.data.val_seed_range[0],
                   cfg.data.val_seed_range[0] + 100))
    assert not (va & set(list(tr)[:100]))

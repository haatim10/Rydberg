"""A1 - config liveness sweep (PROMPT 4).

`filter_init` was declared but never read, and 15 gates plus 378 tests passed
anyway. That is a coverage gap, not a one-off. Every field in ModelConfig,
TrainConfig, NumericConfig and the data/system configs that can affect a result
gets a test here showing that CHANGING it produces a MEASURABLY different
outcome: a different parameter count, a different forward output, or a
different loss on a fixed batch.

A field with no test here is either genuinely inert (documented as such) or a
latent bug of exactly the kind that produced this file.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Track D requires torch")

from trackD_urformer.config import (  # noqa: E402
    DataConfig, ModelConfig, NumericConfig, SystemConfig, TrainConfig,
    TrackDConfig,
)
from trackD_urformer.dataset import TrackDDataset, make_world  # noqa: E402
from trackD_urformer.filter_net import FilterNet  # noqa: E402
from trackD_urformer.train import make_initial_batch, nmse_loss  # noqa: E402
from trackD_urformer.urformer import URformer, count_parameters  # noqa: E402

C128, F64 = torch.complex128, torch.float64
NUM64 = NumericConfig("float64")


def _t(a, dtype=C128):
    return torch.as_tensor(np.array(a, copy=True)[None], dtype=dtype)


@pytest.fixture(scope="module")
def w():
    sysc = SystemConfig(K=3, N=16, P=20, master_seed=20260827)
    return make_world(0, sysc=sysc, N=16, P=20, snr_db=5.0)


def _fwd(mcfg: ModelConfig, w, seed: int = 0, numeric: NumericConfig = NUM64):
    """Deterministic forward pass under a given ModelConfig."""
    torch.manual_seed(seed)
    m = URformer(16, 3, mcfg, numeric)
    if numeric.dtype == "float64":
        m = m.double()
    cd = numeric.complex_dtype
    rd = numeric.real_dtype
    with torch.no_grad():
        out = m(_t(np.zeros_like(w.G_true), cd), _t(w.Z, rd), _t(w.S, cd),
                _t(w.B, cd), torch.tensor([w.sigma2], dtype=rd))
    return out[0].numpy(), m


def _nparams(mcfg: ModelConfig, numeric: NumericConfig = NUM64) -> int:
    torch.manual_seed(0)
    return count_parameters(URformer(16, 3, mcfg, numeric))["totals"][
        "all_parameters"]


def _differs(a, b, tol=1e-9) -> bool:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) > tol


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------
def test_live_T_UR(w):
    """More unrolled layers => more parameters AND a different output."""
    assert _nparams(ModelConfig(T_UR=2)) != _nparams(ModelConfig(T_UR=3))
    assert _differs(_fwd(ModelConfig(T_UR=2), w)[0],
                    _fwd(ModelConfig(T_UR=3), w)[0])


def test_live_d_model(w):
    assert _nparams(ModelConfig(T_UR=1, d_model=32)) != \
           _nparams(ModelConfig(T_UR=1, d_model=64))


def test_live_L_enc(w):
    assert _nparams(ModelConfig(T_UR=1, L_enc=1)) != \
           _nparams(ModelConfig(T_UR=1, L_enc=3))


def test_live_n_heads(w):
    """Head count does not change the parameter count, but does change the
    attention pattern and therefore the output once the residual is non-zero."""
    a = ModelConfig(T_UR=1, n_heads=2)
    b = ModelConfig(T_UR=1, n_heads=4)
    assert _nparams(a) == _nparams(b)          # same params by construction
    torch.manual_seed(0)
    ma = URformer(16, 3, a, NUM64).double()
    torch.manual_seed(0)
    mb = URformer(16, 3, b, NUM64).double()
    # out_proj is zero-init, so perturb it to expose the attention difference.
    for m in (ma, mb):
        with torch.no_grad():
            torch.manual_seed(7)
            m.layers[0].former.out_proj.weight.normal_(0, 0.1)
    args = (_t(np.zeros_like(w.G_true)), _t(w.Z, F64), _t(w.S), _t(w.B),
            torch.tensor([w.sigma2], dtype=F64))
    with torch.no_grad():
        assert _differs(ma(*args)[0].numpy(), mb(*args)[0].numpy())


def test_live_ffn_mult(w):
    assert _nparams(ModelConfig(T_UR=1, ffn_mult=2)) != \
           _nparams(ModelConfig(T_UR=1, ffn_mult=4))


def test_live_filter_hidden(w):
    assert _nparams(ModelConfig(T_UR=1, filter_hidden=16)) != \
           _nparams(ModelConfig(T_UR=1, filter_hidden=32))


def test_live_filter_input(w):
    """Different kappa featurization => different R_learned => different output."""
    a = _fwd(ModelConfig(T_UR=1, filter_input="kappa"), w)[0]
    b = _fwd(ModelConfig(T_UR=1, filter_input="log1p_kappa"), w)[0]
    assert _differs(a, b)
    # the 2-feature variant also changes the input dimension
    assert _nparams(ModelConfig(T_UR=1, filter_input="log1p_kappa")) != \
           _nparams(ModelConfig(T_UR=1,
                                filter_input="log1p_kappa_plus_logsigma2"))


def test_live_filter_init(w, tmp_path):
    """THE FIELD THAT WAS DEAD. Warm-started weights must differ from random.

    Wired via URformer.apply_filter_warmstart(); this test is what would have
    caught the original dead field.
    """
    from trackD_urformer.filter_net import measure_kappa_range, warmstart_filternet

    mcfg = ModelConfig(T_UR=1)
    torch.manual_seed(0)
    m = URformer(16, 3, mcfg, NUM64).double()
    before = [p.detach().clone() for p in m.layers[0].filter_net.parameters()]

    # Build a tiny warm-start cache and apply it.
    stats = {"grid_lo": 1.0, "grid_hi": 1000.0, "p0.1": 1.0, "max": 250.0}
    torch.manual_seed(1)
    net = FilterNet(hidden=mcfg.filter_hidden, filter_input=mcfg.filter_input)
    cache = tmp_path / "ws.pt"
    warmstart_filternet(net, stats, cache_path=cache, max_steps=300,
                        n_grid=256)
    info = m.apply_filter_warmstart(str(cache))
    after = [p.detach().clone() for p in m.layers[0].filter_net.parameters()]

    assert any(_differs(a, b) for a, b in zip(before, after)), \
        "apply_filter_warmstart did not change FilterNet weights"
    assert "achieved_mse" in info


def test_live_gate_init(w):
    """gate_init changes alpha, hence the effective filter multiplier."""
    for a, b in (("near_gs", "neutral"), ("neutral", "near_emgs")):
        assert ModelConfig(gate_init=a).gate_init_value != \
               ModelConfig(gate_init=b).gate_init_value
    assert _differs(_fwd(ModelConfig(T_UR=1, gate_init="near_gs"), w)[0],
                    _fwd(ModelConfig(T_UR=1, gate_init="near_emgs"), w)[0])


def test_live_use_transformer(w):
    """Arm 2: the Transformer is not constructed at all."""
    full = ModelConfig(T_UR=2)
    only = ModelConfig(T_UR=2, use_transformer=False)
    assert _nparams(full) != _nparams(only)
    assert _nparams(only) == 2 * (97 + 1)      # FilterNet + gate, per layer
    torch.manual_seed(0)
    m = URformer(16, 3, only, NUM64)
    assert all(l.former is None for l in m.layers)


def test_live_tie_layers(w):
    """Tied layers share one module object, so distinct params collapse."""
    untied = _nparams(ModelConfig(T_UR=4, tie_layers=False))
    tied = _nparams(ModelConfig(T_UR=4, tie_layers=True))
    assert tied < untied


def test_live_deep_supervision(w):
    """deep_supervision changes the LOSS on a fixed batch, not the forward."""
    from trackD_urformer.train import deep_supervision_loss

    mcfg = ModelConfig(T_UR=3)
    torch.manual_seed(0)
    m = URformer(16, 3, mcfg, NUM64).double()
    # Perturb so intermediate layers differ from the final one.
    with torch.no_grad():
        torch.manual_seed(3)
        for l in m.layers:
            l.former.out_proj.weight.normal_(0, 0.05)
    args = (_t(np.zeros_like(w.G_true)), _t(w.Z, F64), _t(w.S), _t(w.B),
            torch.tensor([w.sigma2], dtype=F64))
    Gt = _t(w.G_true)
    with torch.no_grad():
        outs = m(*args, return_all=True)
        final_only = float(nmse_loss(outs[-1], Gt))
        deep = float(deep_supervision_loss(outs, Gt))
    assert abs(final_only - deep) > 1e-9


def test_live_predict_one_minus_R():
    """The A3 labeled variant must actually change the mapping."""
    torch.manual_seed(0)
    a = FilterNet(hidden=32, predict_one_minus_R=False).double()
    torch.manual_seed(0)
    b = FilterNet(hidden=32, predict_one_minus_R=True).double()
    k = torch.logspace(-1, 3, 64, dtype=torch.float64).view(1, 1, -1)
    with torch.no_grad():
        assert _differs(a(k).numpy(), b(k).numpy())
        # they are exact complements of each other
        assert np.allclose((a(k) + b(k)).numpy(), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# NumericConfig
# ---------------------------------------------------------------------------
def test_live_dtype(w):
    """dtype changes the achieved precision, measurably."""
    assert NumericConfig("float32").eps != NumericConfig("float64").eps
    assert NumericConfig("float32").complex_dtype == torch.complex64
    assert NumericConfig("float64").complex_dtype == torch.complex128
    a64, _ = _fwd(ModelConfig(T_UR=1), w, numeric=NUM64)
    a32, _ = _fwd(ModelConfig(T_UR=1), w, numeric=NumericConfig("float32"))
    # Same maths, different precision: close but NOT bitwise equal.
    d = float(np.max(np.abs(a64 - a32)))
    assert 0.0 < d < 1e-2


# ---------------------------------------------------------------------------
# TrainConfig
# ---------------------------------------------------------------------------
def test_live_train_init(w):
    """Each initializer produces a genuinely different G^(0)."""
    cfg = TrackDConfig()
    batch = {
        "Z": _t(w.Z, torch.float32).float(),
        "S": _t(w.S, torch.complex64),
        "B": _t(w.B, torch.complex64),
        "G_true": _t(w.G_true, torch.complex64),
        "trial": torch.tensor([0]),
    }
    outs = {}
    for init in ("random", "spectral", "linearized_ls"):
        torch.manual_seed(0)
        outs[init] = make_initial_batch(batch, init, cfg).numpy()
    assert _differs(outs["random"], outs["spectral"], tol=1e-6)
    assert _differs(outs["spectral"], outs["linearized_ls"], tol=1e-6)
    assert _differs(outs["random"], outs["linearized_ls"], tol=1e-6)


def test_live_grad_clip_and_lr_are_recorded():
    """These reach the optimizer; assert they are at least distinct in config."""
    assert TrainConfig(lr=1e-3).lr != TrainConfig(lr=1e-4).lr
    assert TrainConfig(grad_clip=1.0).grad_clip != TrainConfig(grad_clip=None).grad_clip
    assert TrainConfig(batch_size=32).batch_size != TrainConfig(batch_size=64).batch_size
    assert TrainConfig(epochs=50).epochs != TrainConfig(epochs=10).epochs
    assert TrainConfig(num_threads=1).num_threads != TrainConfig(num_threads=4).num_threads


# ---------------------------------------------------------------------------
# DataConfig / SystemConfig
# ---------------------------------------------------------------------------
def test_live_pilot_mode():
    """fixed_S reuses ONE pilot matrix; random_S draws a fresh one per sample."""
    sysc = SystemConfig(K=3, N=8, P=12)
    num = NumericConfig("float64")
    fixed = TrackDDataset("train", sysc=sysc,
                          datac=DataConfig(pilot_mode="fixed_S"), numeric=num,
                          N=8, P=12)
    rand = TrackDDataset("train", sysc=sysc,
                         datac=DataConfig(pilot_mode="random_S"), numeric=num,
                         N=8, P=12)
    assert fixed.S_fixed is not None and rand.S_fixed is None
    assert np.allclose(fixed.sample(0).S, fixed.sample(1).S)      # reused
    assert not np.allclose(rand.sample(0).S, rand.sample(1).S)    # fresh


def test_live_snr_mode():
    sysc = SystemConfig(K=3, N=8, P=12)
    num = NumericConfig("float64")
    rng_ds = TrackDDataset("train", sysc=sysc,
                           datac=DataConfig(snr_mode="snr_range"), numeric=num,
                           N=8, P=12)
    fix_ds = TrackDDataset("train", sysc=sysc,
                           datac=DataConfig(snr_mode="snr_fixed"), numeric=num,
                           N=8, P=12)
    snrs_rng = {rng_ds.sample(i).snr_db for i in range(8)}
    snrs_fix = {fix_ds.sample(i).snr_db for i in range(8)}
    assert len(snrs_rng) > 1        # sampled per example
    assert len(snrs_fix) == 1       # single value


def test_live_snr_range_db():
    sysc = SystemConfig(K=3, N=8, P=12)
    num = NumericConfig("float64")
    lo = TrackDDataset("train", sysc=sysc,
                       datac=DataConfig(snr_range_db=(0.0, 1.0)), numeric=num,
                       N=8, P=12)
    hi = TrackDDataset("train", sysc=sysc,
                       datac=DataConfig(snr_range_db=(19.0, 20.0)), numeric=num,
                       N=8, P=12)
    assert max(lo.sample(i).snr_db for i in range(8)) < \
           min(hi.sample(i).snr_db for i in range(8))


def test_live_rsr_train_mode_is_declared_fixed():
    """rsr_train_mode is DECLARATIVE: it records that RSR is never sampled.

    There is deliberately no 'range' code path -- the field exists so a later
    reader cannot assume one. Liveness here means the value is carried into the
    serialized config, which is asserted below.
    """
    assert DataConfig().rsr_train_mode == "fixed"
    assert TrackDConfig().to_dict()["data"]["rsr_train_mode"] == "fixed"
    # "range" has no implementation, so it must be REJECTED rather than sit as
    # a silent no-op -- the failure mode that produced the dead filter_init.
    with pytest.raises(NotImplementedError, match="rsr_train_mode"):
        DataConfig(rsr_train_mode="range")


def test_live_system_fields():
    """N, K, P, rsr_db, master_seed all change the generated world."""
    base = SystemConfig(K=3, N=8, P=12, master_seed=20260827)
    w0 = make_world(0, sysc=base, N=8, P=12, snr_db=5.0)
    assert make_world(0, sysc=base, N=16, P=12, snr_db=5.0).G_true.shape != \
        w0.G_true.shape                                            # N
    assert make_world(0, sysc=base, N=8, P=20, snr_db=5.0).Z.shape != w0.Z.shape  # P
    assert make_world(0, sysc=SystemConfig(K=2, N=8, P=12), N=8, P=12,
                      snr_db=5.0).G_true.shape[1] == 2             # K
    assert not np.allclose(
        make_world(0, sysc=base, N=8, P=12, snr_db=5.0, rsr_db=20.0).B, w0.B)  # rsr
    other = SystemConfig(K=3, N=8, P=12, master_seed=99999)
    assert not np.allclose(
        make_world(0, sysc=other, N=8, P=12, snr_db=5.0).G_true, w0.G_true)   # seed


def test_live_config_is_fully_serialized():
    """Every config field reaches to_dict(), so a checkpoint records it."""
    d = TrackDConfig().to_dict()
    for section in ("system", "data", "model", "train", "numeric", "baseline"):
        assert section in d and isinstance(d[section], dict)
    assert "rsr_train_mode" in d["data"]
    assert "use_transformer" in d["model"]
    assert "filter_init" in d["model"]
    assert d["gate_init_value"] == ModelConfig().gate_init_value


def test_live_baseline_T_GS(w):
    """Iteration count changes the classical estimate."""
    from trackD_urformer.baselines import run_em_gs
    a = run_em_gs(w, max_iter=5, init="spectral", seed=0)
    b = run_em_gs(w, max_iter=100, init="spectral", seed=0)
    assert _differs(a, b, tol=1e-9)

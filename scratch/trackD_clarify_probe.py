"""Track D clarification probe (PROMPT 3 items 2, 3e, 4).

Measures, rather than asserts:
  * the kappa distribution over the ACTUAL training dataset
  * the FilterNet warm-start grid bounds derived from it, and the achieved fit MSE
  * what R_learned(kappa) actually produces at t=0 under RANDOM init
  * exactly which classical estimator (if any) the default untrained URformer equals
  * the seed-ledger disjointness assertion
  * gate J's exact conditions

Writes reports/trackD_clarify.json. Throwaway.

Run:  PYTHONPATH=. python3 scratch/trackD_clarify_probe.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from rydberg_sim.gs import biased_gs_channel_rows, em_gs_channel_rows
from trackD_urformer.config import (
    DataConfig, ModelConfig, NumericConfig, SystemConfig, TrackDConfig,
)
from trackD_urformer.dataset import TrackDDataset, make_world
from trackD_urformer.filter_net import (
    FilterNet, measure_kappa_range, warmstart_filternet,
)
from trackD_urformer.torch_forward import (
    bessel_ratio_torch, em_kappa, forward_field, gs_layer, em_gs_layer,
)
from trackD_urformer.urformer import URformer

OUT = Path("reports/trackD_clarify.json")
C128, F64 = torch.complex128, torch.float64
res: dict = {}


def _t(a, dtype=C128):
    return torch.as_tensor(np.array(a, copy=True)[None], dtype=dtype)


def _rel(a, b):
    b = np.asarray(b)
    return float(np.linalg.norm(np.asarray(a) - b) / np.linalg.norm(b))


cfg = TrackDConfig()
torch.set_num_threads(2)

# ---------------------------------------------------------------------------
# (1) kappa distribution over the ACTUAL training dataset
# ---------------------------------------------------------------------------
print("== measuring kappa over the training dataset ==")
ds = TrackDDataset("train", sysc=cfg.system, datac=cfg.data, numeric=cfg.numeric)
kstats = measure_kappa_range(ds, eps=1e-12, n_samples=512)
print(json.dumps(kstats, indent=2))
res["kappa_stats_training_set"] = kstats

# kappa across the full training SNR range, at the reference config
res["kappa_note"] = (
    "measured at G=0 (the first iteration), where |Y| = |B| and kappa is "
    "largest, over 512 training realizations spanning the full SNR range"
)

# ---------------------------------------------------------------------------
# (2) FilterNet warm start: grid from the measurement, achieved MSE
# ---------------------------------------------------------------------------
print("\n== warm-start fit ==")
torch.manual_seed(0)
net = FilterNet(hidden=cfg.model.filter_hidden, filter_input=cfg.model.filter_input)
info = warmstart_filternet(
    net, kstats, cache_path="reports/trackD_filternet_warmstart.pt"
)
print(json.dumps({k: v for k, v in info.items() if k != "kappa_stats"}, indent=2))
res["warmstart"] = info

# how good is the warm-started net across the measured range?
kg = torch.logspace(np.log10(kstats["grid_lo"]), np.log10(kstats["grid_hi"]),
                    2048, dtype=torch.float64)
with torch.no_grad():
    pred = net(kg.view(1, 1, -1)).view(-1)
tgt = bessel_ratio_torch(kg)
res["warmstart_check"] = {
    "max_abs_err_over_grid": float((pred - tgt).abs().max()),
    "mean_abs_err_over_grid": float((pred - tgt).abs().mean()),
}
print("warm-start max abs err over grid:",
      res["warmstart_check"]["max_abs_err_over_grid"])

# ---------------------------------------------------------------------------
# (3) What does R_learned produce at t=0 under the DEFAULT (random) init?
# ---------------------------------------------------------------------------
print("\n== R_learned at t=0, DEFAULT random init ==")
w = make_world(0, sysc=cfg.system, N=cfg.system.N, P=cfg.system.P, snr_db=5.0)
G0 = np.zeros_like(w.G_true)
Y = forward_field(_t(G0), _t(w.S), _t(w.B))
kap = em_kappa(_t(w.Z, F64), Y, torch.tensor([w.sigma2], dtype=F64), 1e-12)
R_exact = bessel_ratio_torch(kap)

rand_stats = []
for seed in (0, 1, 2, 3, 4):
    torch.manual_seed(seed)
    fn = FilterNet(hidden=cfg.model.filter_hidden,
                   filter_input=cfg.model.filter_input).double()
    with torch.no_grad():
        R = fn(kap)
    rand_stats.append({
        "seed": seed,
        "R_learned_min": float(R.min()), "R_learned_max": float(R.max()),
        "R_learned_mean": float(R.mean()),
        "rel_err_vs_bessel": _rel(R.numpy(), R_exact.numpy()),
        "max_abs_err_vs_bessel": float((R - R_exact).abs().max()),
    })
    print(rand_stats[-1])
res["R_learned_at_init_random"] = rand_stats
res["R_exact_at_this_config"] = {
    "min": float(R_exact.min()), "max": float(R_exact.max()),
    "mean": float(R_exact.mean()),
    "kappa_min": float(kap.min()), "kappa_max": float(kap.max()),
}

# ---------------------------------------------------------------------------
# (4) What does the DEFAULT untrained URformer actually equal?
# ---------------------------------------------------------------------------
print("\n== default untrained URformer vs classical estimators ==")
torch.manual_seed(0)
model = URformer(cfg.system.N, cfg.system.K, ModelConfig(T_UR=1),
                 NumericConfig("float64")).double()
alpha0 = float(model.layers[0].alpha)
args = (_t(G0), _t(w.Z, F64), _t(w.S), _t(w.B),
        torch.tensor([w.sigma2], dtype=F64))
with torch.no_grad():
    out_default = model(*args)[0].numpy()

ref_gs = biased_gs_channel_rows(w.S, w.Z, w.B, max_iter=1, G0=G0).G_hat
ref_em = em_gs_channel_rows(w.S, w.Z, w.B, w.sigma2, max_iter=1, G0=G0).G_hat

# the effective per-element multiplier applied to Y_direct
with torch.no_grad():
    R_lrn = model.layers[0].filter_net(kap).double()
mult = alpha0 * R_lrn + (1.0 - alpha0)

res["untrained_default"] = {
    "alpha_layer0": alpha0,
    "alphas_all_layers": URformer(cfg.system.N, cfg.system.K, cfg.model,
                                  cfg.numeric).initial_alphas(),
    "rel_err_vs_one_GS_step": _rel(out_default, ref_gs),
    "rel_err_vs_one_EMGS_step": _rel(out_default, ref_em),
    "effective_multiplier_on_Y_direct": {
        "min": float(mult.min()), "max": float(mult.max()),
        "mean": float(mult.mean()),
        "formula": "alpha*R_learned + (1-alpha); ==1 exactly iff alpha==0 or R==1",
        "deviation_from_1_max": float((mult - 1.0).abs().max()),
    },
    "transformer_residual_max_abs": None,  # filled below
}
with torch.no_grad():
    G_lin_probe = model.layers[0].former(_t(w.G_true))
res["untrained_default"]["transformer_residual_max_abs"] = float(
    torch.abs(G_lin_probe).max())

print("alpha_0 =", alpha0)
print("rel err vs 1 GS step  :", res["untrained_default"]["rel_err_vs_one_GS_step"])
print("rel err vs 1 EMGS step:", res["untrained_default"]["rel_err_vs_one_EMGS_step"])
print("effective multiplier  :", res["untrained_default"]["effective_multiplier_on_Y_direct"])

# forced modes, for contrast
model._set_test_mode(alpha=0.0, disable_residual=True)
with torch.no_grad():
    forced_gs = model(*args)[0].numpy()
model._set_test_mode(filter_override="exact_bessel", alpha=1.0, disable_residual=True)
with torch.no_grad():
    forced_em = model(*args)[0].numpy()
model._clear_test_mode()
res["forced_modes"] = {
    "alpha0_residual_off_vs_GS": _rel(forced_gs, ref_gs),
    "exact_bessel_alpha1_residual_off_vs_EMGS": _rel(forced_em, ref_em),
}
print("forced GS  :", res["forced_modes"]["alpha0_residual_off_vs_GS"])
print("forced EMGS:", res["forced_modes"]["exact_bessel_alpha1_residual_off_vs_EMGS"])

# What if FilterNet WERE warm-started? Then alpha*R+(1-alpha) with R=Bessel.
mult_ws = alpha0 * R_exact + (1.0 - alpha0)
res["hypothetical_warmstarted_untrained"] = {
    "effective_multiplier_min": float(mult_ws.min()),
    "effective_multiplier_max": float(mult_ws.max()),
    "note": ("even with a perfect warm start the default alpha=0.1192 gives "
             "alpha*R+(1-alpha) != 1 and != R, so it is still neither GS nor EM-GS"),
}

# ---------------------------------------------------------------------------
# (5) Seed ledger - programmatic disjointness assertion
# ---------------------------------------------------------------------------
print("\n== seed ledger ==")
d = cfg.data
ranges = {"train": d.train_seed_range, "val": d.val_seed_range,
          "test": d.test_seed_range}
used = {
    "train": (d.train_seed_range[0], d.train_seed_range[0] + d.n_train),
    "val": (d.val_seed_range[0], d.val_seed_range[0] + d.n_val),
    "test": (d.test_seed_range[0], d.test_seed_range[0] + d.n_test),
}


def overlap(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return max(0, hi - lo)


ledger = {"declared_ranges": {k: list(v) for k, v in ranges.items()},
          "actually_used": {k: list(v) for k, v in used.items()},
          "intersections": {}, "all_disjoint": True}
for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
    n_decl = overlap(ranges[a], ranges[b])
    n_used = overlap(used[a], used[b])
    ledger["intersections"][f"{a}_vs_{b}"] = {
        "declared_overlap_count": n_decl, "used_overlap_count": n_used,
    }
    ledger["all_disjoint"] &= (n_decl == 0 and n_used == 0)
    assert n_decl == 0, f"declared ranges {a}/{b} overlap by {n_decl}"
    assert n_used == 0, f"used ranges {a}/{b} overlap by {n_used}"
print(json.dumps(ledger, indent=2))
res["seed_ledger"] = ledger
res["seed_ledger"]["assertion_output"] = (
    "all three pairwise intersections empty; asserts passed"
)

# ---------------------------------------------------------------------------
# (6) Baseline spread, to justify the 2 dB success threshold
# ---------------------------------------------------------------------------
print("\n== baseline paired spread (for the pre-registered threshold) ==")
from trackD_urformer.baselines import run_em_gs, run_gs, nmse_parts

n_probe = 200
diffs_db, em_db, gs_db = [], [], []
te, td, ge, gd = 0.0, 0.0, 0.0, 0.0
test_ds = TrackDDataset("test", sysc=cfg.system, datac=cfg.data,
                        numeric=cfg.numeric)
for i in range(n_probe):
    s = test_ds.sample(i)
    ghat_em = run_em_gs(s, max_iter=cfg.baseline.T_GS, init="spectral", seed=s.trial)
    ghat_gs = run_gs(s, max_iter=cfg.baseline.T_GS, init="spectral", seed=s.trial)
    e1, d1 = nmse_parts(ghat_em, s.G_true)
    e2, d2 = nmse_parts(ghat_gs, s.G_true)
    te += e1; td += d1; ge += e2; gd += d2
    em_db.append(10 * np.log10(e1 / d1))
    gs_db.append(10 * np.log10(e2 / d2))
    diffs_db.append(10 * np.log10(e2 / d2) - 10 * np.log10(e1 / d1))

em_db = np.array(em_db); gs_db = np.array(gs_db); diffs_db = np.array(diffs_db)
rng = np.random.default_rng(12345)
boot = np.array([np.median(rng.choice(diffs_db, diffs_db.size, replace=True))
                 for _ in range(2000)])
res["baseline_spread"] = {
    "n_trials": n_probe,
    "em_gs_spectral_nmse_db_ratio_of_sums": float(10 * np.log10(te / td)),
    "gs_spectral_nmse_db_ratio_of_sums": float(10 * np.log10(ge / gd)),
    "em_gs_per_trial_db": {"median": float(np.median(em_db)),
                           "std": float(em_db.std(ddof=1)),
                           "p5": float(np.percentile(em_db, 5)),
                           "p95": float(np.percentile(em_db, 95))},
    "paired_gs_minus_emgs_db": {
        "median": float(np.median(diffs_db)),
        "std": float(diffs_db.std(ddof=1)),
        "boot_ci95": [float(np.percentile(boot, 2.5)),
                      float(np.percentile(boot, 97.5))],
    },
    "note": ("the paired GS-minus-EM-GS difference is the natural yardstick: it "
             "is a real but small algorithmic effect measured on the same "
             "realizations, so it calibrates what a 'meaningful' gap looks like"),
}
print(json.dumps(res["baseline_spread"], indent=2))

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
print(f"\nwrote {OUT}")

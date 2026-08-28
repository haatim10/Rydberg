"""PROMPT 4 A4 fix: re-measure kappa and re-fit after widening SNR and grid.

A4 failed both coverage checks:
  SNR   train [0,20] vs eval [-10,20]        -> extrapolation at -10, -5
  kappa grid [1.32, 6513] vs eval [0.009, 2131] -> eval min 147x below grid_lo

Fixes applied:
  DataConfig.snr_range_db  (0,20) -> (-10,20)
  measure_kappa_range grid_lo: p0.1 -> 0.1 * observed min

This re-measures under the widened training distribution, re-fits the warm
start under the max-abs-error < 0.01 criterion, and re-checks BOTH coverage
conditions. Writes reports/trackD_partA4.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from trackD_urformer.config import D1_SNR_GRID_DB, TrackDConfig
from trackD_urformer.dataset import TrackDDataset, make_world
from trackD_urformer.filter_net import (
    FilterNet, measure_kappa_range, warmstart_filternet,
)
from trackD_urformer.torch_forward import bessel_ratio_torch, em_kappa, forward_field

OUT = Path("reports/trackD_partA4.json")
C128, F64 = torch.complex128, torch.float64
cfg = TrackDConfig()
torch.set_num_threads(2)
res: dict = {}


def _t(a, dtype=C128):
    return torch.as_tensor(np.array(a, copy=True)[None], dtype=dtype)


print("== re-measure kappa under the WIDENED training distribution ==")
print(f"   snr_range_db = {cfg.data.snr_range_db}")
ds = TrackDDataset("train", sysc=cfg.system, datac=cfg.data, numeric=cfg.numeric)
kstats = measure_kappa_range(ds, eps=1e-12, n_samples=768)
res["kappa_train"] = kstats
print(json.dumps(kstats, indent=2))

print("\n== kappa across the EVALUATION sweep ==")
ev_min, ev_max = np.inf, -np.inf
per_snr = {}
for snr in D1_SNR_GRID_DB:
    lo, hi = np.inf, -np.inf
    for i in range(60):
        s = make_world(2_000_000 + i, sysc=cfg.system, N=cfg.system.N,
                       P=cfg.system.P, snr_db=snr)
        G0 = torch.zeros((1, s.N, s.K), dtype=C128)
        Y = forward_field(G0, _t(s.S), _t(s.B))
        k = em_kappa(_t(s.Z, F64), Y, torch.tensor([s.sigma2], dtype=F64), 1e-12)
        lo = min(lo, float(k.min())); hi = max(hi, float(k.max()))
    per_snr[str(snr)] = [lo, hi]
    ev_min, ev_max = min(ev_min, lo), max(ev_max, hi)
    print(f"   SNR {snr:6.1f}: kappa in [{lo:12.5f}, {hi:12.2f}]")
res["kappa_eval_per_snr"] = per_snr
res["kappa_eval_range"] = [ev_min, ev_max]

# Grid must cover BOTH training and evaluation.
grid_lo = min(kstats["grid_lo"], 0.1 * ev_min)
grid_hi = max(kstats["grid_hi"], 4.0 * ev_max)
kstats_cov = dict(kstats, grid_lo=grid_lo, grid_hi=grid_hi)
res["grid_used"] = [grid_lo, grid_hi]
print(f"\n   grid widened to cover both: [{grid_lo:.6g}, {grid_hi:.6g}]")

print("\n== re-fit warm start on the covering grid, criterion max abs < 0.01 ==")
rows = []
for width in (32, 64, 128):
    for variant in ("R", "one_minus_R"):
        torch.manual_seed(0)
        net = FilterNet(hidden=width, filter_input=cfg.model.filter_input,
                        predict_one_minus_R=(variant == "one_minus_R"))
        info = warmstart_filternet(net, kstats_cov, cache_path=None,
                                   max_steps=60000, target_mse=1e-12,
                                   target_max_abs=0.01, n_grid=8192)
        rows.append({"hidden": width, "variant": variant,
                     **{k: v for k, v in info.items() if k != "kappa_stats"}})
        print(f"   hidden={width:4d} {variant:12s} mse={info['achieved_mse']:.3e} "
              f"max_abs={info['achieved_max_abs']:.5f} steps={info['steps']:6d} "
              f"{'PASS' if info['achieved_max_abs'] < 0.01 else 'FAIL'}")
res["A3_refit"] = rows

passing = [r for r in rows if r["achieved_max_abs"] < 0.01]
if passing:
    best = min(passing, key=lambda r: (r["hidden"], r["variant"] != "R"))
    res["adopted"] = best
    print(f"\n   adopt hidden={best['hidden']} variant={best['variant']}")
    # Persist the adopted cache for arm 1a / arm 2. Delete any stale cache
    # first: warmstart_filternet short-circuits on an existing file, which
    # would silently keep the OLD weak-criterion fit.
    cache = Path(cfg.model.filter_warmstart_cache)
    if cache.exists():
        cache.unlink()
        print(f"   removed stale cache {cache}")
    torch.manual_seed(0)
    net = FilterNet(hidden=best["hidden"], filter_input=cfg.model.filter_input,
                    predict_one_minus_R=(best["variant"] == "one_minus_R"))
    info = warmstart_filternet(
        net, kstats_cov, cache_path=cfg.model.filter_warmstart_cache,
        max_steps=60000, target_mse=1e-12, target_max_abs=0.01, n_grid=8192)
    net = net.double()
    res["adopted_cache"] = {"path": cfg.model.filter_warmstart_cache, **{
        k: v for k, v in info.items() if k != "kappa_stats"}}
    print(f"   wrote cache {cfg.model.filter_warmstart_cache}")
    kg = torch.logspace(np.log10(grid_lo), np.log10(grid_hi), 80,
                        dtype=torch.float64)
    with torch.no_grad():
        pred = net(kg.view(1, 1, -1)).view(-1)
    tgt = bessel_ratio_torch(kg)
    res["residual_profile"] = [
        {"kappa": float(a), "R_exact": float(b), "R_fit": float(c),
         "abs_err": float(abs(b - c))}
        for a, b, c in zip(kg, tgt, pred)]
    worst = max(res["residual_profile"], key=lambda d: d["abs_err"])
    print(f"   worst residual: kappa={worst['kappa']:.4g} err={worst['abs_err']:.5f}")
else:
    res["adopted"] = None
    print("\n   NO width/variant reached max abs < 0.01 on the covering grid")

# Final coverage verdicts
tr_lo, tr_hi = cfg.data.snr_range_db
ev_lo, ev_hi = min(D1_SNR_GRID_DB), max(D1_SNR_GRID_DB)
res["coverage"] = {
    "snr_train": [tr_lo, tr_hi], "snr_eval": [float(ev_lo), float(ev_hi)],
    "snr_covers_eval": bool(tr_lo <= ev_lo and tr_hi >= ev_hi),
    "kappa_grid": [grid_lo, grid_hi],
    "kappa_train": [kstats["min"], kstats["max"]],
    "kappa_eval": [ev_min, ev_max],
    "kappa_covers_train": bool(grid_lo <= kstats["min"] and grid_hi >= kstats["max"]),
    "kappa_covers_eval": bool(grid_lo <= ev_min and grid_hi >= ev_max),
}
print("\n== A4 coverage verdicts ==")
print(json.dumps(res["coverage"], indent=2))

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
print(f"\nwrote {OUT}")

"""PROMPT 5 Part A: stage-1 diagnostics. NO TRAINING.

A1  re-evaluate the 3 saved checkpoints on VALIDATION in the TEST form
    (paired improvement vs EM-GS, bootstrap CI, median and mean)
A2  paired per-trial difference between arm 1a and arm 1b, on BOTH sets
A3  exchangeability: absolute NMSE for the untrained classical estimators on
    both sets, per method
A4  checkpoint-selection noise (see the LIMITATION note below)

LIMITATION, stated up front: stage 1 retained only best.pt (the selected epoch)
and checkpoint.pt (the final epoch). Per-epoch weights were NOT kept, so the
exact A4 test -- paired validation difference between EVERY epoch's checkpoint
and the selected one -- cannot be computed from existing artifacts. What IS
computable: the selected-vs-final paired difference (a real 2-point paired
measurement), and the per-epoch validation curve against a bootstrap SE of the
validation metric. Both are reported, and the gap is named.

Writes reports/trackD_partA_diag.json.
Run:  PYTHONPATH=. python3 scratch/trackD_partA_diagnostics.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from rydberg_sim.forward import exact_forward
from rydberg_sim.rng import get_operating_point_rngs
from trackD_urformer.baselines import (
    make_initial_G, nmse_parts, run_em_gs, run_gs, run_linearised_ls,
)
from trackD_urformer.config import TrackDConfig
from trackD_urformer.dataset import TrackDDataset
from trackD_urformer.stage1 import ARMS, build_model
from trackD_urformer.torch_forward import least_squares_G

from dataclasses import replace

OUT = Path("reports/trackD_partA_diag.json")
RES = Path("results/track_d/stage1")
C128, F64 = torch.complex128, torch.float64
torch.set_num_threads(2)

cfg = TrackDConfig()
cfg = replace(cfg, train=replace(cfg.train, init="spectral"))   # as stage 1 ran
db = lambda x: 10.0 * float(np.log10(max(float(x), 1e-30)))
res: dict = {}
RNG = np.random.default_rng(20260830)


def boot_median_ci(d, n=4000):
    b = np.array([np.median(RNG.choice(d, d.size, replace=True))
                  for _ in range(n)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def boot_mean_ci(d, n=4000):
    b = np.array([RNG.choice(d, d.size, replace=True).mean() for _ in range(n)])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# ---------------------------------------------------------------------------
# Evaluate every method on a split, per trial. Same code path for val and test.
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_split(split: str, n: int, which=("checkpoint",)) -> dict:
    ds = TrackDDataset(split, sysc=cfg.system, datac=cfg.data,
                       numeric=cfg.numeric, init=cfg.train.init)
    cd, rd = cfg.numeric.complex_dtype, cfg.numeric.real_dtype

    models = {}
    for arm in ARMS:
        for tag in which:
            p = RES / arm / (f"{'best' if tag=='checkpoint' else 'checkpoint'}.pt")
            if not p.exists():
                continue
            m, _ = build_model(cfg, arm)
            blob = torch.load(p, map_location="cpu", weights_only=False)
            m.load_state_dict(blob["model"])
            m.eval()
            models[arm if tag == "checkpoint" else f"{arm}__final"] = m

    keys = ["gs_spectral", "em_gs_spectral", "linearised_ls",
            "oracle_phase"] + list(models)
    per = {k: [] for k in keys}
    energies = {k: [0.0, 0.0] for k in keys}
    snrs = []

    for i in range(n):
        s = ds.sample(i)
        snrs.append(s.snr_db)
        est = {
            "gs_spectral": run_gs(s, max_iter=cfg.baseline.T_GS,
                                  init="spectral", seed=s.trial),
            "em_gs_spectral": run_em_gs(s, max_iter=cfg.baseline.T_GS,
                                        init="spectral", seed=s.trial),
            "linearised_ls": run_linearised_ls(s),
        }
        rngs = get_operating_point_rngs(cfg.system.master_seed, s.trial,
                                        s.snr_db, s.rsr_db)
        ex = exact_forward(s.G_true, s.S, s.B, s.sigma2, rng_noise=rngs.noise)
        of = s.Z * np.exp(1j * np.angle(np.asarray(ex.E)))
        T = lambda a, d: torch.as_tensor(np.array(a, copy=True)[None], dtype=d)
        est["oracle_phase"] = least_squares_G(
            T(of, C128) - T(s.B, C128), T(s.S, C128))[0].numpy()

        G0 = T(ds.g0(i), cd)
        Z, S, B = T(s.Z, rd), T(s.S, cd), T(s.B, cd)
        s2 = torch.tensor([s.sigma2], dtype=rd)
        for name, m in models.items():
            est[name] = m(G0, Z, S, B, s2)[0].detach().numpy()

        for k, gh in est.items():
            e, d_ = nmse_parts(gh, s.G_true)
            energies[k][0] += e
            energies[k][1] += d_
            per[k].append(e / d_)

    return {
        "n": n,
        "snr_db": snrs,
        "per_trial": {k: per[k] for k in keys},
        "absolute": {k: {"ratio_of_sums_db": db(energies[k][0] / energies[k][1]),
                         "median_db": db(np.median(per[k])),
                         "mean_db": db(np.mean(per[k]))} for k in keys},
    }


print("== evaluating VALIDATION set (2000 trials, all methods) ==")
V = eval_split("val", cfg.data.n_val, which=("checkpoint", "final"))
print("   done")

print("== loading TEST per-trial from stage 1 (not recomputed) ==")
S1 = json.load(open("reports/trackD_stage1_results.json"))
T_per = S1["test"]["per_trial_nmse"]

# test SNRs, regenerated from the same deterministic dataset (no model needed)
tds = TrackDDataset("test", sysc=cfg.system, datac=cfg.data, numeric=cfg.numeric)
T_snr = [tds.sample(i).snr_db for i in range(S1["test"]["n_test"])]

# ---------------------------------------------------------------------------
# A3 - exchangeability of the two splits
# ---------------------------------------------------------------------------
print("\n== A3: exchangeability ==")
a3 = {"per_method": {}}
for k in ("gs_spectral", "em_gs_spectral", "linearised_ls", "oracle_phase"):
    v = np.array(V["per_trial"][k])
    t = np.array(T_per[k])
    a3["per_method"][k] = {
        "val_ratio_of_sums_db": V["absolute"][k]["ratio_of_sums_db"],
        "test_ratio_of_sums_db": db(np.mean(t)),   # RoS == mean here (den equal)
        "val_median_db": db(np.median(v)),
        "test_median_db": db(np.median(t)),
        "median_diff_db": db(np.median(v)) - db(np.median(t)),
    }
    m = a3["per_method"][k]
    print(f"   {k:18s} val median {m['val_median_db']:8.3f}  "
          f"test {m['test_median_db']:8.3f}  diff {m['median_diff_db']:+7.3f} dB")

vs, ts = np.array(V["snr_db"]), np.array(T_snr)
a3["snr_distribution"] = {
    "val": {"mean": float(vs.mean()), "median": float(np.median(vs)),
            "std": float(vs.std(ddof=1)), "min": float(vs.min()),
            "max": float(vs.max())},
    "test": {"mean": float(ts.mean()), "median": float(np.median(ts)),
             "std": float(ts.std(ddof=1)), "min": float(ts.min()),
             "max": float(ts.max())},
}
a3["snr_mean_diff_db"] = float(vs.mean() - ts.mean())
print(f"   SNR mean: val {vs.mean():.3f} dB vs test {ts.mean():.3f} dB "
      f"(diff {vs.mean()-ts.mean():+.3f})")
res["A3"] = a3

# ---------------------------------------------------------------------------
# A1 - validation in the TEST form: paired improvement vs EM-GS
# ---------------------------------------------------------------------------
print("\n== A1: validation in test form (paired vs EM-GS) ==")
a1 = {}
for split, per in (("val", V["per_trial"]), ("test", T_per)):
    base = 10 * np.log10(np.array(per["em_gs_spectral"]))
    a1[split] = {}
    for arm in ARMS:
        if arm not in per:
            continue
        d = 10 * np.log10(np.array(per[arm])) - base
        lo, hi = boot_median_ci(d)
        mlo, mhi = boot_mean_ci(d)
        a1[split][arm] = {
            "median_diff_db": float(np.median(d)),
            "median_ci95": [lo, hi],
            "mean_diff_db": float(d.mean()),
            "mean_ci95": [mlo, mhi],
            "win_rate": float(np.mean(d < 0)),
            "absolute_median_db": db(np.median(np.array(per[arm]))),
        }
print(f"   {'arm':28s} {'VAL paired':>12} {'CI':>20} | {'TEST paired':>12} {'CI':>20}")
for arm in ARMS:
    v, t = a1["val"][arm], a1["test"][arm]
    print(f"   {arm:28s} {v['median_diff_db']:12.3f} "
          f"[{v['median_ci95'][0]:7.3f},{v['median_ci95'][1]:7.3f}] | "
          f"{t['median_diff_db']:12.3f} "
          f"[{t['median_ci95'][0]:7.3f},{t['median_ci95'][1]:7.3f}]")
res["A1"] = a1

# ---------------------------------------------------------------------------
# A2 - paired difference between arm 1a and arm 1b, both sets
# ---------------------------------------------------------------------------
print("\n== A2: arm1a - arm1b, paired per trial ==")
a2 = {}
for split, per in (("val", V["per_trial"]), ("test", T_per)):
    d = (10 * np.log10(np.array(per["arm1a_full_warmstart"]))
         - 10 * np.log10(np.array(per["arm1b_full_random"])))
    lo, hi = boot_median_ci(d)
    mlo, mhi = boot_mean_ci(d)
    a2[split] = {
        "median_diff_db": float(np.median(d)),
        "median_ci95": [lo, hi],
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "mean_diff_db": float(d.mean()),
        "mean_ci95": [mlo, mhi],
        "std_db": float(d.std(ddof=1)),
        "frac_1a_better": float(np.mean(d < 0)),
    }
    x = a2[split]
    print(f"   {split:5s} median {x['median_diff_db']:+7.3f} dB  "
          f"CI [{x['median_ci95'][0]:+.3f}, {x['median_ci95'][1]:+.3f}]  "
          f"excl0={x['ci_excludes_zero']}  1a better on {100*x['frac_1a_better']:.1f}%")
res["A2"] = a2

# ---------------------------------------------------------------------------
# A4 - selection noise, within the limits of what was retained
# ---------------------------------------------------------------------------
print("\n== A4: checkpoint-selection noise ==")
a4 = {
    "LIMITATION": (
        "stage 1 retained only best.pt (selected epoch) and checkpoint.pt "
        "(final epoch). Per-epoch weights were not kept, so the exact test -- "
        "paired validation difference between EVERY epoch and the selected one "
        "-- is not computable from existing artifacts. Reported instead: (a) the "
        "selected-vs-final paired difference, a real paired measurement; (b) the "
        "per-epoch validation curve against a bootstrap SE of the validation "
        "metric, which is a MARGINAL SE and therefore an UPPER bound on the "
        "plateau width, since pairing would cancel realization variance."
    ),
}
for arm in ("arm1a_full_warmstart", "arm1b_full_random"):
    rows = list(csv.DictReader(open(RES / arm / "curves.csv")))
    val = np.array([float(r["val_nmse_db"]) for r in rows])
    best_ep = int(S1["arms"][arm]["best_epoch"])

    # (a) selected vs final, PAIRED
    sel = np.array(V["per_trial"][arm])
    fin = np.array(V["per_trial"][arm + "__final"])
    d = 10 * np.log10(fin) - 10 * np.log10(sel)
    lo, hi = boot_median_ci(d)

    # (b) marginal bootstrap SE of the validation ratio-of-sums metric
    sel_lin = sel
    bs = np.array([RNG.choice(sel_lin, sel_lin.size, replace=True).mean()
                   for _ in range(2000)])
    se_db = float(np.std(10 * np.log10(bs), ddof=1))
    within = int(np.sum(val <= val[best_ep] + se_db))

    a4[arm] = {
        "best_epoch": best_ep,
        "best_val_db": float(val[best_ep]),
        "final_val_db": float(val[-1]),
        "selected_vs_final_paired_median_db": float(np.median(d)),
        "selected_vs_final_ci95": [lo, hi],
        "selected_vs_final_excludes_zero": bool(lo > 0 or hi < 0),
        "marginal_val_se_db": se_db,
        "n_epochs_within_1se_of_best": within,
        "epochs_total": len(val),
    }
    x = a4[arm]
    print(f"   {arm}")
    print(f"      best epoch {best_ep}, val {val[best_ep]:.3f} dB, "
          f"final {val[-1]:.3f} dB")
    print(f"      selected vs final, PAIRED: {x['selected_vs_final_paired_median_db']:+.3f} dB "
          f"CI [{lo:+.3f},{hi:+.3f}] excl0={x['selected_vs_final_excludes_zero']}")
    print(f"      marginal val SE {se_db:.3f} dB -> "
          f"{within}/{len(val)} epochs within 1 SE of best (UPPER bound)")
res["A4"] = a4

res["absolute_val"] = V["absolute"]
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
print(f"\nwrote {OUT}")

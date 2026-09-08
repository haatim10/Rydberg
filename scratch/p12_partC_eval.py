"""PROMPT 12 Part C -- evaluate the new arms and score P19, P20, P21.

Written and committed BEFORE any Part C run finishes, so the thresholds cannot
drift to fit whatever comes back. Every threshold below is copied verbatim from
reports/p12/PREREG_P19_P21.md.

Evaluation reuses the stage-4 pattern: one pass over the test set, identical
worlds for every arm, paired per-trial dB differences, bootstrap CI on the
median, reported per SNR bin.

  P19  across-seed spread of Delta_H under focused [5,20] training
  P20  Delta_H under the SNR-balanced loss, per bin
  P21  does a per-sample loss in dB reproduce the balanced-weighting result

Run:  PYTHONPATH=. python3 scratch/p12_partC_eval.py
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from trackD_urformer.baselines import nmse_parts
from trackD_urformer.config import TrackDConfig
from trackD_urformer.dataset import TrackDDataset
from trackD_urformer.stage1 import build_model, db
from trackD_urformer.stage3 import paired
from trackD_urformer.stage4 import BINS, by_bin

PARTC = Path("results/p12/partC")
STAGE4 = Path("results/track_d/stage4")
STAGE5 = Path("results/track_d/stage5")
STAGE2_U1 = Path("results/track_d/stage2/B3_80k_13ep/best.pt")
STAGE3_H1 = Path("results/track_d/stage3/H1_hs_urformer_80k/best.pt")
OUT = Path(os.environ.get("PARTC_OUT", "reports/p12"))   # PROMPT 15: re-score to reports/p15

FULL_SNR = (-10.0, 20.0)
HIGH_SNR = (5.0, 20.0)
N_TEST = 2000

# ---- thresholds, copied verbatim from the pre-registration -----------------
P19_SD_MAX = 0.12          # across-seed SD of Delta_H, dB
P19_RANGE_MAX = 0.25       # across-seed range, dB
P19_DECISION_SPREAD = 0.50  # >= this and the collapse claim is unsupported
P19_DECISION_ANY_SEED = 0.50
P20_BAND = (-0.10, 0.35)   # Delta_H over SNR >= 5
P20_NO_BIN_ABOVE = 0.50
P20_POOLED_ABS_MAX = 0.30
P21_SPAN_MAX = 4.0         # gradient-share span under the log loss
P21_MIN_GAIN_VS_UNBALANCED = 1.2
P21_MARGIN_VS_BALANCED = 0.5


def load_arm(name, *, hankel, P=20):
    """Rebuild an arm and load its best checkpoint from Part C."""
    cfg = TrackDConfig()
    rc = replace(cfg, system=replace(cfg.system, P=P),
                 model=replace(cfg.model, filter_init="random",
                               use_transformer=True, use_hankel=hankel,
                               hankel_rank=7, hankel_mode="fixed",
                               hankel_gate="none"))
    m, _ = build_model(rc, "arm1b_full_random")
    p = PARTC / name / "best.pt"
    m.load_state_dict(torch.load(p, map_location="cpu",
                                 weights_only=False)["model"])
    return m.eval()


def load_path(path, *, hankel, P=20):
    cfg = TrackDConfig()
    rc = replace(cfg, system=replace(cfg.system, P=P),
                 model=replace(cfg.model, filter_init="random",
                               use_transformer=True, use_hankel=hankel,
                               hankel_rank=7, hankel_mode="fixed",
                               hankel_gate="none"))
    m, _ = build_model(rc, "arm1b_full_random")
    m.load_state_dict(torch.load(path, map_location="cpu",
                                 weights_only=False)["model"])
    return m.eval()


def eval_models(models: dict, snr_range, n_test=N_TEST, P=20):
    """One pass, identical worlds for every arm.

    PROMPT 15 A1 FIX. ``TrackDConfig().train.init`` defaults to ``"random"``
    (config.py:324); every stage script overrides it to ``"spectral"`` before
    building an evaluation dataset, and the arms are TRAINED with the spectral
    initialiser. This function did not, so it fed the networks a random
    ``G0`` and every number PROMPT 12 Part C reported was measured off-model.
    The defect was invisible to a reproducibility check because
    ``make_initial_G("random", ..., seed=trial)`` is seeded per trial and
    therefore perfectly deterministic -- just wrong. Determinism and
    correctness are different properties.

    Pinned by ``tests/test_trackd_eval_reproducibility.py``.
    """
    cfg = TrackDConfig()
    cfg = replace(cfg, train=replace(cfg.train, init="spectral"))
    ecfg = replace(cfg, system=replace(cfg.system, P=P),
                   data=replace(cfg.data, snr_range_db=snr_range))
    ds = TrackDDataset("test", sysc=ecfg.system, datac=ecfg.data,
                       numeric=ecfg.numeric, init=ecfg.train.init)
    cd, rd = ecfg.numeric.complex_dtype, ecfg.numeric.real_dtype
    per = {k: [] for k in models}
    snr = []
    T = lambda a, d: torch.as_tensor(np.array(a, copy=True)[None], dtype=d)
    t0 = time.time()
    for i in range(n_test):
        s = ds.sample(i)
        snr.append(s.snr_db)
        G0, Z = T(ds.g0(i), cd), T(s.Z, rd)
        S, B = T(s.S, cd), T(s.B, cd)
        s2 = torch.tensor([s.sigma2], dtype=rd)
        for nm, m in models.items():
            with torch.no_grad():
                gh = m(G0, Z, S, B, s2)[0].numpy()
            e, d_ = nmse_parts(gh, s.G_true)
            per[nm].append(e / d_)
        if (i + 1) % 500 == 0:
            print(f"    eval {i+1}/{n_test}  {(time.time()-t0)/60:.1f} min",
                  flush=True)
    return per, np.asarray(snr)


# ======================================================= P19
def score_p19():
    """Across-seed spread of Delta_H under focused [5,20] training."""
    seeds = {}
    # seed 1 is the committed stage-4 pair
    have_s1 = (STAGE4 / "C_U1_snr5_20" / "best.pt").exists()
    pairs = []
    if have_s1:
        pairs.append(("seed1", load_path(STAGE4 / "C_U1_snr5_20" / "best.pt",
                                         hankel=False),
                      load_path(STAGE4 / "C_H1_snr5_20" / "best.pt",
                                hankel=True)))
    for s in (2, 3):
        u, h = f"C1_U1_seed{s}", f"C1_H1_seed{s}"
        if (PARTC / u / "best.pt").exists() and (PARTC / h / "best.pt").exists():
            pairs.append((f"seed{s}", load_arm(u, hankel=False),
                          load_arm(h, hankel=True)))
    if not pairs:
        return {"status": "NOT RUN"}

    models = {}
    for tag, u, h in pairs:
        models[f"U1_{tag}"] = u
        models[f"H1_{tag}"] = h
    per, snr = eval_models(models, HIGH_SNR)

    for tag, *_ in pairs:
        r = paired(per, f"U1_{tag}", f"H1_{tag}")
        seeds[tag] = {"delta_h_db": r["median_diff_db"],
                      "ci95": r["boot_ci95_median"]}
    vals = np.array([v["delta_h_db"] for v in seeds.values()])
    sd = float(vals.std(ddof=1)) if vals.size > 1 else None
    rng_ = float(vals.max() - vals.min()) if vals.size > 1 else None
    held = (sd is not None and sd <= P19_SD_MAX and rng_ <= P19_RANGE_MAX)
    unsupported = (sd is not None and
                   (rng_ >= P19_DECISION_SPREAD
                    or bool((vals > P19_DECISION_ANY_SEED).any())))
    return {
        "status": ("HELD" if held else "FAILED") if sd is not None else "PARTIAL",
        "n_seeds": len(seeds), "per_seed": seeds,
        "across_seed_sd_db": sd, "across_seed_range_db": rng_,
        "threshold_sd_max": P19_SD_MAX, "threshold_range_max": P19_RANGE_MAX,
        "COLLAPSE_CLAIM_UNSUPPORTED": unsupported,
        "note": "CIs above are over TEST REALISATIONS; the seed spread is the "
                "separate quantity reported here.",
    }


# ======================================================= P20
def score_p20():
    """Delta_H under the SNR-balanced loss, per bin."""
    h = PARTC / "C2_H1_balanced" / "best.pt"
    u = STAGE5 / "C1_snr_balanced_P20" / "best.pt"
    if not h.exists() or not u.exists():
        return {"status": "NOT RUN"}
    models = {"U1_balanced": load_path(u, hankel=False),
              "H1_balanced": load_arm("C2_H1_balanced", hankel=True)}
    per, snr = eval_models(models, FULL_SNR)
    bins = by_bin({k: list(v) for k, v in per.items()},
                  "U1_balanced", "H1_balanced", snr)
    m = snr >= 5
    hi = paired({k: list(np.asarray(v)[m]) for k, v in per.items()},
                "U1_balanced", "H1_balanced")
    pooled = paired({k: list(v) for k, v in per.items()},
                    "U1_balanced", "H1_balanced")
    d = hi["median_diff_db"]
    worst_bin = max(b["median_diff_db"] for b in bins["bins"])
    held = (P20_BAND[0] <= d <= P20_BAND[1]
            and worst_bin <= P20_NO_BIN_ABOVE
            and abs(pooled["median_diff_db"]) <= P20_POOLED_ABS_MAX)
    return {
        "status": "HELD" if held else "FAILED",
        "delta_h_high_snr_ge5_db": d, "ci95": hi["boot_ci95_median"],
        "delta_h_pooled_db": pooled["median_diff_db"],
        "per_bin": bins["bins"],
        "predicted_band_high_snr": list(P20_BAND), "predicted_point": 0.10,
        "max_bin_db": worst_bin, "threshold_no_bin_above": P20_NO_BIN_ABOVE,
        "RECOVERS_ABOVE_BAND": bool(d > P20_BAND[1]),
        "tension_note": "if Delta_H recovers above +0.35 dB the recommendation "
                        "and the negative result are in tension and Paper 2 "
                        "must say so",
    }


# ======================================================= P21
def score_p21():
    """Log-domain loss: gradient share, and gain vs both references."""
    lg = PARTC / "C3_U1_logdb"
    if not (lg / "best.pt").exists():
        return {"status": "NOT RUN"}
    res = json.loads((lg / "result.json").read_text())
    gs = res.get("gradient_shares", {})
    span = gs.get("span_max_over_min")

    models = {"U1_logdb": load_arm("C3_U1_logdb", hankel=False)}
    if STAGE2_U1.exists():
        models["U1_unbalanced"] = load_path(STAGE2_U1, hankel=False)
    ub = STAGE5 / "C1_snr_balanced_P20" / "best.pt"
    if ub.exists():
        models["U1_balanced"] = load_path(ub, hankel=False)
    per, snr = eval_models(models, FULL_SNR)
    m = snr >= 5
    sub = {k: list(np.asarray(v)[m]) for k, v in per.items()}

    out = {"gradient_share": gs, "span_max_over_min": span,
           "threshold_span_max": P21_SPAN_MAX}
    if "U1_unbalanced" in per:
        r = paired(sub, "U1_unbalanced", "U1_logdb")
        out["logdb_vs_unbalanced_high_snr_db"] = r["median_diff_db"]
        out["logdb_vs_unbalanced_ci"] = r["boot_ci95_median"]
    if "U1_balanced" in per:
        r = paired(sub, "U1_balanced", "U1_logdb")
        # positive => logdb better than balanced
        out["logdb_minus_balanced_high_snr_db"] = r["median_diff_db"]
        out["logdb_minus_balanced_ci"] = r["boot_ci95_median"]

    gain = out.get("logdb_vs_unbalanced_high_snr_db")
    marg = out.get("logdb_minus_balanced_high_snr_db")
    held = (span is not None and span < P21_SPAN_MAX
            and gain is not None and gain >= P21_MIN_GAIN_VS_UNBALANCED
            and marg is not None and abs(marg) <= P21_MARGIN_VS_BALANCED)
    out["status"] = "HELD" if held else "FAILED"
    out["REPRODUCES_BALANCED"] = bool(marg is not None
                                      and abs(marg) <= P21_MARGIN_VS_BALANCED)
    out["cost_note"] = ("if this holds, Paper 2 must position per-bin "
                        "reweighting as one of at least two adequate fixes "
                        "rather than as THE fix")
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, fn in (("P19_seed_spread", score_p19),
                     ("P20_delta_h_balanced", score_p20),
                     ("P21_logdb_control", score_p21)):
        print(f"\n### {name}", flush=True)
        try:
            res[name] = fn()
        except Exception as e:                       # keep partial results
            res[name] = {"status": "ERROR", "error": f"{type(e).__name__}: {e}"}
        print(json.dumps(res[name], indent=1)[:2200])
    (OUT / "partC_scores.json").write_text(json.dumps(res, indent=1) + "\n",
                                           encoding="utf-8")
    print(f"\nwrote {OUT/'partC_scores.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""PROMPT 21 Part B -- score P31 on the log-domain loss.

Written and committed BEFORE batch C4 finishes, so no threshold and no decision
branch in it can have been chosen after seeing a number. Everything is
transcribed from reports/p21/PREREG_P31.md, committed standing alone at c4cbc56
before C4 started. Every verdict is a RE-REGISTRATION and is printed as one.

Two contrasts, both against arms already in hand, both mixed-SNR:

    log - balanced    C1_seedN   (bin-reweighted) - C4_U1log_seedN
    log - unbalanced  U1_seedN   (conventional)   - C4_U1log_seedN

by_bin(per, a, b) reports a - b, so positive means the LOG arm is better.

Scorer-correctness check
------------------------
Neither contrast has a published value on valid arms. The pass therefore also
reproduces the published P27 contrast U1 - C1, which shares the worlds, the
path, the bin edges and BOTH comparison arms with the targets. If it
reproduces, the targets are being read correctly.

Gradient share -- and why this file re-implements it
----------------------------------------------------
P31b's falsifier is "span below 1.9", and that 1.9 came from
trackD_urformer/stage5.py's gradient_shares(), which is hard-wired to
weighted_nmse_loss through a `balanced: bool` flag and cannot evaluate a
log-domain loss. The replica below reproduces stage5's ALGORITHM exactly --
bins in the outer loop, per-batch gradient norms averaged over the batches
seen -- and only parameterises the loss, so the number it returns is comparable
to the 1.9 and 30.0 anchors.

It is NOT directly comparable to the invalid arm's 3.32, which came from
scratch/p12_partC.py's different estimator (batches in the outer loop, weighted
by subset size). That is recorded in the output rather than glossed.

Run:  PYTHONPATH=. python3 scratch/p21_score_c4.py
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from trackD_urformer.baselines import nmse_parts
from trackD_urformer.config import TrackDConfig
from trackD_urformer.dataset import TrackDDataset, collate
from trackD_urformer.stage1 import build_model
from trackD_urformer.stage4 import by_bin
from trackD_urformer.stage5 import (
    BIN_EDGES, BIN_WEIGHTS, snr_weights, weighted_nmse_loss,
)
from trackD_urformer.train import make_initial_batch

OUT = Path("reports/p21/p31_score.json")
N_TEST = 2000
GRAD_BATCHES = 24          # stage5's default

# tag -> (bin-reweighted arm, conventional arm, log arm)
ARMS = {
    "seed1": ("results/p16/tier05/C1_seed1/best.pt",
              "results/p16/tier05/U1_seed1/best.pt",
              "results/p17/tier1/C4_U1log_seed1/best.pt"),
    "seed2": ("results/p16/tier05/C1_seed2/best.pt",
              "results/p16/tier05/U1_seed2/best.pt",
              "results/p17/tier1/C4_U1log_seed2/best.pt"),
    "seed3": ("results/p16/tier05/C1_seed3/best.pt",
              "results/p16/tier05/U1_seed3/best.pt",
              "results/p17/tier1/C4_U1log_seed3/best.pt"),
}

# ---- transcribed from PREREG_P31.md, committed c4cbc56 before C4 -----------
P31A_SD_MAX = 0.35          # P31a(i) falsifier
P31A_SD_POINT = 0.15
P31A_MARGIN_ABS_MAX = 0.50  # P31a(ii) falsifier, the equivalence margin
P31A_MARGIN_POINT = 0.15
P31B_SPAN_MIN = 1.9         # P31b FAILS if the measured span is BELOW this
P31B_SPAN_POINT = 2.8
# Anchors, all from stage5's estimator except the last.
SPAN_CONVENTIONAL = 30.0
SHARE_BELOW5_CONVENTIONAL = 0.897
SPAN_BIN_REWEIGHTED = 1.9
SHARE_BELOW5_BIN_REWEIGHTED = 0.427
INVALID_SPAN_DIFFERENT_ESTIMATOR = 3.32
INVALID_SHARE_BELOW5 = 0.359
INVALID_LOG_MINUS_BALANCED = 0.343
INVALID_LOG_MINUS_UNBALANCED = 2.228

# P27's published per-seed values, for the path check.
P27_PUBLISHED = {"seed1": 1.9024, "seed2": 2.2427, "seed3": 2.3691}
P27_TOL_DB = 0.002


def logdb_loss(G_hat, G, snr_db=None):
    """C4's training loss: per-sample error in dB, averaged. Mirrors
    scratch/p17_tier1.py's 'logdb' branch exactly."""
    num = torch.sum(torch.abs(G_hat - G) ** 2, dim=(1, 2))
    den = torch.sum(torch.abs(G) ** 2, dim=(1, 2))
    return torch.mean(10.0 * torch.log10(num / den + 1e-30))


def balanced_loss(G_hat, G, snr_db):
    return weighted_nmse_loss(G_hat, G, snr_weights(snr_db))


def conventional_loss(G_hat, G, snr_db=None):
    return weighted_nmse_loss(G_hat, G, None)


def load(cfg, path: str, hankel: bool = False):
    rc = replace(cfg, model=replace(cfg.model, filter_init="random",
                                    use_transformer=True, use_hankel=hankel,
                                    hankel_rank=7, hankel_mode="fixed",
                                    hankel_gate="none"))
    m, _ = build_model(rc, "arm1b_full_random")
    m.load_state_dict(torch.load(path, map_location="cpu",
                                 weights_only=False)["model"])
    return m.eval()


def provenance(path: str) -> dict:
    b = torch.load(path, map_location="cpu", weights_only=False)
    c = b.get("config") or {}
    tr = c.get("train", {}) if isinstance(c, dict) else {}
    return {"epoch": b.get("epoch"), "init": tr.get("init"),
            "train_seed": tr.get("seed"),
            "spec": b.get("spec")}


def gradient_shares(model, loader, cfg, loss_fn, n_batches=GRAD_BATCHES) -> dict:
    """stage5.gradient_shares' ALGORITHM, with the loss parameterised.

    Bins in the outer loop; per-batch gradient norms averaged over the batches
    actually seen in that bin. Reproduced rather than imported because the
    original hard-wires weighted_nmse_loss behind a boolean.
    """
    shares = np.zeros(len(BIN_WEIGHTS))
    model_was_training = model.training
    model.train()
    for bi in range(len(BIN_WEIGHTS)):
        lo, hi = BIN_EDGES[bi], BIN_EDGES[bi + 1]
        tot_g, seen = 0.0, 0
        for b in loader:
            m = (b["snr_db"] >= lo) & (b["snr_db"] < hi)
            if not bool(m.any()):
                continue
            sub = {k: (v[m] if torch.is_tensor(v) else v) for k, v in b.items()}
            model.zero_grad(set_to_none=True)
            G0 = make_initial_batch(sub, cfg.train.init, cfg)
            est = model(G0, sub["Z"], sub["S"], sub["B"], sub["sigma2"])
            loss_fn(est, sub["G_true"], sub["snr_db"]).backward()
            tot_g += float(torch.sqrt(sum((p.grad ** 2).sum()
                                          for p in model.parameters()
                                          if p.grad is not None)))
            seen += 1
            if seen >= n_batches:
                break
        shares[bi] = tot_g / max(seen, 1)
    model.zero_grad(set_to_none=True)
    if not model_was_training:
        model.eval()
    sh = shares / shares.sum()
    return {"bins": [[float(BIN_EDGES[i]), float(BIN_EDGES[i + 1])]
                     for i in range(len(BIN_WEIGHTS))],
            "grad_share": [round(float(x), 6) for x in sh],
            "grad_share_below_5dB": round(float(sh[:3].sum()), 4),
            "span_max_over_min": round(float(sh.max() / sh.min()), 4),
            "estimator": "stage5 algorithm, loss parameterised"}


@torch.no_grad()
def evaluate(cfg, models: dict) -> tuple[dict, np.ndarray]:
    ds = TrackDDataset("test", sysc=cfg.system,
                       datac=replace(cfg.data, n_test=N_TEST),
                       numeric=cfg.numeric, init=cfg.train.init)
    cd, rd = cfg.numeric.complex_dtype, cfg.numeric.real_dtype
    per, snr = {k: [] for k in models}, []
    T = lambda a, d: torch.as_tensor(np.array(a, copy=True)[None], dtype=d)
    for i in range(N_TEST):
        s = ds.sample(i)
        snr.append(s.snr_db)
        G0, Z = T(ds.g0(i), cd), T(s.Z, rd)
        S, B = T(s.S, cd), T(s.B, cd)
        s2 = torch.tensor([s.sigma2], dtype=rd)
        for n, m in models.items():
            gh = m(G0, Z, S, B, s2)[0].numpy()
            a, b = nmse_parts(gh, s.G_true)
            per[n].append(a / b)
        if (i + 1) % 500 == 0:
            print(f"    eval {i+1}/{N_TEST}", flush=True)
    return per, np.asarray(snr)


def summarise(r: dict) -> dict:
    hs, pl = r["high_snr_ge5"], r["pooled_SAMPLING_DESIGN_DEPENDENT"]
    return {
        "ge5_median_db": hs["median_diff_db"],
        "ge5_ci95_over_test_realisations": hs["boot_ci95_median"],
        "ge5_ci_excludes_zero": hs["ci_excludes_zero"],
        "ge5_ratio_of_sums_db": hs.get("ratio_of_sums_diff_db_SECONDARY"),
        "pooled_median_db": pl["median_diff_db"],
        "pooled_ratio_of_sums_db": pl.get("ratio_of_sums_diff_db_SECONDARY"),
        "per_bin_db": [b["median_diff_db"] for b in r["bins"]],
        "per_bin_ci95_over_test_realisations":
            [b["boot_ci95_median"] for b in r["bins"]],
        "per_bin_ci_excludes_zero": [b["ci_excludes_zero"] for b in r["bins"]],
        "bins": [b["bin"] for b in r["bins"]],
    }


def main() -> int:
    torch.set_num_threads(1)
    base = TrackDConfig()
    cfg = replace(base, train=replace(base.train, init="spectral"))
    assert tuple(cfg.data.snr_range_db) == (-10.0, 20.0)
    assert cfg.system.P == 20

    have = {k: v for k, v in ARMS.items() if all(Path(p).exists() for p in v)}
    missing = sorted(set(ARMS) - set(have))
    if missing:
        print(f"MISSING checkpoints for: {missing}")
    if len(have) < 3:
        print(f"P31 needs three seeds, have {sorted(have)}; aborting")
        return 1

    prov, models = {}, {}
    for tag, (c1, u1, lg) in have.items():
        for nm, path in ((f"C1_{tag}", c1), (f"U1_{tag}", u1),
                         (f"LOG_{tag}", lg)):
            prov[nm] = provenance(path)
            models[nm] = load(cfg, path)

    bad = {k: v for k, v in prov.items() if v["init"] != "spectral"}
    if bad:
        print(f"REFUSING TO SCORE -- non-spectral initialiser: {bad}")
        return 1

    print(f"evaluating {len(models)} arms on {N_TEST} shared worlds", flush=True)
    per, snr = evaluate(cfg, models)

    order = ["seed1", "seed2", "seed3"]
    vs_bal = {t: summarise(by_bin(per, f"C1_{t}", f"LOG_{t}", snr))
              for t in order}
    vs_unb = {t: summarise(by_bin(per, f"U1_{t}", f"LOG_{t}", snr))
              for t in order}
    p27 = {t: by_bin(per, f"U1_{t}", f"C1_{t}", snr)["high_snr_ge5"]
           ["median_diff_db"] for t in order}
    p27_err = {t: abs(p27[t] - P27_PUBLISHED[t]) for t in order}
    p27_ok = all(e <= P27_TOL_DB for e in p27_err.values())

    # --- gradient share on the log arms, under the log loss ----------------
    print("  gradient shares", flush=True)
    tds = TrackDDataset("train", sysc=cfg.system,
                        datac=replace(cfg.data, n_train=4096),
                        numeric=cfg.numeric, P=cfg.system.P,
                        init=cfg.train.init)
    loader = DataLoader(tds, batch_size=cfg.train.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=0)
    shares = {}
    for t in order:
        shares[t] = {
            "under_log_loss": gradient_shares(models[f"LOG_{t}"], loader, cfg,
                                              logdb_loss),
            # same model, conventional loss: isolates model from loss
            "under_conventional_loss": gradient_shares(
                models[f"LOG_{t}"], loader, cfg, conventional_loss),
        }

    # --- verdicts ----------------------------------------------------------
    vals = np.array([vs_bal[t]["ge5_median_db"] for t in order])
    m, s = float(vals.mean()), float(vals.std(ddof=1))
    unb = np.array([vs_unb[t]["ge5_median_db"] for t in order])

    p31a_i = s <= P31A_SD_MAX
    p31a_ii = abs(m) <= P31A_MARGIN_ABS_MAX
    spans = np.array([shares[t]["under_log_loss"]["span_max_over_min"]
                      for t in order])
    span_mean = float(spans.mean())
    p31b = span_mean >= P31B_SPAN_MIN        # HELD if it does NOT fall below

    lo_hi = [(vs_bal[t]["per_bin_db"][0], vs_bal[t]["per_bin_db"][5])
             for t in order]
    p31c_each = [lo > hi for lo, hi in lo_hi]
    p31c = all(p31c_each)

    if (m - 2 * s) > 0:
        branch, title_claim = "LOG_BETTER", (
            "the log loss beats bin reweighting on three seeds; the paper "
            "reports a fix that improves on the published remedy, and the "
            "title may claim both an attribution finding and an improved fix")
    elif abs(m) <= P31A_MARGIN_ABS_MAX:
        branch, title_claim = "EQUIVALENT", (
            "the two are equivalent within the registered margin; the paper "
            "reports bin reweighting as adequate and the log loss as A SIMPLER "
            "ALTERNATIVE THAT IS NO BETTER, in those words, and the title "
            "claims an attribution finding with a replication attached")
    elif (m + 2 * s) < 0:
        branch, title_claim = "LOG_WORSE", (
            "the log loss is worse; the paper says so and recommends bin "
            "reweighting")
    else:
        branch, title_claim = "MARGIN_EXCEEDED", (
            "the margin exceeds +/-0.50 dB, so the equivalence framing is "
            "abandoned and the measured direction is reported as the result")

    out = {
        "prereg": "reports/p21/PREREG_P31.md @ c4cbc56",
        "re_registration": True,
        "re_registration_note": (
            "The invalid random-init arm already reported +0.343 (log - "
            "balanced), +2.228 (log - unbalanced), span 3.32 and share 0.359. "
            "Predictions were made knowing them; every verdict here is a "
            "re-registration and must be labelled one wherever quoted."),
        "P31c_status": (
            "UNSCOREABLE THIS TURN. P31c asked for the per-bin sign sequence "
            "of Delta_H under log-loss training. All three C4 arms are "
            "hankel=False, so no prior-carrying log arm exists; measuring it "
            "needs three further H1-log runs, which PROMPT 21's scope "
            "forbids. Not implied to have been tested."),
        "n_test": N_TEST, "P": int(cfg.system.P),
        "checkpoint_provenance": prov,
        "path_check_P27": {
            "published": P27_PUBLISHED,
            "measured": {t: round(p27[t], 4) for t in order},
            "abs_error_db": {t: round(p27_err[t], 6) for t in order},
            "tolerance_db": P27_TOL_DB, "reproduced": bool(p27_ok),
            "meaning": ("shares worlds, path, bins and BOTH comparison arms "
                        "with the targets; if false, no verdict below holds")},
        "log_minus_balanced": vs_bal,
        "log_minus_unbalanced": vs_unb,
        "bins": vs_bal["seed1"]["bins"],
        "three_seed_mean_log_minus_balanced_db": round(m, 4),
        "seed_spread_sd_db": round(s, 4),
        "seed_spread_range_db": round(float(vals.max() - vals.min()), 4),
        "mean_minus_2sd_db": round(m - 2 * s, 4),
        "three_seed_mean_log_minus_unbalanced_db": round(float(unb.mean()), 4),
        "seed_spread_sd_log_minus_unbalanced_db":
            round(float(unb.std(ddof=1)), 4),
        "gradient_shares": shares,
        "gradient_share_anchors": {
            "conventional_span": SPAN_CONVENTIONAL,
            "conventional_share_below_5dB": SHARE_BELOW5_CONVENTIONAL,
            "bin_reweighted_span": SPAN_BIN_REWEIGHTED,
            "bin_reweighted_share_below_5dB": SHARE_BELOW5_BIN_REWEIGHTED,
            "invalid_arm_span": INVALID_SPAN_DIFFERENT_ESTIMATOR,
            "invalid_arm_share_below_5dB": INVALID_SHARE_BELOW5,
            "comparability_note": (
                "The conventional and bin-reweighted anchors come from "
                "stage5.gradient_shares, whose algorithm this file "
                "reproduces, so the measured spans are comparable to them. "
                "The invalid arm's 3.32 came from scratch/p12_partC.py's "
                "DIFFERENT estimator and is NOT directly comparable to "
                "either.")},
        "P31a": {
            "i_prediction": f"across-seed SD <= {P31A_SD_MAX} dB",
            "i_point_estimate": P31A_SD_POINT,
            "i_measured_sd_db": round(s, 4),
            "i_status": "HELD" if p31a_i else "FAILED",
            "ii_prediction": f"|three-seed mean| <= {P31A_MARGIN_ABS_MAX} dB",
            "ii_point_estimate": P31A_MARGIN_POINT,
            "ii_measured_mean_db": round(m, 4),
            "ii_status": "HELD" if p31a_ii else "FAILED"},
        "P31b": {
            "prediction": f"log-loss gradient-share span is NOT below "
                          f"{P31B_SPAN_MIN}",
            "point_estimate": P31B_SPAN_POINT,
            "measured_span_per_seed": [float(x) for x in spans],
            "measured_span_mean": round(span_mean, 4),
            "status": "HELD" if p31b else "FAILED",
            "if_failed_means": ("the log loss flattens the gradient better "
                                "than explicit reweighting while needing none "
                                "of its machinery")},
        "P31c_prime": {
            "prediction": ("the log-over-balanced margin is larger at low SNR "
                           "than at high SNR on every seed"),
            "per_seed_low_minus_high_db":
                {t: round(float(lo - hi), 4)
                 for t, (lo, hi) in zip(order, lo_hi)},
            "per_seed_held": dict(zip(order, [bool(x) for x in p31c_each])),
            "status": "HELD" if p31c else "FAILED"},
        "decision": {"branch": branch, "consequence": title_claim,
                     "rule": "PREREG_P31, evaluated in its stated order"},
        "note": ("CIs are over TEST REALISATIONS; the seed spread is the "
                 "separate quantity in seed_spread_*. Three seeds give two "
                 "degrees of freedom."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    print("\npath check (published P27 contrast, U1 - C1):")
    for t in order:
        print(f"  {t}: measured {p27[t]:+.4f} vs published "
              f"{P27_PUBLISHED[t]:+.4f}, error {p27_err[t]:.6f}")
    print(f"  -> {'OK' if p27_ok else 'MISMATCH, verdicts below are void'}")
    print(f"\n  log - balanced (positive = log better)")
    print(f"  {'seed':<7} {'SNR>=5':>9} {'CI over test real.':>22} "
          f"{'ratio-of-sums':>14}")
    for t in order:
        z = vs_bal[t]
        print(f"  {t:<7} {z['ge5_median_db']:>+9.4f} "
              f"  [{z['ge5_ci95_over_test_realisations'][0]:+.3f},"
              f"{z['ge5_ci95_over_test_realisations'][1]:+.3f}]"
              f"{'*' if z['ge5_ci_excludes_zero'] else ' '} "
              f"{z['ge5_ratio_of_sums_db']:>+14.4f}")
    print(f"  {'mean':<7} {m:>+9.4f}   seed SD {s:.4f}, mean-2SD {m-2*s:+.4f}")
    print(f"\n  log - unbalanced: "
          f"{[round(float(x),4) for x in unb]}, mean {unb.mean():+.4f}")
    print(f"\n  gradient-share span under the log loss: "
          f"{[round(float(x),3) for x in spans]}, mean {span_mean:.3f}")
    print(f"    anchors: conventional {SPAN_CONVENTIONAL}, bin-reweighted "
          f"{SPAN_BIN_REWEIGHTED}")
    for t in order:
        g = shares[t]["under_log_loss"]
        print(f"    {t} share below 5 dB {g['grad_share_below_5dB']:.4f}")
    print(f"\nP31a(i):  {out['P31a']['i_status']}   SD {s:.4f} "
          f"(max {P31A_SD_MAX})")
    print(f"P31a(ii): {out['P31a']['ii_status']}   mean {m:+.4f} "
          f"(|.| <= {P31A_MARGIN_ABS_MAX})")
    print(f"P31b:     {out['P31b']['status']}   span {span_mean:.3f} "
          f"(fails if < {P31B_SPAN_MIN})")
    print(f"P31c':    {out['P31c_prime']['status']}   "
          f"{out['P31c_prime']['per_seed_low_minus_high_db']}")
    print(f"P31c:     UNSCOREABLE -- no prior-carrying log arm exists")
    print(f"\nDECISION ({branch}): {title_claim}")
    print("  ALL VERDICTS ARE RE-REGISTRATIONS -- label them as such.")
    print("  wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

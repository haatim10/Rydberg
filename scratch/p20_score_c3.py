"""PROMPT 20 Part B -- score P30 on Delta_H under the SNR-balanced loss.

Written and committed BEFORE batch C3 finishes, so no threshold and no decision
branch in it can have been chosen after seeing a number. Everything is
transcribed from reports/p20/PREREG_P30.md, committed standing alone at da632bd
before C3 started.

The contrast
------------
    C1_seedN  (balanced loss, NO prior)  -  C3_H1bal_seedN  (balanced loss, prior)
positive = the structural prior is better. The no-prior side is the Tier 0.5
arm already in hand; the prior side is trained by C3.

The pairing crosses two drivers, which is the confound shape that cost PROMPT
15 a turn. Settled before PREREG_P30 was written: the two drivers build
identical configs in every field and bitwise identical initial weights for a
balanced no-prior arm, and the C3 arm differs from its pairing in exactly one
field, model.use_hankel.

Scorer-correctness check
------------------------
This contrast has no published value -- that is why C3 exists -- so the check
cannot be a reproduction of its own number. Instead the pass also evaluates the
Tier 0.5 U1 arms and reproduces the P27 contrast U1 - C1, whose three seed
values are published in reports/p16/p27_score.json. That shares the worlds, the
path, the bins and the no-prior arms with the target contrast, so if it
reproduces, the target is being read correctly too. It costs three extra arms.

Run:  PYTHONPATH=. python3 scratch/p20_score_c3.py
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from trackD_urformer.baselines import nmse_parts
from trackD_urformer.config import TrackDConfig
from trackD_urformer.dataset import TrackDDataset
from trackD_urformer.stage1 import build_model
from trackD_urformer.stage4 import by_bin

OUT = Path("reports/p20/p30_score.json")
N_TEST = 2000

# tag -> (no-prior balanced arm, prior balanced arm, uniform-loss arm)
ARMS = {
    "seed1": ("results/p16/tier05/C1_seed1/best.pt",
              "results/p17/tier1/C3_H1bal_seed1/best.pt",
              "results/p16/tier05/U1_seed1/best.pt"),
    "seed2": ("results/p16/tier05/C1_seed2/best.pt",
              "results/p17/tier1/C3_H1bal_seed2/best.pt",
              "results/p16/tier05/U1_seed2/best.pt"),
    "seed3": ("results/p16/tier05/C1_seed3/best.pt",
              "results/p17/tier1/C3_H1bal_seed3/best.pt",
              "results/p16/tier05/U1_seed3/best.pt"),
}

# ---- transcribed from PREREG_P30.md, committed da632bd before C3 -----------
P30A_SD_MAX = 0.35            # P30a falsifier
P30A_POINT = 0.12
P30B_MEAN_MAX = 0.60          # P30b falsifier -- THE THESIS TEST
P30B_POINT = 0.25
P30C_SIGNS = ["-", "-", "-", "+", "+", "+"]
# What the INVALID random-init arm showed, recorded because it is not unseen.
INVALID_PRIOR_GE5_DB = 0.174
INVALID_PRIOR_SIGNS = ["-", "-", "-", "-", "+", "+"]
# Anchors this contrast is read against, both three-seed means.
MIXED_UNIFORM_DELTA_H = 1.2862
FOCUSED_DELTA_H = 0.0197

# P27's published per-seed values, for the path check.
P27_PUBLISHED = {"seed1": 1.9024, "seed2": 2.2427, "seed3": 2.3691}
P27_TOL_DB = 0.002


def load(cfg, path: str, hankel: bool):
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
    md = c.get("model", {}) if isinstance(c, dict) else {}
    return {"epoch": b.get("epoch"), "init": tr.get("init"),
            "train_seed": tr.get("seed"), "use_hankel": md.get("use_hankel")}


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
        print(f"P30 needs three seeds, have {sorted(have)}; aborting")
        return 1

    prov, models = {}, {}
    for tag, (c1, h1, u1) in have.items():
        for nm, path, hk in ((f"C1_{tag}", c1, False),
                             (f"H1bal_{tag}", h1, True),
                             (f"U1_{tag}", u1, False)):
            prov[nm] = provenance(path)
            models[nm] = load(cfg, path, hankel=hk)

    bad = {k: v for k, v in prov.items() if v["init"] != "spectral"}
    if bad:
        print(f"REFUSING TO SCORE -- non-spectral initialiser: {bad}")
        return 1

    print(f"evaluating {len(models)} arms on {N_TEST} shared worlds", flush=True)
    per, snr = evaluate(cfg, models)

    order = ["seed1", "seed2", "seed3"]
    # target contrast: balanced no-prior minus balanced prior
    seeds = {t: summarise(by_bin(per, f"C1_{t}", f"H1bal_{t}", snr))
             for t in order}
    # path check: the published P27 contrast, same worlds and same no-prior arms
    p27 = {t: by_bin(per, f"U1_{t}", f"C1_{t}", snr)["high_snr_ge5"]
           ["median_diff_db"] for t in order}
    p27_err = {t: abs(p27[t] - P27_PUBLISHED[t]) for t in order}
    p27_ok = all(e <= P27_TOL_DB for e in p27_err.values())

    vals = np.array([seeds[t]["ge5_median_db"] for t in order])
    m, s = float(vals.mean()), float(vals.std(ddof=1))
    rng, vmin = float(vals.max() - vals.min()), float(vals.min())
    per_bin = np.array([seeds[t]["per_bin_db"] for t in order])
    bin_mean, bin_sd = per_bin.mean(axis=0), per_bin.std(axis=0, ddof=1)

    p30a = s <= P30A_SD_MAX
    p30b = m <= P30B_MEAN_MAX
    sign_rows = [["+" if v > 0 else "-" for v in row] for row in per_bin]
    p30c_each = [r == P30C_SIGNS for r in sign_rows]
    p30c = all(p30c_each)
    matches_invalid = all(r == INVALID_PRIOR_SIGNS for r in sign_rows)

    if p30b:
        branch = ("STRONG: removing the training imbalance dissolves most of "
                  "the prior's advantage, and this now holds under two "
                  "independent ways of removing it -- narrowing the training "
                  "range, and reweighting the loss without narrowing it")
    else:
        branch = ("WEAKER: the attribution argument is specific to the focused "
                  "design. The prior retains substantial value under the "
                  "balanced loss, and the focused result does not generalise "
                  "to every way of fixing the imbalance")

    out = {
        "prereg": "reports/p20/PREREG_P30.md @ da632bd",
        "re_registration": True,
        "re_registration_note": (
            "The invalid random-init arm already reported +0.174 for this "
            "quantity with sign sequence - - - - + +. The predictions were "
            "made knowing that and are correspondingly weaker; every verdict "
            "here must be labelled a re-registration wherever it is quoted."),
        "contrast": ("C1 (balanced, no prior) - H1bal (balanced, prior); "
                     "positive = the structural prior is better"),
        "n_test": N_TEST, "P": int(cfg.system.P),
        "checkpoint_provenance": prov,
        "per_seed": seeds,
        "bins": seeds["seed1"]["bins"],
        "path_check_P27": {
            "contrast": "U1 - C1, the published P27 quantity",
            "published": P27_PUBLISHED,
            "measured": {t: round(p27[t], 4) for t in order},
            "abs_error_db": {t: round(p27_err[t], 6) for t in order},
            "tolerance_db": P27_TOL_DB,
            "reproduced": bool(p27_ok),
            "meaning": ("shares worlds, path, bins and the no-prior arms with "
                        "the target contrast; if false, NO verdict below may "
                        "be trusted")},
        "three_seed_mean_ge5_db": round(m, 4),
        "seed_spread_sd_db": round(s, 4),
        "seed_spread_range_db": round(rng, 4),
        "mean_minus_2sd_db": round(m - 2 * s, 4),
        "three_seed_mean_per_bin_db": [round(float(x), 4) for x in bin_mean],
        "seed_spread_sd_per_bin_db": [round(float(x), 4) for x in bin_sd],
        "anchors": {
            "mixed_uniform_loss_delta_H": MIXED_UNIFORM_DELTA_H,
            "focused_delta_H": FOCUSED_DELTA_H,
            "invalid_random_init_value": INVALID_PRIOR_GE5_DB},
        "P30a": {"prediction": f"across-seed SD <= {P30A_SD_MAX} dB",
                 "point_estimate": P30A_POINT, "measured_sd_db": round(s, 4),
                 "status": "HELD" if p30a else "FAILED"},
        "P30b": {"prediction": f"three-seed mean <= {P30B_MEAN_MAX} dB",
                 "point_estimate": P30B_POINT, "measured_mean_db": round(m, 4),
                 "status": "HELD" if p30b else "FAILED",
                 "is_the_thesis_test": True},
        "P30c": {"prediction": "per-bin signs - - - + + + on every seed",
                 "per_seed_signs": {t: "".join(r)
                                    for t, r in zip(order, sign_rows)},
                 "per_seed_held": dict(zip(order, p30c_each)),
                 "status": "HELD" if p30c else "FAILED",
                 "matches_invalid_arm_pattern": bool(matches_invalid),
                 "note": ("if it fails by reproducing the invalid arm's "
                          "- - - - + + on all three seeds, that is not noise: "
                          "it says the crossing SNR moves with the loss "
                          "design, which is itself reportable")},
        "separation": {"rule": "mean - 2*SD > 0 across seeds",
                       "value": round(m - 2 * s, 4),
                       "separated": bool((m - 2 * s) > 0)},
        "decision": {"branch": "STRONG" if p30b else "WEAKER",
                     "consequence": branch},
        "note": ("CIs in per_seed are over TEST REALISATIONS; the seed spread "
                 "is the separate quantity reported as seed_spread_*. Three "
                 "seeds give two degrees of freedom."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    print("\npath check (published P27 contrast, U1 - C1):")
    for t in order:
        print(f"  {t}: measured {p27[t]:+.4f} vs published "
              f"{P27_PUBLISHED[t]:+.4f}, error {p27_err[t]:.6f}")
    print(f"  -> {'OK' if p27_ok else 'MISMATCH, verdicts below are void'}")
    print(f"\n  {'seed':<7} {'SNR>=5':>9} {'CI over test real.':>22} "
          f"{'ratio-of-sums':>14}")
    for t in order:
        z = seeds[t]
        print(f"  {t:<7} {z['ge5_median_db']:>+9.4f} "
              f"  [{z['ge5_ci95_over_test_realisations'][0]:+.3f},"
              f"{z['ge5_ci95_over_test_realisations'][1]:+.3f}]"
              f"{'*' if z['ge5_ci_excludes_zero'] else ' '} "
              f"{z['ge5_ratio_of_sums_db']:>+14.4f}")
    print(f"  {'mean':<7} {m:>+9.4f}   seed SD {s:.4f}, range {rng:.4f}, "
          f"mean-2SD {m - 2*s:+.4f}")
    print(f"\n  anchors: mixed/uniform {MIXED_UNIFORM_DELTA_H:+.4f}, "
          f"focused {FOCUSED_DELTA_H:+.4f}, invalid arm "
          f"{INVALID_PRIOR_GE5_DB:+.4f}")
    print("  per-bin three-seed mean:", [f"{v:+.3f}" for v in bin_mean])
    for t, r in zip(order, sign_rows):
        print(f"    {t} signs {''.join(r)}")
    print(f"\nP30a: {out['P30a']['status']}  SD {s:.4f} (max {P30A_SD_MAX})")
    print(f"P30b: {out['P30b']['status']}  mean {m:+.4f} "
          f"(max {P30B_MEAN_MAX})  <- THE THESIS TEST")
    print(f"P30c: {out['P30c']['status']}  {out['P30c']['per_seed_signs']}"
          f"{'  [matches the invalid arm]' if matches_invalid else ''}")
    print(f"\nDECISION ({out['decision']['branch']}): {branch}")
    print("  ALL VERDICTS ARE RE-REGISTRATIONS -- label them as such.")
    print("  wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""PROMPT 19 Part B -- score P29 on the three seeds of the +1.209 dB contrast.

Written and committed BEFORE batch C2 finishes, so no threshold and no outcome
trigger in it can have been chosen after seeing a number. Everything is
transcribed from reports/p17/PREREG_P29.md, committed standing alone at 157eca6
before C2 started.

The evaluation path mirrors trackD_urformer/stage3.py's test pass:

  * the FULL test SNR range (-10, 20), so this contrast has SIX bins, where the
    focused design of C1 had three;
  * n_test = 2000, P = 20, spectral initialiser;
  * by_bin(per, "U1", "H1", snr), reporting a - b, so positive means the
    Hankel prior is better -- the sign of the published +1.209.

Verified before this file was committed: all 2000 test-world SNR draws are
identical to the stored stage-3 draws.

Reported for every seed and never pooled together:
  * the SNR >= 5 restriction, which is what +1.209 is;
  * the pooled-over-everything figure, which for seed 1 is +0.129;
  * BOTH pooling rules -- paired per-trial median and ratio of sums -- because
    C1 showed the two disagree in sign;
  * the per-bin values, which is what P29d predicts.

Run:  PYTHONPATH=. python3 scratch/p19_score_c2.py
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

OUT = Path("reports/p19/p29_score.json")
N_TEST = 2000

# seed -> (no-prior checkpoint, Hankel-prior checkpoint). Seed 1 is the
# published stage-2 / stage-3 pair and is NOT retrained by C2.
ARMS = {
    "seed1": ("results/track_d/stage2/B3_80k_13ep/best.pt",
              "results/track_d/stage3/H1_hs_urformer_80k/best.pt"),
    "seed2": ("results/p17/tier1/C2_U1_seed2/best.pt",
              "results/p17/tier1/C2_H1_seed2/best.pt"),
    "seed3": ("results/p17/tier1/C2_U1_seed3/best.pt",
              "results/p17/tier1/C2_H1_seed3/best.pt"),
}

# ---- transcribed from PREREG_P29.md, committed 157eca6 before C2 -----------
P29A_SD_MAX = 0.45           # P29a falsifier
P29A_POINT = 0.20            # pre-registered point estimate
OUTCOME_B_SD_FRACTION = 0.40  # s >= 0.40*m triggers outcome (b)
OUTCOME_A_MEAN_MIN = 0.90    # 75% of +1.209
# P29d: three negatives then three positives, flipping between [0,5) and [5,10)
P29D_SIGNS = ["-", "-", "-", "+", "+", "+"]

# The published seed-1 anchor. Transcribed so a scorer that silently evaluates
# the wrong worlds fails visibly on seed 1 instead of quietly reporting.
SEED1_PUBLISHED_GE5_DB = 1.209
SEED1_PUBLISHED_GE5_CI = [1.139, 1.330]
SEED1_PUBLISHED_POOLED_DB = 0.1289
SEED1_PUBLISHED_PER_BIN = [-0.111, -0.506, -0.451, 0.398, 1.305, 2.226]
REPRODUCTION_TOL_DB = 0.005


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
    cfgd = b.get("config") or {}
    tr = cfgd.get("train", {}) if isinstance(cfgd, dict) else {}
    md = cfgd.get("model", {}) if isinstance(cfgd, dict) else {}
    return {"epoch": b.get("epoch"), "init": tr.get("init"),
            "train_seed": tr.get("seed"), "use_hankel": md.get("use_hankel"),
            "snr_range_db": (cfgd.get("data", {}) or {}).get("snr_range_db")}


@torch.no_grad()
def evaluate(cfg, models: dict) -> tuple[dict, np.ndarray]:
    """One pass, all arms, identical worlds. Mirrors stage3.evaluate_stage3."""
    ds = TrackDDataset("test", sysc=cfg.system,
                       datac=replace(cfg.data, n_test=N_TEST),
                       numeric=cfg.numeric, init=cfg.train.init)
    cd, rd = cfg.numeric.complex_dtype, cfg.numeric.real_dtype
    per = {k: [] for k in models}
    snr = []
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


def main() -> int:
    torch.set_num_threads(1)
    base = TrackDConfig()
    cfg = replace(base, train=replace(base.train, init="spectral"))
    assert cfg.train.init == "spectral"
    assert tuple(cfg.data.snr_range_db) == (-10.0, 20.0), "mixed design"
    assert cfg.system.P == 20

    have = {k: v for k, v in ARMS.items()
            if Path(v[0]).exists() and Path(v[1]).exists()}
    missing = sorted(set(ARMS) - set(have))
    if missing:
        print(f"MISSING checkpoints for: {missing}")
    if "seed1" not in have:
        print("seed 1 is the reproduction check and must be present; aborting")
        return 1
    if len(have) < 3:
        print(f"P29 needs three seeds, have {sorted(have)}; aborting")
        return 1

    prov, models = {}, {}
    for tag, (u, h) in have.items():
        prov[f"U1_{tag}"] = provenance(u)
        prov[f"H1_{tag}"] = provenance(h)
        models[f"U1_{tag}"] = load(cfg, u, hankel=False)
        models[f"H1_{tag}"] = load(cfg, h, hankel=True)

    bad = {k: v for k, v in prov.items() if v["init"] != "spectral"}
    if bad:
        print(f"REFUSING TO SCORE -- arms not trained with the spectral "
              f"initialiser: {bad}")
        return 1

    print(f"evaluating {len(models)} arms on {N_TEST} shared worlds, "
          f"SNR {tuple(cfg.data.snr_range_db)}", flush=True)
    per, snr = evaluate(cfg, models)

    seeds = {}
    for tag in have:
        r = by_bin(per, f"U1_{tag}", f"H1_{tag}", snr)
        hs, pl = r["high_snr_ge5"], r["pooled_SAMPLING_DESIGN_DEPENDENT"]
        seeds[tag] = {
            "ge5_median_db": hs["median_diff_db"],
            "ge5_ci95_over_test_realisations": hs["boot_ci95_median"],
            "ge5_ci_excludes_zero": hs["ci_excludes_zero"],
            "ge5_ratio_of_sums_db": hs.get("ratio_of_sums_diff_db_SECONDARY"),
            "ge5_n": hs["n"],
            "pooled_median_db": pl["median_diff_db"],
            "pooled_ratio_of_sums_db":
                pl.get("ratio_of_sums_diff_db_SECONDARY"),
            "per_bin_db": [b["median_diff_db"] for b in r["bins"]],
            "per_bin_ci95_over_test_realisations":
                [b["boot_ci95_median"] for b in r["bins"]],
            "per_bin_ci_excludes_zero":
                [b["ci_excludes_zero"] for b in r["bins"]],
            "bins": [b["bin"] for b in r["bins"]],
        }

    got1 = seeds["seed1"]["ge5_median_db"]
    repro_err = abs(got1 - SEED1_PUBLISHED_GE5_DB)
    repro_ok = repro_err <= REPRODUCTION_TOL_DB

    order = ["seed1", "seed2", "seed3"]
    vals = np.array([seeds[t]["ge5_median_db"] for t in order])
    m, s = float(vals.mean()), float(vals.std(ddof=1))
    rng, vmin = float(vals.max() - vals.min()), float(vals.min())
    per_bin = np.array([seeds[t]["per_bin_db"] for t in order])
    bin_mean, bin_sd = per_bin.mean(axis=0), per_bin.std(axis=0, ddof=1)

    p29a = s <= P29A_SD_MAX
    rank1 = int(np.argsort(np.argsort(vals))[0])   # 0 smallest, 2 largest
    p29b = rank1 == 1
    p29c = bool(np.all(vals > 0))
    sign_rows = [["+" if v > 0 else "-" for v in row] for row in per_bin]
    p29d_per_seed = [r == P29D_SIGNS for r in sign_rows]
    p29d = all(p29d_per_seed)

    # --- outcome, evaluated in the order PREREG_P29 fixes -------------------
    if (vmin <= 0) or ((m - 2 * s) <= 0) or (s >= OUTCOME_B_SD_FRACTION * m):
        outcome, why = "b", []
        if vmin <= 0:
            why.append(f"min seed {vmin:.4f} <= 0")
        if (m - 2 * s) <= 0:
            why.append(f"mean-2SD {m - 2*s:.4f} <= 0")
        if s >= OUTCOME_B_SD_FRACTION * m:
            why.append(f"SD {s:.4f} >= {OUTCOME_B_SD_FRACTION}*mean "
                       f"{OUTCOME_B_SD_FRACTION * m:.4f}")
    elif m >= OUTCOME_A_MEAN_MIN:
        outcome, why = "a", [f"mean {m:.4f} >= {OUTCOME_A_MEAN_MIN}"]
    else:
        outcome, why = "c", [f"mean {m:.4f} < {OUTCOME_A_MEAN_MIN}, "
                             f"separated from zero"]
    CONSEQUENCE = {
        "a": ("the collapse claim survives in corrected, stronger form: a "
              "stable substantial gain under mixed-SNR training against a "
              "quantity with no stable sign under focused training"),
        "b": ("no training design yields a stable Delta_H; the finding becomes "
              "that this quantity is not measurable at the resolution the "
              "literature reports it to, and the structural-prior section is "
              "written around the per-bin pattern, not a scalar"),
        "c": ("+1.209 was a favourable seed; the paper reports the three-seed "
              "mean plainly and never quotes +1.209 as a point value again"),
    }

    out = {
        "prereg": "reports/p17/PREREG_P29.md @ 157eca6",
        "contrast": ("U1 (no prior) - H1 (Hankel prior, rank 7); positive = "
                     "the structural prior is better"),
        "design": "mixed-SNR training and test on [-10,20]",
        "n_seeds": len(seeds), "n_test": N_TEST, "P": int(cfg.system.P),
        "checkpoint_provenance": prov,
        "per_seed": seeds,
        "bins": seeds["seed1"]["bins"],
        "seed1_reproduction": {
            "published_ge5_db": SEED1_PUBLISHED_GE5_DB,
            "published_ge5_ci95": SEED1_PUBLISHED_GE5_CI,
            "published_pooled_db": SEED1_PUBLISHED_POOLED_DB,
            "published_per_bin_db": SEED1_PUBLISHED_PER_BIN,
            "measured_ge5_db": round(got1, 4),
            "measured_pooled_db": round(seeds["seed1"]["pooled_median_db"], 4),
            "abs_error_db": round(repro_err, 6),
            "tolerance_db": REPRODUCTION_TOL_DB,
            "reproduced": bool(repro_ok),
            "meaning": ("if false, the scorer is reading different worlds or a "
                        "different path and NO verdict below may be trusted")},
        "three_seed_mean_ge5_db": round(m, 4),
        "seed_spread_sd_db": round(s, 4),
        "seed_spread_range_db": round(rng, 4),
        "mean_minus_2sd_db": round(m - 2 * s, 4),
        "three_seed_mean_per_bin_db": [round(float(x), 4) for x in bin_mean],
        "seed_spread_sd_per_bin_db": [round(float(x), 4) for x in bin_sd],
        "P29a": {"prediction": f"across-seed SD <= {P29A_SD_MAX} dB",
                 "point_estimate": P29A_POINT,
                 "measured_sd_db": round(s, 4),
                 "status": "HELD" if p29a else "FAILED"},
        "P29b": {"prediction": "+1.209 is the MIDDLE of the three seed values",
                 "seed_values_db": [round(float(v), 4) for v in vals],
                 "seed1_rank_0_is_smallest": rank1,
                 "status": "HELD" if p29b else "FAILED"},
        "P29c": {"prediction": "all three seeds give Delta_H > 0 at SNR >= 5",
                 "measured_min_db": round(vmin, 4),
                 "status": "HELD" if p29c else "FAILED"},
        "P29d": {"prediction": "per-bin signs are - - - + + + on every seed",
                 "per_seed_signs": {t: "".join(r)
                                    for t, r in zip(order, sign_rows)},
                 "per_seed_held": dict(zip(order, p29d_per_seed)),
                 "status": "HELD" if p29d else "FAILED"},
        "outcome": {"fired": outcome, "because": why,
                    "consequence": CONSEQUENCE[outcome],
                    "rule": "PREREG_P29 order: b-tests first, then a, else c"},
        "note": ("CIs in per_seed are over TEST REALISATIONS. The seed spread "
                 "is the separate quantity reported as seed_spread_*. Three "
                 "seeds give two degrees of freedom."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    print(f"\nseed-1 reproduction: measured {got1:+.4f} vs published "
          f"{SEED1_PUBLISHED_GE5_DB:+.4f}, error {repro_err:.6f} -> "
          f"{'OK' if repro_ok else 'MISMATCH, verdicts below are void'}")
    print(f"\n  {'seed':<7} {'SNR>=5':>9} {'CI over test real.':>22} "
          f"{'ratio-of-sums':>14} {'pooled med':>11}")
    for t in order:
        z = seeds[t]
        print(f"  {t:<7} {z['ge5_median_db']:>+9.4f} "
              f"  [{z['ge5_ci95_over_test_realisations'][0]:+.3f},"
              f"{z['ge5_ci95_over_test_realisations'][1]:+.3f}]"
              f"{'*' if z['ge5_ci_excludes_zero'] else ' '} "
              f"{z['ge5_ratio_of_sums_db']:>+14.4f} "
              f"{z['pooled_median_db']:>+11.4f}")
    print(f"  {'mean':<7} {m:>+9.4f}   seed SD {s:.4f}, range {rng:.4f}, "
          f"mean-2SD {m - 2*s:+.4f}")
    print("\n  per-bin three-seed mean:", [f"{v:+.3f}" for v in bin_mean])
    for t, r in zip(order, sign_rows):
        print(f"    {t} signs {''.join(r)}")
    print(f"\nP29a: {out['P29a']['status']}  SD {s:.4f} (max {P29A_SD_MAX})")
    print(f"P29b: {out['P29b']['status']}  seed 1 rank {rank1} of 0..2")
    print(f"P29c: {out['P29c']['status']}  min {vmin:+.4f}")
    print(f"P29d: {out['P29d']['status']}  {out['P29d']['per_seed_signs']}")
    print(f"\nOUTCOME ({outcome}) because {why}")
    print(f"  -> {CONSEQUENCE[outcome]}")
    print("  wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

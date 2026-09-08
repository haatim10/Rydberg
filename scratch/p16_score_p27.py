"""PROMPT 16 B1 -- score P27 on the three seeds of the +1.902 dB contrast.

Reuses scratch/trackD_stage5_eval.py's evaluation path exactly, so the seed-1
number this reproduces is the published one rather than a near-miss: same
TrackDDataset worlds, same P=20, same spectral initialiser, same n_test=2000,
same by_bin contrast with the same bin edges.

Every threshold below is transcribed from reports/p16/PREREG_P27.md, committed
at b4a1dc8 before any run started.

Two quantities are reported separately and never pooled:
  * CI over TEST REALISATIONS  -- bootstrap over the 2000 paired trials, per
    seed. This is what by_bin returns.
  * SEED SPREAD                -- SD and range of the three per-seed gains.
Conflating them is the error that produced the withdrawn P19.

Run:  PYTHONPATH=. python3 scratch/p16_score_p27.py
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

OUT = Path("reports/p16/p27_score.json")
N_TEST = 2000
P = 20

# seed -> (unbalanced checkpoint, balanced checkpoint)
ARMS = {
    "seed1": ("results/track_d/stage2/B3_80k_13ep/best.pt",
              "results/track_d/stage5/C1_snr_balanced_P20/best.pt"),
    "seed2": ("results/p16/tier05/U1_seed2/best.pt",
              "results/p16/tier05/C1_seed2/best.pt"),
    "seed3": ("results/p16/tier05/U1_seed3/best.pt",
              "results/p16/tier05/C1_seed3/best.pt"),
}

# ---- transcribed from PREREG_P27.md, committed before any run -------------
P27A_SD_MAX = 0.45
P27A_POINT = 0.25
P27B_MEAN_MIN = 1.40
P27B_ANY_SEED_MIN = 1.00
P27B_TOP_BIN_MIN = 2.00
P27B_LOW_BIN_FLOOR = -0.10
SEED1_PER_BIN = [0.039, 0.044, 0.511, 1.353, 1.974, 2.628]


def load(cfg, path):
    rc = replace(cfg, model=replace(cfg.model, filter_init="random",
                                    use_transformer=True, use_hankel=False,
                                    hankel_rank=7, hankel_mode="fixed",
                                    hankel_gate="none"))
    m, _ = build_model(rc, "arm1b_full_random")
    m.load_state_dict(torch.load(path, map_location="cpu",
                                 weights_only=False)["model"])
    return m.eval()


@torch.no_grad()
def evaluate(cfg, models: dict) -> tuple[dict, np.ndarray]:
    ds = TrackDDataset("test", sysc=cfg.system,
                       datac=replace(cfg.data, n_test=N_TEST),
                       numeric=cfg.numeric, P=P, init=cfg.train.init)
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
    cfg = replace(TrackDConfig(),
                  train=replace(TrackDConfig().train, init="spectral"))
    assert cfg.train.init == "spectral"

    have = {k: v for k, v in ARMS.items()
            if Path(v[0]).exists() and Path(v[1]).exists()}
    missing = sorted(set(ARMS) - set(have))
    if missing:
        print(f"MISSING checkpoints for: {missing}")
    if len(have) < 2:
        print("need at least two seeds; aborting")
        return 1

    models = {}
    for tag, (u, c) in have.items():
        models[f"U1_{tag}"] = load(cfg, u)
        models[f"C1_{tag}"] = load(cfg, c)
    print(f"evaluating {len(models)} arms on {N_TEST} shared worlds", flush=True)
    per, snr = evaluate(cfg, models)

    # by_bin(per, a, b) reports a - b, so U1 - C1 is positive when the balanced
    # arm is better -- the same convention as the published +1.902.
    seeds = {}
    for tag in have:
        r = by_bin(per, f"U1_{tag}", f"C1_{tag}", snr)
        seeds[tag] = {
            "high_snr_ge5_db": r["high_snr_ge5"]["median_diff_db"],
            "ci95_over_test_realisations": r["high_snr_ge5"]["boot_ci95_median"],
            "n": r["high_snr_ge5"]["n"],
            "per_bin_db": [b["median_diff_db"] for b in r["bins"]],
            "per_bin_ci95_over_test_realisations":
                [b["boot_ci95_median"] for b in r["bins"]],
            "bins": [b["bin"] for b in r["bins"]],
        }

    vals = np.array([seeds[t]["high_snr_ge5_db"] for t in seeds])
    sd = float(vals.std(ddof=1)) if vals.size > 1 else None
    rng = float(vals.max() - vals.min()) if vals.size > 1 else None
    mean = float(vals.mean())
    per_bin = np.array([seeds[t]["per_bin_db"] for t in seeds])
    bin_mean = per_bin.mean(axis=0)

    p27a_held = sd is not None and sd <= P27A_SD_MAX
    monotone = bool(np.all(np.diff(bin_mean) >= 0))
    p27b_fail = []
    if mean < P27B_MEAN_MIN:
        p27b_fail.append(f"three-seed mean {mean:.3f} < {P27B_MEAN_MIN}")
    if float(vals.min()) < P27B_ANY_SEED_MIN:
        p27b_fail.append(f"min seed {vals.min():.3f} < {P27B_ANY_SEED_MIN}")
    if not monotone:
        p27b_fail.append("per-bin means not monotone non-decreasing")
    if bin_mean[-1] < P27B_TOP_BIN_MIN:
        p27b_fail.append(f"top bin {bin_mean[-1]:.3f} < {P27B_TOP_BIN_MIN}")
    if float(bin_mean[:2].min()) < P27B_LOW_BIN_FLOOR:
        p27b_fail.append(f"low bin {bin_mean[:2].min():.3f} < {P27B_LOW_BIN_FLOOR}")

    separated = sd is not None and (mean - 2 * sd) > 0

    out = {
        "prereg": "reports/p16/PREREG_P27.md @ b4a1dc8",
        "contrast": "U1 (uniform loss) - C1 (balanced loss); positive = balanced better",
        "n_seeds": len(seeds), "n_test": N_TEST, "P": P,
        "per_seed": seeds,
        "bins": seeds[list(seeds)[0]]["bins"],
        "three_seed_mean_high_snr_db": round(mean, 4),
        "seed_spread_sd_db": None if sd is None else round(sd, 4),
        "seed_spread_range_db": None if rng is None else round(rng, 4),
        "three_seed_mean_per_bin_db": [round(v, 4) for v in bin_mean],
        "seed1_per_bin_published": SEED1_PER_BIN,
        "P27a": {"threshold_sd_max": P27A_SD_MAX, "point_estimate": P27A_POINT,
                 "measured_sd_db": None if sd is None else round(sd, 4),
                 "status": "HELD" if p27a_held else "FAILED"},
        "P27b": {"status": "HELD" if not p27b_fail else "FAILED",
                 "failures": p27b_fail, "monotone": monotone},
        "separation": {
            "rule": "mean - 2*SD > 0 across seeds",
            "value": None if sd is None else round(mean - 2 * sd, 4),
            "separated": separated,
            "consequence": ("headline may be stated with the seed spread "
                            "alongside it" if separated else
                            "HEADLINE IS SINGLE-SEED and Paper 2 must say so "
                            "in those words, in the abstract and at first "
                            "quotation")},
        "note": ("CIs in per_seed are over TEST REALISATIONS. The seed spread "
                 "is the separate quantity reported as seed_spread_*."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    print(f"\nP27a: {out['P27a']['status']}  SD {sd} (max {P27A_SD_MAX})")
    print(f"P27b: {out['P27b']['status']}  {p27b_fail or ''}")
    print(f"separation: mean-2SD = {out['separation']['value']} -> "
          f"{'SEPARATED' if separated else 'NOT SEPARATED'}")
    print(f"\n  {'seed':<7} {'SNR>=5':>9}  CI over test realisations")
    for t in seeds:
        s = seeds[t]
        print(f"  {t:<7} {s['high_snr_ge5_db']:>+9.4f}  "
              f"[{s['ci95_over_test_realisations'][0]:+.3f}, "
              f"{s['ci95_over_test_realisations'][1]:+.3f}]")
    print(f"  {'mean':<7} {mean:>+9.4f}   seed SD {sd}, range {rng}")
    print("\n  per-bin three-seed mean:",
          [f"{v:+.3f}" for v in bin_mean])
    print("  wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

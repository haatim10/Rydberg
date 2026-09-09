"""PROMPT 17 Part C -- score P28 on the three seeds of the +0.078 dB contrast.

Written and committed BEFORE batch C1 finishes, so no threshold in it can have
been chosen after seeing a number. Every threshold is transcribed from
reports/p16/PREREG_P28.md, committed standing alone at e95c7ec before C1
started.

The evaluation path mirrors trackD_urformer/stage4.py's Part C exactly, which
is what makes the seed-1 reproduction a real check rather than a near-miss:

  * test SNR range (5, 20) -- the focused arms are trained AND tested there,
    so this contrast has THREE bins, not the six of the mixed-SNR designs;
  * n_test = 2000, P = 20 (the config default the stage-4 run used);
  * spectral initialiser;
  * by_bin(per, "U1", "H1", snr), which reports a - b, so a positive number
    means the Hankel prior is better -- the sign of the published +0.0778.

Two quantities are reported separately and never pooled:
  * CI over TEST REALISATIONS -- bootstrap over the 2000 paired trials, per
    seed, which is what by_bin returns;
  * SEED SPREAD -- SD and range of the three per-seed gains.
Conflating them is the error that produced the withdrawn P19.

Run:  PYTHONPATH=. python3 scratch/p17_score_c1.py
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
from trackD_urformer.stage4 import HIGH_SNR, by_bin

OUT = Path("reports/p17/p28_score.json")
N_TEST = 2000

# seed -> (no-prior checkpoint, Hankel-prior checkpoint). Seed 1 is the
# published stage-4 pair and is NOT retrained by C1; both arms read
# init='spectral', seed 20260827, epoch 9 from their own checkpoints.
ARMS = {
    "seed1": ("results/track_d/stage4/C_U1_snr5_20/best.pt",
              "results/track_d/stage4/C_H1_snr5_20/best.pt"),
    "seed2": ("results/p17/tier1/C1_U1_seed2/best.pt",
              "results/p17/tier1/C1_H1_seed2/best.pt"),
    "seed3": ("results/p17/tier1/C1_U1_seed3/best.pt",
              "results/p17/tier1/C1_H1_seed3/best.pt"),
}

# ---- transcribed from PREREG_P28.md, committed e95c7ec before C1 -----------
P28A_SD_MAX = 0.35          # P28a falsifier
P28A_POINT = 0.18           # pre-registered point estimate
GATE_RANGE_MAX = 0.50       # gate on C2-C4: three-seed range above this stops

# The published seed-1 anchor, read from reports/trackD_stage4_results.json
# ["part_c"]["contrasts"]["C_H1_vs_C_U1"]. Transcribed here so that a scorer
# that silently evaluates the wrong worlds fails visibly on seed 1.
SEED1_PUBLISHED_DB = 0.0778
SEED1_PUBLISHED_CI = [0.0497, 0.1119]
SEED1_PUBLISHED_PER_BIN = [-0.0757, 0.1297, 0.2322]
REPRODUCTION_TOL_DB = 0.001


def load(cfg, path: str, hankel: bool):
    rc = replace(cfg, model=replace(cfg.model, filter_init="random",
                                    use_transformer=True, use_hankel=hankel,
                                    hankel_rank=7, hankel_mode="fixed",
                                    hankel_gate="none"))
    m, _ = build_model(rc, "arm1b_full_random")
    m.load_state_dict(torch.load(path, map_location="cpu",
                                 weights_only=False)["model"])
    return m.eval()


def checkpoint_provenance(path: str) -> dict:
    """What the checkpoint says about itself. Read rather than assumed: the
    PROMPT 15 initialiser defect was invisible to every other check."""
    b = torch.load(path, map_location="cpu", weights_only=False)
    cfgd = b.get("config") or {}
    tr = cfgd.get("train", {}) if isinstance(cfgd, dict) else {}
    md = cfgd.get("model", {}) if isinstance(cfgd, dict) else {}
    return {"epoch": b.get("epoch"), "init": tr.get("init"),
            "train_seed": tr.get("seed"), "use_hankel": md.get("use_hankel")}


@torch.no_grad()
def evaluate(cfg, models: dict) -> tuple[dict, np.ndarray]:
    """One pass, all arms, identical worlds. Mirrors stage4.evaluate."""
    ecfg = replace(cfg, data=replace(cfg.data, snr_range_db=HIGH_SNR,
                                     n_test=N_TEST))
    ds = TrackDDataset("test", sysc=ecfg.system, datac=ecfg.data,
                       numeric=ecfg.numeric, init=ecfg.train.init)
    cd, rd = ecfg.numeric.complex_dtype, ecfg.numeric.real_dtype
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
    assert cfg.system.P == 20, "stage 4 Part C ran at P=20"

    have = {k: v for k, v in ARMS.items()
            if Path(v[0]).exists() and Path(v[1]).exists()}
    missing = sorted(set(ARMS) - set(have))
    if missing:
        print(f"MISSING checkpoints for: {missing}")
    if "seed1" not in have:
        print("seed 1 is the reproduction check and must be present; aborting")
        return 1
    if len(have) < 3:
        print(f"P28 needs three seeds, have {sorted(have)}; aborting")
        return 1

    prov = {}
    models = {}
    for tag, (u, h) in have.items():
        prov[f"U1_{tag}"] = checkpoint_provenance(u)
        prov[f"H1_{tag}"] = checkpoint_provenance(h)
        models[f"U1_{tag}"] = load(cfg, u, hankel=False)
        models[f"H1_{tag}"] = load(cfg, h, hankel=True)

    bad = {k: v for k, v in prov.items() if v["init"] != "spectral"}
    if bad:
        print(f"REFUSING TO SCORE -- arms not trained with the spectral "
              f"initialiser: {bad}")
        return 1

    print(f"evaluating {len(models)} arms on {N_TEST} shared worlds, "
          f"SNR {HIGH_SNR}", flush=True)
    per, snr = evaluate(cfg, models)

    seeds = {}
    for tag in have:
        r = by_bin(per, f"U1_{tag}", f"H1_{tag}", snr)
        seeds[tag] = {
            "high_snr_ge5_db": r["high_snr_ge5"]["median_diff_db"],
            "ci95_over_test_realisations": r["high_snr_ge5"]["boot_ci95_median"],
            "ci_excludes_zero": r["high_snr_ge5"]["ci_excludes_zero"],
            "n": r["high_snr_ge5"]["n"],
            "ratio_of_sums_db_SECONDARY":
                r["high_snr_ge5"].get("ratio_of_sums_diff_db_SECONDARY"),
            "per_bin_db": [b["median_diff_db"] for b in r["bins"]],
            "per_bin_ci95_over_test_realisations":
                [b["boot_ci95_median"] for b in r["bins"]],
            "per_bin_ci_excludes_zero":
                [b["ci_excludes_zero"] for b in r["bins"]],
            "bins": [b["bin"] for b in r["bins"]],
        }

    # --- reproduction check on seed 1 --------------------------------------
    got1 = seeds["seed1"]["high_snr_ge5_db"]
    repro_err = abs(got1 - SEED1_PUBLISHED_DB)
    repro_ok = repro_err <= REPRODUCTION_TOL_DB

    order = ["seed1", "seed2", "seed3"]
    vals = np.array([seeds[t]["high_snr_ge5_db"] for t in order])
    mean, sd, rng = (float(vals.mean()), float(vals.std(ddof=1)),
                     float(vals.max() - vals.min()))
    per_bin = np.array([seeds[t]["per_bin_db"] for t in order])
    bin_mean = per_bin.mean(axis=0)
    bin_sd = per_bin.std(axis=0, ddof=1)

    # --- P28a: across-seed SD ---------------------------------------------
    p28a_held = sd <= P28A_SD_MAX

    # --- P28b: is +0.078 the MIDDLE of the three, not an extreme? ----------
    # The literal "does it lie in [min, max]" reading is vacuous by
    # construction and PREREG_P28 says so; this is the non-vacuous version.
    rank1 = int(np.argsort(np.argsort(vals))[0])   # 0 = smallest, 2 = largest
    p28b_held = rank1 == 1

    # --- P28c: sign stability ---------------------------------------------
    p28c_held = bool(np.all(vals > 0))

    separated = (mean - 2 * sd) > 0
    gate_stops = (not p28a_held) or rng > GATE_RANGE_MAX

    out = {
        "prereg": "reports/p16/PREREG_P28.md @ e95c7ec",
        "contrast": ("U1 (no prior) - H1 (Hankel prior, rank 7); "
                     "positive = the structural prior is better"),
        "design": "focused training, SNR in [5,20] for train and test",
        "n_seeds": len(seeds), "n_test": N_TEST, "P": int(cfg.system.P),
        "test_snr_range": list(HIGH_SNR),
        "checkpoint_provenance": prov,
        "per_seed": seeds,
        "bins": seeds["seed1"]["bins"],
        "seed1_reproduction": {
            "published_db": SEED1_PUBLISHED_DB,
            "published_ci95": SEED1_PUBLISHED_CI,
            "published_per_bin_db": SEED1_PUBLISHED_PER_BIN,
            "measured_db": round(got1, 4),
            "abs_error_db": round(repro_err, 6),
            "tolerance_db": REPRODUCTION_TOL_DB,
            "reproduced": bool(repro_ok),
            "meaning": ("if false, the scorer is reading different worlds or a "
                        "different path and NO verdict below may be trusted"),
        },
        "three_seed_mean_high_snr_db": round(mean, 4),
        "seed_spread_sd_db": round(sd, 4),
        "seed_spread_range_db": round(rng, 4),
        "three_seed_mean_per_bin_db": [round(float(v), 4) for v in bin_mean],
        "seed_spread_sd_per_bin_db": [round(float(v), 4) for v in bin_sd],
        "P28a": {"prediction": f"across-seed SD <= {P28A_SD_MAX} dB",
                 "point_estimate": P28A_POINT,
                 "measured_sd_db": round(sd, 4),
                 "status": "HELD" if p28a_held else "FAILED"},
        "P28b": {"prediction": "+0.078 is the MIDDLE of the three seed values",
                 "seed_values_db": [round(float(v), 4) for v in vals],
                 "seed1_rank_0_is_smallest": rank1,
                 "prior_probability_under_exchangeability": 1 / 3,
                 "status": "HELD" if p28b_held else "FAILED"},
        "P28c": {"prediction": "all three seeds give Delta_H > 0 at SNR >= 5",
                 "measured_min_db": round(float(vals.min()), 4),
                 "status": "HELD" if p28c_held else "FAILED"},
        "separation": {
            "rule": "mean - 2*SD > 0 across seeds",
            "value": round(mean - 2 * sd, 4),
            "separated": bool(separated),
            "consequence": (
                "Paper 2 may state that the structural prior retains a small "
                "but non-zero advantage under focused training, quoting the "
                "mean with the seed spread alongside"
                if separated else
                "Paper 2 must say the focused-training advantage is WITHIN "
                "SEED VARIATION and cannot be distinguished from zero at this "
                "sample size -- NOT that it is zero, which three seeds cannot "
                "establish either")},
        "gate_on_C2_C4": {
            "rule": f"stop if P28a fails or three-seed range > {GATE_RANGE_MAX}",
            "range_db": round(rng, 4),
            "stop": bool(gate_stops),
            "consequence": ("batches C2-C4 do not launch this turn"
                            if gate_stops else
                            "C2 may launch once P29 is committed standing alone")},
        "note": ("CIs in per_seed are over TEST REALISATIONS. The seed spread "
                 "is the separate quantity reported as seed_spread_*. Three "
                 "seeds give two degrees of freedom, so the SD itself carries "
                 "large sampling uncertainty and any statement made from it "
                 "must say it was evaluated on three seeds."),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    print(f"\nseed-1 reproduction: measured {got1:+.4f} vs published "
          f"{SEED1_PUBLISHED_DB:+.4f}, error {repro_err:.6f} -> "
          f"{'OK' if repro_ok else 'MISMATCH, verdicts below are void'}")
    print(f"\n  {'seed':<7} {'SNR>=5':>9}  CI over test realisations")
    for t in order:
        s = seeds[t]
        print(f"  {t:<7} {s['high_snr_ge5_db']:>+9.4f}  "
              f"[{s['ci95_over_test_realisations'][0]:+.3f}, "
              f"{s['ci95_over_test_realisations'][1]:+.3f}]"
              f"{'  *' if s['ci_excludes_zero'] else ''}")
    print(f"  {'mean':<7} {mean:>+9.4f}   seed SD {sd:.4f}, range {rng:.4f}")
    print("\n  per-bin three-seed mean:", [f"{v:+.3f}" for v in bin_mean],
          " (bins", out["bins"], ")")
    print(f"\nP28a: {out['P28a']['status']}  SD {sd:.4f} (max {P28A_SD_MAX}, "
          f"point {P28A_POINT})")
    print(f"P28b: {out['P28b']['status']}  seed 1 rank {rank1} of 0..2")
    print(f"P28c: {out['P28c']['status']}  min {vals.min():+.4f}")
    print(f"separation: mean-2SD = {mean - 2*sd:+.4f} -> "
          f"{'SEPARATED' if separated else 'NOT SEPARATED'}")
    print(f"gate: {'STOP -- C2-C4 do not launch' if gate_stops else 'PROCEED'}")
    print("  wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

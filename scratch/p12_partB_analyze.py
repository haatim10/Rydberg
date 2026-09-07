"""PROMPT 12 -- score Part B against the pre-registrations P16, P17, P18.

Reads results/p12/{B1,B2,B3,B4,B5}.json and prints the verdicts. Scores each
pre-registration held/failed against the thresholds written down in
reports/p12/PREREG_P16_P18.md BEFORE the runs.

Run:  PYTHONPATH=. python3 scratch/p12_partB_analyze.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SRC = Path("results/p12")
OUT = Path("reports/p12")

# Committed crossings, from reports/trackD_partB9_analysis.json B1_collapse.
CROSSINGS = {16: 0.5876, 32: 0.518, 64: 0.5440}
# rho achieved by the B2 cells, measured (scratch/p12_partA_rho.rho_for).
B2_RHO = {"N16_L9": 0.608, "N16_L14": 0.735, "N64_L38": 0.596, "N64_L48": 0.669}


def load(name):
    f = SRC / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else None


def score_p16(d):
    if d is None:
        return {"status": "NOT RUN"}
    pts = d["points"]
    med = np.array([p["delta_median_db"] for p in pts])
    mean = float(med.mean())
    in_band = all(2.0 <= m <= 3.2 for m in med)
    mean_ok = 2.0 <= mean <= 3.1
    ci_ok = all(not (p["boot_ci95_median"][1] < 1.5
                     or p["boot_ci95_median"][0] > 3.5) for p in pts)
    held = mean_ok and ci_ok
    return {
        "status": "HELD" if held else "FAILED",
        "per_bin_median_db": med.tolist(),
        "six_point_mean_db": mean,
        "predicted_mean_band": [2.2, 2.9], "predicted_point": 2.45,
        "falsifier_mean_band": [2.0, 3.1],
        "all_bins_in_2p0_3p2": bool(in_band),
        "mean_inside_falsifier_band": bool(mean_ok),
        "no_bin_CI_wholly_outside_1p5_3p5": bool(ci_ok),
        "ratio_of_sums_per_point": [p["delta_ratio_of_sums_db"] for p in pts],
        "mean_L_hat": [p["mean_L_hat"] for p in pts],
    }


def score_p17(d):
    if d is None:
        return {"status": "NOT RUN"}
    pts = sorted(d["points"], key=lambda p: p["admissible_ranks"])
    hi = [p for p in pts if p["admissible_ranks"] >= 8]
    lo = [p for p in pts if p["admissible_ranks"] < 8]
    hv = [p["delta_median_db"] for p in hi]
    span_hi = float(max(hv) - min(hv)) if hv else None
    # monotone in the admissible-rank count over the >=8 group?
    order = [p["delta_median_db"] for p in sorted(hi, key=lambda p: p["admissible_ranks"])]
    monotone = bool(all(a <= b for a, b in zip(order, order[1:]))
                    or all(a >= b for a, b in zip(order, order[1:])))
    quantisation_supported = bool(monotone and span_hi is not None and span_hi > 0.5)
    # my prediction: span < 0.30 and NOT monotone over the >=8 group
    pred_held = bool(span_hi is not None and span_hi < 0.30 and not monotone)
    lo_drop = None
    if lo and hv:
        lo_drop = float(np.mean(hv) - min(p["delta_median_db"] for p in lo))
    return {
        "status": "HELD" if pred_held else "FAILED",
        "points": [{"pencil": p["pencil"], "adm": p["admissible_ranks"],
                    "delta_db": p["delta_median_db"],
                    "ci": p["boot_ci95_median"],
                    "mean_L_hat": p["mean_L_hat"]} for p in pts],
        "span_over_adm_ge_8_db": span_hi,
        "monotone_over_adm_ge_8": monotone,
        "predicted_span_lt_0p30_and_not_monotone": pred_held,
        "QUANTISATION_ACCOUNT_SUPPORTED": quantisation_supported,
        "drop_at_adm_5_vs_mean_of_ge8_db": lo_drop,
        "predicted_drop_at_adm5_band": [0.3, 1.5],
        "confound": "at fixed N the pencil sets admissible-rank COUNT and "
                    "maximum representable RANK together; this experiment "
                    "cannot separate them (stated in the pre-registration)",
    }


def score_p18(d):
    if d is None:
        return {"status": "NOT RUN"}
    pts = d["points"]
    vals = [p["interleaved_minus_posthoc_db"] for p in pts]
    neg_sig = [p for p in pts if p["boot_ci95_median"][1] < 0]
    pooled = float(np.mean(vals))
    growth = float(vals[-1] - vals[0])
    held = (len(neg_sig) == 0) and (0.3 <= pooled <= 1.5)
    return {
        "status": "HELD" if held else "FAILED",
        "per_bin": [{"snr_db": p["snr_db"],
                     "interleaved_minus_posthoc_db":
                         p["interleaved_minus_posthoc_db"],
                     "ci": p["boot_ci95_median"],
                     "sig": p["ci_excludes_zero"]} for p in pts],
        "mean_over_six_points_db": pooled,
        "predicted_pooled_band": [0.3, 1.5], "predicted_point": 0.8,
        "bins_where_posthoc_significantly_better": [p["snr_db"] for p in neg_sig],
        "ONE_SIGNED": len(neg_sig) == 0,
        "top_minus_bottom_bin_db": growth,
        "predicted_growth_band": [0.2, 1.5],
    }


def score_b2(d):
    if d is None:
        return {"status": "NOT RUN"}
    rows = []
    for p in d["points"]:
        rho = B2_RHO.get(p["tag"])
        rows.append({"tag": p["tag"], "N": p["N"], "L": p["L"],
                     "rho_measured": rho, "n": p["n"],
                     "delta_db": p["delta_median_db"],
                     "ci": p["boot_ci95_median"],
                     "negative_and_significant": p["negative_and_significant"]})
    brack = {}
    for N in (16, 64):
        cells = [r for r in rows if r["N"] == N]
        brack[N] = {
            "crossing": CROSSINGS[N],
            "bracketed": any(r["negative_and_significant"] for r in cells),
            "cells": [(r["tag"], r["rho_measured"], r["delta_db"],
                       r["negative_and_significant"]) for r in cells],
        }
    brack[32] = {"crossing": CROSSINGS[32], "bracketed": True,
                 "cells": "already bracketed before this turn"}
    return {"rows": rows, "brackets": brack,
            "ALL_THREE_BRACKETED": all(v["bracketed"] for v in brack.values())}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {
        "P16_bridging": score_p16(load("B1")),
        "P17_pencil": score_p17(load("B3")),
        "P18_posthoc_vs_interleaved": score_p18(load("B5")),
        "B2_bracketing": score_b2(load("B2")),
        "B4_coarse_to_fine": load("B4"),
    }
    print("=" * 72)
    for k, v in res.items():
        print(f"\n### {k}")
        if v is None:
            print("  NOT RUN")
            continue
        print(json.dumps(v, indent=1)[:2600])
    (OUT / "partB_scores.json").write_text(json.dumps(res, indent=1) + "\n",
                                           encoding="utf-8")
    print(f"\nwrote {OUT/'partB_scores.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

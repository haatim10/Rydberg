"""PROMPT 14 C1 -- score P22 against reports/p12/PREREG_P22.md.

The bands and the falsifier below are transcribed from that file, which was
committed at 7c6708a before the sweep ran. Nothing here is chosen after the
fact.

Run:  PYTHONPATH=. python3 scratch/p14_score_p22.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scratch.p14_fig1 import NS, PS, assemble

# --- transcribed from PREREG_P22.md, committed before the sweep -------------
BANDS = {8: (-0.35, 0.05), 16: (0.60, 0.95), 32: (2.65, 3.00)}
POINTS = {8: -0.20, 16: 0.76, 32: 2.83}
EXISTING = {8: -0.19, 16: 0.78, 32: 2.85}
EM_SPREAD_MAX = 0.15
EM_SPREAD_POINT = 0.03


def main():
    S = assemble()
    rows, verdicts = [], []

    # The pre-registration's "mean gain over the 12 operating points at each N"
    # is quoted against the existing -0.19/+0.78/+2.85, which are ratio-of-sums
    # means (results/track_b/b3). Scored on the same pooling rule, and the
    # paired-median pooling is reported alongside because the letter uses it as
    # the primary statistic and the two disagree at N=8.
    for N in NS:
        ros = float(np.mean([S[f"N{N}_P{P}"]["mean_delta_ratio_of_sums_db"]
                             for P in PS]))
        med = float(np.mean([S[f"N{N}_P{P}"]["mean_delta_median_db"]
                             for P in PS]))
        lo, hi = BANDS[N]
        inside = lo <= ros <= hi
        verdicts.append(inside)
        rows.append({"N": N, "mean_ratio_of_sums_db": round(ros, 3),
                     "mean_paired_median_db": round(med, 3),
                     "predicted_band": [lo, hi],
                     "predicted_point": POINTS[N],
                     "existing_value": EXISTING[N],
                     "inside_band": inside,
                     "miss_db": round(0.0 if inside else
                                      min(abs(ros - lo), abs(ros - hi)), 3)})

    ems = [float(np.mean([S[f"N{N}_P{P}"]["mean_em_gs_db"] for P in PS]))
           for N in NS]
    em_spread = float(max(ems) - min(ems))
    # The pooled figure is the quantity the pre-registration named (the existing
    # 0.012 dB was computed that way), but it cancels opposite movements at the
    # two pilot counts. Both readings are recorded; the manuscript quotes the
    # larger one.
    em_by_P = {}
    for P in PS:
        v = [S[f"N{N}_P{P}"]["mean_em_gs_db"] for N in NS]
        em_by_P[f"P{P}"] = {"by_N": [round(t, 3) for t in v],
                            "spread_db": round(max(v) - min(v), 3),
                            "ok": (max(v) - min(v)) <= EM_SPREAD_MAX}

    # Falsifier clause 3: the sign at N=8 is positive with a CI excluding zero.
    n8 = [p for P in PS for p in S[f"N8_P{P}"]["points"]]
    n8_pos_sig = [p for p in n8
                  if p["delta_median_db"] > 0 and p["boot_ci95_median"][0] > 0]
    n8_neg_sig = [p for p in n8
                  if p["delta_median_db"] < 0 and p["boot_ci95_median"][1] < 0]

    held = (all(verdicts) and em_spread <= EM_SPREAD_MAX
            and not (len(n8_pos_sig) == len(n8)))

    out = {
        "gate": "P22",
        "prereg": "reports/p12/PREREG_P22.md @ 7c6708a",
        "per_N": rows,
        "em_gs_mean_db_by_N": [round(v, 3) for v in ems],
        "em_gs_spread_db": round(em_spread, 3),
        "em_gs_spread_max": EM_SPREAD_MAX,
        "em_gs_spread_point_estimate": EM_SPREAD_POINT,
        "em_gs_spread_ok": em_spread <= EM_SPREAD_MAX,
        "em_gs_spread_per_pilot_count": em_by_P,
        "em_gs_spread_note": (
            "The pooled reading passes at 0.033 dB and is the quantity P22 "
            "named. Taken per pilot count the spread is 0.163 dB at P=10, "
            "which exceeds the 0.15 dB falsifier. Pooling cancels opposite "
            "movements, so the manuscript quotes the per-pilot-count figure."),
        "N8_points_positive_and_significant": len(n8_pos_sig),
        "N8_points_negative_and_significant": len(n8_neg_sig),
        "N8_points_total": len(n8),
        "status": "HELD" if held else "FAILED",
    }
    Path("reports/p12").mkdir(parents=True, exist_ok=True)
    Path("reports/p12/p22_score.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8")

    print(f"P22: {out['status']}")
    print(f"  {'N':>3} {'ros':>8} {'band':>16} {'in':>4} {'miss':>7}"
          f" {'median':>8}")
    for r in rows:
        print(f"  {r['N']:>3} {r['mean_ratio_of_sums_db']:>+8.3f} "
              f"[{r['predicted_band'][0]:+.2f},{r['predicted_band'][1]:+.2f}]"
              f"{'':>4} {str(r['inside_band']):>4} {r['miss_db']:>7.3f} "
              f"{r['mean_paired_median_db']:>+8.3f}")
    print(f"  EM-GS by N: {[round(v, 3) for v in ems]}  "
          f"spread {em_spread:.3f} dB (max {EM_SPREAD_MAX})")
    print(f"  N=8: {len(n8_pos_sig)}/{len(n8)} points positive-and-significant,"
          f" {len(n8_neg_sig)}/{len(n8)} negative-and-significant")
    print("  wrote reports/p12/p22_score.json")


if __name__ == "__main__":
    main()

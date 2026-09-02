"""PROMPT 9 Part B analysis: collapse tests, oracle gaps, cost table.

Reads the per-cell JSONs written by ``trackD_partB9_sweeps.py`` and produces

  B1  the array-size collapse, indexed by ``r_eff/cap`` (tests P12)
  B2  K-invariance at fixed pilot adequacy P/2K (tests P11)
  B3  ``Delta_HS(SNR)`` under ADAPTIVE rank at the default configuration
  B4  the unstructured-LS oracle gap for every cell and every method
  B6  Xiao's Saleh-Valenzuela channel against the A2 prediction of +1.30 dB

Primary statistic throughout: the paired per-trial median, per SNR bin.
Pooled values are carried but explicitly labelled sampling-design-dependent,
because the SNR draw is uniform by construction and a pooled median therefore
reports the design as much as the estimator.

Run:  PYTHONPATH=. python3 scratch/trackD_partB9_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trackD_urformer.stage3 import paired
from trackD_urformer.stage4 import BINS, by_bin

SRC = Path("results/track_d/partB9")
OUT_JSON = Path("reports/trackD_partB9_analysis.json")

# From reports/trackD_normalization.md A2 (Track B Experiment C re-indexing;
# median Roy-Vetterli effective rank of the noiseless channel columns).
# Cell tag -> (r_eff, cap, r_eff/cap).
R_EFF = {
    "B1_N16_L2":  (1.88, 8),   "B1_N16_L4":  (3.10, 8),   "B1_N16_L7":  (4.35, 8),
    "B1_N64_L8":  (6.33, 32),  "B1_N64_L14": (9.89, 32),  "B1_N64_L29": (16.24, 32),
}
# The N=32 reference row, measured in Track B Experiment C, quoted in A2.
N32_REF = [(0.212, 3.556), (0.285, 1.792), (0.356, 1.038), (0.408, 0.577),
           (0.460, 0.266), (0.507, 0.046), (0.546, -0.117)]
N32_CROSSING = 0.518
A2_PREDICTION = {"B6_xiao_clustered": 1.30, "B6_xiao_literal": -0.12}

EM, HS, OR = "EM-GS", "HS-EM-GS-auto", "oracle"


def per_trial(cell: dict) -> dict:
    return {k: (np.asarray(cell["num"][k]) / np.asarray(cell["den"][k])).tolist()
            for k in cell["num"]}


def se_db(per: dict, a: str, b: str) -> float:
    """Paired standard error of the mean per-trial dB difference."""
    d = 10 * np.log10(np.asarray(per[a])) - 10 * np.log10(np.asarray(per[b]))
    return float(d.std(ddof=1) / np.sqrt(d.size))


def summarize(cell: dict) -> dict:
    per = per_trial(cell)
    snr = np.asarray(cell["snr_db"])
    out = {k: cell[k] for k in ("tag", "N", "K", "P", "L", "channel", "n",
                                "seconds", "cap")}
    out["mean_L_hat"] = float(np.mean(cell["L_hat"]))
    out["median_L_hat"] = float(np.median(cell["L_hat"]))
    out["median_db"] = {k: float(10 * np.log10(np.median(np.asarray(v))))
                        for k, v in per.items()}
    out["delta_hs"] = by_bin(per, EM, HS, snr)          # EM-GS - HS: >0 = HS wins
    out["paired_se_db"] = se_db(per, EM, HS)
    out["vs_oracle"] = {k: by_bin(per, k, OR, snr) for k in (EM, HS)}
    return out


def crossing(points: list[tuple[float, float]]) -> float | None:
    """Zero crossing of Delta vs r_eff/cap, linear in the bracketing pair.

    Extrapolates from the last two points when every cell is still positive,
    which is flagged by the caller rather than hidden.
    """
    p = sorted(points)
    for (x0, y0), (x1, y1) in zip(p, p[1:]):
        if (y0 > 0) != (y1 > 0):
            return float(x0 + (x1 - x0) * y0 / (y0 - y1))
    (x0, y0), (x1, y1) = p[-2], p[-1]
    if y0 == y1:
        return None
    return float(x0 + (x1 - x0) * y0 / (y0 - y1))


def band(sub: dict, lo: float, hi: float) -> float | None:
    """Median Delta over an SNR half, or None when the cell lacks that half."""
    key = "high_snr_ge5" if lo >= 5 else "low_snr_lt5"
    r = sub.get(key)
    return None if r is None else r["median_diff_db"]


def main() -> int:
    cells = {}
    for f in sorted(SRC.glob("*.json")):
        c = json.loads(f.read_text())
        cells[c["tag"]] = summarize(c)
    if not cells:
        print(f"no cells in {SRC}")
        return 1
    res = {"cells": cells}

    def hdr(t):
        print(f"\n{t}\n" + "-" * len(t))

    # ---------------- B3 : Delta_HS(SNR), adaptive rank, default cfg --------
    if "B3_default" in cells:
        c = cells["B3_default"]
        hdr(f"B3  adaptive-rank Delta_HS(SNR), default cfg  "
            f"(n={c['n']}, mean L_hat {c['mean_L_hat']:.2f})")
        print("  bin        n   Delta_HS      CI95            EM-GS   HS    oracle")
        for r, ro_e, ro_h in zip(c["delta_hs"]["bins"],
                                 c["vs_oracle"][EM]["bins"],
                                 c["vs_oracle"][HS]["bins"]):
            print(f"  [{r['bin'][0]:+3d},{r['bin'][1]:+3d}) {r['n']:4d} "
                  f"{r['median_diff_db']:+7.3f}  "
                  f"[{r['boot_ci95_median'][0]:+.3f},{r['boot_ci95_median'][1]:+.3f}]"
                  f"   gap-to-oracle {ro_e['median_diff_db']:+6.2f} "
                  f"{ro_h['median_diff_db']:+6.2f}")

    # ---------------- B2 : K-invariance at fixed P/2K ----------------------
    b2 = {t: c for t, c in cells.items() if t.startswith("B2_")}
    if b2:
        hdr("B2  K-invariance at fixed pilot adequacy P/2K = 3.33   (tests P11)")
        print("  K   P    n    Delta_HS   CI95              SNR>=5    SNR<5   "
              "median NMSE (EM-GS)")
        rows = []
        for t in sorted(b2, key=lambda s: cells[s]["K"]):
            c = b2[t]
            d = c["delta_hs"]
            p = d["pooled_SAMPLING_DESIGN_DEPENDENT"]
            rows.append((c["K"], p["median_diff_db"], p["boot_ci95_median"]))
            print(f"  {c['K']}  {c['P']:3d} {c['n']:4d}  {p['median_diff_db']:+7.3f}  "
                  f"[{p['boot_ci95_median'][0]:+.3f},{p['boot_ci95_median'][1]:+.3f}]"
                  f"   {band(d,5,20) if band(d,5,20) is None else f'{band(d,5,20):+7.3f}'}"
                  f"  {band(d,-10,5) if band(d,-10,5) is None else f'{band(d,-10,5):+7.3f}'}"
                  f"   {c['median_db'][EM]:+7.2f}")
        vals = [v for _, v, _ in rows]
        spread = max(vals) - min(vals)
        mono = all(b > a for a, b in zip(vals, vals[1:])) or \
               all(b < a for a, b in zip(vals, vals[1:]))
        adj = max(abs(b - a) for a, b in zip(vals, vals[1:])) if len(vals) > 1 else 0.0
        res["B2_K_invariance"] = {
            "delta_by_K": {str(k): v for k, v, _ in rows},
            "spread_db": float(spread), "max_adjacent_gap_db": float(adj),
            "monotone": bool(mono),
            "P11_prediction": "spread < +/-0.15 dB, no monotone trend",
            "P11_falsifier": "monotone trend > 0.3 dB, or adjacent pair > 0.3 dB "
                             "with CIs excluding each other",
            "P11_holds": bool(spread <= 0.30 and not (mono and spread > 0.30)),
            "P11_within_stated_tolerance": bool(spread <= 0.30)}
        print(f"  spread {spread:.3f} dB across K; monotone={mono}; "
              f"max adjacent gap {adj:.3f} dB")

    # ---------------- B1 : array-size collapse in r_eff/cap ----------------
    b1 = {t: c for t, c in cells.items() if t.startswith("B1_")}
    if b1:
        hdr("B1  array-size collapse indexed by r_eff/cap   (tests P12)")
        print("   N  cap   L  r_eff  r_eff/cap    n   Delta_HS   CI95")
        by_n: dict[int, list[tuple[float, float]]] = {}
        for t in sorted(b1, key=lambda s: (cells[s]["N"], cells[s]["L"])):
            c = b1[t]
            reff, cap = R_EFF[t][0], R_EFF[t][1]
            x = reff / cap
            p = c["delta_hs"]["pooled_SAMPLING_DESIGN_DEPENDENT"]
            by_n.setdefault(c["N"], []).append((x, p["median_diff_db"]))
            c["r_eff"], c["r_eff_over_cap"] = reff, x
            print(f"  {c['N']:3d} {cap:4d} {c['L']:3d} {reff:6.2f}   {x:.3f}   "
                  f"{c['n']:4d}  {p['median_diff_db']:+7.3f}  "
                  f"[{p['boot_ci95_median'][0]:+.3f},{p['boot_ci95_median'][1]:+.3f}]")
        # N=32 reference interpolated at each measured abscissa.
        ref_x = np.array([x for x, _ in N32_REF])
        ref_y = np.array([y for _, y in N32_REF])
        dev = {}
        for N, pts in by_n.items():
            for x, y in pts:
                if ref_x.min() <= x <= ref_x.max():
                    dev[f"N{N}_x{x:.3f}"] = float(y - np.interp(x, ref_x, ref_y))
        cross = {str(N): crossing(pts) for N, pts in by_n.items()}
        cross["32"] = N32_CROSSING
        spread_dev = (max(dev.values()) - min(dev.values())) if dev else None
        max_dev = max(abs(v) for v in dev.values()) if dev else None
        cs = [v for v in cross.values() if v is not None]
        res["B1_collapse"] = {
            "delta_by_N": {str(N): pts for N, pts in by_n.items()},
            "deviation_from_N32_curve_db": dev,
            "max_abs_deviation_db": max_dev,
            "zero_crossing_r_eff_over_cap": cross,
            "crossing_spread": float(max(cs) - min(cs)) if len(cs) > 1 else None,
            "P12_prediction": "curves within +/-0.3 dB at matched r_eff/cap; "
                              "each N crosses zero at 0.52 +/- 0.08",
            "P12_falsifier": "systematic ordering by N > 0.5 dB, or crossings "
                             "differing by more than 0.15 in r_eff/cap"}
        print("  deviation from the N=32 reference curve at matched r_eff/cap:")
        for k, v in dev.items():
            print(f"    {k}  {v:+.3f} dB")
        print("  zero crossings (r_eff/cap): " +
              "  ".join(f"N={k} {'--' if v is None else f'{v:.3f}'}"
                        for k, v in sorted(cross.items(), key=lambda kv: int(kv[0]))))

    # ---------------- B6 : Xiao SV against the A2 prediction ---------------
    b6 = {t: c for t, c in cells.items() if t.startswith("B6_")}
    if b6:
        hdr("B6  Xiao Saleh-Valenzuela channel vs the A2 prediction   (tests A2)")
        print("  reading            n   predicted   measured   CI95              "
              "err     SNR>=5    SNR<5")
        res["B6_xiao"] = {}
        for t, c in sorted(b6.items()):
            p = c["delta_hs"]["pooled_SAMPLING_DESIGN_DEPENDENT"]
            pred = A2_PREDICTION[t]
            m = p["median_diff_db"]
            lo, hi = p["boot_ci95_median"]
            res["B6_xiao"][t] = {
                "predicted_db": pred, "measured_db": m, "ci95": [lo, hi],
                "error_db": float(m - pred),
                "prediction_inside_ci": bool(lo <= pred <= hi),
                "high_snr_ge5": band(c["delta_hs"], 5, 20),
                "low_snr_lt5": band(c["delta_hs"], -10, 5),
                "mean_L_hat": c["mean_L_hat"]}
            hb, lb = band(c["delta_hs"], 5, 20), band(c["delta_hs"], -10, 5)
            print(f"  {t[9:]:16s} {c['n']:4d}  {pred:+7.2f}   {m:+7.3f}  "
                  f"[{lo:+.3f},{hi:+.3f}]  {m-pred:+6.2f}"
                  f"  {'--' if hb is None else f'{hb:+7.3f}'}"
                  f"  {'--' if lb is None else f'{lb:+7.3f}'}")

    # ---------------- B4 : oracle gap, every cell --------------------------
    hdr("B4  gap to the unstructured-LS oracle (dB, pooled median; >0 = short of it)")
    print("  cell              EM-GS   HS-auto   fraction of the EM-GS gap closed")
    res["B4_oracle"] = {}
    for t, c in sorted(cells.items()):
        ge = c["vs_oracle"][EM]["pooled_SAMPLING_DESIGN_DEPENDENT"]["median_diff_db"]
        gh = c["vs_oracle"][HS]["pooled_SAMPLING_DESIGN_DEPENDENT"]["median_diff_db"]
        frac = (ge - gh) / ge if ge > 1e-9 else float("nan")
        res["B4_oracle"][t] = {"em_gs_gap_db": ge, "hs_auto_gap_db": gh,
                               "fraction_closed": float(frac)}
        print(f"  {t:16s} {ge:+7.2f}  {gh:+7.2f}   {frac:6.1%}")

    # ---------------- B5 : cost -------------------------------------------
    hdr("B5  measured cost per trial (single thread), from this sweep")
    print("  cell              n   seconds   s/trial   cap  mean L_hat")
    res["B5_cost"] = {}
    for t, c in sorted(cells.items()):
        s = c["seconds"] / max(c["n"], 1)
        res["B5_cost"][t] = {"n": c["n"], "seconds": c["seconds"],
                             "sec_per_trial_all_three_methods": float(s),
                             "cap": c["cap"], "mean_L_hat": c["mean_L_hat"]}
        print(f"  {t:16s} {c['n']:4d} {c['seconds']:8.0f}  {s:7.3f}  "
              f"{c['cap']:4d}  {c['mean_L_hat']:.2f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_JSON}  ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

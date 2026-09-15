"""PROMPT 22 Part B -- score P32: L vs L/r_max vs rho as predictors.

Pre-registration: reports/p22/PREREG_P32.md, committed standing alone before
this script ran.

Re-analysis only. No estimator is run; every number comes from stored per-trial
results and the committed Experiment C table.

Run:  PYTHONPATH=. python3 scratch/p22_score_p32.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "results/track_d/partB9"
EXPC = REPO / "trackB_hankel_emgs/results/experiment_C_path_count.csv"
OUT = REPO / "reports/p22"

EM, HS = "EM-GS", "HS-EM-GS-auto"
BIN = (5, 10)          # primary: paired per-trial median in [5,10)
N_BOOT = 10000
SEED = 0x9032

# Held-out cells. r_eff from the committed R_EFF table in
# scratch/trackD_partB9_analysis.py (median Roy-Vetterli effective rank of the
# noiseless channel columns); cap = r_max = ceil(N/2).
HELDOUT = {
    "B1_N16_L2":  dict(N=16, L=2,  cap=8,  r_eff=1.88),
    "B1_N16_L4":  dict(N=16, L=4,  cap=8,  r_eff=3.10),
    "B1_N16_L7":  dict(N=16, L=7,  cap=8,  r_eff=4.35),
    "B1_N64_L8":  dict(N=64, L=8,  cap=32, r_eff=6.33),
    "B1_N64_L14": dict(N=64, L=14, cap=32, r_eff=9.89),
    "B1_N64_L29": dict(N=64, L=29, cap=32, r_eff=16.24),
}


def load_table():
    """Experiment C: the fixed eight-point reference. N=32 so r_max=16."""
    rows = []
    with open(EXPC, newline="") as fh:
        for r in csv.DictReader(fh):
            L = float(r["L"])
            rows.append(dict(L=L, gain=float(r["gain_db"]),
                             lo=float(r["gain_ci_lo"]), hi=float(r["gain_ci_hi"])))
    # rho for each table row: r_eff = rho_published * cap. The committed
    # N32_REF row of scratch/trackD_partB9_analysis.py carries rho directly.
    rho = [0.119, 0.212, 0.285, 0.356, 0.408, 0.460, 0.507, 0.546]
    assert len(rho) == len(rows), (len(rho), len(rows))
    for r, x in zip(rows, rho):
        r["rho"] = x
        r["L_over_rmax"] = r["L"] / 16.0
    return rows


def predict(table, xkey, x):
    """Piecewise-linear on the fixed table; linear extrapolation outside it.

    No regression, no spline -- the rule registered in PREREG_P32.md and
    verified there to reproduce the manuscript's 1.30 and 0.648 dB.
    """
    pts = sorted((r[xkey], r["gain"]) for r in table)
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    if x <= xs[0]:
        x0, y0, x1, y1 = xs[0], ys[0], xs[1], ys[1]
    elif x >= xs[-1]:
        x0, y0, x1, y1 = xs[-2], ys[-2], xs[-1], ys[-1]
    else:
        i = int(np.searchsorted(xs, x) - 1)
        x0, y0, x1, y1 = xs[i], ys[i], xs[i + 1], ys[i + 1]
    return float(y0 + (y1 - y0) * (x - x0) / (x1 - x0))


def per_trial_gain(cell):
    """Per-trial dB gain, EM-GS minus HS-GS. Positive means HS-GS wins."""
    num, den = cell["num"], cell["den"]
    e = 10 * np.log10(np.asarray(num[EM]) / np.asarray(den[EM]))
    h = 10 * np.log10(np.asarray(num[HS]) / np.asarray(den[HS]))
    return e - h


def measured(tag):
    cell = json.loads((SRC / f"{tag}.json").read_text())
    d = per_trial_gain(cell)
    snr = np.asarray(cell["snr_db"])
    m = (snr >= BIN[0]) & (snr < BIN[1])
    rng = np.random.default_rng(SEED)
    sel = d[m]
    boot = np.median(rng.choice(sel, size=(N_BOOT, sel.size)), axis=1)
    hi_m = snr >= 5
    return dict(
        n_bin=int(m.sum()),
        median_db=float(np.median(sel)),
        ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        high_snr_ge5_median_db=float(np.median(d[hi_m])),
        n_high=int(hi_m.sum()))


def crossing_with_ci(table, xkey):
    """Zero crossing by linear interpolation in the bracketing pair, with a
    bootstrap interval over the table rows' own uncertainty.

    Each row's gain is resampled from a normal matched to its published 95%
    CI (the CI is symmetric enough here for that to be fair); the crossing is
    recomputed each time. Reported because the manuscript quotes the crossing
    locations bare.
    """
    pts = sorted((r[xkey], r["gain"], r["lo"], r["hi"]) for r in table)
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    sd = np.array([(p[3] - p[2]) / (2 * 1.96) for p in pts])

    def cross(y):
        for i in range(len(xs) - 1):
            if (y[i] > 0) != (y[i + 1] > 0):
                return xs[i] + (xs[i + 1] - xs[i]) * y[i] / (y[i] - y[i + 1])
        return np.nan

    rng = np.random.default_rng(SEED)
    draws = rng.normal(ys[None, :], sd[None, :], size=(N_BOOT, len(xs)))
    vals = np.array([cross(d) for d in draws])
    vals = vals[np.isfinite(vals)]
    return dict(point=float(cross(ys)),
                ci95=[float(np.percentile(vals, 2.5)),
                      float(np.percentile(vals, 97.5))],
                frac_bracketed=float(vals.size / N_BOOT))


def main():
    table = load_table()

    # Rule check: must reproduce the manuscript's published predictions.
    chk = {"rho=0.331": predict(table, "rho", 0.331),
           "rho=0.400": predict(table, "rho", 0.400)}

    keys = {"L": "L", "L_over_rmax": "L_over_rmax", "rho": "rho"}
    cells, errs = {}, {k: [] for k in keys}
    for tag, meta in HELDOUT.items():
        obs = measured(tag)
        x = {"L": meta["L"],
             "L_over_rmax": meta["L"] / meta["cap"],
             "rho": meta["r_eff"] / meta["cap"]}
        pred = {k: predict(table, keys[k], x[k]) for k in keys}
        cells[tag] = dict(meta=meta, x=x, measured=obs, predicted=pred,
                          error={k: pred[k] - obs["median_db"] for k in keys})
        for k in keys:
            errs[k].append(pred[k] - obs["median_db"])

    summ = {}
    for k in keys:
        e = np.array(errs[k])
        summ[k] = dict(MAE=float(np.mean(np.abs(e))),
                       RMSE=float(np.sqrt(np.mean(e ** 2))),
                       max_abs=float(np.max(np.abs(e))))

    # Decision rule: paired difference of absolute errors, rho vs L/r_max.
    from scipy import stats
    a = np.abs(np.array(errs["rho"]))
    b = np.abs(np.array(errs["L_over_rmax"]))
    d = a - b
    n = d.size
    tcrit = float(stats.t.ppf(0.975, n - 1))
    half = tcrit * d.std(ddof=1) / np.sqrt(n)
    paired = dict(mean_diff_abs_err=float(d.mean()),
                  ci95=[float(d.mean() - half), float(d.mean() + half)],
                  excludes_zero=bool((d.mean() - half) * (d.mean() + half) > 0),
                  per_cell=[float(v) for v in d])
    rho_wins = bool(summ["rho"]["MAE"] < summ["L_over_rmax"]["MAE"]
                    and paired["excludes_zero"])

    # Sign classification -- weak at n=6, reported as counts not accuracy.
    signs = {k: int(sum(1 for t in cells
                        if (cells[t]["predicted"][k] > 0)
                        == (cells[t]["measured"]["median_db"] > 0)))
             for k in keys}

    crossings = {k: crossing_with_ci(table, keys[k]) for k in keys}

    out = dict(
        prereg="reports/p22/PREREG_P32.md",
        rule="piecewise-linear on the fixed 8-point Experiment C table",
        rule_check_against_manuscript=chk,
        bin=list(BIN), n_boot=N_BOOT, seed=SEED,
        cells=cells, summary=summ, paired_rho_vs_Lrmax=paired,
        rho_beats_Lrmax_by_registered_rule=rho_wins,
        sign_correct_of_6=signs,
        crossing_n32_table_with_ci=crossings)
    (OUT / "p32_score.json").write_text(json.dumps(out, indent=2) + "\n")

    print("rule check (must match manuscript 1.30 / 0.648):",
          {k: round(v, 3) for k, v in chk.items()})
    print()
    print(f"{'cell':<12}{'meas':>8}{'predL':>8}{'predL/r':>9}{'predrho':>9}"
          f"{'|eL|':>8}{'|eL/r|':>8}{'|erho|':>8}")
    for t, c in cells.items():
        m = c["measured"]["median_db"]
        print(f"{t:<12}{m:8.3f}{c['predicted']['L']:8.3f}"
              f"{c['predicted']['L_over_rmax']:9.3f}{c['predicted']['rho']:9.3f}"
              f"{abs(c['error']['L']):8.3f}{abs(c['error']['L_over_rmax']):8.3f}"
              f"{abs(c['error']['rho']):8.3f}")
    print()
    for k in keys:
        print(f"  {k:<14} MAE {summ[k]['MAE']:6.3f}  RMSE {summ[k]['RMSE']:6.3f}"
              f"  max {summ[k]['max_abs']:6.3f}  sign {signs[k]}/6")
    print()
    print("paired |e_rho| - |e_L/rmax|: mean %.3f  95%% CI [%.3f, %.3f]  "
          "excludes zero: %s" % (paired["mean_diff_abs_err"],
                                 paired["ci95"][0], paired["ci95"][1],
                                 paired["excludes_zero"]))
    print("rho beats L/r_max by the registered rule:", rho_wins)
    print()
    for k, v in crossings.items():
        print(f"  crossing vs {k:<14} {v['point']:.4f}  "
              f"95% CI [{v['ci95'][0]:.4f}, {v['ci95'][1]:.4f}]  "
              f"bracketed in {100*v['frac_bracketed']:.1f}% of draws")
    print(f"\nwrote {OUT/'p32_score.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

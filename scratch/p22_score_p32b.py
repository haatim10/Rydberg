"""PROMPT 22 Part B -- P32 addendum: the PRE-REGISTERED secondary statistic.

Why this exists. PREREG_P32.md registered the paired median in the [5,10) SNR
bin as primary and `high_snr_ge5` as secondary. It also asserted "840 trials per
cell". That assertion is wrong: it holds for the three N=16 cells (853, 840,
874) but the three N=64 cells carry only 84, 85 and 83 trials. In the [5,10)
bin that leaves n = 14, 15, 14 -- below the 20-trial minimum this project's own
`by_bin` applies before it will report a bin at all.

So the registered PRIMARY statistic is weaker than registered, through an error
in the pre-registration rather than in the data. The remedy is not to pick a
new statistic after seeing the answer. It is to score the statistic that was
ALSO registered, under the IDENTICAL decision rule, and report both.

No new choices are made here. Same table, same rule, same decision criterion.

Run:  PYTHONPATH=. python3 scratch/p22_score_p32b.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "results/track_d/partB9"
OUT = REPO / "reports/p22"

import sys
sys.path.insert(0, str(REPO))
from scratch.p22_score_p32 import (EM, HS, HELDOUT, N_BOOT, SEED, load_table,
                                   per_trial_gain, predict)

MIN_BIN = 20        # the project's own minimum before a bin is reported


def measured_high(tag):
    """Paired per-trial median over SNR >= 5 dB, with a bootstrap interval."""
    cell = json.loads((SRC / f"{tag}.json").read_text())
    d = per_trial_gain(cell)
    snr = np.asarray(cell["snr_db"])
    sel = d[snr >= 5]
    rng = np.random.default_rng(SEED)
    boot = np.median(rng.choice(sel, size=(N_BOOT, sel.size)), axis=1)
    return dict(n=int(sel.size), median_db=float(np.median(sel)),
                ci95=[float(np.percentile(boot, 2.5)),
                      float(np.percentile(boot, 97.5))])


def crossing_from_cells(cells_meta, measured_by_tag, xkey, n_boot=N_BOOT):
    """Zero crossing of gain vs predictor across the cells of one array size.

    Bootstrap over the per-cell medians: each cell's median is resampled from a
    normal matched to its own bootstrap interval, and the crossing recomputed.
    Reports what fraction of draws BRACKET the crossing rather than
    extrapolating to it -- an extrapolated crossing is a weaker number and the
    existing analysis code already flags it as such.
    """
    pts = []
    for tag, meta in cells_meta:
        x = {"L": meta["L"], "L_over_rmax": meta["L"] / meta["cap"],
             "rho": meta["r_eff"] / meta["cap"]}[xkey]
        m = measured_by_tag[tag]
        sd = (m["ci95"][1] - m["ci95"][0]) / (2 * 1.96)
        pts.append((x, m["median_db"], sd))
    pts.sort()
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    sd = np.array([p[2] for p in pts])

    def cross(y):
        for i in range(len(xs) - 1):
            if (y[i] > 0) != (y[i + 1] > 0):
                return (xs[i] + (xs[i + 1] - xs[i]) * y[i] / (y[i] - y[i + 1]),
                        True)
        if len(xs) >= 2 and ys.size and y[-2] != y[-1]:
            return (xs[-2] + (xs[-1] - xs[-2]) * y[-2] / (y[-2] - y[-1]), False)
        return (np.nan, False)

    rng = np.random.default_rng(SEED)
    draws = rng.normal(ys[None, :], sd[None, :], size=(n_boot, len(xs)))
    res = [cross(d) for d in draws]
    vals = np.array([r[0] for r in res])
    brk = np.array([r[1] for r in res])
    ok = np.isfinite(vals)
    pt, pt_brk = cross(ys)
    return dict(point=float(pt), bracketed=bool(pt_brk),
                ci95=[float(np.percentile(vals[ok], 2.5)),
                      float(np.percentile(vals[ok], 97.5))],
                frac_bracketed=float(brk[ok].mean()),
                n_points=len(xs))


def main():
    table = load_table()
    keys = ("L", "L_over_rmax", "rho")

    meas = {t: measured_high(t) for t in HELDOUT}
    cells, errs = {}, {k: [] for k in keys}
    for tag, meta in HELDOUT.items():
        x = {"L": meta["L"], "L_over_rmax": meta["L"] / meta["cap"],
             "rho": meta["r_eff"] / meta["cap"]}
        pred = {k: predict(table, k, x[k]) for k in keys}
        cells[tag] = dict(meta=meta, measured=meas[tag], predicted=pred,
                          error={k: pred[k] - meas[tag]["median_db"]
                                 for k in keys})
        for k in keys:
            errs[k].append(pred[k] - meas[tag]["median_db"])

    summ = {k: dict(MAE=float(np.mean(np.abs(errs[k]))),
                    RMSE=float(np.sqrt(np.mean(np.square(errs[k])))),
                    max_abs=float(np.max(np.abs(errs[k])))) for k in keys}

    a, b = np.abs(errs["rho"]), np.abs(errs["L_over_rmax"])
    d = np.asarray(a) - np.asarray(b)
    half = float(stats.t.ppf(0.975, d.size - 1)) * d.std(ddof=1) / np.sqrt(d.size)
    paired = dict(mean_diff_abs_err=float(d.mean()),
                  ci95=[float(d.mean() - half), float(d.mean() + half)],
                  excludes_zero=bool((d.mean() - half) * (d.mean() + half) > 0),
                  per_cell=[float(v) for v in d])
    rho_wins = bool(summ["rho"]["MAE"] < summ["L_over_rmax"]["MAE"]
                    and paired["excludes_zero"])
    signs = {k: int(sum(1 for t in cells
                        if (cells[t]["predicted"][k] > 0)
                        == (cells[t]["measured"]["median_db"] > 0)))
             for k in keys}

    by_n = {}
    for N in (16, 64):
        sub = [(t, m) for t, m in HELDOUT.items() if m["N"] == N]
        by_n[str(N)] = {k: crossing_from_cells(sub, meas, k) for k in keys}

    out = dict(
        note="PRE-REGISTERED SECONDARY statistic (high_snr_ge5), identical rule.",
        why=("the registered primary bin [5,10) holds only n=14,15,14 for the "
             "three N=64 cells, below this project's own 20-trial minimum; the "
             "prereg's '840 trials per cell' was wrong for those three cells"),
        min_bin_threshold=MIN_BIN,
        cells=cells, summary=summ, paired_rho_vs_Lrmax=paired,
        rho_beats_Lrmax_by_registered_rule=rho_wins,
        sign_correct_of_6=signs,
        crossings_by_array_size=by_n)
    (OUT / "p32_score_secondary.json").write_text(json.dumps(out, indent=2) + "\n")

    print("PRE-REGISTERED SECONDARY: paired median over SNR >= 5 dB\n")
    print(f"{'cell':<12}{'n':>5}{'meas':>8}{'predL':>8}{'predL/r':>9}"
          f"{'predrho':>9}{'|eL|':>8}{'|eL/r|':>8}{'|erho|':>8}")
    for t, c in cells.items():
        m = c["measured"]
        print(f"{t:<12}{m['n']:5d}{m['median_db']:8.3f}"
              f"{c['predicted']['L']:8.3f}{c['predicted']['L_over_rmax']:9.3f}"
              f"{c['predicted']['rho']:9.3f}{abs(c['error']['L']):8.3f}"
              f"{abs(c['error']['L_over_rmax']):8.3f}{abs(c['error']['rho']):8.3f}")
    print()
    for k in keys:
        print(f"  {k:<14} MAE {summ[k]['MAE']:6.3f}  RMSE {summ[k]['RMSE']:6.3f}"
              f"  max {summ[k]['max_abs']:6.3f}  sign {signs[k]}/6")
    print()
    print("paired |e_rho| - |e_L/rmax|: mean %+.3f  95%% CI [%+.3f, %+.3f]  "
          "excludes zero: %s" % (paired["mean_diff_abs_err"], paired["ci95"][0],
                                 paired["ci95"][1], paired["excludes_zero"]))
    print("rho beats L/r_max by the registered rule:", rho_wins)
    print()
    print("crossings from the held-out cells (3 points each):")
    for N, dd in by_n.items():
        for k, v in dd.items():
            print(f"  N={N:<3} vs {k:<14} {v['point']:8.4f} "
                  f"[{v['ci95'][0]:7.4f}, {v['ci95'][1]:7.4f}]  "
                  f"bracketed {'yes' if v['bracketed'] else 'NO (extrapolated)'}"
                  f", {100*v['frac_bracketed']:.0f}% of draws")
    print(f"\nwrote {OUT/'p32_score_secondary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

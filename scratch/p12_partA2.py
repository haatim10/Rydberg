"""PROMPT 12 A2 -- per-bin U1 vs U1+post vs H1, paper-ready table and figure.

No new runs. Reads the stage-3 per-trial rows.

  U1        the unrolled estimator, no structural prior
  U1+post   U1 followed by ONE Cadzow projection applied post hoc
  H1        the structural prior interleaved INSIDE the unrolled loop

The contrast that matters for the pro-unrolling argument is
``internal_vs_posthoc`` = H1 - U1+post: does putting the projection inside the
loop beat bolting it on afterwards?

Run:  PYTHONPATH=. python3 scratch/p12_partA2.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SRC = Path("reports/trackD_stage3_results.json")
OUT = Path("reports/p12")
FIG = Path("paper/fig/p12")
BINS = [(-10, -5), (-5, 0), (0, 5), (5, 10), (10, 15), (15, 20)]

U1, POST, H1 = "U1_urformer_80k", "U1_plus_post", "H1_hs_urformer_80k"


def boot_ci_median(x, n_boot=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = np.median(x[rng.integers(0, x.size, size=(n_boot, x.size))], axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def contrast(per, snr, a, b):
    """Per-bin paired per-trial median of 10log10(a) - 10log10(b), in dB.

    Positive means b is BETTER than a (b has the smaller NMSE).
    """
    A = 10 * np.log10(np.asarray(per[a], float))
    B = 10 * np.log10(np.asarray(per[b], float))
    d = A - B
    snr = np.asarray(snr, float)
    rows = []
    for lo, hi in BINS:
        m = (snr >= lo) & (snr < hi)
        if m.sum() < 10:
            continue
        v = d[m]
        ci = boot_ci_median(v)
        rows.append({"bin": [lo, hi], "n": int(m.sum()),
                     "median_diff_db": float(np.median(v)),
                     "boot_ci95_median": list(ci),
                     "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0)})
    m = snr >= 5
    v = d[m]
    ci = boot_ci_median(v)
    hi5 = {"median_diff_db": float(np.median(v)), "boot_ci95_median": list(ci),
           "n": int(m.sum()), "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0)}
    return rows, hi5


def main():
    d = json.loads(SRC.read_text())
    per = d["test"]["per_trial_nmse"]
    snr = d["test"]["snr_db"]

    out = {}
    specs = [
        ("H1_vs_U1", U1, H1, "structural prior inside the loop, vs no prior"),
        ("post_vs_U1", U1, POST, "one post-hoc projection, vs no prior"),
        ("H1_vs_post", POST, H1, "INTERNAL vs POST-HOC (the pro-unrolling contrast)"),
    ]
    for name, a, b, desc in specs:
        rows, hi5 = contrast(per, snr, a, b)
        out[name] = {"desc": desc, "minuend": a, "subtrahend": b,
                     "bins": rows, "high_snr_ge5": hi5}

    print("=== A2: per-bin contrasts, paired per-trial median, bootstrap CI95 ===")
    print("    positive = the second arm is better\n")
    hdr = f"{'SNR bin':>12s} " + "".join(f"{s:>26s}" for _, _, _, s in
                                         [(0, 0, 0, 'H1 - U1'),
                                          (0, 0, 0, 'U1+post - U1'),
                                          (0, 0, 0, 'H1 - U1+post')])
    print(hdr)
    for i, b in enumerate(BINS):
        line = f"  [{b[0]:+3d},{b[1]:+3d})".rjust(12)
        for name, *_ in specs:
            r = [x for x in out[name]["bins"] if x["bin"] == list(b)]
            if not r:
                line += f"{'--':>26s}"
                continue
            r = r[0]
            star = "*" if r["ci_excludes_zero"] else " "
            line += (f"{r['median_diff_db']:+8.3f} "
                     f"[{r['boot_ci95_median'][0]:+6.3f},"
                     f"{r['boot_ci95_median'][1]:+6.3f}]{star}").rjust(26)
        print(line)
    line = "  SNR >= 5".rjust(12)
    for name, *_ in specs:
        h = out[name]["high_snr_ge5"]
        star = "*" if h["ci_excludes_zero"] else " "
        line += (f"{h['median_diff_db']:+8.3f} "
                 f"[{h['boot_ci95_median'][0]:+6.3f},"
                 f"{h['boot_ci95_median'][1]:+6.3f}]{star}").rjust(26)
    print(line)
    print("\n  * = bootstrap CI excludes zero")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "a2_internal_vs_posthoc.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8")

    # ------------------------------------------------------------ figure
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "lines.linewidth": 1.2, "lines.markersize": 4,
        "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
        "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    styles = {"H1_vs_U1": ("C0", "o-", "prior inside loop $-$ no prior"),
              "post_vs_U1": ("C7", "s:", "post-hoc only $-$ no prior"),
              "H1_vs_post": ("C3", "^-", "inside loop $-$ post-hoc")}
    for name in ("H1_vs_U1", "post_vs_U1", "H1_vs_post"):
        rows = out[name]["bins"]
        x = [0.5 * (r["bin"][0] + r["bin"][1]) for r in rows]
        y = [r["median_diff_db"] for r in rows]
        lo = [r["median_diff_db"] - r["boot_ci95_median"][0] for r in rows]
        hi = [r["boot_ci95_median"][1] - r["median_diff_db"] for r in rows]
        c, m, lab = styles[name]
        ax.errorbar(x, y, yerr=[lo, hi], fmt=m, color=c, capsize=2,
                    elinewidth=0.8, label=lab)
    ax.axhline(0, color="0.5", lw=0.8, ls=":")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("improvement (dB)")
    ax.legend(frameon=False, loc="upper left", handlelength=1.8)
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"a2_internal_vs_posthoc.{ext}", facecolor="white")
    plt.close(fig)
    print(f"\n  wrote {OUT/'a2_internal_vs_posthoc.json'}")
    print(f"  wrote {FIG/'a2_internal_vs_posthoc.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

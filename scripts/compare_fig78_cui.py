"""Quantitative comparison of our Fig. 7/8 curves against Cui's published ones.

Cui's curves were extracted from the rendered PDF by colour segmentation and
axis calibration; see scripts/audit_38901/. Extraction artifacts (legend
swatch bleed) are filtered by requiring monotone decreasing BER.

For each algorithm the discrepancy is reported two ways:
  * vertical  — ratio of BERs at the same SNR
  * horizontal — the SNR shift that best aligns our curve to Cui's, which is
    the meaningful measure for a waterfall curve
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
EXTRACT = Path("/tmp/claude-0/-home-user-Rydberg/"
               "bdbd557a-da34-5c9f-a184-91d2140d5ec4/scratchpad/"
               "cui_fig78_extracted.json")


def wilson_upper(k: int, n: int, z: float = 1.6448536269514722) -> float:
    """One-sided 95% upper bound (z = 1.645)."""
    if n <= 0:
        return float("nan")
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return min(1.0, centre + half)


def clean(series: dict) -> dict:
    """Drop extraction artifacts: keep the longest monotone-decreasing run."""
    items = sorted((float(k), v) for k, v in series.items())
    out, last = {}, np.inf
    for x, y in items:
        if y <= last * 1.05:
            out[x] = y
            last = min(last, y)
    return out


def snr_shift(ours: dict, theirs: dict) -> float | None:
    """SNR offset minimising squared log-BER distance between the curves."""
    common = [x for x in ours if x in theirs and ours[x] > 0 and theirs[x] > 0]
    if len(common) < 4:
        return None
    xs = np.array(sorted(common))
    lo = np.log10([ours[x] for x in xs])
    lt = np.log10([theirs[x] for x in xs])

    def cost(d):
        shifted = np.interp(xs + d, xs, lo, left=np.nan, right=np.nan)
        m = ~np.isnan(shifted)
        return np.mean((shifted[m] - lt[m]) ** 2) if m.sum() >= 3 else np.inf

    grid = np.arange(-3.0, 3.01, 0.05)
    return float(grid[int(np.argmin([cost(d) for d in grid]))])


def main(fig: str, sweep: str = "snr_db") -> None:
    agg = json.loads((REPO / "results/track_a" / fig / "aggregate.json").read_text())
    ours: dict[str, dict] = {}
    counts: dict[str, dict] = {}
    for r in agg["aggregate"]:
        ours.setdefault(r["algorithm"], {})[float(r[sweep])] = r["ber"]
        counts.setdefault(r["algorithm"], {})[float(r[sweep])] = (
            r["bit_errors"], r["bit_count"])

    print(f"=== {fig}: zero / near-zero points, one-sided 95% Wilson bounds ===")
    print(f"{'algorithm':<15}{sweep:>7} {'errors':>7} {'bits':>10} "
          f"{'BER':>11} {'95% upper':>11}")
    any_zero = False
    for a in sorted(counts):
        for x in sorted(counts[a]):
            k, n = counts[a][x]
            if k <= 5:
                any_zero = True
                ub = wilson_upper(k, n)
                est = f"{k/n:.3e}" if k else "0"
                print(f"{a:<15}{x:7.1f} {k:7d} {n:10d} {est:>11} {ub:11.3e}")
    if not any_zero:
        print("  (none)")

    if not EXTRACT.exists():
        print("\n(no extracted Cui curves available)")
        return
    cui = json.loads(EXTRACT.read_text()).get(fig)
    if not cui:
        print(f"\n(no extracted Cui curves for {fig})")
        return

    print(f"\n=== {fig}: ours vs Cui ===")
    print(f"{'algorithm':<15} {'SNR shift (dB)':>15} {'median BER ratio':>18} "
          f"{'n pts':>6}")
    for a in ("biased_gs", "em_gs", "exhaustive_ls", "exhaustive_ml", "genie_zf"):
        if a not in ours or a not in cui:
            continue
        theirs = clean(cui[a])
        common = [x for x in ours[a]
                  if x in theirs and ours[a][x] > 0 and theirs[x] > 0]
        if len(common) < 3:
            continue
        ratio = np.median([ours[a][x] / theirs[x] for x in common])
        d = snr_shift({x: ours[a][x] for x in common},
                      {x: theirs[x] for x in common})
        ds = f"{d:+15.2f}" if d is not None else f"{'n/a':>15}"
        print(f"{a:<15} {ds} {ratio:18.3f} {len(common):6d}")
    print("\nSNR shift < 0 means our curve reaches the same BER at a LOWER SNR")
    print("(i.e. we are better); BER ratio < 1 means our BER is lower.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fig7a",
         sys.argv[2] if len(sys.argv) > 2 else "snr_db")

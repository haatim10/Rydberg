"""Decide which B3/B4 points need more than 400 trials, and why.

Rule (fixed before looking at the numbers, applied mechanically):
extend a point only if the bootstrap 95% CI on the HS-GS vs EM-GS pooled
gain is too wide to settle the comparison, i.e. either

    (a) the CI contains 0, so the SIGN of the difference is undetermined; or
    (b) the CI is wider than WIDTH_MAX dB, so the MAGNITUDE is undetermined.

Points whose CI already excludes 0 with a narrow width are left at 400.
Nothing is recomputed for them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WIDTH_MAX = 1.5          # dB
EXTEND_TO = 1200         # 3x the base budget


def decide(store: Path):
    rows = json.loads((store / "summary.json").read_text())
    keep, extend = [], []
    for r in rows:
        lo, hi = r["gain_ci95_db"]
        width = hi - lo
        straddles = lo <= 0.0 <= hi
        too_wide = width > WIDTH_MAX
        rec = dict(N=r["N"], P=r["P"], snr_db=r["snr_db"],
                   gain=r["gain_hs_vs_em_db"], ci=[lo, hi], width=width,
                   reason=("CI contains 0: sign undetermined" if straddles
                           else f"CI width {width:.2f} dB > {WIDTH_MAX}: "
                                "magnitude undetermined" if too_wide else ""))
        (extend if (straddles or too_wide) else keep).append(rec)
    return keep, extend


def main():
    store = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results/track_b/b3"
    keep, extend = decide(store)
    print(f"{store.name}: {len(keep)} points stay at 400, "
          f"{len(extend)} extend to {EXTEND_TO}\n")
    if extend:
        print(f"{'N':>3}{'P':>4}{'SNR':>6} {'gain':>7} {'95% CI':>17} "
              f"{'width':>6}  reason")
        for r in extend:
            print(f"{r['N']:3d}{r['P']:4d}{r['snr_db']:6.1f} {r['gain']:+7.2f} "
                  f"[{r['ci'][0]:+6.2f},{r['ci'][1]:+6.2f}] {r['width']:6.2f}  "
                  f"{r['reason']}")
    widest = max(keep, key=lambda r: r["width"])
    print(f"\nwidest CI among the points left at 400: "
          f"N={widest['N']} P={widest['P']} SNR={widest['snr_db']:+.1f}, "
          f"width {widest['width']:.2f} dB, CI "
          f"[{widest['ci'][0]:+.2f},{widest['ci'][1]:+.2f}]")
    (store / "extension_plan.json").write_text(json.dumps(
        {"rule": {"width_max_db": WIDTH_MAX, "extend_to": EXTEND_TO,
                  "criteria": ["CI contains 0 (sign undetermined)",
                               f"CI width > {WIDTH_MAX} dB (magnitude "
                               "undetermined)"]},
         "extend": extend, "unchanged": keep}, indent=2))
    print(f"wrote {store/'extension_plan.json'}")


if __name__ == "__main__":
    main()

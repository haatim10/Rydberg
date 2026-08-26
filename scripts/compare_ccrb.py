"""Point-by-point old-vs-new comparison of the CRLB/CCRB after the theta/psi
Jacobian fix, plus the effect on every bound-crossing claim in the paper.

    python scripts/compare_ccrb.py OLD.json [NEW.json]

The unconstrained rank-1 bound does not involve the Jacobian at all, so it
MUST be unchanged; the script checks that as a control. Only the constrained
bound should move.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    old = json.loads(Path(sys.argv[1]).read_text())
    new_path = (Path(sys.argv[2]) if len(sys.argv) > 2
                else REPO / "results/track_b/constrained_crlb.json")
    new = json.loads(new_path.read_text())

    print("=" * 88)
    print("CONTROL - unconstrained rank-1 bound must be UNCHANGED (no Jacobian)")
    print("=" * 88)
    worst_unc = 0.0
    for kind in ("b3", "b4", "b6"):
        for k, v in old["unconstrained_rank1"][kind].items():
            if k in new["unconstrained_rank1"][kind]:
                worst_unc = max(worst_unc, abs(new["unconstrained_rank1"][kind][k] - v))
    print(f"  max |new - old| over all points: {worst_unc:.2e} dB "
          f"{'OK' if worst_unc < 1e-6 else '<-- UNEXPECTED'}\n")

    print("=" * 88)
    print("CONSTRAINED BOUND (CCRB) - old vs new, b3 grid")
    print("=" * 88)
    print(f"{'N':>3}{'P':>4}{'SNR':>5} | {'old unc':>9} {'new unc':>9} | "
          f"{'old CCRB':>9} {'new CCRB':>9} {'shift dB':>9}")
    shifts = {}
    for k in sorted(old["constrained"]["b3"],
                    key=lambda s: (int(s.split("_")[0][1:]),
                                   int(s.split("_P")[1].split("_")[0]),
                                   float(s.split("snr")[1]))):
        if k not in new["constrained"]["b3"]:
            continue
        N = int(k.split("_")[0][1:])
        P = int(k.split("_P")[1].split("_")[0])
        s = float(k.split("snr")[1])
        ou = old["unconstrained_rank1"]["b3"][k]
        nu = new["unconstrained_rank1"]["b3"][k]
        oc = old["constrained"]["b3"][k]
        nc = new["constrained"]["b3"][k]
        shifts[k] = nc - oc
        print(f"{N:3d}{P:4d}{s:5.0f} | {ou:9.3f} {nu:9.3f} | "
              f"{oc:9.3f} {nc:9.3f} {nc-oc:+9.3f}")

    v = np.array(list(shifts.values()))
    kmax = max(shifts, key=lambda x: abs(shifts[x]))
    kmin = min(shifts, key=lambda x: abs(shifts[x]))
    print(f"\n  CCRB shift over {v.size} b3 points:")
    print(f"    max  {v.max():+.3f} dB     min  {v.min():+.3f} dB")
    print(f"    mean {v.mean():+.3f} dB    mean |shift| {np.abs(v).mean():.3f} dB")
    print(f"    largest |shift|: {kmax} ({shifts[kmax]:+.3f} dB)")
    print(f"    smallest |shift|: {kmin} ({shifts[kmin]:+.3f} dB)")

    # ---- effect on the paper's crossing claims, recomputed from the stores --
    print()
    print("=" * 88)
    print("EFFECT ON PAPER CLAIMS - estimator curves vs the two bounds")
    print("=" * 88)
    fin = json.loads((REPO / "results/track_b/final_results.json").read_text()) \
        if (REPO / "results/track_b/final_results.json").exists() else None
    if fin is None:
        print("  final_results.json not found; skipping crossing recount")
        return 0

    def curve(est):
        out = {}
        for rec in fin if isinstance(fin, list) else fin.get("points", []):
            if rec.get("estimator") == est:
                key = (f"N{rec.get('N', 8)}_P{rec.get('P')}"
                       f"_snr{float(rec.get('snr_db')):+.0f}")
                out[key] = float(rec["nmse_db"])
        return out

    hs = curve("hs_gs") or curve("structured_hankel")
    if not hs:
        print("  HS-GS curve not found in final_results.json; "
              "recount deferred to verify_paper.py")
        return 0

    for label, bound in (("old CCRB", old["constrained"]["b3"]),
                         ("new CCRB", new["constrained"]["b3"])):
        common = [k for k in hs if k in bound]
        above = [k for k in common if hs[k] > bound[k]]
        gaps = [hs[k] - bound[k] for k in above]
        print(f"  HS-GS above {label}: {len(above)}/{len(common)}"
              + (f", by {min(gaps):.2f}-{max(gaps):.2f} dB" if gaps else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

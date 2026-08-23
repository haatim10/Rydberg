"""FIX 20 — fail if a number in the report disagrees with the stored data.

Fixes 3, 5, 16 and 17 were one failure repeated: a value in prose that no
longer matched the data behind it. This closes that loop. Run it before
committing the report; it exits non-zero on any mismatch.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MD = REPO / "TECHNICAL_REPORT.md"
FAIL: list[str] = []


def load(name):
    p = REPO / "results/track_b" / name
    return json.loads(p.read_text()) if p.exists() else None


def want(text: str, label: str):
    if text not in MD_TEXT:
        FAIL.append(f"{label}: expected to find {text!r}")


MD_TEXT = MD.read_text()
NUM, CAL, TIM = load("report_numbers.json"), load("calibration.json"), load("timing.json")

# --- trial counts ---------------------------------------------------------
t = NUM["trials"]
assert sum(t["b3_per_point"].values()) == t["b3_total"]
assert (t["grand_total_unique"] == t["b3_total"] + t["b4_new_only"]
        + t.get("b6_total", 0))
want(f"{t['grand_total_unique']:,}", "grand total")
want(f"{t['b3_total']:,}", "B3 total")

# the n column of the printed B3 table must sum to the B3 total
rows = re.findall(r"^\| (\d+) \| (\d+) \| ([+-]\d+) \| (\d+) \|", MD_TEXT, re.M)
if rows:
    printed = sum(int(r[3]) for r in rows)
    if printed != t["b3_total"]:
        FAIL.append(f"B3 table n column sums to {printed}, "
                    f"stored total is {t['b3_total']}")

# --- rho(N) must satisfy its own definition -------------------------------
for N, v in NUM["rho"].items():
    lhs = v["rho"] * v["dof_geometric"]
    if abs(lhs - v["dof_unstructured"]) > 1e-6:
        FAIL.append(f"rho({N})*3E[sumL] = {lhs} != 2NK = {v['dof_unstructured']}")
    want(f"{v['rho']:.3f}×", f"rho(N={N})")

# --- E[L] must be the same number in section 8 and audit check 6 ----------
want(f"{NUM['L']['mean_L_per_user']:.4f}", "E[L_k]")
if MD_TEXT.count(f"{NUM['L']['mean_L_per_user']:.4f}") < 2:
    FAIL.append("E[L_k] should appear in both Sec. 8 and audit check 6")
for stale in ("44.6 ", "mean 5.0111"):
    if stale in MD_TEXT:
        FAIL.append(f"stale E[L]-derived value still present: {stale!r}")

# --- B5 aggregates --------------------------------------------------------
for N, v in NUM["b5"].items():
    want(f"{v['mean_gain_db']:+.3f}", f"B5 mean gain N={N}")
    want(f"{v['mean_win_rate_unweighted']:.2%}", f"B5 win rate N={N}")

# --- calibration ----------------------------------------------------------
for k in ("snr", "rsr"):
    v = CAL[k]
    if not v["target_inside_ci"]:
        FAIL.append(f"calibration {k}: target outside CI, report claims otherwise")
want("2.82 dB and 12.15 dB", "the old calibration value must remain, quoted as superseded")

# --- timing ---------------------------------------------------------------
for r in TIM:
    if not r["projection_off_matches_chained_em_gs"]:
        FAIL.append(f"timing N={r['N']}: projection-off does not match chained baseline")
    want(f"{r['ratio_to_em_gs']['hs_gs_auto']:.1f}×", f"HS-GS cost ratio N={r['N']}")

# --- claims that must NOT be present --------------------------------------
# Sec. 10 deliberately QUOTES the superseded wrong claims in order to record
# them, so the ban applies to the body of the report, not to that section.
_i = MD_TEXT.index("## 10. Corrections")
_j = MD_TEXT.index("## 11. Final claim audit")
BODY = MD_TEXT[:_i] + MD_TEXT[_j:]
for banned, why in (
    ("precisely the set of channels", "Fix 4: the constraint is a relaxation"),
    ("GS is a contraction", "Fix 7: no contraction is claimed"),
    ("21 700", "Fix 3: superseded trial count"),
    ("24 100", "Fix 3: superseded grand total"),
    ("$L_k\\ge5$ — 60%", "Fix 2: wrong vacuity threshold"),
):
    if banned in BODY:
        FAIL.append(f"banned claim present ({why}): {banned!r}")

if FAIL:
    print(f"{len(FAIL)} REPORT/DATA MISMATCHES\n")
    for f in FAIL:
        print("  -", f)
    sys.exit(1)
print(f"report checked against stored data: OK "
      f"({len(MD_TEXT.splitlines())} lines, {t['grand_total_unique']:,} trials)")

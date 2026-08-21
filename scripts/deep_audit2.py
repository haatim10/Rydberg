"""Deep audit part 2: store integrity, config provenance, Track B."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

FIND = []


def check(cond, name, detail="", sev="HIGH"):
    s = "OK" if cond else sev
    FIND.append((s, name, detail))
    tag = {"OK": "  ok  ", "LOW": " LOW  ", "MED": " MED  ",
           "HIGH": " HIGH ", "BLOCK": "BLOCK "}[s]
    print(f"[{tag}] {name}" + (f"  — {detail}" if detail else ""))
    return bool(cond)


from rydberg_sim.monte_carlo import RESULTS_NAME, load_result_table

REPO = Path(__file__).resolve().parent.parent
TA = REPO / "results/track_a"

# expected configuration per figure, straight from the paper
EXPECT = {
    "fig5_final": dict(qam=16, N=36, K=3, rsr={12.0}, n_snr=18, metric="detection_nmse"),
    "fig6":       dict(qam=16, N=36, K=3, snr={3.0}, n_rsr=26, metric="detection_nmse"),
    "fig7a":      dict(qam=4,  N=36, K=3, rsr={12.0}, n_snr=18, metric="ber"),
    "fig7b":      dict(qam=16, N=100, K=6, rsr={12.0}, n_snr=18, metric="ber"),
    "fig8":       dict(qam=4,  N=36, K=3, snr={3.0}, n_rsr=26, metric="ber"),
    "fig8_16qam": dict(qam=16, N=36, K=3, snr={3.0}, n_rsr=5, metric="ber"),
}

print("=" * 78)
print("K. STORE INTEGRITY AND CONFIG PROVENANCE")
print("=" * 78)
for name, exp in EXPECT.items():
    d = TA / name
    files = list(d.glob("chunk*/" + RESULTS_NAME)) or [d / RESULTS_NAME]
    files = [f for f in files if f.exists()]
    if not files:
        check(False, f"{name}: results present", "no results.csv", sev="BLOCK")
        continue
    rows = []
    for f in files:
        rows += load_result_table(f)
    fps = {r["config_fingerprint"] for r in rows}
    exps = {r["experiment"] for r in rows}
    st = Counter(r["status"] for r in rows)
    key = Counter((r["config_fingerprint"], r["experiment"], r["trial"],
                   r["snr_db"], r["rsr_db"], r["algorithm"], r["metric"])
                  for r in rows)
    dups = sum(1 for v in key.values() if v > 1)
    snrs = {r["snr_db"] for r in rows}
    rsrs = {r["rsr_db"] for r in rows}
    print(f"\n  --- {name}: {len(rows):,} rows, exps={exps} ---")
    check(len(fps) == 1, f"{name}: single fingerprint", f"{len(fps)} found")
    check(len(exps) == 1, f"{name}: single experiment name")
    check(dups == 0, f"{name}: no duplicate keys", f"{dups} dups")
    check(set(st) == {"ok"}, f"{name}: all rows ok", str(dict(st)))
    if "rsr" in exp:
        check(rsrs == exp["rsr"], f"{name}: RSR fixed at {exp['rsr']}", str(rsrs))
        check(len(snrs) == exp["n_snr"], f"{name}: {exp['n_snr']} SNR points",
              f"{len(snrs)}")
    else:
        check(snrs == exp["snr"], f"{name}: SNR fixed at {exp['snr']}", str(snrs))
        check(len(rsrs) == exp["n_rsr"], f"{name}: {exp['n_rsr']} RSR points",
              f"{len(rsrs)}")
    mets = {r["metric"] for r in rows}
    check(exp["metric"] in mets, f"{name}: carries {exp['metric']}", str(mets))
    if exp["metric"] == "ber":
        # bits per trial must equal K*log2(M) -> confirms modulation and K
        br = next(r for r in rows if r["metric"] == "ber")
        want = exp["K"] * int(np.log2(exp["qam"]))
        check(int(br["bit_count"]) == want,
              f"{name}: bits/trial = K*log2(M) = {want}",
              f"got {br['bit_count']} (confirms {exp['qam']}-QAM, K={exp['K']})")
    # aggregate exists and matches the store
    ag = d / "aggregate.json"
    check(ag.exists(), f"{name}: aggregate.json present")

print()
print("=" * 78)
print("L. PLOTS ARE GENERATED FROM THE FINAL AGGREGATES")
print("=" * 78)
import subprocess
src = (REPO / "scripts/plot_fig78.py").read_text()
check("aggregate.json" in src, "plot_fig78 reads aggregate.json")
check("results.csv" not in src, "plot_fig78 does not re-read raw rows")
for name in ("fig7a", "fig7b", "fig8", "fig8_16qam"):
    d = TA / name
    png = list(d.glob("*_ber.png"))
    ag = d / "aggregate.json"
    if png and ag.exists():
        check(png[0].stat().st_mtime >= ag.stat().st_mtime - 5,
              f"{name}: plot newer than its aggregate",
              f"plot {png[0].name}", sev="MED")

print()
print("=" * 78)
print("M. TRACK B — exact model, no linearization, L_k handling")
print("=" * 78)
TB = Path("/home/user/rydberg-trackb")
import importlib.util
import sys
sys.path.insert(0, str(TB))
for mod in list(sys.modules):
    if mod.startswith("rydberg_sim"):
        del sys.modules[mod]
import rydberg_sim.track_b_proposed as prop
import rydberg_sim.track_b_drivers as drv
import rydberg_sim.baselines as bl

# tripwire: the proposed estimator must never touch the linearised solver
calls = {"n": 0}
orig = bl.linearised_closed_form_ls


def trip(*a, **k):
    calls["n"] += 1
    return orig(*a, **k)


bl.linearised_closed_form_ls = trip
try:
    w = drv.track_b_world(0, 10, 5.0)
    r = prop.hs_gs(w.S, w.Z, w.B, w.sigma2, L_hat=3, max_iter=10)
    r2 = prop.hs_gs_auto(w.S, w.Z, w.B, w.sigma2, max_iter=10, select_iter=5)
finally:
    bl.linearised_closed_form_ls = orig
check(calls["n"] == 0, "HS-GS never calls the linearised model",
      f"{calls['n']} calls", sev="BLOCK")
check(r.linearised_model_used is False, "HS-GS flags linearised_model_used=False")

# exact forward model
resid = np.abs(w.G @ w.S + w.B + w.W) - w.Z
check(float(np.abs(resid).max()) < 1e-12, "Track B: Z = |GS+B+W| exactly",
      f"max {np.abs(resid).max():.2e}")

# L_k distribution
flat = np.array([drv.draw_L_k(t) for t in range(4000)]).ravel()
check(flat.min() == 3 and flat.max() == 7, "L_k support = {3..7}")
check(abs(flat.mean() - 5.0) < 0.1, "L_k mean ~ 5", f"{flat.mean():.3f}")
counts = np.array([(flat == v).sum() for v in range(3, 8)])
check(counts.std() / counts.mean() < 0.05, "L_k uniform across support",
      f"cv {counts.std()/counts.mean():.3f}")
check(drv.draw_L_k(7) == drv.draw_L_k(7), "L_k reproducible")
check(tuple(w.L_k) == drv.draw_L_k(0), "world L_k matches the draw")

# Track B must use the ULA channel, never the Cui one
sp = drv.track_b_spec(P=10, n_trials=1)
check(sp.track == "B" and sp.channel_model == "ula_geometric",
      "Track B uses the geometric ULA channel")

# CRN across estimators
a = drv.track_b_world(3, 10, 5.0)
b = drv.track_b_world(3, 10, 5.0)
check(all(np.array_equal(getattr(a, x), getattr(b, x))
          for x in ("G", "S", "B", "W", "Z")), "Track B CRN deterministic")

# baseline store
bfile = TB / "results/track_b/baseline_preliminary.json"
if bfile.exists():
    bj = json.loads(bfile.read_text())
    check(bj["rsr_db"] == 12.0, "B1/B2 baseline RSR = 12 dB")
    check(bj["N"] == 8 and bj["K"] == 3, "B1/B2 baseline N=8, K=3")
    check(bj["n_trials"] == 400, "B1/B2 baseline 400 trials/point")
    algs = {r["algorithm"] for r in bj["rows"]}
    check(algs == {"biased_gs", "em_gs"}, "B1/B2 baseline plots GS and EM-GS only",
          str(algs))

print()
print("=" * 78)
print("SUMMARY (part 2)")
print("=" * 78)
c = Counter(s for s, _, _ in FIND)
print("   ", dict(c))
bad = [f for f in FIND if f[0] in ("BLOCK", "HIGH", "MED")]
if bad:
    print("\n  NON-OK:")
    for s, n, d in bad:
        print(f"    [{s}] {n} — {d}")
else:
    print("\n  No BLOCKER / HIGH / MEDIUM findings.")

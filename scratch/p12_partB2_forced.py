"""PROMPT 12 B2 addendum -- what is abstention masking?

B2 hit a wall that is a property of the ESTIMATOR, not of the coordinate.
At N=16 (cap 8) with L=14, the held-out selector returns L_hat >= cap in
essentially every trial, the projection becomes the identity, and HS-GS is
bit-for-bit EM-GS. The measured Delta_HS is therefore exactly +0.000 with a
zero-width CI -- it cannot go negative, so the crossing cannot be bracketed
from the negative side by raising rho.

That leaves a question the sweep alone cannot answer:

    does Delta_HS actually go negative at high rho, or does abstention
    merely prevent us from ever observing it?

This diagnostic answers it by running the SAME operator with the order FORCED
to a value strictly below cap, so the projection is always active. It is a
DIAGNOSTIC, not a proposed estimator: an estimator that cannot abstain is not
the method Paper 1 proposes, and this must never be quoted as HS-GS.

No new architecture, no new generator: this is hs_gs with a fixed L_hat, which
already exists (rydberg_sim/track_b_proposed.hs_gs).

Run:  PYTHONPATH=. python3 scratch/p12_partB2_forced.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from rydberg_sim.track_b_proposed import hankel_rank_cap, hs_gs, hs_gs_auto
from scratch.p12_partB import (
    TD_KW, boot_ci_median, em_gs_only, nmse_parts, trackd_world,
)

OUT = Path("results/p12")

# (tag, N, L, rho measured by scratch/p12_partA_rho.rho_for)
CELLS = [("N16_L7",  16, 7,  0.542),
         ("N16_L9",  16, 9,  0.608),
         ("N16_L14", 16, 14, 0.735)]


def run(n_trials=250):
    rows = []
    for tag, N, L, rho in CELLS:
        cap = hankel_rank_cap(N)
        forced = cap - 1              # strictly below cap: always active
        num_e, num_a, num_f, den = [], [], [], []
        lhat, abst = [], []
        t0 = time.time()
        rng = np.random.default_rng(np.random.SeedSequence([650_000, N, L]))
        for t in range(n_trials):
            snr = float(np.round(rng.uniform(-10, 20), 3))
            w = trackd_world(650_000 + 11 * t, N=N, P=20, snr_db=snr, L=L)
            G = np.asarray(w.G_true)
            e = em_gs_only(w, TD_KW["max_iter"])
            # the proposed estimator, with abstention
            r = hs_gs_auto(w.S, w.Z, w.B, w.sigma2, **TD_KW)
            lhat.append(int(r.L_hat))
            abst.append(int(r.L_hat >= cap))
            # the diagnostic: projection forced active
            f = hs_gs(w.S, w.Z, w.B, w.sigma2, L_hat=forced,
                      exact_step="em_gs", max_iter=TD_KW["max_iter"],
                      cadzow_iter=4).G_hat
            a, b = nmse_parts(e, G); num_e.append(a); den.append(b)
            num_a.append(nmse_parts(r.G_hat, G)[0])
            num_f.append(nmse_parts(f, G)[0])
        den = np.asarray(den)
        e_db = 10 * np.log10(np.asarray(num_e) / den)
        d_auto = e_db - 10 * np.log10(np.asarray(num_a) / den)
        d_forced = e_db - 10 * np.log10(np.asarray(num_f) / den)
        ci_a, ci_f = boot_ci_median(d_auto), boot_ci_median(d_forced)
        rows.append({
            "tag": tag, "N": N, "L": L, "cap": cap, "forced_order": forced,
            "rho_measured": rho, "n": n_trials,
            "seconds": round(time.time() - t0, 1),
            "abstention_rate": float(np.mean(abst)),
            "mean_L_hat": float(np.mean(lhat)),
            "delta_with_abstention_db": float(np.median(d_auto)),
            "ci_with_abstention": list(ci_a),
            "delta_forced_active_db": float(np.median(d_forced)),
            "ci_forced_active": list(ci_f),
            "forced_negative_and_significant": bool(ci_f[1] < 0),
        })
        print(f"  [B2f] {tag:9s} rho={rho:.3f} abstain={100*np.mean(abst):5.1f}%  "
              f"auto {rows[-1]['delta_with_abstention_db']:+.3f} "
              f"CI[{ci_a[0]:+.3f},{ci_a[1]:+.3f}]   "
              f"forced {rows[-1]['delta_forced_active_db']:+.3f} "
              f"CI[{ci_f[0]:+.3f},{ci_f[1]:+.3f}]"
              f"{'  NEG-SIG' if ci_f[1] < 0 else ''}  "
              f"{rows[-1]['seconds']:.0f}s", flush=True)
    return {"cell": "B2_forced_order_diagnostic",
            "config": {"K": 3, "P": 20, "rsr_db": 10.0, "operator": TD_KW,
                       "generator": "trackD make_world",
                       "forced_order": "cap - 1, so the projection is always "
                                       "active",
                       "WARNING": "DIAGNOSTIC ONLY. An estimator that cannot "
                                  "abstain is not the method Paper 1 proposes. "
                                  "Never quote the forced column as HS-GS."},
            "points": rows}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "B2_forced.json"
    if f.exists():
        print(f"skip (done): {f}")
        raise SystemExit(0)
    print("=== PROMPT 12 B2 addendum: forced-order diagnostic ===", flush=True)
    r = run()
    f.write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {f}")

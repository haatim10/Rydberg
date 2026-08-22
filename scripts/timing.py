"""FIX 12 — what does HS-GS actually cost?

"+2.85 dB at N=32" invites the question "at what price". HS-GS runs a Cadzow
projection (an SVD of a roughly ceil(N/2)-square Hankel matrix, per user, per
iteration) inside all 50 iterations, and before that an order search over L
with a held-out refit at each candidate. That is plausibly one to two orders
of magnitude more work than EM-GS.

Acceptance check: with the projection disabled (L_hat at the rank cap, a
no-op) HS-GS must reproduce the *chained* EM-GS baseline -- 50 separate
max_iter=1 calls -- to within noise. It does NOT reproduce a single
max_iter=50 call, and should not be expected to: HS-GS must re-enter the
solver every iteration so the projection can be interleaved (eq. 23), and
that call structure alone costs 1.77-1.79x. That overhead is architectural,
not projection work, so it is measured and reported separately rather than
charged to the structural step.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from rydberg_sim.gs import biased_gs_channel_rows, em_gs_channel_rows
from rydberg_sim.track_b_drivers import TRACK_B_K, track_b_world
from rydberg_sim.track_b_proposed import hankel_rank_cap, hs_gs, hs_gs_auto

REPS = 12
HS = dict(exact_step="em_gs", max_iter=50, select_iter=20)


def med_ms(fn, n=REPS):
    ts = []
    for t in range(n):
        w = track_b_world(t, 30, 5.0, N=fn.N)
        a = time.perf_counter()
        fn(w)
        ts.append((time.perf_counter() - a) * 1e3)
    return statistics.median(ts)


def main():
    rows = []
    for N in (8, 16, 32):
        cap = hankel_rank_cap(N)

        def gs(w): return biased_gs_channel_rows(w.S, w.Z, w.B, max_iter=50)
        def em(w): return em_gs_channel_rows(w.S, w.Z, w.B, w.sigma2, max_iter=50)
        def hs(w): return hs_gs_auto(w.S, w.Z, w.B, w.sigma2, **HS)
        # projection only, order fixed: isolates the Cadzow cost
        def hs_fixed(w): return hs_gs(w.S, w.Z, w.B, w.sigma2, L_hat=3,
                                      exact_step="em_gs", max_iter=50)
        # projection disabled (L_hat == cap is a no-op): must match EM-GS
        def hs_off(w): return hs_gs(w.S, w.Z, w.B, w.sigma2, L_hat=cap,
                                    exact_step="em_gs", max_iter=50)
        # the honest baseline for HS-GS: same call structure, no projection
        def em_chained(w):
            G = None
            for _ in range(50):
                G = em_gs_channel_rows(w.S, w.Z, w.B, w.sigma2,
                                       max_iter=1, G0=G).G_hat
            return G
        for f in (gs, em, hs, hs_fixed, hs_off, em_chained):
            f.N = N
        t = {k: med_ms(f) for k, f in
             (("biased_gs", gs), ("em_gs", em), ("em_gs_chained", em_chained),
              ("hs_gs_auto", hs), ("hs_gs_fixed_L", hs_fixed),
              ("hs_gs_projection_off", hs_off))}
        n_L = cap                       # candidates searched by the order rule
        rows.append({
            "N": N, "rank_cap": cap, "median_ms": t,
            "ratio_to_em_gs": {k: v / t["em_gs"] for k, v in t.items()},
            "order_search_share": (t["hs_gs_auto"] - t["hs_gs_fixed_L"])
                                  / t["hs_gs_auto"],
            "projection_share": (t["hs_gs_fixed_L"] - t["hs_gs_projection_off"])
                                / t["hs_gs_auto"],
            "n_order_candidates": n_L,
            "flops_note": {
                "hs_gs_svd": f"iters x K x |L-grid| x O(cap^3) = "
                             f"50 x {TRACK_B_K} x {n_L} x O({cap}^3)",
                "em_gs": f"iters x N x O(K^2 P) = 50 x {N} x O({TRACK_B_K}^2 x 30)",
            },
        })
        # acceptance: projection-off must match the CHAINED baseline
        off, chained = t["hs_gs_projection_off"], t["em_gs_chained"]
        rel = abs(off - chained) / chained
        rows[-1]["projection_off_matches_chained_em_gs"] = bool(rel < 0.15)
        rows[-1]["projection_off_rel_diff_vs_chained"] = rel
        rows[-1]["chained_call_overhead"] = t["em_gs_chained"] / t["em_gs"]

    (REPO / "results/track_b/timing.json").write_text(json.dumps(rows, indent=2))
    hdr = (f"{'N':>3} {'cap':>4} | {'GS':>8} {'EM-GS':>8} {'HS-GS':>9} "
           f"{'HS fixed L':>11} {'HS proj off':>12} | {'HS/EM':>7} "
           f"{'order%':>7} {'proj%':>6}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        t = r["median_ms"]
        print(f"{r['N']:3d} {r['rank_cap']:4d} | {t['biased_gs']:8.1f} "
              f"{t['em_gs']:8.1f} {t['hs_gs_auto']:9.1f} "
              f"{t['hs_gs_fixed_L']:11.1f} {t['hs_gs_projection_off']:12.1f} | "
              f"{r['ratio_to_em_gs']['hs_gs_auto']:6.1f}x "
              f"{r['order_search_share']:7.0%} {r['projection_share']:6.0%}")
    print("\n  acceptance — projection disabled vs the CHAINED EM-GS baseline:")
    for r in rows:
        print(f"    N={r['N']:2d}: rel diff "
              f"{r['projection_off_rel_diff_vs_chained']:.1%} -> "
              f"{'PASS' if r['projection_off_matches_chained_em_gs'] else 'FAIL'}"
              f"   (chained-call overhead alone: "
              f"{r['chained_call_overhead']:.2f}x a single max_iter=50 call)")
    print(f"\nwrote {REPO/'results/track_b/timing.json'}")


if __name__ == "__main__":
    main()

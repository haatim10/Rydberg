"""PROMPT 12 Part A -- repository checks and re-analysis. NO NEW COMPUTE.

Everything here reads already-committed results files and source constants.

  A1  exactly what separates family B3 from family TD, field by field, and
      whether a TD cell already exists at N=32, K=3, P=30 (which would make
      the B1 bridging run unnecessary).
  A2  per-bin U1 vs U1+post vs H1, paper-ready table + figure.
  A3  abstention rate as an observable proxy for rho, across every cell run
      with adaptive order.
  A4  the two internal inconsistencies: r_eff(L=16, N=32), and Xu et al. pages.

Run:  PYTHONPATH=. python3 scratch/p12_partA.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path("reports/p12")
FIG = Path("paper/fig/p12")


# ----------------------------------------------------------------- helpers
def boot_ci_median(x, n_boot=2000, seed=0):
    """Bootstrap 95% CI of the median of a paired difference vector."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    m = np.median(x[idx], axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


BINS = [(-10, -5), (-5, 0), (0, 5), (5, 10), (10, 15), (15, 20)]


# ============================================================== A1
def a1_family_table():
    """Field-by-field B3 vs TD, read off the source, not off the manuscripts."""
    from rydberg_sim import track_b_drivers as tbd
    from rydberg_sim.track_b_proposed import hs_gs_auto
    from trackD_urformer.config import TrackDConfig
    import inspect

    sig = inspect.signature(hs_gs_auto).parameters
    default_cadzow = sig["cadzow_iter"].default
    default_project_every = sig["project_every"].default

    # B3 call site: scripts/run_b3.py:145  ->  hs_gs_auto(..., **HS_KW)
    #   HS_KW = dict(exact_step="em_gs", max_iter=50, select_iter=20)
    # TD call site: scratch/trackD_partB9_sweeps.py:171
    #   hs_gs_auto(..., exact_step="em_gs", max_iter=100, select_iter=25)
    # NEITHER passes cadzow_iter, so both take the default.
    td = TrackDConfig().system

    rows = [
        # field, B3, TD, moves Delta_HS?, direction
        ("estimator entry point", "hs_gs_auto", "hs_gs_auto",
         "no", "identical function, same file"),
        ("order selection", "held-out, adaptive", "held-out, adaptive",
         "no", "select_order_heldout in both; no oracle in either"),
        ("Cadzow sweeps n_cz", f"{default_cadzow} (default)",
         f"{default_cadzow} (default)", "no",
         "NEITHER call site passes cadzow_iter"),
        ("project_every", f"{default_project_every} (default)",
         f"{default_project_every} (default)", "no", "identical"),
        ("selection iterations", "20", "25", "marginal",
         "longer selection run -> slightly better L_hat -> favours TD"),
        ("estimator iterations T", "50", "100", "yes, small",
         "both arms converge further; EM-GS gains too, so Delta_HS ~flat"),
        ("RSR (dB)", f"{tbd.TRACK_B_RSR_DB}", f"{td.rsr_db}", "yes",
         "lower RSR -> harder phase retrieval -> larger headroom -> "
         "TD favoured on Delta_HS"),
        ("pilots P", "{10, 30}", "20", "yes",
         "fewer pilots -> larger Delta_HS (B7 measured this)"),
        ("array sizes N", "{8, 16, 32}", "{16, 32, 64}", "yes",
         "aperture sets capacity; not comparable across the two grids"),
        ("users K", f"{tbd.TRACK_B_K}", f"{td.K}", "no", "same"),
        ("paths L_k", f"U{{{tbd.TRACK_B_L_MIN}..{tbd.TRACK_B_L_MAX}}}",
         f"U{{{td.L_min}..{td.L_max}}}", "no", "same"),
        ("master seed", f"{tbd.TRACK_B_MASTER_SEED}", f"{td.master_seed}",
         "no", "different worlds, same distribution"),
        ("SNR handling", "fixed 6-point grid", "U[-10,20] binned post hoc",
         "yes", "TD includes a -10 dB bin B3 has no cell for"),
        ("trials", "400 per point", "time-budgeted, 60-1000 per cell",
         "no", "affects CI width, not the estimate"),
        ("pooling rule", "ratio-of-sums within a point",
         "paired per-trial median within a bin", "yes",
         "different statistics of the same quantity; Jensen gap"),
        ("world generator", "track_b_world", "trackD make_world",
         "no", "both geometric ULA, psi = pi sin(theta)"),
    ]

    # Does a TD cell already exist at N=32, K=3, P=30?
    cells = sorted(Path("results/track_d/partB9").glob("*.json"))
    have = []
    for f in cells:
        d = json.loads(f.read_text())
        have.append((d["tag"], d["N"], d["K"], d["P"], d.get("channel")))
    match = [h for h in have if (h[1], h[2], h[3]) == (32, 3, 30)]

    return rows, have, match


# ============================================================== A2
def a2_posthoc_vs_internal():
    """Per-bin U1 vs U1+post vs H1 from the stage-3 per-trial rows."""
    p = Path("reports/trackD_stage3_results.json")
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d


# ============================================================== A3
def a3_abstention_vs_rho():
    """(rho, abstention rate) for every adaptive-order cell already run."""
    from rydberg_sim.track_b_proposed import hankel_rank_cap
    rows = []
    for f in sorted(Path("results/track_d/partB9").glob("*.json")):
        d = json.loads(f.read_text())
        cap = d["cap"]
        lhat = np.asarray(d["L_hat"], int)
        # abstention == the selector returned an order at or above capacity,
        # which makes the projection the identity.
        abst = float(np.mean(lhat >= cap))
        rows.append({"tag": d["tag"], "N": d["N"], "K": d["K"], "P": d["P"],
                     "channel": d.get("channel"), "cap": cap, "n": d["n"],
                     "abstention": abst, "mean_L_hat": float(lhat.mean())})
    return rows


# ============================================================== A4
def a4_reff_L16_N32():
    """Recompute r_eff of the L=16, N=32 channel from the stored tables."""
    hits = {}
    for f in sorted(Path("reports").glob("*.json")) + \
             sorted(Path("results/track_d").rglob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        s = json.dumps(d)
        if "r_eff" in s or "reff" in s:
            hits[str(f)] = d
    return hits


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rows, have, match = a1_family_table()
    print("=== A1: B3 vs TD, field by field ===")
    for r in rows:
        print(f"  {r[0]:24s} B3={r[1]:24s} TD={r[2]:28s} moves={r[3]}")
    print(f"\n  TD cells on disk: {len(have)}")
    for h in have:
        print(f"    {h[0]:22s} N={h[1]:3d} K={h[2]} P={h[3]:3d} ch={h[4]}")
    print(f"\n  TD cell at N=32,K=3,P=30: {match if match else 'NONE -> B1 required'}")

    print("\n=== A3: abstention vs capacity, per adaptive-order cell ===")
    for r in a3_abstention_vs_rho():
        print(f"  {r['tag']:22s} N={r['N']:3d} cap={r['cap']:3d} "
              f"n={r['n']:4d} abstain={100*r['abstention']:5.1f}%  "
              f"mean_Lhat={r['mean_L_hat']:.2f}")

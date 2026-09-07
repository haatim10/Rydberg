"""PROMPT 12 Part B -- classical sweeps. No training.

  B1  THE BRIDGING RUN. TD operator at the B3 principal operating point:
      N=32, K=3, P=30, RSR 12 dB, six-point B3 SNR grid, 400 paired trials
      per point. Scores P16.
  B2  bracket the two extrapolated rho crossings at N=16 and N=64.
  B3  pencil sweep at fixed N=32 and fixed rho. Scores P17.
  B5  classical post-hoc (EM-GS -> one Cadzow) vs interleaved HS-GS. Scores P18.

The operator is the SAME in both families (Part A, A1): hs_gs_auto, adaptive
held-out order, cadzow_iter=4 by default. What "TD operator" means here is
max_iter=100 / select_iter=25 versus B3's 50 / 20.

Worlds come from the Track B generator at RSR 12 dB for B1/B5 (so the cell is
genuinely the B3 operating point) and from the Track D generator for B2/B3
(so those stay comparable to the committed TD cells they extend).

Run one group:
  PYTHONPATH=. python3 scratch/p12_partB.py --group B1
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from rydberg_sim.track_b_proposed import (
    cadzow_project, hankel_rank_cap, hs_gs, hs_gs_auto, select_order_heldout,
)

OUT = Path("results/p12")
BINS = [(-10, -5), (-5, 0), (0, 5), (5, 10), (10, 15), (15, 20)]
B3_SNR = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)

# TD operator settings (Part A, A1).
TD_KW = dict(exact_step="em_gs", max_iter=100, select_iter=25)


def nmse_parts(G_hat, G):
    d = np.asarray(G_hat) - np.asarray(G)
    return float(np.sum(np.abs(d) ** 2)), float(np.sum(np.abs(np.asarray(G)) ** 2))


def boot_ci_median(x, n_boot=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = np.median(x[rng.integers(0, x.size, size=(n_boot, x.size))], axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ------------------------------------------------------------- world makers
def trackb_world(trial, *, N, P, snr_db, rsr_db=12.0, K=3, L=None):
    """Track B generator -- used for B1/B5 so the cell IS the B3 operating point."""
    import sys
    if "trackB_hankel_emgs" not in sys.path:
        sys.path.insert(0, "trackB_hankel_emgs")
    from system_model import make_world as tb_make_world
    return tb_make_world(trial, N=N, P=P, snr_db=snr_db, L=L, rsr_db=rsr_db)


def trackd_world(trial, *, N, P, snr_db, K=3, L=None):
    from trackD_urformer.config import TrackDConfig
    from trackD_urformer.dataset import make_world
    cfg = TrackDConfig()
    return make_world(trial, sysc=replace(cfg.system, K=K), N=N, P=P,
                      snr_db=snr_db, L=L)


def em_gs_only(w, max_iter):
    from rydberg_sim.gs import em_gs_channel_rows
    return em_gs_channel_rows(w.S, w.Z, w.B, w.sigma2, max_iter=max_iter).G_hat


# ============================================================== B1
def run_b1(seconds_per_point=100000, n_trials=400):
    """TD operator at the B3 principal cell. Scores P16."""
    rows = []
    for snr in B3_SNR:
        num_e, num_h, den, lhat, act = [], [], [], [], []
        t0 = time.time()
        for t in range(n_trials):
            w = trackb_world(600_000 + t, N=32, P=30, snr_db=snr, rsr_db=12.0)
            G = np.asarray(w.G)
            e = em_gs_only(w, TD_KW["max_iter"])
            r = hs_gs_auto(w.S, w.Z, w.B, w.sigma2, **TD_KW)
            a, b = nmse_parts(e, G); num_e.append(a); den.append(b)
            a2, _ = nmse_parts(r.G_hat, G); num_h.append(a2)
            lhat.append(int(r.L_hat))
            act.append(bool(r.L_hat < hankel_rank_cap(32)))
        d = 10 * np.log10(np.asarray(num_e) / np.asarray(den)) - \
            10 * np.log10(np.asarray(num_h) / np.asarray(den))
        ci = boot_ci_median(d)
        rows.append({
            "snr_db": snr, "n": n_trials, "seconds": round(time.time() - t0, 1),
            "delta_median_db": float(np.median(d)), "boot_ci95_median": list(ci),
            "delta_ratio_of_sums_db": float(
                10 * np.log10(np.sum(num_e) / np.sum(den))
                - 10 * np.log10(np.sum(num_h) / np.sum(den))),
            "em_gs_db": float(10 * np.log10(np.sum(num_e) / np.sum(den))),
            "hs_gs_db": float(10 * np.log10(np.sum(num_h) / np.sum(den))),
            "mean_L_hat": float(np.mean(lhat)),
            "active_frac": float(np.mean(act)),
            "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
        })
        print(f"  [B1] SNR {snr:+5.1f}  delta_med {rows[-1]['delta_median_db']:+.3f} "
              f"CI[{ci[0]:+.3f},{ci[1]:+.3f}]  ros {rows[-1]['delta_ratio_of_sums_db']:+.3f} "
              f"Lhat {rows[-1]['mean_L_hat']:.2f}  {rows[-1]['seconds']:.0f}s", flush=True)
    return {"cell": "B1_bridging_TDop_at_B3_point",
            "config": {"N": 32, "K": 3, "P": 30, "rsr_db": 12.0,
                       "snr_grid": list(B3_SNR), "n_trials": n_trials,
                       "operator": TD_KW, "cadzow_iter": 4,
                       "generator": "trackB_hankel_emgs/system_model.make_world"},
            "points": rows}


# ============================================================== B5
def run_b5(n_trials=300):
    """Interleaved HS-GS vs EM-GS + ONE post-hoc Cadzow. Scores P18."""
    rows = []
    for snr in B3_SNR:
        num_i, num_p, den = [], [], []
        t0 = time.time()
        for t in range(n_trials):
            w = trackb_world(610_000 + t, N=32, P=30, snr_db=snr, rsr_db=12.0)
            G = np.asarray(w.G)
            # One shared L_hat, so the contrast isolates PLACEMENT.
            L_hat, _ = select_order_heldout(
                w.S, w.Z, w.B, w.sigma2, exact_step="em_gs",
                max_iter=TD_KW["select_iter"], cadzow_iter=4)
            inter = hs_gs(w.S, w.Z, w.B, w.sigma2, L_hat=L_hat,
                          exact_step="em_gs", max_iter=TD_KW["max_iter"],
                          project_every=1, cadzow_iter=4).G_hat
            base = em_gs_only(w, TD_KW["max_iter"])
            post = np.array(base, copy=True)
            for k in range(post.shape[1]):
                post[:, k] = cadzow_project(post[:, k], L_hat, n_iter=1)
            a, b = nmse_parts(inter, G); num_i.append(a); den.append(b)
            a2, _ = nmse_parts(post, G); num_p.append(a2)
        d = 10 * np.log10(np.asarray(num_p) / np.asarray(den)) - \
            10 * np.log10(np.asarray(num_i) / np.asarray(den))
        ci = boot_ci_median(d)
        rows.append({"snr_db": snr, "n": n_trials,
                     "seconds": round(time.time() - t0, 1),
                     "interleaved_minus_posthoc_db": float(np.median(d)),
                     "boot_ci95_median": list(ci),
                     "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0)})
        print(f"  [B5] SNR {snr:+5.1f}  interleaved-minus-posthoc "
              f"{rows[-1]['interleaved_minus_posthoc_db']:+.3f} "
              f"CI[{ci[0]:+.3f},{ci[1]:+.3f}]  {rows[-1]['seconds']:.0f}s", flush=True)
    return {"cell": "B5_posthoc_vs_interleaved_classical",
            "config": {"N": 32, "K": 3, "P": 30, "rsr_db": 12.0,
                       "n_trials": n_trials, "operator": TD_KW,
                       "note": "positive = interleaved better; shared L_hat"},
            "points": rows}


# ============================================================== B3 pencil
def run_b3_pencil(n_trials=250):
    """Vary the pencil at fixed N=32 and fixed rho. Scores P17."""
    rows = []
    for p in (4, 6, 8, 12, 16, 24):
        adm = min(32 - p, p + 1)
        num_e, num_h, den, lhat = [], [], [], []
        t0 = time.time()
        rng = np.random.default_rng(np.random.SeedSequence([620_000, p]))
        for t in range(n_trials):
            snr = float(np.round(rng.uniform(-10, 20), 3))
            w = trackd_world(620_000 + t, N=32, P=20, snr_db=snr)
            G = np.asarray(w.G_true)
            e = em_gs_only(w, TD_KW["max_iter"])
            r = hs_gs_auto(w.S, w.Z, w.B, w.sigma2, pencil=p, **TD_KW)
            a, b = nmse_parts(e, G); num_e.append(a); den.append(b)
            a2, _ = nmse_parts(r.G_hat, G); num_h.append(a2)
            lhat.append(int(r.L_hat))
        d = 10 * np.log10(np.asarray(num_e) / np.asarray(den)) - \
            10 * np.log10(np.asarray(num_h) / np.asarray(den))
        ci = boot_ci_median(d)
        rows.append({"pencil": p, "hankel_shape": [32 - p, p + 1],
                     "admissible_ranks": adm, "n": n_trials,
                     "seconds": round(time.time() - t0, 1),
                     "delta_median_db": float(np.median(d)),
                     "boot_ci95_median": list(ci),
                     "mean_L_hat": float(np.mean(lhat))})
        print(f"  [B3pencil] p={p:2d} shape={32-p:2d}x{p+1:2d} adm={adm:2d}  "
              f"delta {rows[-1]['delta_median_db']:+.3f} "
              f"CI[{ci[0]:+.3f},{ci[1]:+.3f}] Lhat {rows[-1]['mean_L_hat']:.2f} "
              f"{rows[-1]['seconds']:.0f}s", flush=True)
    return {"cell": "B3_pencil_sweep", "config": {
        "N": 32, "K": 3, "P": 20, "rsr_db": 10.0, "n_trials": n_trials,
        "snr": "U[-10,20] binned post hoc", "operator": TD_KW,
        "generator": "trackD make_world",
        "confound": "at fixed N the pencil sets admissible-rank COUNT and "
                    "maximum representable RANK together; see P17"},
            "points": rows}


# ============================================================== B2 brackets
def run_b2(n_trials=300):
    """Add high-rho cells at N=16 and N=64 to bracket the crossings."""
    # L read off the committed r_eff(L) behaviour so the target rho is hit by
    # construction: at N=16 cap=8, at N=64 cap=32.
    cells = [("N16_L9", 16, 9), ("N16_L12", 16, 12),
             ("N64_L38", 64, 38), ("N64_L50", 64, 50)]
    rows = []
    for tag, N, L in cells:
        cap = hankel_rank_cap(N)
        num_e, num_h, den, lhat = [], [], [], []
        t0 = time.time()
        rng = np.random.default_rng(np.random.SeedSequence([630_000, N, L]))
        budget = n_trials if N == 16 else 120     # N=64 is ~30x the cost
        for t in range(budget):
            snr = float(np.round(rng.uniform(-10, 20), 3))
            w = trackd_world(630_000 + 7 * t, N=N, P=20, snr_db=snr, L=L)
            G = np.asarray(w.G_true)
            e = em_gs_only(w, TD_KW["max_iter"])
            r = hs_gs_auto(w.S, w.Z, w.B, w.sigma2, **TD_KW)
            a, b = nmse_parts(e, G); num_e.append(a); den.append(b)
            a2, _ = nmse_parts(r.G_hat, G); num_h.append(a2)
            lhat.append(int(r.L_hat))
        d = 10 * np.log10(np.asarray(num_e) / np.asarray(den)) - \
            10 * np.log10(np.asarray(num_h) / np.asarray(den))
        ci = boot_ci_median(d)
        rows.append({"tag": tag, "N": N, "L": L, "cap": cap, "n": budget,
                     "seconds": round(time.time() - t0, 1),
                     "delta_median_db": float(np.median(d)),
                     "boot_ci95_median": list(ci),
                     "negative_and_significant": bool(ci[1] < 0),
                     "mean_L_hat": float(np.mean(lhat))})
        print(f"  [B2] {tag:9s} N={N:2d} L={L:2d} cap={cap:2d} n={budget:4d}  "
              f"delta {rows[-1]['delta_median_db']:+.3f} "
              f"CI[{ci[0]:+.3f},{ci[1]:+.3f}] "
              f"{'NEG-SIG' if ci[1] < 0 else ''} {rows[-1]['seconds']:.0f}s",
              flush=True)
    return {"cell": "B2_bracket_crossings",
            "config": {"K": 3, "P": 20, "rsr_db": 10.0, "operator": TD_KW,
                       "generator": "trackD make_world"},
            "points": rows}


GROUPS = {"B1": run_b1, "B5": run_b5, "B3": run_b3_pencil, "B2": run_b2}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=sorted(GROUPS))
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / f"{a.group}.json"
    if f.exists():
        print(f"skip (done): {f}")
        return 0
    print(f"=== PROMPT 12 Part B, group {a.group} ===", flush=True)
    r = GROUPS[a.group]()
    f.write_text(json.dumps(r, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

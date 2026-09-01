"""PROMPT 8 Part C -- Hankel spectral diagnostics. No training, no estimator tuning.

Pure linear algebra on generated channels, plus (for the "what the estimator
actually sees" half) the converged EM-GS estimate. Nothing here is trained and
nothing here is tuned.

Metrics, per column, on the Hankel embedding H(g) with the repository's pencil
p = N//2 (rows = N-p, cols = p+1):

  cumulative energy   E(q) = sum_{i<=q} s_i^2 / sum_i s_i^2
  effective rank      exp(-sum p_i ln p_i), p_i = s_i / sum_j s_j   (Roy-Vetterli)
  effective rank^2    the same with p_i = s_i^2 / sum_j s_j^2       (energy form)
  stable rank         ||H||_F^2 / ||H||_2^2 = sum s_i^2 / s_1^2
  tail energy         1 - E(r), the fraction destroyed by rank-r truncation

C4 -- Xiao's Saleh-Valenzuela channel, TWO readings
---------------------------------------------------
Xiao Table I gives L=4 clusters, C_l=10 rays, DoA theta_{l,c} ~ U(-pi/2, pi/2).
Subscripted (l,c), that reads as an independent draw per ray, i.e. 40
independent DoAs and no angular clustering at all. But a Saleh-Valenzuela
channel normally has rays concentrated around a cluster centre, and Table I
does not give an intra-cluster angular spread.

This is the SAME ambiguity the repository already met and resolved for Cui's
Table I, which lists "incident angles Uniform(-90,90)" and "max cluster AS
Uniform(-5,5)" -- resolved in rydberg_sim/channel_cui.py:62-67 as cluster
centres uniform with rays offset within +-5 degrees. Both readings are
reported here rather than picking one:

  "literal"   40 independent DoAs ~ U(-pi/2, pi/2)         -- pessimistic bound
  "clustered" 4 centres ~ U(-pi/2, pi/2), 10 rays each
              offset ~ U(-5, +5) degrees                    -- Cui precedent

Run:  PYTHONPATH=. python3 scratch/trackD_spectral_diagnostics.py
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from rydberg_sim.track_b_proposed import hankel_rank_cap
from rydberg_sim.track_b_structure import hankel_matrix
from trackD_urformer.baselines import run_em_gs
from trackD_urformer.config import TrackDConfig
from trackD_urformer.dataset import make_world

OUT = Path("reports/trackD_spectral_diagnostics.json")
R_GRID = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
CUI_RAY_OFFSET_DEG = 5.0     # rydberg_sim/channel_cui.py:67


def spectrum(g: np.ndarray, pencil: int | None = None) -> np.ndarray:
    return np.linalg.svd(hankel_matrix(g, pencil), compute_uv=False)


def stats_from_sv(s: np.ndarray, r_grid=R_GRID) -> dict:
    s = np.asarray(s, dtype=np.float64)
    s = s[s > 0]
    if s.size == 0:
        return {}
    e = s ** 2
    cum = np.cumsum(e) / e.sum()
    p1 = s / s.sum()
    p2 = e / e.sum()
    ent1 = -np.sum(p1 * np.log(np.maximum(p1, 1e-300)))
    ent2 = -np.sum(p2 * np.log(np.maximum(p2, 1e-300)))
    return {
        "n_sv": int(s.size),
        "cum_energy": cum.tolist(),
        "effective_rank": float(np.exp(ent1)),
        "effective_rank_energy": float(np.exp(ent2)),
        "stable_rank": float(e.sum() / e[0]),
        "tail_energy": {int(r): float(1.0 - cum[min(r, cum.size) - 1])
                        for r in r_grid if r <= s.size},
    }


def aggregate(cols) -> dict:
    """Mean cumulative-energy curve and median scalar metrics over columns."""
    per = [stats_from_sv(spectrum(c)) for c in cols]
    per = [p for p in per if p]
    n = max(p["n_sv"] for p in per)
    cum = np.zeros(n)
    for p in per:                       # pad short curves with 1.0 (all energy in)
        c = np.array(p["cum_energy"])
        cum += np.concatenate([c, np.ones(n - c.size)])
    cum /= len(per)
    rs = sorted({r for p in per for r in p["tail_energy"]})
    return {
        "n_cols": len(per),
        "cum_energy_mean": cum.tolist(),
        "effective_rank_median": float(np.median([p["effective_rank"] for p in per])),
        "effective_rank_energy_median":
            float(np.median([p["effective_rank_energy"] for p in per])),
        "stable_rank_median": float(np.median([p["stable_rank"] for p in per])),
        "tail_energy_median": {int(r): float(np.median(
            [p["tail_energy"][r] for p in per if r in p["tail_energy"]]))
            for r in rs},
    }


# --------------------------------------------------------------- generators
def true_columns(cfg, *, N, L, n_trials, snr_db=5.0, seed0=900_000):
    """Noiseless TRUE channel columns with exactly L paths per user."""
    cols = []
    for t in range(n_trials):
        s = make_world(seed0 + t, sysc=cfg.system, N=N, P=cfg.system.P,
                       snr_db=snr_db, L=L)
        cols.extend(np.asarray(s.G_true)[:, k] for k in range(s.K))
    return cols


def emgs_columns(cfg, *, N, L, n_trials, snr_db, seed0=900_000):
    """Columns of the converged EM-GS estimate -- what the estimator sees."""
    cols = []
    for t in range(n_trials):
        s = make_world(seed0 + t, sysc=cfg.system, N=N, P=cfg.system.P,
                       snr_db=snr_db, L=L)
        gh = run_em_gs(s, max_iter=100, init="spectral", seed=s.trial)
        cols.extend(np.asarray(gh)[:, k] for k in range(s.K))
    return cols


def sv_columns(N, *, n_clusters=4, rays=10, mode="clustered", n_cols=600,
               seed=20260902):
    """Xiao's Saleh-Valenzuela channel column: sum of n_clusters*rays rays."""
    rng = np.random.default_rng(seed)
    n = np.arange(N)
    out = []
    for _ in range(n_cols):
        if mode == "literal":
            th = rng.uniform(-np.pi / 2, np.pi / 2, n_clusters * rays)
        else:
            ctr = rng.uniform(-np.pi / 2, np.pi / 2, n_clusters)
            off = np.deg2rad(rng.uniform(-CUI_RAY_OFFSET_DEG,
                                         CUI_RAY_OFFSET_DEG,
                                         (n_clusters, rays)))
            th = np.clip(ctr[:, None] + off, -np.pi / 2, np.pi / 2).ravel()
        D = th.size
        a = (rng.standard_normal(D) + 1j * rng.standard_normal(D)) / np.sqrt(2 * D)
        psi = np.pi * np.sin(th)                       # repo convention
        out.append((a[None, :] * np.exp(-1j * psi[None, :] * n[:, None])).sum(1))
    return out


def main() -> int:
    cfg = TrackDConfig()
    res: dict = {"pencil_rule": "p = N//2 (repository convention)",
                 "rank_cap": {int(N): int(hankel_rank_cap(N)) for N in
                              (8, 16, 32, 64)},
                 "r_grid": list(R_GRID)}

    # ---- C1: L sweep at M=32, noiseless and on the EM-GS estimate ---------
    print("=== C1: L sweep at M=32 ===", flush=True)
    c1: dict = {"noiseless": {}, "emgs_5dB": {}}
    for L in (1, 2, 3, 5, 7, 10, 13, 16):
        c1["noiseless"][L] = aggregate(true_columns(cfg, N=32, L=L, n_trials=150))
        c1["emgs_5dB"][L] = aggregate(
            emgs_columns(cfg, N=32, L=L, n_trials=60, snr_db=5.0))
        a, b = c1["noiseless"][L], c1["emgs_5dB"][L]
        print(f"  L={L:2d}  erank true {a['effective_rank_median']:5.2f} / "
              f"est {b['effective_rank_median']:5.2f}   "
              f"srank {a['stable_rank_median']:5.2f} / {b['stable_rank_median']:5.2f}"
              f"   tail@7 {a['tail_energy_median'][7]:.4f} / "
              f"{b['tail_energy_median'][7]:.4f}", flush=True)
    res["C1_L_sweep_M32"] = c1

    # ---- C2: M sweep at fixed L, noiseless -------------------------------
    print("=== C2: M sweep at L=5, noiseless ===", flush=True)
    c2 = {}
    for N in (16, 32, 64):
        c2[N] = aggregate(true_columns(cfg, N=N, L=5, n_trials=150))
        print(f"  M={N:2d}  cap {hankel_rank_cap(N):2d}  "
              f"erank {c2[N]['effective_rank_median']:5.2f}  "
              f"srank {c2[N]['stable_rank_median']:5.2f}", flush=True)
    res["C2_M_sweep_L5"] = c2

    # ---- C3: SNR sweep on the estimate, L ~ the project default ----------
    print("=== C3: SNR sweep on the EM-GS estimate, L=5, M=32 ===", flush=True)
    c3 = {}
    for snr in (-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0):
        c3[snr] = aggregate(emgs_columns(cfg, N=32, L=5, n_trials=60,
                                         snr_db=snr))
        print(f"  SNR {snr:+6.1f}  erank {c3[snr]['effective_rank_median']:5.2f}"
              f"  srank {c3[snr]['stable_rank_median']:5.2f}"
              f"  tail@7 {c3[snr]['tail_energy_median'][7]:.4f}", flush=True)
    res["C3_SNR_sweep_estimate"] = c3

    # ---- C4: Xiao's own channel ------------------------------------------
    print("=== C4: Xiao Saleh-Valenzuela, M=32, 4 clusters x 10 rays ===",
          flush=True)
    c4 = {}
    for mode in ("clustered", "literal"):
        c4[mode] = aggregate(sv_columns(32, mode=mode, n_cols=600))
        d = c4[mode]
        best = min(d["tail_energy_median"], key=lambda r: (
            d["tail_energy_median"][r] > 0.01, r))
        print(f"  {mode:10s} erank {d['effective_rank_median']:5.2f}  "
              f"srank {d['stable_rank_median']:5.2f}  "
              f"tail@7 {d['tail_energy_median'][7]:.4f}  "
              f"tail@13 {d['tail_energy_median'][13]:.4f}  "
              f"smallest r with tail<1%: {best}", flush=True)
    res["C4_xiao_SV_M32"] = c4

    OUT.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

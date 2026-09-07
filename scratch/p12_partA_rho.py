"""PROMPT 12 Part A -- A3 (abstention vs rho) and A4 (r_eff reconciliation).

Channel generation only. No estimator is run, so this is re-analysis, not new
compute: rho is a property of the channel DISTRIBUTION at a configuration,
which is exactly how the committed R_EFF table in
scratch/trackD_partB9_analysis.py was built (median Roy-Vetterli effective
rank of the noiseless channel columns).

Run:  PYTHONPATH=. python3 scratch/p12_partA_rho.py
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from rydberg_sim.track_b_proposed import hankel_rank_cap
from trackD_urformer.config import TrackDConfig
from trackD_urformer.dataset import make_world

# Same channel generators the TD cells used.
from scratch.trackD_partB9_sweeps import sv_channel

OUT = Path("reports/p12")
N_DRAW = 400            # channel draws per configuration


def hankel(g, pencil=None):
    g = np.asarray(g).ravel()
    N = g.size
    p = pencil if pencil is not None else N // 2
    rows = N - p
    return np.lib.stride_tricks.sliding_window_view(g, p + 1)[:rows]


def eff_rank(g, pencil=None):
    """Roy-Vetterli effective rank of the Hankel lifting of one column."""
    s = np.linalg.svd(hankel(g, pencil), compute_uv=False)
    s = s[s > 0]
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(np.maximum(p, 1e-300)))))


def rho_for(*, N, K, L, channel, n_draw=N_DRAW, seed=0xA3):
    """Median r_eff over columns, and rho = r_eff / cap."""
    cfg = TrackDConfig()
    cap = hankel_rank_cap(N)
    vals = []
    rng = np.random.default_rng(seed)
    for t in range(n_draw):
        if channel.startswith("sv_"):
            G = sv_channel(N, K, rng, mode=channel[3:])
        else:
            w = make_world(seed + t, sysc=replace(cfg.system, K=K), N=N,
                           P=20, snr_db=5.0, L=L)
            G = np.asarray(w.G_true)
        for k in range(G.shape[1]):
            vals.append(eff_rank(G[:, k]))
    r = float(np.median(vals))
    return r, cap, r / cap, len(vals)


def a4_reff_L16_N32():
    """Settle 8.73 (trackD_normalization.md:55) vs 8.54
    (trackD_generalization_audit.md:233). Same definition in both; only the
    draw differs. Recompute at increasing sample size to see which converges."""
    print("=== A4: r_eff of the L=16, N=32 noiseless channel ===")
    print("  definition: median over columns of exp(-sum p_i ln p_i),")
    print("              p_i = sigma_i / sum sigma_j, on the Hankel lifting")
    out = {}
    for n in (200, 400, 1000, 2000):
        r, cap, rho, ncols = rho_for(N=32, K=3, L=16, channel="ula", n_draw=n)
        out[n] = {"r_eff": r, "cap": cap, "rho": rho, "n_cols": ncols}
        print(f"  {n:5d} trials ({ncols:5d} columns):  r_eff = {r:.3f}   "
              f"rho = {rho:.4f}")
    # Also the L=14 row, to check the whole N32_REF column reproduces.
    r14, cap14, rho14, _ = rho_for(N=32, K=3, L=14, channel="ula", n_draw=1000)
    print(f"  cross-check L=14: r_eff = {r14:.3f}  rho = {rho14:.4f}  "
          f"(committed table: 8.11 / 0.507)")
    return out


def a3_table():
    """rho and abstention for every already-run adaptive-order cell."""
    rows = []
    for f in sorted(Path("results/track_d/partB9").glob("*.json")):
        d = json.loads(f.read_text())
        lhat = np.asarray(d["L_hat"], int)
        cap = d["cap"]
        r, _, rho, _ = rho_for(N=d["N"], K=d["K"], L=d["L"],
                               channel=d.get("channel", "ula"))
        rows.append({
            "tag": d["tag"], "N": d["N"], "K": d["K"], "P": d["P"],
            "L": d["L"], "channel": d.get("channel"), "cap": cap,
            "n_trials": d["n"], "r_eff": r, "rho": rho,
            "abstention": float(np.mean(lhat >= cap)),
            "mean_L_hat": float(lhat.mean()),
        })
    return rows


def spearman(x, y):
    """Spearman rank correlation, no scipy dependency."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx @ ry) / np.sqrt((rx @ rx) * (ry @ ry)))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    a4 = a4_reff_L16_N32()

    print("\n=== A3: abstention rate as an observable proxy for rho ===")
    rows = a3_table()
    print(f"  {'tag':22s} {'N':>3s} {'cap':>4s} {'rho':>6s} {'abstain%':>9s} "
          f"{'mean_Lhat':>10s}  n")
    for r in sorted(rows, key=lambda r: r["rho"]):
        print(f"  {r['tag']:22s} {r['N']:3d} {r['cap']:4d} {r['rho']:6.3f} "
              f"{100*r['abstention']:9.1f} {r['mean_L_hat']:10.2f}  {r['n_trials']}")

    rho = [r["rho"] for r in rows]
    ab = [r["abstention"] for r in rows]
    rs = spearman(rho, ab)
    print(f"\n  Spearman rank correlation rho vs abstention: {rs:+.3f} "
          f"over {len(rows)} cells")

    # Does abstention alone separate the useful region (rho <~ 0.5)?
    useful = [r for r in rows if r["rho"] <= 0.5]
    vacuous = [r for r in rows if r["rho"] > 0.5]
    print(f"  cells with rho <= 0.5: {len(useful)}   rho > 0.5: {len(vacuous)}")
    if useful and vacuous:
        mu = max(r["abstention"] for r in useful)
        mv = min(r["abstention"] for r in vacuous)
        print(f"  max abstention among useful  : {100*mu:.1f}%")
        print(f"  min abstention among vacuous : {100*mv:.1f}%")
        print(f"  SEPARABLE by a threshold: {mu < mv}")

    (OUT / "a3_abstention_vs_rho.json").write_text(
        json.dumps({"cells": rows, "spearman": rs, "a4_reff_L16_N32": a4},
                   indent=1) + "\n", encoding="utf-8")
    print(f"\n  wrote {OUT/'a3_abstention_vs_rho.json'}")

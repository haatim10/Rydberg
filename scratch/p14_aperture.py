"""PROMPT 14 C1 -- aperture sweep at the PRINCIPAL operator. No training.

N in {8,16,32} x P in {10,30} x six SNR points x 400 paired trials, with
n_cz = 4, T = 100, selection budget 25, RSR 12 dB, Track B generator --
i.e. exactly the operator and operating point of the headline cell, so panel
(a) and panel (b) of Fig. 1 can share one configuration.

The N=32, P=30 cell is NOT recomputed: results/p12/B1.json already is that
cell at this exact operator (same 400 trials, same six SNR points), so it is
reused. Reusing it is what brings the sweep inside budget.

Scores P22 (reports/p12/PREREG_P22.md), committed before this ran.

Run one shard:
  PYTHONPATH=. python3 scratch/p14_aperture.py --shard 0 --n-shards 4
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from rydberg_sim.track_b_proposed import hankel_rank_cap, hs_gs_auto
from scratch.p12_partB import (
    TD_KW, boot_ci_median, em_gs_only, nmse_parts, trackb_world,
)

OUT = Path("results/p14/aperture")
SNRS = (-5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
NS = (8, 16, 32)
PS = (10, 30)
N_TRIALS = 400
SEED0 = 700_000


def cells():
    """All 6 (N,P) cells; the N=32,P=30 one is served from B1 instead."""
    return [(N, P) for N in NS for P in PS]


def one_cell(N, P):
    rows = []
    for si, snr in enumerate(SNRS):
        num_e, num_h, den, lhat, act = [], [], [], [], []
        t0 = time.time()
        for t in range(N_TRIALS):
            w = trackb_world(SEED0 + 10_000 * N + 1000 * P + 100 * si + t,
                             N=N, P=P, snr_db=snr, rsr_db=12.0)
            G = np.asarray(w.G)
            e = em_gs_only(w, TD_KW["max_iter"])
            r = hs_gs_auto(w.S, w.Z, w.B, w.sigma2, **TD_KW)
            a, b = nmse_parts(e, G); num_e.append(a); den.append(b)
            num_h.append(nmse_parts(r.G_hat, G)[0])
            lhat.append(int(r.L_hat))
            act.append(bool(r.L_hat < hankel_rank_cap(N)))
        den_a = np.asarray(den)
        d = 10 * np.log10(np.asarray(num_e) / den_a) - \
            10 * np.log10(np.asarray(num_h) / den_a)
        ci = boot_ci_median(d)
        em_db = float(10 * np.log10(np.sum(num_e) / np.sum(den_a)))
        hs_db = float(10 * np.log10(np.sum(num_h) / np.sum(den_a)))
        rows.append({
            "snr_db": snr, "n": N_TRIALS, "seconds": round(time.time() - t0, 1),
            "em_gs_db": em_db, "hs_gs_db": hs_db,
            "delta_ratio_of_sums_db": em_db - hs_db,
            "delta_median_db": float(np.median(d)),
            "boot_ci95_median": list(ci),
            "mean_L_hat": float(np.mean(lhat)),
            "active_frac": float(np.mean(act)),
        })
        print(f"  [ap] N={N:2d} P={P:2d} SNR {snr:+5.1f}  "
              f"ros {rows[-1]['delta_ratio_of_sums_db']:+.3f}  "
              f"med {rows[-1]['delta_median_db']:+.3f}  "
              f"act {rows[-1]['active_frac']:.3f}  "
              f"{rows[-1]['seconds']:.0f}s", flush=True)
    return {"N": N, "P": P, "cap": hankel_rank_cap(N),
            "config": {"n_cz": 4, "T": TD_KW["max_iter"],
                       "select_iter": TD_KW["select_iter"], "rsr_db": 12.0,
                       "K": 3, "n_trials": N_TRIALS,
                       "generator": "trackB_hankel_emgs/system_model"},
            "points": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    cs = cells()
    for i in range(a.shard, len(cs), a.n_shards):
        N, P = cs[i]
        f = OUT / f"N{N}_P{P}.json"
        if f.exists():
            print(f"skip (done): {f.name}", flush=True)
            continue
        if (N, P) == (32, 30):
            print("skip (served from results/p12/B1.json): N32_P30", flush=True)
            continue
        f.write_text(json.dumps(one_cell(N, P), indent=1) + "\n",
                     encoding="utf-8")
        print(f"wrote {f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

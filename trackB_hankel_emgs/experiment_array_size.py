"""Experiment B -- Hankel gain versus array size N.

Sweeps N in {8, 16, 32} across the full SNR grid, everything else fixed at the
Track-B defaults (K = 3, P = 30, RSR = 12 dB, L_k ~ U{3..7}).

The N = 8 column is the SAME store experiment A writes, and is reused rather
than recomputed. Running experiment A first therefore makes this cheaper;
running this alone also produces experiment A's data.

Reports BOTH, and never conflates them:
  * mean gain  -- unweighted mean of the per-(N,SNR) gains at each N;
  * max gain   -- the largest single operating-point gain at each N.

    python experiment_array_size.py
"""
from __future__ import annotations

import os
import sys

import config as cfg
from runner import FP, RESULTS, sweep

P = cfg.P_DEFAULT


def points(n_small: int, n_large: int):
    out = []
    for N in cfg.N_GRID:
        n_trials = {cfg.N_DEFAULT: n_small, 32: cfg.N_TRIALS_N32}.get(N, n_large)
        for snr in cfg.SNR_GRID_DB:
            out.append(dict(
                path=str(RESULTS / "grid" / f"N{N:02d}_P{P}_snr{snr:+05.1f}.npz"),
                N=N, P=P, snr_db=snr, L=None, n_trials=n_trials,
                tag=f"B N={N} SNR={snr:+.0f}"))
    # cheapest first, so partial results cover the whole N range early
    return sorted(out, key=lambda d: (d["N"], d["snr_db"]))


def main():
    n = int(os.environ.get("N_TRIALS", cfg.N_TRIALS))
    nl = int(os.environ.get("N_TRIALS_LARGE", cfg.N_TRIALS_LARGE))
    procs = int(os.environ.get("PROCS", 4))
    print(f"Experiment B: gain vs array size | N in {cfg.N_GRID}, P={P}, "
          f"K={cfg.K}, RSR={cfg.RSR_DB} dB")
    print(f"  trials/point: N={cfg.N_DEFAULT}->{n}, N=16->{nl}, N=32->{cfg.N_TRIALS_N32}; "
          f"{len(cfg.N_GRID) * len(cfg.SNR_GRID_DB)} points, fingerprint {FP}")
    print(f"  rank ceilings ceil(N/2): " +
          ", ".join(f"N={N}->{-(-N // 2)}" for N in cfg.N_GRID))
    sweep(points(n, nl), procs=procs)
    print("Experiment B done")


if __name__ == "__main__":
    sys.exit(main())

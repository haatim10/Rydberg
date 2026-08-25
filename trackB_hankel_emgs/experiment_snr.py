"""Experiment A -- NMSE vs SNR, EM-GS versus Hankel-EM-GS.

Track-B default operating point: N = 8, K = 3, P = 30, RSR = 12 dB,
L_k ~ U{3..7} drawn i.i.d. per user per realisation (the established
behaviour; L is NOT controlled here -- that is experiment C).

The stores written here are shared with experiment B, which reuses the N = 8
column rather than recomputing it: for a given (N, P, SNR, trial) the world is
bit-identical, so there is nothing to gain from a second run.

    python experiment_snr.py            # 600 trials/point, 4 processes
    N_TRIALS=100 python experiment_snr.py   # quick look
"""
from __future__ import annotations

import os
import sys

import config as cfg
from runner import FP, RESULTS, sweep

N = cfg.N_DEFAULT
P = cfg.P_DEFAULT


def points(n_trials: int):
    out = []
    for snr in cfg.SNR_GRID_DB:
        out.append(dict(
            path=str(RESULTS / "grid" / f"N{N:02d}_P{P}_snr{snr:+05.1f}.npz"),
            N=N, P=P, snr_db=snr, L=None, n_trials=n_trials,
            tag=f"A N={N} SNR={snr:+.0f}"))
    return out


def main():
    n = int(os.environ.get("N_TRIALS", cfg.N_TRIALS))
    procs = int(os.environ.get("PROCS", 4))
    print(f"Experiment A: NMSE vs SNR | N={N} P={P} K={cfg.K} RSR={cfg.RSR_DB} dB")
    print(f"  L_k ~ U{{{cfg.L_MIN}..{cfg.L_MAX}}} i.i.d.; SNR grid {cfg.SNR_GRID_DB}")
    print(f"  {n} trials/point, {len(cfg.SNR_GRID_DB)} points, fingerprint {FP}")
    sweep(points(n), procs=procs)
    print("Experiment A done")


if __name__ == "__main__":
    sys.exit(main())

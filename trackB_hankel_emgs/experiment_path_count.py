"""Experiment C -- Hankel gain versus TRUE path count L. The mechanism test.

Experiments A and B draw L_k ~ U{3..7} at random, so neither isolates the
effect of path count. Here L_k = L is FIXED and identical for all K users, and
L is swept from very sparse up to the Hankel rank ceiling.

    N = 32  ->  r_max = ceil(N/2) = 16,  so L_GRID = 2,4,...,16 reaches the
    ceiling exactly. At L = r_max the constraint "rank <= L" is satisfied by
    every vector and carries no information.

PREDICTION, RECORDED BEFORE THE RUN: if the gain comes from the low-rank
Hankel structure, then as L grows the Hankel rank grows, the prior carries
less information, and the gain should shrink -- reaching zero around
L = r_max. If instead the gain came from generic denoising it would persist at
large L, where the estimate is just as noisy.

Also records L_hat per trial so rank-selection error (L_hat - L) can be
examined; under-selection truncates genuine channel components.

    python experiment_path_count.py
"""
from __future__ import annotations

import os
import sys

import config as cfg
from runner import FP, RESULTS, sweep

N = cfg.EXP_C_N
P = cfg.P_DEFAULT
SNR = cfg.EXP_C_SNR


def points(n_trials: int):
    return [dict(path=str(RESULTS / "pathcount" / f"L{L:02d}.npz"),
                 N=N, P=P, snr_db=SNR, L=L, n_trials=n_trials,
                 tag=f"C L={L:2d}")
            for L in cfg.L_GRID]


def main():
    n = int(os.environ.get("N_TRIALS", cfg.N_TRIALS_PATH))
    procs = int(os.environ.get("PROCS", 4))
    r_max = -(-N // 2)
    print(f"Experiment C: gain vs path count | N={N} (r_max={r_max}) P={P} "
          f"SNR={SNR} dB RSR={cfg.RSR_DB} dB")
    print(f"  L FIXED and identical across users, grid {cfg.L_GRID}")
    print(f"  {n} trials/point, {len(cfg.L_GRID)} points, fingerprint {FP}")
    sweep(points(n), procs=procs)
    print("Experiment C done")


if __name__ == "__main__":
    sys.exit(main())

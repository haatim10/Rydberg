"""Track-B B4: channel NMSE vs pilot length P at N = 16.

N = 16 is fixed in advance, not chosen from the curves. It is the smallest
array in the tested set for which the Hankel structural constraint is
non-vacuous across the whole intended L_k support: the rank cap is
ceil(N/2), so cap(8) = 4 < max L_k = 7 (the constraint is vacuous for
L_k >= 5, i.e. 60% of the U{3..7} prior), while cap(16) = 8 > 7 leaves the
constraint informative for every draw. N = 32 would also qualify but is
further from the frozen baseline size.

Pilot grid and SNR are the frozen B2 values. The estimators, hyper-
parameters, world generator and per-trial storage format are identical to
B3, so the two experiments are directly comparable; the (P = 10, SNR = 5)
and (P = 30, SNR = 5) points are the *same* CRN worlds B3 already
evaluates and are reused from the B3 store rather than recomputed.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import run_b3 as rb3
from rydberg_sim.track_b_drivers import B2_P, B2_SNR_DB

B4_N = 16
B4_P = B2_P                 # (6, 10, 14, 20, 30, 40)
B4_SNR = B2_SNR_DB          # 5.0 dB

B3_STORE = REPO / "results" / "track_b" / "b3"
B4_STORE = REPO / "results" / "track_b" / "b4"


def reuse_from_b3() -> list[str]:
    """Copy the points B3 already computed. Identical worlds, so recomputing
    them would violate the no-rerun rule and waste hours."""
    done = []
    B4_STORE.mkdir(parents=True, exist_ok=True)
    for P in B4_P:
        name = f"N{B4_N}_P{P}_snr{B4_SNR:+05.1f}.npz"
        src, dst = B3_STORE / name, B4_STORE / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            done.append(name)
    return done


def main() -> None:
    import json
    import multiprocessing as mp

    rb3.OUT = B4_STORE                       # inherited by forked workers
    reused = reuse_from_b3()
    for r in reused:
        print(f"  reused from B3 (identical CRN worlds): {r}", flush=True)

    n_proc = int(os.environ.get("B4_PROCS", "4"))
    n_trials = int(os.environ.get("B4_TRIALS", str(rb3.N_TRIALS)))
    B4_STORE.mkdir(parents=True, exist_ok=True)
    (B4_STORE / "config.json").write_text(json.dumps({
        "fingerprint": rb3.FP, "N": B4_N, "P_grid": B4_P, "snr_db": B4_SNR,
        "n_trials": n_trials, "estimators": rb3.ESTIMATORS,
        "hs_kwargs": rb3.HS_KW, "gs_max_iter": rb3.GS_MAX_ITER,
        "reused_from_b3": reused,
        "why_N16": "smallest tested N whose Hankel rank cap (8) exceeds "
                   "max L_k (7), so the constraint is non-vacuous across the "
                   "whole L_k support; fixed a priori, not chosen from curves",
        "observation": "EXACT Z = |G S + B + W|, no linearization",
    }, indent=2))
    jobs = [(B4_N, P, float(B4_SNR), n_trials) for P in B4_P]
    print(f"B4 fingerprint {rb3.FP}, N={B4_N}, {n_trials} trials/point",
          flush=True)
    with mp.Pool(n_proc) as pool:
        for msg in pool.imap_unordered(rb3.run_point, jobs):
            print(" ", msg, flush=True)
    print("B4 done", flush=True)


if __name__ == "__main__":
    main()

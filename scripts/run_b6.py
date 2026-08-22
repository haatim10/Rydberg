"""Track-B B6: does the structural advantage survive at weak reference?

RSR is the atomic receiver's defining design parameter and the reason this
is *biased* phase retrieval. Every earlier Track-B experiment fixed
RSR = 12 dB, so nothing said whether the HS-GS advantage holds when the
reference is weak -- the regime where the problem is hardest and least like
conventional channel estimation.

Design (fixed before running, and not altered afterwards):
    RSR  in {0, 6, 12, 18, 24} dB
    N    in {8, 32}          -- the two ends, where the effect sign differs
    P = 30, SNR = 5 dB, K = 3, L_k ~ U{3..7}, t0 = 50
    400 trials/point, the SAME adaptive extension rule as B3
Same CRN world function, same estimators, same per-trial storage format.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import run_b3 as rb3

B6_RSR = (0.0, 6.0, 12.0, 18.0, 24.0)
B6_N = (8, 32)
B6_P = 30
B6_SNR = 5.0
B6_STORE = REPO / "results" / "track_b" / "b6"


def point_path(N, P, snr, rsr):
    return B6_STORE / f"N{N}_P{P}_snr{snr:+05.1f}_rsr{rsr:+05.1f}.npz"


def run_point(args):
    """Same body as B3's, but the world carries an explicit RSR."""
    import time

    import numpy as np
    from rydberg_sim.gs import biased_gs_channel_rows, em_gs_channel_rows
    from rydberg_sim.track_b_drivers import TRACK_B_K, track_b_world
    from rydberg_sim.track_b_proposed import hs_gs_auto

    N, P, snr, rsr, n_trials = args
    path = point_path(N, P, snr, rsr)
    rb3.OUT = B6_STORE
    d = rb3.load_point(path, TRACK_B_K)
    have = {int(t) for t in d["trial"]}
    todo = [t for t in range(n_trials) if t not in have]
    if not todo:
        return f"N={N} RSR={rsr:+.1f}  already complete ({len(have)})"

    t0 = time.time()
    buf = {k: [] for k in ("trial", "denom", "L_hat", "active", "L_true")}
    buf.update({f"num_{e}": [] for e in rb3.ESTIMATORS})

    def flush():
        if not buf["trial"]:
            return
        for k in list(d):
            new = np.asarray(buf[k], dtype=d[k].dtype).reshape(
                (len(buf[k]),) + d[k].shape[1:])
            d[k] = np.concatenate([d[k], new], axis=0)
            buf[k] = []
        rb3.save_point(path, d)

    for i, t in enumerate(todo):
        w = track_b_world(t, P, float(snr), rsr_db=float(rsr), N=N)
        G = {"biased_gs": biased_gs_channel_rows(
                 w.S, w.Z, w.B, max_iter=rb3.GS_MAX_ITER).G_hat,
             "em_gs": em_gs_channel_rows(
                 w.S, w.Z, w.B, w.sigma2, max_iter=rb3.GS_MAX_ITER).G_hat}
        r = hs_gs_auto(w.S, w.Z, w.B, w.sigma2, **rb3.HS_KW)
        assert not r.linearised_model_used
        G["hs_gs"] = r.G_hat
        buf["trial"].append(t)
        buf["denom"].append(float(np.linalg.norm(w.G, "fro") ** 2))
        buf["L_hat"].append(int(r.L_hat))
        buf["active"].append(bool(r.constraint_active))
        buf["L_true"].append([int(v) for v in w.L_k])
        for e in rb3.ESTIMATORS:
            buf[f"num_{e}"].append(float(np.sum(np.abs(G[e] - w.G) ** 2)))
        if (i + 1) % rb3.CHUNK == 0:
            flush()
    flush()
    return (f"N={N} RSR={rsr:+.1f}  +{len(todo)} trials "
            f"(total {len(d['trial'])})  {(time.time()-t0)/60:.1f} min")


def main():
    import multiprocessing as mp
    n_proc = int(os.environ.get("B6_PROCS", "4"))
    n_trials = int(os.environ.get("B6_TRIALS", "400"))
    B6_STORE.mkdir(parents=True, exist_ok=True)
    (B6_STORE / "config.json").write_text(json.dumps({
        "fingerprint": rb3.FP, "rsr_db_grid": B6_RSR, "N_grid": B6_N,
        "P": B6_P, "snr_db": B6_SNR, "n_trials": n_trials,
        "estimators": rb3.ESTIMATORS, "hs_kwargs": rb3.HS_KW,
        "gs_max_iter": rb3.GS_MAX_ITER,
        "rationale": "RSR was fixed at 12 dB in B1-B5; this asks whether the "
                     "structural advantage survives at weak reference",
        "observation": "EXACT Z = |G S + B + W|, no linearization",
    }, indent=2))
    jobs = [(N, B6_P, B6_SNR, r, n_trials) for N in B6_N for r in B6_RSR]
    jobs.sort(key=lambda j: -j[0])
    print(f"B6 fingerprint {rb3.FP}, {len(jobs)} points, {n_trials}/pt",
          flush=True)
    with mp.Pool(n_proc) as pool:
        for msg in pool.imap_unordered(run_point, jobs):
            print(" ", msg, flush=True)
    print("B6 done", flush=True)


if __name__ == "__main__":
    main()

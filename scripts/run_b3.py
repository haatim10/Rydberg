"""Track-B B3: channel NMSE vs SNR for N in {8,16,32}, P in {10,30}.

Three estimators on identical CRN worlds, all on the EXACT nonlinear
observation Z = |G S + B + W|:

    biased_gs   Cui Alg. 1 row adapter
    em_gs       Cui Alg. 2 row adapter
    hs_gs       proposed Hankel-structured exact GS (order from held-out pilots)

Per trial the error numerator ||Ghat-G||_F^2 and the denominator ||G||_F^2
are stored separately, so the pooled ratio-of-sums NMSE can be
reconstructed exactly, and so can any subset/bootstrap of it.

Checkpointing: one .npz per (N, P, SNR) point, flushed every CHUNK trials.
A rerun loads what is there and computes only the missing trial indices.
Completed trials are never recomputed.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from rydberg_sim.gs import biased_gs_channel_rows, em_gs_channel_rows
from rydberg_sim.track_b_drivers import (
    B1_SNR_DB, TRACK_B_K, TRACK_B_L_MAX, TRACK_B_L_MIN, TRACK_B_MASTER_SEED,
    TRACK_B_RSR_DB, track_b_world,
)
from rydberg_sim.track_b_proposed import hankel_rank_cap, hs_gs_auto

OUT = REPO / "results" / "track_b" / "b3"

# --- frozen B3 design -----------------------------------------------------
N_GRID = (8, 16, 32)
P_GRID = (10, 30)                 # the frozen B1 panels
SNR_GRID = B1_SNR_DB              # (-5, 0, 5, 10, 15, 20)
N_TRIALS = 400                    # starting budget; extended only where needed
ESTIMATORS = ("biased_gs", "em_gs", "hs_gs")
CHUNK = 25                        # checkpoint flush interval

# HS-GS hyperparameters: exactly those of the audited N=8/16/32 smoke run.
HS_KW = dict(exact_step="em_gs", max_iter=50, select_iter=20)
GS_MAX_ITER = 50                  # Cui t0 = 50


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fingerprint() -> str:
    """Config fingerprint. Deliberately excludes the sweep grids."""
    payload = json.dumps({
        "master_seed": TRACK_B_MASTER_SEED, "K": TRACK_B_K,
        "L_min": TRACK_B_L_MIN, "L_max": TRACK_B_L_MAX,
        "rsr_db": TRACK_B_RSR_DB, "gs_max_iter": GS_MAX_ITER,
        "hs": HS_KW, "estimators": ESTIMATORS,
        "model": "ula_geometric psi=pi sin(theta); Z=|GS+B+W| exact",
        "src": {n: _sha(REPO / "rydberg_sim" / n) for n in
                ("track_b_proposed.py", "track_b_drivers.py",
                 "track_b_structure.py", "gs.py", "monte_carlo.py")},
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


FP = fingerprint()


def point_path(N: int, P: int, snr: float) -> Path:
    return OUT / f"N{N}_P{P}_snr{snr:+05.1f}.npz"


def _blank(K: int) -> dict:
    d = {"trial": np.zeros(0, np.int64), "denom": np.zeros(0),
         "L_hat": np.zeros(0, np.int64), "active": np.zeros(0, bool),
         "L_true": np.zeros((0, K), np.int64)}
    d.update({f"num_{e}": np.zeros(0) for e in ESTIMATORS})
    return d


def load_point(path: Path, K: int) -> dict:
    if not path.exists():
        return _blank(K)
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files if k != "fingerprint"}
        fp = str(z["fingerprint"]) if "fingerprint" in z.files else ""
    if fp != FP:
        raise SystemExit(f"fingerprint mismatch in {path.name}: {fp} != {FP}")
    return d


def save_point(path: Path, d: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    order = np.argsort(d["trial"], kind="stable")
    d = {k: v[order] for k, v in d.items()}
    assert len(np.unique(d["trial"])) == len(d["trial"]), "duplicate trial index"
    tmp = path.with_name(path.stem + ".tmp.npz")   # savez appends .npz itself
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, fingerprint=np.array(FP), **d)
    tmp.replace(path)


def run_point(args) -> str:
    N, P, snr, n_trials = args
    path = point_path(N, P, snr)
    d = load_point(path, TRACK_B_K)
    have = set(int(t) for t in d["trial"])
    todo = [t for t in range(n_trials) if t not in have]
    if not todo:
        return f"N={N} P={P} SNR={snr:+.1f}  already complete ({len(have)})"

    t_start = time.time()
    buf = {k: [] for k in ("trial", "denom", "L_hat", "active", "L_true")}
    buf.update({f"num_{e}": [] for e in ESTIMATORS})

    def flush():
        if not buf["trial"]:
            return
        for k in list(d):
            new = np.asarray(buf[k], dtype=d[k].dtype).reshape(
                (len(buf[k]),) + d[k].shape[1:])
            d[k] = np.concatenate([d[k], new], axis=0)
            buf[k] = []
        save_point(path, d)

    for i, t in enumerate(todo):
        w = track_b_world(t, P, float(snr), N=N)          # identical CRN world
        G_hat = {
            "biased_gs": biased_gs_channel_rows(
                w.S, w.Z, w.B, max_iter=GS_MAX_ITER).G_hat,
            "em_gs": em_gs_channel_rows(
                w.S, w.Z, w.B, w.sigma2, max_iter=GS_MAX_ITER).G_hat,
        }
        r = hs_gs_auto(w.S, w.Z, w.B, w.sigma2, **HS_KW)
        assert not r.linearised_model_used
        G_hat["hs_gs"] = r.G_hat
        buf["trial"].append(t)
        buf["denom"].append(float(np.linalg.norm(w.G, "fro") ** 2))
        buf["L_hat"].append(int(r.L_hat))
        buf["active"].append(bool(r.constraint_active))
        buf["L_true"].append([int(v) for v in w.L_k])
        for e in ESTIMATORS:
            buf[f"num_{e}"].append(float(np.sum(np.abs(G_hat[e] - w.G) ** 2)))
        if (i + 1) % CHUNK == 0:
            flush()
    flush()
    dt = time.time() - t_start
    return (f"N={N} P={P} SNR={snr:+.1f}  +{len(todo)} trials "
            f"(total {len(d['trial'])})  {dt/60:.1f} min")


def jobs(n_trials=N_TRIALS):
    # heaviest first so the pool drains evenly
    js = [(N, P, s, n_trials) for N in N_GRID for P in P_GRID for s in SNR_GRID]
    return sorted(js, key=lambda j: -(j[0] * j[1]))


def main() -> None:
    import multiprocessing as mp
    n_proc = int(os.environ.get("B3_PROCS", "4"))
    n_trials = int(os.environ.get("B3_TRIALS", str(N_TRIALS)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "config.json").write_text(json.dumps({
        "fingerprint": FP, "N_grid": N_GRID, "P_grid": P_GRID,
        "snr_db_grid": SNR_GRID, "n_trials": n_trials,
        "estimators": ESTIMATORS, "hs_kwargs": HS_KW,
        "gs_max_iter": GS_MAX_ITER, "rsr_db": TRACK_B_RSR_DB,
        "master_seed": TRACK_B_MASTER_SEED, "K": TRACK_B_K,
        "L_k": f"U{{{TRACK_B_L_MIN}..{TRACK_B_L_MAX}}} iid per user per trial",
        "rank_cap": {str(N): hankel_rank_cap(N) for N in N_GRID},
        "observation": "EXACT Z = |G S + B + W|, no linearization",
    }, indent=2))
    print(f"B3 fingerprint {FP}, {n_trials} trials/point, {n_proc} procs",
          flush=True)
    with mp.Pool(n_proc) as pool:
        for msg in pool.imap_unordered(run_point, jobs(n_trials)):
            print(" ", msg, flush=True)
    print("B3 done", flush=True)


if __name__ == "__main__":
    main()

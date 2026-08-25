"""Shared Monte Carlo driver for experiments A, B and C.

One paired trial = one frozen world handed to BOTH estimators. Per trial we
store the NMSE numerator for each estimator and the shared denominator
separately, so any pooling (ratio-of-sums, mean-of-ratios, median, bootstrap
over any subset) can be reconstructed later without rerunning anything.

Checkpointing: one .npz per operating point, flushed every CHUNK trials. A
rerun loads what exists and computes only the missing trial indices, so the
sweep is resumable and idempotent.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

import config as cfg
import em_gs
import hankel_em_gs as hem
from system_model import channel_nmse_parts, make_world

RESULTS = Path(__file__).resolve().parent / "results"

FIELDS = {
    "trial": np.int64, "denom": np.float64,
    "num_em_gs": np.float64, "num_hankel_em_gs": np.float64,
    "L_hat": np.int64, "active": np.bool_, "r_max": np.int64,
    "L_true_sum": np.int64, "paired_ok": np.bool_,
}


def fingerprint() -> str:
    """Fingerprint of everything that can change a stored NUMBER.

    Deliberately covers the model and estimator parameters plus the source of
    the algorithm modules -- but NOT config.py itself and NOT the trial counts.
    Trial counts change how many results a store holds, never their values, and
    stores at different budgets are poolable. Hashing config.py wholesale would
    invalidate valid stores every time a comment or a budget was edited.
    """
    src = {n: hashlib.sha256((Path(__file__).parent / n).read_bytes()).hexdigest()[:16]
           for n in ("system_model.py", "em_gs.py", "hankel_projection.py",
                     "hankel_em_gs.py")}
    payload = json.dumps({
        "seed": cfg.MASTER_SEED, "K": cfg.K, "L": [cfg.L_MIN, cfg.L_MAX],
        "rsr": cfg.RSR_DB, "iters": cfg.GS_MAX_ITER, "step": cfg.EXACT_STEP,
        "cadzow": cfg.CADZOW_ITER, "project_every": cfg.PROJECT_EVERY,
        "select_iter": cfg.SELECT_ITER, "val_frac": cfg.VAL_FRAC,
        "ridge": cfg.RIDGE,
        "model": "ula_geometric psi=pi sin(theta); Z=|GS+B+W| exact",
        "src": src,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


FP = fingerprint()


def _load(path: Path) -> dict:
    if path.exists():
        d = np.load(path)
        if str(d["fingerprint"]) != FP:
            raise SystemExit(
                f"{path.name}: fingerprint {d['fingerprint']} != {FP}. The code or "
                f"config changed since this store was written. Delete it and rerun "
                f"rather than mixing incompatible results.")
        return {k: d[k] for k in FIELDS}
    return {k: np.empty((0,), dtype=t) for k, t in FIELDS.items()}


def _save(path: Path, d: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, fingerprint=np.array(FP), **d)
    tmp.replace(path)


def run_point(path: Path, *, N: int, P: int, snr_db: float, L: int | None,
              n_trials: int, tag: str) -> str:
    """Run (or resume) one operating point. Returns a one-line status."""
    d = _load(path)
    have = {int(t) for t in d["trial"]}
    todo = [t for t in range(n_trials) if t not in have]
    if not todo:
        return f"{tag}: complete ({len(have)} trials)"

    t0 = time.time()
    buf = {k: [] for k in FIELDS}

    def flush():
        if not buf["trial"]:
            return
        for k, dt in FIELDS.items():
            d[k] = np.concatenate([d[k], np.asarray(buf[k], dtype=dt)])
            buf[k] = []
        _save(path, d)

    for i, t in enumerate(todo):
        w = make_world(t, N=N, P=P, snr_db=snr_db, L=L)

        # --- both estimators receive the SAME frozen world object ----------
        G_base = em_gs.em_gs(w.S, w.Z, w.B, w.sigma2, max_iter=cfg.GS_MAX_ITER)
        res = hem.hankel_em_gs(w.S, w.Z, w.B, w.sigma2, max_iter=cfg.GS_MAX_ITER)

        # pairing guard: the world must not have been mutated by either run
        paired_ok = bool(
            np.array_equal(w.Z, np.abs(w.G @ w.S + w.B + w.W))
            and np.isfinite(G_base).all() and np.isfinite(res.G_hat).all()
        )

        e_base, den = channel_nmse_parts(G_base, w.G)
        e_hank, _ = channel_nmse_parts(res.G_hat, w.G)

        buf["trial"].append(t)
        buf["denom"].append(den)
        buf["num_em_gs"].append(e_base)
        buf["num_hankel_em_gs"].append(e_hank)
        buf["L_hat"].append(res.L_hat)
        buf["active"].append(res.active)
        buf["r_max"].append(res.r_max)
        buf["L_true_sum"].append(int(np.sum(w.L_k)))
        buf["paired_ok"].append(paired_ok)

        if (i + 1) % cfg.CHUNK == 0:
            flush()
    flush()
    return f"{tag}: +{len(todo)} trials ({(time.time() - t0) / 60:.1f} min)"


def sweep(points, procs: int = 4) -> None:
    """Run a list of point dicts in parallel."""
    import multiprocessing as mp
    with mp.Pool(int(procs)) as pool:
        for msg in pool.imap_unordered(_run_one, points):
            print("  " + msg, flush=True)


def _run_one(pt: dict) -> str:
    return run_point(Path(pt["path"]), N=pt["N"], P=pt["P"],
                     snr_db=pt["snr_db"], L=pt.get("L"),
                     n_trials=pt["n_trials"], tag=pt["tag"])


__all__ = ["FP", "FIELDS", "run_point", "sweep", "RESULTS"]

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


#: Wrapper modules in this package.
_LOCAL_SRC = ("system_model.py", "em_gs.py", "hankel_projection.py",
              "hankel_em_gs.py")
#: Load-bearing implementation modules. The wrappers above are thin: every
#: number actually comes from here -- channel generator, pilots, reference,
#: forward model, EM-GS solver, Hankel lifting, Cadzow, rank selector, RNG and
#: the Monte-Carlo driver. Hashing only the wrappers (the pre-audit behaviour)
#: meant a change to Cadzow or to the rank selector left every stored result
#: looking valid. See scripts/audit_verify.py and AUDIT.md.
_IMPL_SRC = ("gs.py", "channel.py", "pilots.py", "reference.py", "forward.py",
             "monte_carlo.py", "spectral.py", "rng.py", "config.py",
             "baselines.py", "metrics.py", "track_b_structure.py",
             "track_b_proposed.py", "track_b_drivers.py")


def fingerprint() -> str:
    """Fingerprint of everything that can change a stored NUMBER.

    Covers the model and estimator parameters, the wrapper sources, AND the
    implementation modules they delegate to -- but NOT this package's
    config.py trial counts. Trial counts change how many results a store
    holds, never their values, and stores at different budgets are poolable.
    """
    here = Path(__file__).resolve().parent
    impl_dir = here.parent / "rydberg_sim"
    src = {n: hashlib.sha256((here / n).read_bytes()).hexdigest()[:16]
           for n in _LOCAL_SRC}
    src.update({
        f"rydberg_sim/{n}": hashlib.sha256(
            (impl_dir / n).read_bytes()).hexdigest()[:16]
        for n in _IMPL_SRC if (impl_dir / n).exists()
    })
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


def meta_of(*, N: int, P: int, snr_db: float, rsr_db: float,
            L: int | None) -> dict:
    """Operating-point identity stored INSIDE each result file.

    The filename alone is not evidence: a store can be copied, renamed, or
    written by a sweep whose point definition later changed. Recording the
    point and checking it on resume makes a mismatch loud instead of silent.
    """
    return {"N": int(N), "P": int(P), "snr_db": float(snr_db),
            "rsr_db": float(rsr_db), "L": (-1 if L is None else int(L))}


def _load(path: Path, meta: dict | None = None) -> dict:
    if path.exists():
        d = np.load(path)
        if str(d["fingerprint"]) != FP:
            raise SystemExit(
                f"{path.name}: fingerprint {d['fingerprint']} != {FP}. The code or "
                f"config changed since this store was written. Delete it and rerun "
                f"rather than mixing incompatible results.")
        if meta is not None:
            if "meta" not in d.files:
                raise SystemExit(
                    f"{path.name}: store predates operating-point metadata. "
                    f"Re-stamp it with migrate_stores.py (which re-verifies the "
                    f"trials) or delete it and rerun.")
            stored = json.loads(str(d["meta"]))
            if stored != meta:
                raise SystemExit(
                    f"{path.name}: stored operating point {stored} does not match "
                    f"the requested {meta}. Refusing to append trials from a "
                    f"different point into one file.")
        return {k: d[k] for k in FIELDS}
    return {k: np.empty((0,), dtype=t) for k, t in FIELDS.items()}


def _save(path: Path, d: dict, meta: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    extra = {} if meta is None else {"meta": np.array(json.dumps(meta, sort_keys=True))}
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, fingerprint=np.array(FP), **extra, **d)
    tmp.replace(path)


def run_point(path: Path, *, N: int, P: int, snr_db: float, L: int | None,
              n_trials: int, tag: str, rsr_db: float = cfg.RSR_DB) -> str:
    """Run (or resume) one operating point. Returns a one-line status."""
    meta = meta_of(N=N, P=P, snr_db=snr_db, rsr_db=rsr_db, L=L)
    d = _load(path, meta)
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
        _save(path, d, meta)

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

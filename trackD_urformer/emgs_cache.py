"""Cached converged EM-GS estimates - the input features for arm X1.

X1 is "EM-GS (converged, ``T_GS=100``) followed by a SINGLE Transformer
post-processor". The EM-GS stage is a deterministic function of
``(S, Z, B, sigma2, seed)``, none of which change across epochs, so running it
inside the training loop would repeat the same 318 ms per sample thirteen
times. It is computed ONCE here and memory-mapped during training.

Cost, measured on this container: 317.9 ms/sample single-threaded, so the 80k
train split is 7.1 core-hours. Sharded across workers it is ~2.5 h wall on 3
cores, which is why this is a separate entry point rather than a step inside
:mod:`stage3` -- it runs alongside H1 training rather than in front of it.

**This is not a shortcut and not an approximation.** ``test_emgs_cache_exact``
pins cached == freshly computed. The same argument already justified the
``g0`` memoization in :class:`dataset.TrackDDataset`.

Note on the honest cost of X1
-----------------------------
PROMPT 6 budgets X1 at "~30 min". The *training* is indeed ~30 min -- one
158k-parameter Transformer, no unrolling. The 7 core-hours of EM-GS in front
of it are real compute that the URformer does not pay, and the report says so
rather than quoting the 30 minutes alone.

Run one shard::

    PYTHONPATH=. python3 -m trackD_urformer.emgs_cache --split train \
        --n 80000 --shard 0 --n-shards 3
"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from .baselines import run_em_gs
from .config import TrackDConfig
from .dataset import TrackDDataset

__all__ = ["cache_path", "build_shard", "load_cache", "emgs_estimate"]

CACHE = Path("results") / "track_d" / "emgs_cache"


def cache_path(split: str, n: int, T_GS: int, shard: int | None = None,
               n_shards: int = 1) -> Path:
    """One file per shard; shards are concatenated by :func:`load_cache`.

    ``n`` and ``T_GS`` are in the filename so a cache built for a different
    budget or a different iteration count can never be silently reused.
    """
    stem = f"{split}_n{n}_T{T_GS}"
    if shard is None:
        return CACHE / f"{stem}.npy"
    return CACHE / f"{stem}_shard{shard}of{n_shards}.npy"


def emgs_estimate(sample, *, T_GS: int) -> np.ndarray:
    """Converged EM-GS from the spectral initializer. The X1 front end.

    Identical call to the ``em_gs_spectral`` baseline that stages 1 and 2
    report, so X1's input is exactly the arm it is being compared against.
    """
    return run_em_gs(sample, max_iter=T_GS, init="spectral", seed=sample.trial)


def build_shard(split: str, n: int, *, shard: int, n_shards: int,
                T_GS: int = 100, cfg: TrackDConfig | None = None,
                report_every: int = 250) -> Path:
    """Compute EM-GS for this worker's stride of the split and save it.

    Shards are strided (``idx % n_shards == shard``) rather than blocked, so
    every worker sees the same mix of SNRs and finishes at the same time.
    Resumable: a completed shard file is not recomputed.
    """
    cfg = cfg or TrackDConfig()
    cfg = replace(cfg, data=replace(cfg.data, **{f"n_{split}": n}))
    out = cache_path(split, n, T_GS, shard, n_shards)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"[{split} shard {shard}] already complete: {out}")
        return out

    # cache=False: each sample is visited exactly once here, so holding 80k
    # worlds in memory (~1.4 GB) would buy nothing.
    ds = TrackDDataset(split, sysc=cfg.system, datac=cfg.data,
                       numeric=cfg.numeric, cache=False)
    idx = np.arange(shard, n, n_shards)
    buf = np.empty((idx.size, cfg.system.N, cfg.system.K), dtype=np.complex128)

    t0 = time.time()
    for j, i in enumerate(idx):
        buf[j] = emgs_estimate(ds.sample(int(i)), T_GS=T_GS)
        if report_every and (j + 1) % report_every == 0:
            el = time.time() - t0
            print(f"[{split} shard {shard}] {j+1}/{idx.size}  "
                  f"{el/60:.1f} min elapsed, "
                  f"{el/(j+1)*(idx.size-j-1)/60:.1f} min left", flush=True)

    tmp = out.with_suffix(".partial.npy")
    np.save(tmp, buf)
    tmp.rename(out)          # atomic: a shard file exists only when complete
    print(f"[{split} shard {shard}] wrote {out} in {(time.time()-t0)/60:.1f} min")
    return out


def load_cache(split: str, n: int, T_GS: int = 100, n_shards: int = 1
               ) -> np.ndarray:
    """Reassemble the strided shards into index order ``0..n-1``."""
    whole = cache_path(split, n, T_GS)
    if whole.exists():
        arr = np.load(whole)
        if arr.shape[0] != n:
            raise ValueError(f"{whole} holds {arr.shape[0]} rows, expected {n}")
        return arr

    parts = [cache_path(split, n, T_GS, s, n_shards) for s in range(n_shards)]
    missing = [p for p in parts if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"EM-GS cache incomplete for {split}/n={n}: missing {missing}. "
            f"Build with: python3 -m trackD_urformer.emgs_cache --split {split} "
            f"--n {n} --shard <s> --n-shards {n_shards}")

    N_, K_ = np.load(parts[0], mmap_mode="r").shape[1:]
    out = np.empty((n, N_, K_), dtype=np.complex128)
    for s, p in enumerate(parts):
        out[np.arange(s, n, n_shards)] = np.load(p)
    np.save(whole, out)      # collapse to one file so later loads are one read
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the EM-GS feature cache")
    ap.add_argument("--split", required=True, choices=("train", "val", "test"))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--T-GS", type=int, default=100)
    a = ap.parse_args(argv)
    build_shard(a.split, a.n, shard=a.shard, n_shards=a.n_shards, T_GS=a.T_GS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

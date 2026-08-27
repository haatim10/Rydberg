"""Track D dataset - a thin wrapper over the repository's validated generator.

**No physics is reimplemented here.** Every realization comes from
``rydberg_sim.monte_carlo.generate_channel_estimation_trial``, with the path
counts drawn by ``rydberg_sim.track_b_drivers.draw_L_k`` so the channel
distribution is bit-identical to Track B's (``L_k ~ U{3..7}`` i.i.d. per user).

What this module adds, and only this:

* disjoint train / validation / test trial-index ranges (no leakage),
* a ``fixed_S`` mode that reuses one pilot matrix across the dataset,
* per-sample SNR sampling,
* conversion to batched torch tensors.

Track B is untouched: ``trackB_hankel_emgs`` is not importable as a package
(it does ``import config``, expecting its own cwd), so Track D reuses the
underlying ``rydberg_sim.track_b_drivers`` helpers directly.

fixed_S and the noise stream
----------------------------
``generate_channel_estimation_trial`` draws a fresh ``S`` per trial. For
``fixed_S`` we re-derive the trial's own noise generator from the same stable
key and re-run the repository's ``exact_forward`` with the shared pilot matrix.
Because ``get_operating_point_rngs`` is a pure function of
``(master_seed, trial, snr_db, rsr_db)``, the re-derived noise stream yields a
**bit-identical** ``W``; only ``S`` (and hence ``Z``) changes. This reuses the
repository's forward model rather than recomputing ``Z`` by hand.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from rydberg_sim.forward import exact_forward
from rydberg_sim.monte_carlo import generate_channel_estimation_trial
from rydberg_sim.pilots import generate_gaussian_pilots
from rydberg_sim.rng import get_operating_point_rngs
from rydberg_sim.track_b_drivers import draw_L_k, track_b_spec

from .config import DataConfig, NumericConfig, SystemConfig

__all__ = ["TrackDSample", "make_world", "make_fixed_pilots", "TrackDDataset",
           "collate", "SPLITS"]

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class TrackDSample:
    """One realization. Only fields that cannot be reconstructed cheaply."""

    Z: np.ndarray          # (N, P) float64
    S: np.ndarray          # (K, P) complex128
    B: np.ndarray          # (N, P) complex128
    sigma2: float
    G_true: np.ndarray     # (N, K) complex128
    trial: int
    snr_db: float
    rsr_db: float
    N: int
    K: int
    P: int
    L_k: tuple[int, ...]


def make_fixed_pilots(K: int, P: int, seed: int) -> np.ndarray:
    """One pilot matrix reused across a whole ``fixed_S`` dataset.

    Uses the repository's generator, so the distribution and the full-row-rank
    rejection sampling are exactly those of the per-trial path.
    """
    rng = np.random.default_rng(np.random.SeedSequence([int(seed)]))
    return np.asarray(generate_gaussian_pilots(K=K, P=P, rng=rng).S)


def make_world(
    trial: int,
    *,
    sysc: SystemConfig,
    N: int,
    P: int,
    snr_db: float,
    rsr_db: float | None = None,
    S_fixed: np.ndarray | None = None,
    L: int | None = None,
):
    """One frozen trial world from the repository generator.

    Parameters
    ----------
    L
        If given, all ``K`` users get exactly this many paths. If ``None``
        (the default and the Track-B convention), ``L_k ~ U{L_min..L_max}``
        i.i.d. per user.
    S_fixed
        If given, the world's per-trial pilots are replaced by this matrix and
        ``Z`` is recomputed through ``exact_forward`` with the trial's own
        (bit-identical) noise draw.
    """
    rsr = sysc.rsr_db if rsr_db is None else float(rsr_db)
    if L is None:
        L_k = draw_L_k(trial, sysc.K, master_seed=sysc.master_seed,
                       L_min=sysc.L_min, L_max=sysc.L_max)
    else:
        L_k = (int(L),) * sysc.K

    spec = track_b_spec(
        P=P, n_trials=trial + 1, N=N, K=sysc.K, L=L_k,
        master_seed=sysc.master_seed, experiment="trackD_urformer",
    )
    world = generate_channel_estimation_trial(spec, trial, float(snr_db), rsr)

    if S_fixed is None:
        return TrackDSample(
            Z=np.asarray(world.Z), S=np.asarray(world.S), B=np.asarray(world.B),
            sigma2=float(world.sigma2), G_true=np.asarray(world.G),
            trial=trial, snr_db=float(snr_db), rsr_db=rsr,
            N=N, K=sysc.K, P=P, L_k=L_k,
        )

    if S_fixed.shape != (sysc.K, P):
        raise ValueError(
            f"S_fixed.shape={S_fixed.shape}, expected {(sysc.K, P)}"
        )
    # Re-derive the SAME noise stream from the same stable key, then re-run the
    # repository's forward model with the shared pilots. W is bit-identical.
    rngs = get_operating_point_rngs(sysc.master_seed, trial, float(snr_db), rsr)
    exact = exact_forward(
        world.G, S_fixed, world.B, float(world.sigma2), rng_noise=rngs.noise
    )
    return TrackDSample(
        Z=np.asarray(exact.Z), S=np.asarray(S_fixed), B=np.asarray(world.B),
        sigma2=float(world.sigma2), G_true=np.asarray(world.G),
        trial=trial, snr_db=float(snr_db), rsr_db=rsr,
        N=N, K=sysc.K, P=P, L_k=L_k,
    )


class TrackDDataset(Dataset):
    """Deterministic, index-addressable dataset over a disjoint seed range.

    Realizations are generated lazily and cached in memory, so two epochs see
    identical data and no leakage is possible across splits.
    """

    def __init__(
        self,
        split: str,
        *,
        sysc: SystemConfig,
        datac: DataConfig,
        numeric: NumericConfig,
        N: int | None = None,
        P: int | None = None,
        snr_db: float | None = None,
        cache: bool = True,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        self.split = split
        self.sysc = sysc
        self.datac = datac
        self.numeric = numeric
        self.N = int(sysc.N if N is None else N)
        self.P = int(sysc.P if P is None else P)
        self.snr_override = snr_db

        self.n_items = {"train": datac.n_train, "val": datac.n_val,
                        "test": datac.n_test}[split]
        self.seed_lo, self.seed_hi = {
            "train": datac.train_seed_range,
            "val": datac.val_seed_range,
            "test": datac.test_seed_range,
        }[split]

        self.S_fixed = (
            make_fixed_pilots(sysc.K, self.P, datac.fixed_S_seed)
            if datac.pilot_mode == "fixed_S" else None
        )
        self._cache: dict[int, TrackDSample] = {} if cache else None  # type: ignore

    def __len__(self) -> int:
        return self.n_items

    def _snr_for(self, trial: int) -> float:
        if self.snr_override is not None:
            return float(self.snr_override)
        if self.datac.snr_mode == "snr_fixed":
            return float(self.datac.snr_fixed_db)
        lo, hi = self.datac.snr_range_db
        # Own substream, keyed by trial: independent of channel/pilot/noise.
        rng = np.random.default_rng(
            np.random.SeedSequence([int(self.sysc.master_seed), int(trial), 0x534E52])
        )
        # Quantize to the millidB grid: rydberg_sim.rng.db_to_key encodes the
        # operating point as an integer number of millidB and rejects anything
        # off that grid. The repository's convention wins over a continuous
        # draw; 0.001 dB resolution is far finer than any effect we measure.
        return float(np.round(rng.uniform(lo, hi), 3))

    def sample(self, idx: int) -> TrackDSample:
        if idx < 0 or idx >= self.n_items:
            raise IndexError(f"index {idx} out of range for {self.n_items} items")
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]
        trial = self.seed_lo + idx
        s = make_world(
            trial, sysc=self.sysc, N=self.N, P=self.P,
            snr_db=self._snr_for(trial), S_fixed=self.S_fixed,
        )
        if self._cache is not None:
            self._cache[idx] = s
        return s

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.sample(idx)
        cd, rd = self.numeric.complex_dtype, self.numeric.real_dtype
        # np.array(copy=True): the repository freezes its arrays read-only, and
        # torch refuses to wrap a non-writable buffer without warning.
        def _t(a, dtype):
            return torch.as_tensor(np.array(a, copy=True), dtype=dtype)
        return {
            "Z": _t(s.Z, rd),
            "S": _t(s.S, cd),
            "B": _t(s.B, cd),
            "sigma2": torch.as_tensor(s.sigma2, dtype=rd),
            "G_true": _t(s.G_true, cd),
            "snr_db": torch.as_tensor(s.snr_db, dtype=rd),
            "trial": torch.as_tensor(s.trial, dtype=torch.int64),
        }


def collate(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack a list of samples into a batch. All shapes must agree."""
    return {k: torch.stack([it[k] for it in items]) for k in items[0]}

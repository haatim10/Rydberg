"""Frozen simulation configuration (Step 1).

Only parameters required by the geometric ULA channel generator and the
per-trial RNG policy are included. Later-stage quantities (pilots, SNR,
RSR, QAM, ...) are intentionally omitted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Union

import numpy as np

IntLike = Union[int, np.integer]
FloatLike = Union[float, np.floating]
IntVec = Union[IntLike, Sequence[IntLike], np.ndarray]
FloatVec = Union[FloatLike, Sequence[FloatLike], np.ndarray]


def _reject_bool(value: object, name: str) -> None:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must not be a boolean, got {value!r}")


def _as_tuple_int(value: IntVec, K: int, name: str) -> tuple[int, ...]:
    _reject_bool(value, name)
    if isinstance(value, (int, np.integer)):
        return (int(value),) * K
    values = tuple(int(v) for v in np.asarray(value).reshape(-1))
    if len(values) != K:
        raise ValueError(
            f"{name} must be a scalar or a length-K sequence "
            f"(K={K}), got length {len(values)}"
        )
    return values


def _as_tuple_float(value: FloatVec, K: int, name: str) -> tuple[float, ...]:
    _reject_bool(value, name)
    if isinstance(value, (int, np.integer, float, np.floating)):
        return (float(value),) * K
    values = tuple(float(v) for v in np.asarray(value, dtype=np.float64).reshape(-1))
    if len(values) != K:
        raise ValueError(
            f"{name} must be a scalar or a length-K sequence "
            f"(K={K}), got length {len(values)}"
        )
    return values


@dataclass(frozen=True)
class SimulationConfig:
    """Immutable simulation parameters for the ULA channel model.

    Parameters
    ----------
    N
        Number of receive ULA elements. Must be ``> 0``.
    K
        Number of users. Must be ``> 0``.
    L_k
        Paths per user. A scalar is expanded to all ``K`` users; otherwise a
        length-``K`` sequence. Each entry must satisfy ``1 <= L_k <= N``.
    beta_k
        Large-scale channel power per user. A scalar is expanded to all
        ``K`` users; otherwise a length-``K`` sequence. Each entry must be
        ``> 0``.
    master_seed
        Seed used with ``trial_index`` to construct independent per-trial
        RNG streams.
    c
        Common known polarization/conversion scalar applied as ``G = c H``.
        The default ``c = 1.0`` is a **numerical normalization** for
        simulations, not a claim that the physical atomic conversion gain
        equals 1.

    Notes
    -----
    Use :meth:`create` to pass the aliases ``L`` / ``beta`` in place of
    ``L_k`` / ``beta_k``.
    """

    N: int
    K: int
    L_k: tuple[int, ...]
    beta_k: tuple[float, ...]
    master_seed: int
    c: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.N, (bool, np.bool_)) or int(self.N) != self.N:
            raise TypeError(f"N must be an integer, got {self.N!r}")
        if isinstance(self.K, (bool, np.bool_)) or int(self.K) != self.K:
            raise TypeError(f"K must be an integer, got {self.K!r}")
        object.__setattr__(self, "N", int(self.N))
        object.__setattr__(self, "K", int(self.K))
        object.__setattr__(self, "master_seed", int(self.master_seed))
        object.__setattr__(self, "c", float(self.c))

        if self.N <= 0:
            raise ValueError(f"N must be > 0, got {self.N}")
        if self.K <= 0:
            raise ValueError(f"K must be > 0, got {self.K}")
        if not np.isfinite(self.c):
            raise ValueError(f"c must be finite, got {self.c}")

        L_k = _as_tuple_int(self.L_k, self.K, "L_k")
        beta_k = _as_tuple_float(self.beta_k, self.K, "beta_k")
        object.__setattr__(self, "L_k", L_k)
        object.__setattr__(self, "beta_k", beta_k)

        for k, Lk in enumerate(L_k):
            if not 1 <= Lk <= self.N:
                raise ValueError(
                    f"L_k[{k}]={Lk} must satisfy 1 <= L_k <= N={self.N}"
                )
        for k, bk in enumerate(beta_k):
            if not np.isfinite(bk) or bk <= 0.0:
                raise ValueError(f"beta_k[{k}]={bk} must be finite and > 0")

    @classmethod
    def create(
        cls,
        *,
        N: int,
        K: int,
        master_seed: int,
        L: IntVec | None = None,
        L_k: IntVec | None = None,
        beta: FloatVec | None = None,
        beta_k: FloatVec | None = None,
        c: float = 1.0,
    ) -> "SimulationConfig":
        """Build a config, accepting scalar or per-user ``L`` / ``beta``.

        Provide exactly one of ``L`` or ``L_k``, and exactly one of
        ``beta`` or ``beta_k``. Scalars are expanded across all users.
        """
        if (L is None) == (L_k is None):
            raise ValueError("provide exactly one of L or L_k")
        if (beta is None) == (beta_k is None):
            raise ValueError("provide exactly one of beta or beta_k")
        return cls(
            N=N,
            K=K,
            L_k=L if L is not None else L_k,  # type: ignore[arg-type]
            beta_k=beta if beta is not None else beta_k,  # type: ignore[arg-type]
            master_seed=master_seed,
            c=c,
        )

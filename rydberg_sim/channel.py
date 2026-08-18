"""Geometric ULA channel generator from SystemModel.pdf (Step 2).

This is **not** Cui's 3GPP TR 38.901 generator. There is no random
per-element polarization. The polarization/conversion factor is a
common known scalar ``c``. For normalized simulations ``c = 1``, which
is a numerical convention, not a physical claim that the atomic
conversion gain equals 1.

Model
-----
For user ``k`` and path ``ell``:

    theta_{ell,k} ~ Uniform[-pi/2, pi/2]

Because ``d = lambda/2``:

    psi_{ell,k} = pi * sin(theta_{ell,k})

The ULA steering vector is

    a(theta) = [1, exp(-j psi), exp(-j 2 psi), ..., exp(-j (N-1) psi)]^T

so every entry has magnitude 1 and ``||a(theta)||_2^2 = N``.

Path gains:

    alpha_{ell,k} ~ CN(0, beta_k / L_k)

i.e. real and imaginary parts are independent N(0, (beta_k / L_k) / 2),
which gives ``E[|alpha|^2] = beta_k / L_k``.

With ``A_k = [a(theta_1k), ..., a(theta_{L_k,k})]`` of shape ``(N, L_k)``
and ``alpha_k`` of shape ``(L_k,)``:

    h_k = A_k @ alpha_k
    H   = [h_1, ..., h_K]          # shape (N, K)
    G   = c * H

Rank / degeneracy handling
--------------------------
Config already enforces ``L_k <= N``. For distinct spatial frequencies
the Vandermonde structure of ``A_k`` has full column rank ``L_k``.
Exact angle collisions have probability zero under the continuous
Uniform[-pi/2, pi/2] law, including the identification
``a(pi/2) = a(-pi/2)`` (because ``psi = ±pi`` yield the same steering
vector).

Pathological near-collisions are rejected and redrawn using **two**
explicit numerical criteria, **not** ``numpy.linalg.matrix_rank``'s
default tolerance (``max(M, N) * eps * sigma_max``):

1. Circular spatial-frequency separation. Identify ``psi ~ psi + 2 pi``
   and require

       min_{i != j} d_circ(psi_i, psi_j) >= PSI_SEP_MIN = 1e-10

   This is many orders of magnitude below the Rayleigh scale ``2 pi / N``
   and does not materially alter the Uniform[-pi/2, pi/2] distribution.

2. Relative singular-value test. ``A_k`` is full column rank iff

       sigma_min(A_k) >= RANK_SV_REL_TOL * sigma_max(A_k)

   with ``RANK_SV_REL_TOL = 1e-8``.

A user's angles are redrawn (same channel RNG) until both pass, or a
hard error is raised after ``MAX_ANGLE_DRAWS`` attempts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimulationConfig
from .rng import get_trial_rngs

# Circular spatial-frequency floor. Not a Rayleigh-resolution gap.
PSI_SEP_MIN: float = 1e-10

# Relative singular-value floor for full column rank of A_k.
RANK_SV_REL_TOL: float = 1e-8

# Rejection sampling budget for a single user's path angles.
MAX_ANGLE_DRAWS: int = 64


@dataclass(frozen=True, eq=False)
class ChannelRealization:
    """One geometric ULA channel draw, with ground-truth path parameters.

    Attributes
    ----------
    G
        Atomic-domain channel, ``c * H``, shape ``(N, K)``, complex128.
    H
        RF-domain channel, shape ``(N, K)``, complex128.
    theta
        Length-``K`` tuple of AoA arrays (radians), each shape ``(L_k,)``,
        float64, supported on ``[-pi/2, pi/2]``.
    psi
        Length-``K`` tuple of spatial-frequency arrays, each shape
        ``(L_k,)``, float64. ``psi = pi * sin(theta)``.
    alpha
        Length-``K`` tuple of path-gain vectors, each shape ``(L_k,)``,
        complex128.
    A_k
        Length-``K`` tuple of steering matrices, each shape ``(N, L_k)``,
        complex128.
    L_k
        Path counts, shape ``(K,)``, int64.
    beta_k
        Large-scale powers, shape ``(K,)``, float64.
    cfg
        Config used to generate this realization.
    """

    G: np.ndarray
    H: np.ndarray
    theta: tuple[np.ndarray, ...]
    psi: tuple[np.ndarray, ...]
    alpha: tuple[np.ndarray, ...]
    A_k: tuple[np.ndarray, ...]
    L_k: np.ndarray
    beta_k: np.ndarray
    cfg: SimulationConfig


def spatial_frequency(theta: np.ndarray | float) -> np.ndarray:
    """``psi = pi * sin(theta)`` for half-wavelength ULA spacing."""
    return np.pi * np.sin(np.asarray(theta, dtype=np.float64))


def steering_matrix(theta: np.ndarray | float, N: int) -> np.ndarray:
    """Stack steering vectors as columns: shape ``(N, L)``, complex128."""
    if N <= 0:
        raise ValueError(f"N must be > 0, got {N}")
    theta_arr = np.atleast_1d(np.asarray(theta, dtype=np.float64))
    psi = spatial_frequency(theta_arr)
    n = np.arange(N, dtype=np.float64)[:, np.newaxis]
    return np.exp(-1j * n * psi).astype(np.complex128, copy=False)


def steering_vector(theta: float, N: int) -> np.ndarray:
    """ULA steering vector ``a(theta)`` of shape ``(N,)``.

    ``||a(theta)||_2^2 = N`` because every entry has magnitude 1.
    """
    return steering_matrix(np.asarray([float(theta)], dtype=np.float64), N)[:, 0]


def min_circular_psi_separation(psi: np.ndarray) -> float:
    """Minimum pairwise circular distance on ``psi ~ psi + 2 pi``.

    For a single path the separation is defined as ``+inf``.
    """
    psi = np.asarray(psi, dtype=np.float64).reshape(-1)
    if psi.size <= 1:
        return float("inf")
    ang = np.mod(psi, 2.0 * np.pi)
    ang.sort()
    gaps = np.diff(ang)
    wrap = ang[0] + 2.0 * np.pi - ang[-1]
    return float(np.min(np.concatenate([gaps, np.asarray([wrap])])))


def is_full_column_rank(A: np.ndarray, *, rel_tol: float = RANK_SV_REL_TOL) -> bool:
    """Return True iff ``A`` has full column rank under the SV criterion.

    ``A`` (shape ``(N, L)``) is full column rank when

        sigma_min(A) >= rel_tol * sigma_max(A)

    with the documented ``RANK_SV_REL_TOL`` default. Empty or zero
    matrices are not full column rank. ``L == 0`` is rejected.
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError(f"A must be 2-D, got shape {A.shape}")
    n_rows, n_cols = A.shape
    if n_cols == 0 or n_rows < n_cols:
        return False
    singular_values = np.linalg.svd(A, compute_uv=False)
    sigma_max = float(singular_values[0])
    sigma_min = float(singular_values[-1])
    if not np.isfinite(sigma_max) or sigma_max == 0.0:
        return False
    return sigma_min >= rel_tol * sigma_max


def _paths_degenerate(psi: np.ndarray, A: np.ndarray) -> bool:
    if min_circular_psi_separation(psi) < PSI_SEP_MIN:
        return True
    return not is_full_column_rank(A)


def _complex_normal(
    rng: np.random.Generator, size: int, variance: float
) -> np.ndarray:
    """i.i.d. ``CN(0, variance)`` with ``E[|x|^2] = variance``.

    Real and imaginary parts are independent ``N(0, variance / 2)``.
    """
    scale = np.sqrt(variance / 2.0)
    real = rng.standard_normal(size)
    imag = rng.standard_normal(size)
    return (scale * real + 1j * scale * imag).astype(np.complex128, copy=False)


def _draw_user_paths(
    N: int,
    L: int,
    beta: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw ``(theta, psi, alpha, A)`` for one user, rejecting degeneracy."""
    for _ in range(MAX_ANGLE_DRAWS):
        theta = rng.uniform(-0.5 * np.pi, 0.5 * np.pi, size=L).astype(
            np.float64, copy=False
        )
        psi = spatial_frequency(theta)
        A = steering_matrix(theta, N)
        if not _paths_degenerate(psi, A):
            alpha = _complex_normal(rng, size=L, variance=beta / float(L))
            theta.flags.writeable = False
            psi.flags.writeable = False
            alpha.flags.writeable = False
            A.flags.writeable = False
            return theta, psi, alpha, A
    raise RuntimeError(
        f"Failed to draw L={L} non-degenerate ULA paths in {MAX_ANGLE_DRAWS} "
        f"attempts (PSI_SEP_MIN={PSI_SEP_MIN}, RANK_SV_REL_TOL={RANK_SV_REL_TOL})"
    )


def generate_ula_channel(
    cfg: SimulationConfig,
    trial_index: int,
    *,
    rng: np.random.Generator | None = None,
) -> ChannelRealization:
    """Generate one geometric ULA channel realization.

    Randomness is taken from the **channel** substream of
    ``get_trial_rngs(cfg.master_seed, trial_index)``, unless ``rng`` is
    supplied (tests / injected generators). Pilots, reference, and noise
    streams are not consumed.
    """
    if rng is None:
        rng = get_trial_rngs(cfg.master_seed, trial_index).channel

    N, K = cfg.N, cfg.K
    H = np.zeros((N, K), dtype=np.complex128)
    theta_all: list[np.ndarray] = []
    psi_all: list[np.ndarray] = []
    alpha_all: list[np.ndarray] = []
    A_all: list[np.ndarray] = []

    for k in range(K):
        Lk = cfg.L_k[k]
        beta = cfg.beta_k[k]
        theta, psi, alpha, A = _draw_user_paths(N, Lk, beta, rng)
        H[:, k] = A @ alpha
        theta_all.append(theta)
        psi_all.append(psi)
        alpha_all.append(alpha)
        A_all.append(A)

    H.flags.writeable = False
    G = np.asarray(cfg.c * H, dtype=np.complex128)
    G.flags.writeable = False

    L_k = np.asarray(cfg.L_k, dtype=np.int64)
    L_k.flags.writeable = False
    beta_k = np.asarray(cfg.beta_k, dtype=np.float64)
    beta_k.flags.writeable = False

    return ChannelRealization(
        G=G,
        H=H,
        theta=tuple(theta_all),
        psi=tuple(psi_all),
        alpha=tuple(alpha_all),
        A_k=tuple(A_all),
        L_k=L_k,
        beta_k=beta_k,
        cfg=cfg,
    )

"""
System model for pilot-based channel estimation at a Rydberg atomic MIMO receiver.

Implements the model of SystemModel.pdf:

    E = G S + B + W,     W ~ CN(0, sigma^2)
    Z = |E|                                        (elementwise magnitude)

with the channel generated from ULA geometry (Xu et al., WCL 2025):

    g[n,k] = c * sum_l alpha[l,k] * exp(-1j*(n-1)*psi[l,k]),
    psi[l,k] = 2*pi*d/lam * sin(theta[l,k])  =  pi*sin(theta[l,k])   for d = lam/2

Conventions
-----------
N : receive elements (vapour cells)      K : users        P : pilot instants
L_k : resolvable paths of user k

SNR = E|(GS)_{n,p}|^2 / sigma^2
RSR = E|b_{n,p}|^2 / E|g_{n,k} s_{k,p}|^2
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["SystemConfig", "Realization", "steering_vector", "simulate"]


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

@dataclass
class SystemConfig:
    """Static description of the system. One config -> many random realizations."""

    N: int = 8                      # receive elements (ULA)
    K: int = 3                      # users
    P: int = 30                     # pilot instants
    L_min: int = 3                  # L_k ~ Uniform{L_min, ..., L_max}
    L_max: int = 7
    beta: float = 1.0               # large-scale gain per user (beta_k, uniform here)
    c: float = 1.0                  # atomic conversion gain (NMSE is invariant to it)
    snr_db: float = 10.0
    rsr_db: float = 12.0
    ref_time_varying: bool = True   # True: s_{b,p} unit-modulus random (yours / Xu)
                                    # False: s_{b,p} == 1 (Cui, fixed LO)
    theta_min_deg: float = -90.0
    theta_max_deg: float = 90.0
    min_angle_sep: float = 0.0      # optional minimum |Dpsi| between paths of a user

    @property
    def snr(self) -> float:
        return 10.0 ** (self.snr_db / 10.0)

    @property
    def rsr(self) -> float:
        return 10.0 ** (self.rsr_db / 10.0)

    @property
    def sigma2(self) -> float:
        """Noise variance giving the requested SNR.

        E|(GS)_{n,p}|^2 = sum_k E|g_{n,k}|^2 = K * c^2 * beta   (pilots are CN(0,1)),
        because alpha[l,k] ~ CN(0, beta/L_k) sums to E|g_{n,k}|^2 = c^2 * beta.
        """
        return self.K * (self.c ** 2) * self.beta / self.snr


@dataclass
class Realization:
    """One random draw: channel, pilots, reference, observation."""

    G: np.ndarray          # (N, K) complex  -- ground-truth atomic-domain channel
    S: np.ndarray          # (K, P) complex  -- pilot matrix (known)
    B: np.ndarray          # (N, P) complex  -- reference field (known)
    Z: np.ndarray          # (N, P) real >=0 -- observation (magnitude only)
    W: np.ndarray          # (N, P) complex  -- noise realization (for diagnostics)
    sigma2: float
    L: np.ndarray          # (K,) int        -- true path counts
    theta: list            # list of (L_k,) arrays, true AoAs in radians
    alpha: list            # list of (L_k,) complex arrays, true path gains
    cfg: SystemConfig = field(repr=False, default=None)

    @property
    def E(self) -> np.ndarray:
        """Noiseless-plus-noise complex field before the magnitude operation."""
        return self.G @ self.S + self.B + self.W

    def nmse(self, G_hat: np.ndarray) -> float:
        """Normalized MSE of a channel estimate against ground truth."""
        return float(
            np.sum(np.abs(self.G - G_hat) ** 2) / np.sum(np.abs(self.G) ** 2)
        )


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------

def steering_vector(psi: np.ndarray | float, N: int) -> np.ndarray:
    """ULA array manifold a(psi)[n] = exp(-1j*(n-1)*psi), n = 1..N.

    Returns shape (N,) for scalar psi, or (N, L) for psi of shape (L,).
    """
    psi = np.atleast_1d(np.asarray(psi, dtype=float))
    n = np.arange(N)[:, None]                      # (N, 1)
    A = np.exp(-1j * n * psi[None, :])             # (N, L)
    return A[:, 0] if A.shape[1] == 1 else A


def _draw_angles(L: int, cfg: SystemConfig, rng: np.random.Generator) -> np.ndarray:
    """Draw L angles of arrival, optionally enforcing a minimum spatial-frequency gap."""
    lo, hi = np.deg2rad(cfg.theta_min_deg), np.deg2rad(cfg.theta_max_deg)
    if cfg.min_angle_sep <= 0:
        return rng.uniform(lo, hi, size=L)

    for _ in range(1000):                          # rejection sampling
        theta = rng.uniform(lo, hi, size=L)
        psi = np.pi * np.sin(theta)
        if L == 1 or np.min(np.diff(np.sort(psi))) >= cfg.min_angle_sep:
            return theta
    raise RuntimeError("could not satisfy min_angle_sep; relax it or reduce L")


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------

def simulate(cfg: SystemConfig, rng: np.random.Generator | int | None = None,
             L_fixed: int | None = None) -> Realization:
    """Draw one realization of the system.

    Parameters
    ----------
    cfg      : SystemConfig
    rng      : np.random.Generator, seed, or None
    L_fixed  : if given, every user gets exactly this many paths (useful for tests)
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    N, K, P, c = cfg.N, cfg.K, cfg.P, cfg.c

    # --- channel: multipath over the ULA manifold ---------------------------
    if L_fixed is not None:
        L = np.full(K, int(L_fixed))
    else:
        L = rng.integers(cfg.L_min, cfg.L_max + 1, size=K)

    G = np.zeros((N, K), dtype=complex)
    theta_all, alpha_all = [], []
    for k in range(K):
        Lk = int(L[k])
        theta = _draw_angles(Lk, cfg, rng)
        psi = np.pi * np.sin(theta)                            # d = lam/2
        # alpha ~ CN(0, beta/L_k)  so that E|g_{n,k}|^2 = c^2 * beta
        scale = np.sqrt(cfg.beta / Lk / 2.0)
        alpha = scale * (rng.standard_normal(Lk) + 1j * rng.standard_normal(Lk))
        A = steering_vector(psi, N).reshape(N, Lk)
        G[:, k] = c * (A @ alpha)
        theta_all.append(theta)
        alpha_all.append(alpha)

    # --- pilots: complex, required to resolve the conjugate ambiguity -------
    S = (rng.standard_normal((K, P)) + 1j * rng.standard_normal((K, P))) / np.sqrt(2)

    # --- reference field: single LOS path from known geometry ---------------
    theta_b = rng.uniform(np.deg2rad(cfg.theta_min_deg), np.deg2rad(cfg.theta_max_deg))
    psi_b = np.pi * np.sin(theta_b)
    a_b = steering_vector(psi_b, N)                            # (N,)

    # |alpha_b|^2 set DETERMINISTICALLY so that |b_{n,p}|^2 = RSR * c^2 * beta
    # exactly, for every realization (not just in expectation over many draws).
    # Since a_b and s_b both have unit modulus, |B_{n,p}| = c*|alpha_b| is constant
    # across (n,p); fixing |alpha_b| pins that constant to the requested RSR.
    # Only the phase is randomized (uniform per realization) -- physically, an
    # unknown local-oscillator phase -- since it does not affect achieved RSR and
    # B is fully known to the estimators regardless.
    #
    # Previously alpha_b was drawn as complex Gaussian with E|alpha_b|^2 =
    # rsr*beta: |alpha_b|^2 is then exponentially distributed with std equal to
    # its mean, so a single realization's achieved RSR could differ from the
    # requested value by several dB even though the average over many
    # realizations was correct. The fix below removes that per-realization
    # scatter entirely.
    alpha_b_pow = cfg.rsr * cfg.beta
    phase_b = rng.uniform(0.0, 2.0 * np.pi)
    alpha_b = np.sqrt(alpha_b_pow) * np.exp(1j * phase_b)

    if cfg.ref_time_varying:
        s_b = np.exp(1j * rng.uniform(0, 2 * np.pi, size=P))   # unit modulus
    else:
        s_b = np.ones(P)                                       # Cui: fixed LO
    B = c * alpha_b * np.outer(a_b, s_b)                       # (N, P)

    # --- noise and magnitude-only observation -------------------------------
    sigma2 = cfg.sigma2
    W = np.sqrt(sigma2 / 2.0) * (rng.standard_normal((N, P))
                                 + 1j * rng.standard_normal((N, P)))
    Z = np.abs(G @ S + B + W)

    return Realization(G=G, S=S, B=B, Z=Z, W=W, sigma2=sigma2,
                       L=L, theta=theta_all, alpha=alpha_all, cfg=cfg)

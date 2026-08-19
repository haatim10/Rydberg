"""Cui 3GPP TR 38.901-style clustered channel for Track A (detection).

Source
------
Mingyao Cui, Qunsong Zeng, and Kaibin Huang, "Towards Atomic MIMO
Receivers," arXiv:2404.04864 (IEEE JSAC 2025). The PDF is **not** in
this repository; equations and Table I below are taken from that paper.

This module does **not** import :mod:`rydberg_sim.channel` (Track-B
geometric ULA). Track A must not silently reuse that generator.

Cui observation model (eq. 22)
------------------------------
    z = |A^H s + b + w|

    A ∈ C^{K × N}     (eq. 16: A = [a_1, …, a_N])
    s ∈ C^K           unknown QAM, E|s_k|² = 1
    b, w, z length N
    w ~ CN(0, σ² I)

Path model (eq. 10, 15, 16)
---------------------------
    a_{n,k} = Σ_ℓ (1/ℏ) (μ_eg^T ε_{n k ℓ}) √P_k ρ_{n k ℓ} exp(-j φ_{n k ℓ})

Table I (paper §VI-A) is **not** a full 38.901 CDL drop-in. Cui states
that coefficients "are generated using the standard 3GPP TR 38.901
model, whose key parameters are given in Table I":

    clusters              23
    paths per cluster     20
    path gains            CN(0, 1)
    incident angles       Uniform(-90°, 90°)
    max cluster AS        Uniform(-5°, 5°)
    max delay spread      Uniform(0 ns, 30 ns)

Where Table I is silent, this file records an explicit assumption
instead of inventing a 38.901 CDL profile. See
``results/track_a/README.md``.

Polarization (paper §VI-A)
--------------------------
ε_{n k ℓ} and ε_{b,n} are drawn on the unit circle perpendicular to the
path's incident direction. μ_eg is y-directed
``[0, 1785.916 q a_0, 0]``; only the **direction** is used here. The
physical |μ|/ℏ × ρ √P scale is absorbed by the SNR/RSR normalization
below, because Table I already sets path gains to CN(0, 1).

SNR (eq. 37) and RSR (eq. 38)
-----------------------------
    SNR = E(|a_n^H s|²) / E(|w_n|²)
    RSR = E(|b_n|²) / E(|a_{n k} s_k|²)     (one user in the denominator)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Paper Table I / §VI-A.
CHANNEL_MODEL_CUI: str = "cui_38901"
CUI_N_CLUSTERS: int = 23
CUI_RAYS_PER_CLUSTER: int = 20
CUI_CARRIER_HZ: float = 5.0e9
CUI_ANGLE_MIN_DEG: float = -90.0
CUI_ANGLE_MAX_DEG: float = 90.0
CUI_CLUSTER_RAY_OFFSET_DEG: float = 5.0
CUI_DELAY_SPREAD_MAX_S: float = 30.0e-9
# Paper: μ_eg = [0, 1785.916 q a_0, 0]^T. Direction only; magnitude is
# absorbed by SNR normalization (Table I path gains are already CN(0,1)).
CUI_MU_EG_DIRECTION: tuple[float, float, float] = (0.0, 1.0, 0.0)
SPEED_OF_LIGHT: float = 299_792_458.0

# Documented geometry assumption: Table I gives AoA but not the array
# layout. A half-wavelength ULA is the closest manifold that makes
# incident angles well-defined. This is **not** Track-B
# ``generate_ula_channel`` (no β_k/L_k, no conversion gain c, no
# per-user geometric steering from channel.py).
CUI_ARRAY_GEOMETRY: str = "ula_half_wavelength"


@dataclass(frozen=True)
class CuiChannelParams:
    """Table I parameters plus documented geometry/LO assumptions."""

    n_clusters: int = CUI_N_CLUSTERS
    n_rays_per_cluster: int = CUI_RAYS_PER_CLUSTER
    carrier_hz: float = CUI_CARRIER_HZ
    angle_min_deg: float = CUI_ANGLE_MIN_DEG
    angle_max_deg: float = CUI_ANGLE_MAX_DEG
    cluster_ray_offset_deg: float = CUI_CLUSTER_RAY_OFFSET_DEG
    delay_spread_max_s: float = CUI_DELAY_SPREAD_MAX_S
    lo_azimuth_deg: float = 0.0
    array_geometry: str = CUI_ARRAY_GEOMETRY
    mu_eg_direction: tuple[float, float, float] = CUI_MU_EG_DIRECTION
    normalize_rows: bool = True
    channel_model: str = CHANNEL_MODEL_CUI

    def __post_init__(self) -> None:
        if not isinstance(self.normalize_rows, (bool, np.bool_)):
            raise TypeError(
                f"normalize_rows must be a bool, got {self.normalize_rows!r}"
            )
        object.__setattr__(self, "normalize_rows", bool(self.normalize_rows))
        if self.channel_model != CHANNEL_MODEL_CUI:
            raise ValueError(
                f"CuiChannelParams.channel_model must be {CHANNEL_MODEL_CUI!r}, "
                f"got {self.channel_model!r}"
            )
        if self.array_geometry != CUI_ARRAY_GEOMETRY:
            raise NotImplementedError(
                f"only {CUI_ARRAY_GEOMETRY!r} is implemented (Table I does "
                f"not specify an array); got {self.array_geometry!r}"
            )
        if int(self.n_clusters) != self.n_clusters or self.n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer")
        if int(self.n_rays_per_cluster) != self.n_rays_per_cluster or self.n_rays_per_cluster <= 0:
            raise ValueError("n_rays_per_cluster must be a positive integer")
        if not np.isfinite(self.carrier_hz) or self.carrier_hz <= 0.0:
            raise ValueError("carrier_hz must be finite and > 0")

    def as_fingerprint_dict(self) -> dict[str, object]:
        return {
            "channel_model": self.channel_model,
            "n_clusters": int(self.n_clusters),
            "n_rays_per_cluster": int(self.n_rays_per_cluster),
            "carrier_hz": float(self.carrier_hz),
            "angle_min_deg": float(self.angle_min_deg),
            "angle_max_deg": float(self.angle_max_deg),
            "cluster_ray_offset_deg": float(self.cluster_ray_offset_deg),
            "delay_spread_max_s": float(self.delay_spread_max_s),
            "lo_azimuth_deg": float(self.lo_azimuth_deg),
            "array_geometry": self.array_geometry,
            "mu_eg_direction": [float(x) for x in self.mu_eg_direction],
            # Audit M4: the per-realization row-power normalization is the most
            # consequential Track-A modelling switch. It must be part of the
            # experiment identity so a normalized and an unnormalized run can
            # never share a config fingerprint (and therefore a result store).
            "normalize_rows": bool(self.normalize_rows),
        }


@dataclass(frozen=True, eq=False)
class CuiChannelRealization:
    """One Track-A Cui channel draw. ``A`` is K × N as in eq. 16."""

    A: np.ndarray
    theta_deg: tuple[np.ndarray, ...]
    tau_s: tuple[np.ndarray, ...]
    params: CuiChannelParams
    N: int
    K: int
    mean_abs_sq_per_user: np.ndarray
    channel_model: str = CHANNEL_MODEL_CUI


def _unit(vec: np.ndarray) -> np.ndarray:
    nrm = float(np.linalg.norm(vec))
    if nrm == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return vec / nrm


def _k_hat(theta_rad: float) -> np.ndarray:
    """Unit arrival direction in the x–z plane.

    θ = 0 is broadside (+z → origin), θ = +π/2 is endfire from +x.
    Assumption: Table I does not specify the coordinate frame.
    """
    return np.array([np.sin(theta_rad), 0.0, -np.cos(theta_rad)], dtype=np.float64)


def _polarization_basis(theta_rad: float) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal basis of the plane perpendicular to the incident direction."""
    k = _k_hat(theta_rad)
    y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    e1 = y_axis - np.dot(y_axis, k) * k
    if float(np.linalg.norm(e1)) < 1e-12:
        e1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        e1 = e1 - np.dot(e1, k) * k
    e1 = _unit(e1)
    e2 = _unit(np.cross(k, e1))
    return e1, e2


def _polarization_couplings(
    rng: np.random.Generator, theta_rad: float, mu: np.ndarray, N: int
) -> np.ndarray:
    """Per-element ``μ·ε_n`` with ε_n on the incident-normal unit circle."""
    e1, e2 = _polarization_basis(theta_rad)
    psi = rng.uniform(0.0, 2.0 * np.pi, size=N)
    # ε = cosψ e1 + sinψ e2  →  μ·ε = (μ·e1) cosψ + (μ·e2) sinψ
    return (float(np.dot(mu, e1)) * np.cos(psi)) + (float(np.dot(mu, e2)) * np.sin(psi))


def _mu_hat(params: CuiChannelParams) -> np.ndarray:
    return _unit(np.asarray(params.mu_eg_direction, dtype=np.float64))


def _array_phase(n_index: np.ndarray, theta_rad: float) -> np.ndarray:
    """Half-wavelength ULA phase ``exp(-j n π sin θ)``, n = 0..N-1.

    Assumption A1 in ``results/track_a/README.md``. Implemented here so
    Track A never calls :func:`rydberg_sim.channel.steering_vector`.
    """
    return np.exp(-1j * n_index * np.pi * np.sin(theta_rad))


def _cn01(rng: np.random.Generator) -> complex:
    """Table I path gain: CN(0, 1), E[|α|²] = 1."""
    scale = 1.0 / np.sqrt(2.0)
    return complex(scale * rng.standard_normal() + 1j * scale * rng.standard_normal())


def generate_cui_channel(
    N: int,
    K: int,
    rng: np.random.Generator,
    params: CuiChannelParams | None = None,
    *,
    normalize_rows: bool | None = None,
) -> CuiChannelRealization:
    """Draw one Cui clustered channel ``A ∈ C^{K × N}``.

    Randomness: path gains, cluster AoA, ray offsets, delays, and
    **per-element** polarizations all come from ``rng`` (the Step-14
    **channel** stream). Does not touch pilots/noise/data/solver.

    Row normalization scales each user row so ``mean_n |a_{nk}|² = 1``
    (production Track A). Skipping it returns the Table I draw at its raw
    scale. The RNG consumption is identical either way; only the final
    per-row scale is skipped.

    The fingerprinted source of truth is ``params.normalize_rows``, which
    :meth:`CuiChannelParams.as_fingerprint_dict` includes in the experiment
    identity (audit M4). ``normalize_rows=None`` (the default) reads that
    field. Passing an explicit ``bool`` overrides it for a **direct
    diagnostic call only** and does *not* change any fingerprint — trial
    generators must go through ``params`` so that the switch and the
    recorded experiment identity can never disagree.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy Generator, got {type(rng)!r}")
    if isinstance(N, (bool, np.bool_)) or int(N) != N or N <= 0:
        raise ValueError(f"N must be a positive integer, got {N!r}")
    if isinstance(K, (bool, np.bool_)) or int(K) != K or K <= 0:
        raise ValueError(f"K must be a positive integer, got {K!r}")
    N = int(N)
    K = int(K)
    p = params if params is not None else CuiChannelParams()
    # params is the fingerprinted source of truth; an explicit keyword is a
    # direct-call diagnostic override (audit M4).
    do_normalize = bool(p.normalize_rows if normalize_rows is None else normalize_rows)
    mu = _mu_hat(p)
    n_idx = np.arange(N, dtype=np.float64)
    two_pi_f = 2.0 * np.pi * float(p.carrier_hz)

    A = np.zeros((K, N), dtype=np.complex128)
    theta_all: list[np.ndarray] = []
    tau_all: list[np.ndarray] = []
    n_paths = int(p.n_clusters) * int(p.n_rays_per_cluster)

    for k in range(K):
        # Table I: "Maximum delay spread ~ U(0 ns, 30 ns)".
        delay_spread = float(rng.uniform(0.0, float(p.delay_spread_max_s)))
        thetas = np.empty(n_paths, dtype=np.float64)
        taus = np.empty(n_paths, dtype=np.float64)
        path_i = 0
        for _c in range(int(p.n_clusters)):
            theta_c = float(
                rng.uniform(float(p.angle_min_deg), float(p.angle_max_deg))
            )
            tau_c = float(rng.uniform(0.0, delay_spread)) if delay_spread > 0.0 else 0.0
            for _r in range(int(p.n_rays_per_cluster)):
                offset = float(
                    rng.uniform(
                        -float(p.cluster_ray_offset_deg),
                        float(p.cluster_ray_offset_deg),
                    )
                )
                theta_deg = float(
                    np.clip(
                        theta_c + offset,
                        float(p.angle_min_deg),
                        float(p.angle_max_deg),
                    )
                )
                theta_rad = np.deg2rad(theta_deg)
                alpha = _cn01(rng)
                delay_phase = np.exp(-1j * two_pi_f * tau_c)
                array_ph = _array_phase(n_idx, theta_rad)
                couplings = _polarization_couplings(rng, theta_rad, mu, N)
                A[k, :] += alpha * delay_phase * couplings * array_ph
                thetas[path_i] = theta_deg
                taus[path_i] = tau_c
                path_i += 1
        theta_all.append(thetas)
        tau_all.append(taus)

    # Production Track A: per-user row power → 1 so eq. 37/38 have
    # E|a_{nk}|^2 = 1 after averaging over n. Table I CN(0,1) gains plus
    # |μ·ε| do not already guarantee that scale. The unnormalized path
    # is a diagnostic; it does not change the production definition.
    mean_pow = np.empty(K, dtype=np.float64)
    for k in range(K):
        mean_pow[k] = float(np.mean(np.abs(A[k, :]) ** 2))
        if mean_pow[k] <= 0.0:
            raise RuntimeError(f"user {k} channel has zero power")
        if do_normalize:
            A[k, :] /= np.sqrt(mean_pow[k])
            mean_pow[k] = 1.0

    A.setflags(write=False)
    mean_pow.setflags(write=False)
    frozen_theta = []
    frozen_tau = []
    for th, ta in zip(theta_all, tau_all):
        th = np.array(th, dtype=np.float64, copy=True)
        ta = np.array(ta, dtype=np.float64, copy=True)
        th.setflags(write=False)
        ta.setflags(write=False)
        frozen_theta.append(th)
        frozen_tau.append(ta)
    return CuiChannelRealization(
        A=A,
        theta_deg=tuple(frozen_theta),
        tau_s=tuple(frozen_tau),
        params=p,
        N=N,
        K=K,
        mean_abs_sq_per_user=mean_pow,
        channel_model=CHANNEL_MODEL_CUI,
    )


def generate_cui_reference(
    N: int,
    rng: np.random.Generator,
    rsr_db: float,
    params: CuiChannelParams | None = None,
) -> np.ndarray:
    """LoS reference ``b ∈ C^N`` with per-element random polarization.

    Paper §III-B: LO-to-receiver is LoS. §VI-A: ε_{b,n} on the unit
    circle perpendicular to the LO incident angle. Scale so
    mean_n |b_n|² = RSR_lin, matching eq. 38 when E|a_{nk} s_k|² = 1.

    The LO azimuth is **not** in Table I (assumption A2: default 0°
    broadside). Uses the Step-14 **reference** stream.
    """
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy Generator, got {type(rng)!r}")
    N = int(N)
    p = params if params is not None else CuiChannelParams()
    rsr_lin = 10.0 ** (float(rsr_db) / 10.0)
    if not np.isfinite(rsr_lin) or rsr_lin <= 0.0:
        raise ValueError(f"rsr_db must map to a positive linear RSR, got {rsr_db!r}")
    mu = _mu_hat(p)
    theta_rad = np.deg2rad(float(p.lo_azimuth_deg))
    n_idx = np.arange(N, dtype=np.float64)
    array_ph = _array_phase(n_idx, theta_rad)
    couplings = _polarization_couplings(rng, theta_rad, mu, N)
    b = (couplings * array_ph).astype(np.complex128, copy=False)
    power = float(np.mean(np.abs(b) ** 2))
    if power <= 0.0:
        raise RuntimeError("reference vector has zero power")
    b *= np.sqrt(rsr_lin / power)
    b.setflags(write=False)
    return b

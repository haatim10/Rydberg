"""Cui spectral initialization in the canonical solver convention (Step 8).

Canonical model
---------------
    z = |M^H u + b + w|

    M ∈ C^{D × Q},   u ∈ C^D,   z, b, w length Q
    (z is stored as a real nonnegative amplitude)

This initializer is completely generic. It does **not** know whether ``u``
is Cui's unknown QAM symbol vector or ``conj(g_n)`` from the
channel-estimation adapter. Those mappings live outside
:func:`spectral_initialize`.

When ``|b_q|`` is much larger than the columns of ``M``, every ``mbar_q``
points nearly along the last axis and ``u0`` collapses toward 0. That is
the high-RSR limit of this initializer, not a conjugate-transpose bug.
The weak ``||u0-u||/||u|| < 0.5`` sanity test therefore uses a strong
*nonzero* reference with ``|b|`` on the same order as ``|M^H u|``.

Algorithm (Cui Alg. 1/2, steps 1–4)
-----------------------------------
1. Augment the known reference into the linear model:

       ubar = [u ; 1] ∈ C^{D+1}
       Mbar^H = [M^H , b] ∈ C^{Q × (D+1)}
       Mbar   = (Mbar^H)^H ∈ C^{(D+1) × Q}

   Column ``q`` of ``Mbar`` is ``mbar_q``, and
   ``z_q = |mbar_q^H ubar + w_q|``.

2. Spectral matrix, weighted by amplitudes **z_q** (not ``z_q^2``):

       M_spec = Σ_q z_q  mbar_q mbar_q^H  ∈ C^{(D+1)×(D+1)}

   Then Hermitian-symmetrize:
   ``M_spec ← 0.5 (M_spec + M_spec^H)``.

3. Principal eigenvector of the Hermitian ``M_spec`` (largest eigenvalue).

4. Magnitude scale ``rbar`` from the inner product of ``z`` with
   ``|Mbar^H v|`` (elementwise magnitudes, not a matrix product).

5. ``ubar0 = rbar v``.

6. Global-phase anchor using the **final** augmented entry (the known
   coefficient 1). Only after that rotation is ``u0 = ubar0[:D]`` returned.
   Discarding the last entry before anchoring would leave an arbitrary
   global phase.

What this module does **not** implement (Step 9+)
-------------------------------------------------
biased GS, EM-GS, Bessel ratio, CRLB, GD/PGD, Monte Carlo sweeps,
figures, BER. The plan's comparison

    "GS from spectral initialization should beat GS from random
    initialization at SNR = -5 dB"

is a **future Step-9** acceptance test. Do not implement GS here to
satisfy it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Guard for ||projection||_2^2 == 0. Documented; not a Tikhonov ridge on M_spec.
PROJECTION_DENOM_FLOOR = 1e-300

FUTURE_GS_SPECTRAL_VS_RANDOM_TEST = (
    "GS from spectral initialization should beat GS from random "
    "initialization at SNR = -5 dB. Biased GS is Step 9; not implemented."
)


@dataclass(frozen=True, eq=False)
class SpectralInitResult:
    """Canonical spectral initialization and the diagnostics used to test it.

    Attributes
    ----------
    u0
        Initial estimate, shape ``(D,)``. Phase-anchored.
    MbarH
        ``[M^H, b]``, shape ``(Q, D+1)``.
    Mbar
        ``(MbarH)^H``, shape ``(D+1, Q)``.
    M_spec
        Hermitian spectral matrix, shape ``(D+1, D+1)``.
    eigenvalues
        Eigenvalues of ``M_spec`` in ascending order, shape ``(D+1,)``.
    principal_eigenvalue
        Largest eigenvalue.
    v
        Principal eigenvector, shape ``(D+1,)``. Arbitrary eigensolver phase.
    projection
        ``|Mbar^H v|``, shape ``(Q,)``.
    rbar
        Nonnegative magnitude scale.
    ubar0
        ``rbar * v`` **before** phase anchoring, shape ``(D+1,)``.
    ubar0_anchored
        After rotating so the last entry is real and nonnegative,
        shape ``(D+1,)``.
    """

    u0: np.ndarray
    MbarH: np.ndarray
    Mbar: np.ndarray
    M_spec: np.ndarray
    eigenvalues: np.ndarray
    principal_eigenvalue: float
    v: np.ndarray
    projection: np.ndarray
    rbar: float
    ubar0: np.ndarray
    ubar0_anchored: np.ndarray


@dataclass(frozen=True, eq=False)
class ChannelSpectralInitResult:
    """Per-element channel-row adapter around :func:`spectral_initialize`.

    Physical model for receive element ``n``::

        z_n = |S^T g_n + b_n + w_n|

    Canonical mapping (conjugation of the whole observation; **not** baked
    into :func:`spectral_initialize`)::

        M = S
        u_true = conj(g_n)
        b_solver = conj(b_n)

    Then ``g0 = conj(u0)``. Each element builds its own ``(K+1)×(K+1)``
    spectral matrix from its own ``z_n`` and ``b_n``. ``M = S`` is shared;
    the initializer is **not** cached across ``N``.
    """

    G0: np.ndarray
    row_results: tuple[SpectralInitResult, ...]


def _require_finite(arr: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")


def _as_complex_matrix(value: object, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.complex128)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {arr.shape}")
    _require_finite(arr, name)
    return np.array(arr, dtype=np.complex128, copy=True)


def _as_complex_vector(value: object, name: str, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.complex128)
    if arr.ndim == 2 and arr.shape in ((length, 1), (1, length)):
        arr = arr.reshape(length)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size != length:
        raise ValueError(f"{name} must have length {length}, got {arr.size}")
    _require_finite(arr, name)
    return np.array(arr, dtype=np.complex128, copy=True)


def _as_real_vector(value: object, name: str, length: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 2 and arr.shape in ((length, 1), (1, length)):
        arr = arr.reshape(length)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {arr.shape}")
    if arr.size != length:
        raise ValueError(f"{name} must have length {length}, got {arr.size}")
    _require_finite(arr, name)
    return np.array(arr, dtype=np.float64, copy=True)


def build_augmented_dictionary(
    M: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(MbarH, Mbar)`` for the known-reference augmentation.

    ``MbarH = [M^H, b]`` has shape ``(Q, D+1)``.
    ``Mbar = (MbarH)^H`` has shape ``(D+1, Q)``.
    """
    M_arr = _as_complex_matrix(M, "M")
    d, q = M_arr.shape
    if d < 1 or q < 1:
        raise ValueError(f"M must have positive dimensions, got {M_arr.shape}")
    b_arr = _as_complex_vector(b, "b", q)
    mbar_h = np.concatenate(
        [M_arr.conj().T, b_arr.reshape(q, 1)],
        axis=1,
    ).astype(np.complex128, copy=False)
    mbar = mbar_h.conj().T
    if mbar_h.shape != (q, d + 1):
        raise RuntimeError(
            f"internal MbarH shape {mbar_h.shape}, expected {(q, d + 1)}"
        )
    if mbar.shape != (d + 1, q):
        raise RuntimeError(
            f"internal Mbar shape {mbar.shape}, expected {(d + 1, q)}"
        )
    return mbar_h, mbar


def spectral_matrix_from_columns(
    Mbar: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """Explicit ``Σ_q z[q] * outer(mbar_q, conj(mbar_q))``.

    Reference implementation for the vectorized production path. Uses
    ``z_q``, not ``z_q^2``.
    """
    mbar = _as_complex_matrix(Mbar, "Mbar")
    d1, q = mbar.shape
    z_arr = _as_real_vector(z, "z", q)
    m_spec = np.zeros((d1, d1), dtype=np.complex128)
    for k in range(q):
        col = mbar[:, k]
        m_spec += z_arr[k] * np.outer(col, np.conjugate(col))
    return m_spec


def _spectral_matrix_vectorized(Mbar: np.ndarray, z: np.ndarray) -> np.ndarray:
    """``(Mbar * z) @ Mbar^H == Σ_q z_q mbar_q mbar_q^H``."""
    return (Mbar * z[np.newaxis, :]) @ Mbar.conj().T


def scale_and_anchor(
    MbarH: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    *,
    denom_floor: float = PROJECTION_DENOM_FLOOR,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Magnitude LS scale plus last-entry phase anchor.

    Parameters
    ----------
    MbarH
        Shape ``(Q, D+1)``.
    v
        Length ``D+1`` (principal eigenvector, any global phase).
    z
        Length ``Q`` amplitudes.

    Returns
    -------
    projection, ubar0, rbar, ubar0_anchored, u0
    """
    mbar_h = _as_complex_matrix(MbarH, "MbarH")
    q, d1 = mbar_h.shape
    if d1 < 2:
        raise ValueError(f"MbarH must have at least 2 columns, got shape {mbar_h.shape}")
    v_arr = _as_complex_vector(v, "v", d1)
    z_arr = _as_real_vector(z, "z", q)
    if not np.isfinite(denom_floor) or denom_floor <= 0.0:
        raise ValueError(f"denom_floor must be finite and > 0, got {denom_floor!r}")

    projection = np.abs(mbar_h @ v_arr).astype(np.float64, copy=False)
    numerator = float(np.sum(projection * z_arr))
    denominator = float(np.sum(projection**2))
    # Documented safeguard: a numerically zero projection would make rbar
    # undefined. This is not regularization of M_spec.
    rbar = numerator / max(denominator, float(denom_floor))
    ubar0 = (rbar * v_arr).astype(np.complex128, copy=False)
    # Do NOT drop the last entry before this rotation.
    phase_anchor = np.exp(-1j * np.angle(ubar0[-1]))
    ubar0_anchored = (phase_anchor * ubar0).astype(np.complex128, copy=False)
    u0 = ubar0_anchored[:-1].copy()
    return projection, ubar0, rbar, ubar0_anchored, u0


def spectral_initialize(
    M: np.ndarray,
    z: np.ndarray,
    b: np.ndarray,
    *,
    denom_floor: float = PROJECTION_DENOM_FLOOR,
) -> SpectralInitResult:
    """Cui spectral initializer for ``z = |M^H u + b + w|``.

    ``M.shape == (D, Q)``, ``z.shape == (Q,)``, ``b.shape == (Q,)``.
    Returns ``u0`` of shape ``(D,)`` plus diagnostics.

    This function has no channel-estimation or QAM special case.
    """
    M_arr = _as_complex_matrix(M, "M")
    d, q = M_arr.shape
    z_arr = _as_real_vector(z, "z", q)
    b_arr = _as_complex_vector(b, "b", q)
    if np.any(z_arr < 0.0):
        raise ValueError("z must be nonnegative amplitudes")

    mbar_h, mbar = build_augmented_dictionary(M_arr, b_arr)
    m_spec = _spectral_matrix_vectorized(mbar, z_arr)
    # Tiny Hermitian asymmetry from rounding; eigh requires Hermitian.
    m_spec = 0.5 * (m_spec + m_spec.conj().T)

    eigenvalues, eigenvectors = np.linalg.eigh(m_spec)
    # eigh returns ascending eigenvalues; the principal vector is the last.
    v = eigenvectors[:, -1].astype(np.complex128, copy=False)
    lam_max = float(np.real(eigenvalues[-1]))

    projection, ubar0, rbar, ubar0_anchored, u0 = scale_and_anchor(
        mbar_h, v, z_arr, denom_floor=denom_floor
    )
    if u0.shape != (d,):
        raise RuntimeError(f"internal u0 shape {u0.shape}, expected {(d,)}")

    return SpectralInitResult(
        u0=u0,
        MbarH=mbar_h,
        Mbar=mbar,
        M_spec=np.asarray(m_spec, dtype=np.complex128),
        eigenvalues=np.real(eigenvalues).astype(np.float64, copy=False),
        principal_eigenvalue=lam_max,
        v=v,
        projection=projection,
        rbar=float(rbar),
        ubar0=ubar0,
        ubar0_anchored=ubar0_anchored,
    )


def spectral_initialize_channel_rows(
    S: np.ndarray,
    Z: np.ndarray,
    B: np.ndarray,
    *,
    denom_floor: float = PROJECTION_DENOM_FLOOR,
) -> ChannelSpectralInitResult:
    """Channel-estimation adapter: loop over receive elements.

    Does **not** change :func:`spectral_initialize`. For each ``n`` it
    calls the canonical initializer with ``M = S`` and
    ``b_solver = conj(B[n])``, then ``G0[n] = conj(u0)``.

    ``M_spec`` is rebuilt from ``(z_n, b_n)`` every time; one spectral
    matrix is never reused across ``N``.
    """
    S_arr = _as_complex_matrix(S, "S")
    Z_arr = np.asarray(Z, dtype=np.float64)
    B_arr = _as_complex_matrix(B, "B")
    if Z_arr.ndim != 2:
        raise ValueError(f"Z must be 2-D (N, P), got shape {Z_arr.shape}")
    _require_finite(Z_arr, "Z")
    n_rx, n_pilots = Z_arr.shape
    n_users, p_s = S_arr.shape
    if p_s != n_pilots:
        raise ValueError(
            f"incompatible Z and S: Z.shape={Z_arr.shape}, S.shape={S_arr.shape}"
        )
    if B_arr.shape != (n_rx, n_pilots):
        raise ValueError(
            f"incompatible B: B.shape={B_arr.shape}, expected {(n_rx, n_pilots)}"
        )

    G0 = np.empty((n_rx, n_users), dtype=np.complex128)
    rows: list[SpectralInitResult] = []
    for n in range(n_rx):
        # Adapter conjugation only. Canonical function still sees (M, z, b).
        row = spectral_initialize(
            S_arr, Z_arr[n], np.conjugate(B_arr[n]), denom_floor=denom_floor
        )
        G0[n] = np.conjugate(row.u0)
        rows.append(row)
    return ChannelSpectralInitResult(G0=G0, row_results=tuple(rows))

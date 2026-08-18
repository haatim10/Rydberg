"""Deterministic error metrics (Step 13).

This module is **side-effect free**. It never generates channels, noise,
pilots, or Monte Carlo trials. Callers pass truth and prediction arrays
and receive scalars plus small diagnostic dataclasses.

Detection NMSE (Cui, continuous output before QAM demapping)
------------------------------------------------------------
    NMSE_detection = E||s - s̃||₂² / E||s||₂²

For unit-average-energy QAM with ``K = D`` users the default
denominator is ``E||s||₂² = K``, **not** a per-trial ``||s_true||₂²``.
Pass ``expected_symbol_energy`` to override. ``s̃`` is the continuous
detector output.

Channel NMSE
------------
    NMSE_channel = E||Ĝ - G||_F² / E||G||_F²

A single realization reports ``error_energy = ||Ĝ - G||_F²`` and
``true_energy = ||G||_F²``, hence an *instantaneous* ratio
``error_energy / true_energy``. That per-trial ratio is **not** the
Monte Carlo estimator. The preferred aggregation is

    (Σ error energies) / (Σ true energies)

not ``mean(per_trial_nmse_db)`` and not necessarily
``mean(per_trial_nmse_linear)``. Theoretical normalization uses
:func:`expected_channel_frobenius_energy`: ``c² N Σ_k β_k``
(``N K`` when ``c = 1``, ``β_k = 1``).

The **primary** channel NMSE compares ``Ĝ`` to ``G`` with **no** phase
alignment (``B`` is known and nonzero). Per-row phase alignment is a
diagnostic only and never overwrites ``Ĝ``.

BER
---
Continuous output → nearest Step-4 QAM point → Gray bits → compare.
Accumulate ``bit_errors`` and ``bit_count`` as integers. Global BER is
``total_bit_errors / total_bit_count``, not the mean of per-trial BERs.

dB conversion
-------------
NMSE is a **power** ratio: ``nmse_db = 10 log10(nmse_linear)``.
``nmse_linear = 0`` returns ``-inf``. Never ``20 log10``.

What this module does **not** implement (Step 14+)
-------------------------------------------------
Monte Carlo trial driver, common-random-number harness, parallel
execution, result caching, confidence intervals, figure sweeps, Track-C,
GD/PGD, machine learning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .linearised_crlb import expected_channel_frobenius_energy
from .qam import (
    QAMConstellation,
    build_qam_constellation,
    nearest_qam_indices,
)


def _require_finite(arr: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")


def _as_complex_vector(value: object, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.complex128)
    if arr.ndim == 2 and min(arr.shape) == 1:
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector, got shape {arr.shape}")
    _require_finite(arr, name)
    return np.array(arr, dtype=np.complex128, copy=True)


def _as_complex_matrix(value: object, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.complex128)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {arr.shape}")
    _require_finite(arr, name)
    return np.array(arr, dtype=np.complex128, copy=True)


def _as_positive_energy(value: object, name: str) -> float:
    try:
        energy = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number, got {value!r}") from exc
    if not np.isfinite(energy) or energy <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return energy


def nmse_to_db(nmse_linear: float) -> float:
    """``10 log10(nmse)``. Power ratio, not amplitude. ``0 → -inf``."""
    try:
        x = float(nmse_linear)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"nmse_linear must be a real number, got {nmse_linear!r}") from exc
    if not np.isfinite(x) or x < 0.0:
        raise ValueError(f"nmse_linear must be finite and >= 0, got {nmse_linear!r}")
    if x == 0.0:
        return float(-np.inf)
    return float(10.0 * np.log10(x))


@dataclass(frozen=True, eq=False)
class DetectionNMSEResult:
    """Cui detection NMSE on the **continuous** symbol estimate.

    ``nmse_linear = error_energy / expected_energy``. Default
    ``expected_energy = K`` (vector length) for unit-energy QAM.
    """

    error_energy: float
    expected_energy: float
    realization_energy: float
    nmse_linear: float
    nmse_db: float


def detection_nmse(
    s_hat_continuous: np.ndarray,
    s_true: np.ndarray,
    *,
    expected_symbol_energy: float | None = None,
) -> DetectionNMSEResult:
    """``||ŝ - s||₂² / E||s||₂²`` with explicit expected-energy denominator.

    ``s_hat_continuous`` is the detector output **before** QAM demapping.
    The default denominator is ``K = s.size``, not ``||s_true||₂²``.
    """
    hat = _as_complex_vector(s_hat_continuous, "s_hat_continuous")
    true = _as_complex_vector(s_true, "s_true")
    if hat.shape != true.shape:
        raise ValueError(
            f"s_hat_continuous and s_true shapes must match, got {hat.shape} and {true.shape}"
        )
    k = int(true.size)
    if expected_symbol_energy is None:
        energy = float(k)
    else:
        energy = _as_positive_energy(expected_symbol_energy, "expected_symbol_energy")
    error = float(np.sum(np.abs(hat - true) ** 2))
    realization = float(np.sum(np.abs(true) ** 2))
    nmse = error / energy
    return DetectionNMSEResult(
        error_energy=error,
        expected_energy=energy,
        realization_energy=realization,
        nmse_linear=nmse,
        nmse_db=nmse_to_db(nmse),
    )


def phase_align_channel_rows(
    G_hat: np.ndarray,
    G: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Best per-row complex phase: ``φ_n = argmin ||e^{-jφ} Ĝ[n] - G[n]||₂²``.

    Closed form: ``φ_n = -angle(Ĝ[n]ᴴ G[n])``. Returns ``(aligned, phi)``
    without modifying ``G_hat``. Diagnostic only.
    """
    hat = _as_complex_matrix(G_hat, "G_hat")
    true = _as_complex_matrix(G, "G")
    if hat.shape != true.shape:
        raise ValueError(
            f"G_hat and G shapes must match, got {hat.shape} and {true.shape}"
        )
    n_rx = hat.shape[0]
    aligned = np.array(hat, dtype=np.complex128, copy=True)
    phi = np.zeros(n_rx, dtype=np.float64)
    for n in range(n_rx):
        inner = np.vdot(hat[n], true[n])
        if inner != 0.0:
            phi[n] = float(-np.angle(inner))
            aligned[n] = np.exp(-1j * phi[n]) * hat[n]
    return aligned, phi


@dataclass(frozen=True, eq=False)
class ChannelNMSEResult:
    """Channel NMSE. Primary metric is **unaligned**.

    ``instantaneous_nmse = error_energy / true_energy`` uses this
    realization's ``||G||_F²``. ``nmse_linear`` uses
    ``expected_channel_energy`` when that argument is given, otherwise
    the instantaneous ratio. Monte Carlo should accumulate
    ``error_energy`` and ``true_energy`` (or expected energies) in
    linear units via :class:`NmseAccumulator`.
    """

    error_energy: float
    true_energy: float
    expected_energy: float | None
    instantaneous_nmse: float
    nmse_linear: float
    nmse_db: float
    phase_aligned_error_energy: float
    phase_aligned_nmse_linear: float
    phase_aligned_nmse_db: float
    phase_aligned_instantaneous_nmse: float
    row_phases: np.ndarray
    likely_phase_anchor_problem: bool


def channel_nmse(
    G_hat: np.ndarray,
    G: np.ndarray,
    *,
    expected_channel_energy: float | None = None,
) -> ChannelNMSEResult:
    """Primary NMSE: ``||Ĝ - G||_F²`` over the chosen energy, no alignment."""
    hat = _as_complex_matrix(G_hat, "G_hat")
    true = _as_complex_matrix(G, "G")
    if hat.shape != true.shape:
        raise ValueError(
            f"G_hat.shape={hat.shape} must equal G.shape={true.shape}"
        )
    error = float(np.linalg.norm(hat - true, ord="fro") ** 2)
    true_energy = float(np.linalg.norm(true, ord="fro") ** 2)
    if true_energy == 0.0:
        raise ValueError("||G||_F^2 is 0; cannot form a channel NMSE")
    instantaneous = error / true_energy
    expected: float | None
    if expected_channel_energy is None:
        expected = None
        primary = instantaneous
    else:
        expected = _as_positive_energy(
            expected_channel_energy, "expected_channel_energy"
        )
        primary = error / expected

    aligned, phi = phase_align_channel_rows(hat, true)
    aligned_error = float(np.linalg.norm(aligned - true, ord="fro") ** 2)
    aligned_inst = aligned_error / true_energy
    aligned_primary = (
        aligned_error / expected if expected is not None else aligned_inst
    )
    suspicious = bool(primary > 1e-6 and aligned_primary < 0.1 * primary)
    return ChannelNMSEResult(
        error_energy=error,
        true_energy=true_energy,
        expected_energy=expected,
        instantaneous_nmse=instantaneous,
        nmse_linear=primary,
        nmse_db=nmse_to_db(primary),
        phase_aligned_error_energy=aligned_error,
        phase_aligned_nmse_linear=aligned_primary,
        phase_aligned_nmse_db=nmse_to_db(aligned_primary),
        phase_aligned_instantaneous_nmse=aligned_inst,
        row_phases=phi,
        likely_phase_anchor_problem=suspicious,
    )


def decoded_bits(
    s_tilde: np.ndarray,
    constellation: QAMConstellation | int,
) -> np.ndarray:
    """Nearest-neighbour QAM → Gray bits. Shape ``(n_symbols, bits_per_symbol)``."""
    const = (
        constellation
        if isinstance(constellation, QAMConstellation)
        else build_qam_constellation(int(constellation))
    )
    idx = nearest_qam_indices(s_tilde, const)
    return const.bit_labels[idx].copy()


@dataclass(frozen=True, eq=False)
class BERResult:
    """One block of demapped bits. SER is diagnostic; BER is required."""

    bit_errors: int
    bit_count: int
    ber: float
    symbol_errors: int
    symbol_count: int
    ser: float
    bits_hat: np.ndarray
    symbols_hat: np.ndarray


def detection_ber(
    s_tilde: np.ndarray,
    bits_true: np.ndarray,
    constellation: QAMConstellation | int,
) -> BERResult:
    """Project ``s_tilde`` onto Step-4 QAM and count Gray bit errors.

    ``bits_true`` has shape ``(n_symbols, log2(M))`` or a flat packed
    vector of that length.
    """
    const = (
        constellation
        if isinstance(constellation, QAMConstellation)
        else build_qam_constellation(int(constellation))
    )
    tilde = np.asarray(s_tilde, dtype=np.complex128).reshape(-1)
    _require_finite(tilde, "s_tilde")
    bit_arr = np.asarray(bits_true, dtype=np.uint8)
    if bit_arr.ndim == 1:
        if bit_arr.size % const.bits_per_symbol != 0:
            raise ValueError(
                f"flat bits length {bit_arr.size} is not a multiple of "
                f"{const.bits_per_symbol}"
            )
        bit_arr = bit_arr.reshape(-1, const.bits_per_symbol)
    if bit_arr.ndim != 2 or bit_arr.shape[1] != const.bits_per_symbol:
        raise ValueError(
            f"bits_true must have shape (n, {const.bits_per_symbol}), got {bit_arr.shape}"
        )
    if np.any((bit_arr != 0) & (bit_arr != 1)):
        raise ValueError("bits_true must be 0 or 1")
    if tilde.size != bit_arr.shape[0]:
        raise ValueError(
            f"s_tilde length {tilde.size} != number of symbols {bit_arr.shape[0]}"
        )
    idx = nearest_qam_indices(tilde, const)
    bits_hat = const.bit_labels[idx].copy()
    symbols_hat = const.points[idx].astype(np.complex128, copy=True)
    bit_errors = int(np.sum(bits_hat != bit_arr))
    bit_count = int(bit_arr.size)
    symbol_errors = int(np.sum(np.any(bits_hat != bit_arr, axis=1)))
    symbol_count = int(bit_arr.shape[0])
    return BERResult(
        bit_errors=bit_errors,
        bit_count=bit_count,
        ber=(bit_errors / bit_count) if bit_count else float("nan"),
        symbol_errors=symbol_errors,
        symbol_count=symbol_count,
        ser=(symbol_errors / symbol_count) if symbol_count else float("nan"),
        bits_hat=bits_hat,
        symbols_hat=symbols_hat,
    )


@dataclass
class NmseAccumulator:
    """Sum linear energies. NMSE = total_error_energy / total_true_energy.

    Do **not** average per-trial dB values. For detection, pass
    ``expected_energy`` as the second summand (Cui ``K`` per trial).
    For channel Monte Carlo, pass realization ``true_energy`` or the
    theoretical expected energy, consistently.
    """

    total_error_energy: float = 0.0
    total_true_energy: float = 0.0

    def add(self, error_energy: float, true_energy: float) -> None:
        err = float(error_energy)
        tru = float(true_energy)
        if not np.isfinite(err) or err < 0.0:
            raise ValueError(f"error_energy must be finite and >= 0, got {error_energy!r}")
        if not np.isfinite(tru) or tru <= 0.0:
            raise ValueError(f"true_energy must be finite and > 0, got {true_energy!r}")
        self.total_error_energy += err
        self.total_true_energy += tru

    def add_detection(self, result: DetectionNMSEResult) -> None:
        self.add(result.error_energy, result.expected_energy)

    def add_channel(
        self, result: ChannelNMSEResult, *, use_expected: bool = False
    ) -> None:
        if use_expected:
            if result.expected_energy is None:
                raise ValueError("use_expected=True requires expected_channel_energy")
            self.add(result.error_energy, result.expected_energy)
        else:
            self.add(result.error_energy, result.true_energy)

    @property
    def nmse_linear(self) -> float:
        if self.total_true_energy == 0.0:
            raise ValueError("no energy accumulated")
        return self.total_error_energy / self.total_true_energy

    @property
    def nmse_db(self) -> float:
        return nmse_to_db(self.nmse_linear)


@dataclass
class BerAccumulator:
    """Global BER = total_bit_errors / total_bit_count, not mean(per-trial BER)."""

    total_bit_errors: int = 0
    total_bit_count: int = 0

    def add(self, bit_errors: int, bit_count: int) -> None:
        if int(bit_errors) != bit_errors or int(bit_count) != bit_count:
            raise TypeError("bit_errors and bit_count must be integers")
        err = int(bit_errors)
        n = int(bit_count)
        if err < 0 or n <= 0 or err > n:
            raise ValueError(
                f"need 0 <= bit_errors <= bit_count and bit_count > 0, got {err}, {n}"
            )
        self.total_bit_errors += err
        self.total_bit_count += n

    def add_result(self, result: BERResult) -> None:
        self.add(result.bit_errors, result.bit_count)

    @property
    def ber(self) -> float:
        if self.total_bit_count == 0:
            raise ValueError("no bits accumulated")
        return self.total_bit_errors / self.total_bit_count

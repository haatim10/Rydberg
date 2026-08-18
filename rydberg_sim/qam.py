"""Gray-mapped unit-energy square QAM (Step 4, Part B).

This is a **data** symbol generator for later Track A / Track C
detection. It is **not** the channel-estimation pilot matrix ``S``.
Estimation pilots are i.i.d. CN(0, 1) and live in ``pilots.py``.

Constellation
-------------
Square M-QAM with Gray labeling on each PAM axis independently.
Bits of each symbol are packed as ``[I bits | Q bits]`` (I is the
high half). Axis bit strings use the binary-reflected Gray code of
the PAM level index, with levels running from most negative to most
positive.

2-PAM (4-QAM, 1 bit/axis), unnormalized levels ``{-1, +1}``:

    bits 0 -> -1
    bits 1 -> +1

    Average |±1 ± j|² = 2, so points are divided by √2.

4-PAM (16-QAM, 2 bits/axis), unnormalized levels ``{-3, -1, +1, +3}``:

    bits 00 -> -3
    bits 01 -> -1
    bits 11 -> +1
    bits 10 -> +3

    Average energy of all I/Q combinations is 10, so points are
    divided by √10.

The constellation lookup table is built once per ``M`` and cached.
``c = 1`` normalization of the ULA channel is unrelated: here the
scale is only so that ``mean(|x|²) = 1`` over the alphabet.

RNG
---
Random bit/symbol generation takes an explicit ``numpy.random.Generator``.
Callers who want trial-addressable data should pass
``get_trial_rngs(master_seed, trial_index).data``. That ``data`` stream
is a *fifth* SeedSequence child and does not change the channel,
pilots, reference, or noise streams.

This module does not implement a detector. ``qam_to_bits`` performs an
exact constellation lookup (or accepts integer indices), not
nearest-neighbor slicing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rng import get_trial_rngs

_CONSTELLATION_CACHE: dict[int, "QAMConstellation"] = {}


def _binary_to_gray(n: np.ndarray | int) -> np.ndarray:
    n_arr = np.asarray(n, dtype=np.int64)
    return n_arr ^ (n_arr >> 1)


@dataclass(frozen=True, eq=False)
class QAMConstellation:
    """Cached square Gray-mapped QAM alphabet of order ``M``.

    Attributes
    ----------
    M
        Alphabet size. Must be square: ``M = 2^{2m}`` for integer ``m ≥ 1``.
    bits_per_symbol
        ``log2(M)``.
    points
        Constellation points, shape ``(M,)``, complex128, indexed by the
        integer value of the Gray bit label. ``mean(|points|²) = 1``.
    bit_labels
        Bit tuples, shape ``(M, bits_per_symbol)``, uint8. Row ``i`` is
        the bit label of ``points[i]``.
    scale
        Normalization applied to the unnormalized integer grid
        (``1/√2`` for 4-QAM, ``1/√10`` for 16-QAM).
    axis_levels
        Unnormalized PAM levels, e.g. ``(-3, -1, 1, 3)`` for 16-QAM.
    """

    M: int
    bits_per_symbol: int
    points: np.ndarray
    bit_labels: np.ndarray
    scale: float
    axis_levels: np.ndarray


@dataclass(frozen=True, eq=False)
class QAMSequence:
    """A block of QAM symbols with their exact bits and indices."""

    symbols: np.ndarray
    bits: np.ndarray
    indices: np.ndarray
    constellation: QAMConstellation


def _validate_M(M: int) -> tuple[int, int]:
    if isinstance(M, (bool, np.bool_)) or int(M) != M:
        raise TypeError(f"M must be an integer, got {M!r}")
    M = int(M)
    if M < 4:
        raise ValueError(f"M must be a square QAM order >= 4, got {M}")
    bits = int(np.round(np.log2(M)))
    if (1 << bits) != M:
        raise ValueError(f"M must be a power of two, got {M}")
    if bits % 2 != 0:
        raise ValueError(f"M must be square QAM (even log2(M)), got {M}")
    return M, bits


def build_qam_constellation(M: int) -> QAMConstellation:
    """Return the cached Gray-mapped unit-energy constellation of order ``M``."""
    M, bits_per_symbol = _validate_M(M)
    cached = _CONSTELLATION_CACHE.get(M)
    if cached is not None:
        return cached

    bits_per_axis = bits_per_symbol // 2
    n_levels = 1 << bits_per_axis
    axis_levels = (np.arange(n_levels, dtype=np.float64) * 2.0) - (n_levels - 1)
    # E[I^2] = (L^2 - 1) / 3, E[|x|^2] = 2 E[I^2] = 2(L^2 - 1)/3.
    unnormalized_energy = 2.0 * (n_levels**2 - 1) / 3.0
    scale = 1.0 / np.sqrt(unnormalized_energy)

    gray_of_index = _binary_to_gray(np.arange(n_levels))
    # Inverse: Gray label g -> PAM index.
    pam_index_from_gray = np.empty(n_levels, dtype=np.int64)
    pam_index_from_gray[gray_of_index] = np.arange(n_levels)

    indices = np.arange(M, dtype=np.int64)
    i_gray = indices >> bits_per_axis
    q_gray = indices & ((1 << bits_per_axis) - 1)
    i_level = axis_levels[pam_index_from_gray[i_gray]]
    q_level = axis_levels[pam_index_from_gray[q_gray]]
    points = ((i_level + 1j * q_level) * scale).astype(np.complex128, copy=False)

    shifts = np.arange(bits_per_symbol - 1, -1, -1, dtype=np.int64)
    bit_labels = ((indices[:, np.newaxis] >> shifts) & 1).astype(np.uint8)

    points.flags.writeable = False
    bit_labels.flags.writeable = False
    axis_levels.flags.writeable = False

    const = QAMConstellation(
        M=M,
        bits_per_symbol=bits_per_symbol,
        points=points,
        bit_labels=bit_labels,
        scale=float(scale),
        axis_levels=axis_levels,
    )
    _CONSTELLATION_CACHE[M] = const
    return const


def _pack_bits(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.ndim != 2:
        raise ValueError(f"bits must be 2-D (n_symbols, bits_per_symbol), got {bits.shape}")
    n_bits = bits.shape[1]
    weights = (1 << np.arange(n_bits - 1, -1, -1, dtype=np.int64)).astype(np.int64)
    return bits.astype(np.int64) @ weights


def bits_to_qam(
    bits: np.ndarray,
    M: int,
) -> np.ndarray:
    """Map bits to unit-energy QAM symbols via the lookup table.

    ``bits`` has shape ``(n_symbols, log2(M))`` with entries in {0, 1},
    or shape ``(n_symbols * log2(M),)`` packed row-major.
    """
    const = build_qam_constellation(M)
    bit_arr = np.asarray(bits, dtype=np.uint8)
    if bit_arr.ndim == 1:
        if bit_arr.size % const.bits_per_symbol != 0:
            raise ValueError(
                f"flat bits length {bit_arr.size} is not a multiple of "
                f"{const.bits_per_symbol}"
            )
        bit_arr = bit_arr.reshape(-1, const.bits_per_symbol)
    if bit_arr.ndim != 2 or bit_arr.shape[1] != const.bits_per_symbol:
        raise ValueError(
            f"bits must have shape (n, {const.bits_per_symbol}), got {bit_arr.shape}"
        )
    if np.any((bit_arr != 0) & (bit_arr != 1)):
        raise ValueError("bits must be 0 or 1")
    indices = _pack_bits(bit_arr)
    return const.points[indices]


def _match_points(symbols: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Exact lookup of constellation points (not a detector)."""
    symbols = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    # Nearest table entry, then demand a numerically exact hit.
    delta = symbols[:, np.newaxis] - points[np.newaxis, :]
    indices = np.argmin(np.abs(delta), axis=1)
    matched = points[indices]
    if not np.allclose(symbols, matched, rtol=0.0, atol=1e-10):
        raise ValueError(
            "symbols are not exact constellation points; "
            "qam_to_bits does not implement a detector"
        )
    return indices


def qam_to_bits(
    symbols_or_indices: np.ndarray,
    M: int,
) -> np.ndarray:
    """Inverse lookup: exact symbols or integer indices -> bits.

    This is **not** a nearest-neighbor detector. Symbols must match the
    constellation table to floating-point precision.
    """
    const = build_qam_constellation(M)
    arr = np.asarray(symbols_or_indices)
    if arr.dtype.kind in "iu":
        indices = np.asarray(arr, dtype=np.int64).reshape(-1)
        if np.any((indices < 0) | (indices >= const.M)):
            raise ValueError(f"indices must be in 0..{const.M - 1}")
    else:
        indices = _match_points(arr, const.points)
    return const.bit_labels[indices]


def generate_qam(
    rng: np.random.Generator | None,
    num_symbols: int,
    M: int,
    *,
    master_seed: int | None = None,
    trial_index: int | None = None,
) -> QAMSequence:
    """Draw i.i.d. uniform QAM symbols with Gray bits.

    Pass an explicit ``Generator``. To use the trial-addressable data
    stream without touching pilots/channel/reference/noise, omit
    ``rng`` and pass ``master_seed`` and ``trial_index``.
    """
    if isinstance(num_symbols, (bool, np.bool_)) or int(num_symbols) != num_symbols:
        raise TypeError(f"num_symbols must be an integer, got {num_symbols!r}")
    num_symbols = int(num_symbols)
    if num_symbols <= 0:
        raise ValueError(f"num_symbols must be > 0, got {num_symbols}")

    if rng is None:
        if master_seed is None or trial_index is None:
            raise ValueError("provide rng or both master_seed and trial_index")
        rng = get_trial_rngs(int(master_seed), int(trial_index)).data
    elif not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a numpy Generator, got {type(rng)!r}")

    const = build_qam_constellation(M)
    bits = rng.integers(0, 2, size=(num_symbols, const.bits_per_symbol), dtype=np.uint8)
    indices = _pack_bits(bits)
    symbols = const.points[indices]
    bits.flags.writeable = False
    indices.flags.writeable = False
    symbols.flags.writeable = False
    return QAMSequence(
        symbols=symbols,
        bits=bits,
        indices=indices,
        constellation=const,
    )

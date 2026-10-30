"""Deterministic per-trial RNG policy (Step 1) and operating-point keys (Step 14).

Every Monte Carlo trial is independently reproducible from
``(master_seed, trial_index)``. Trial ``t`` produces the same stream
whether it is drawn alone or after trials ``0, ..., t-1``.

This module never touches the global NumPy RNG
(``np.random.seed``, ``np.random.randn``, ``np.random.random``, ...).
All randomness comes from ``numpy.random.Generator`` objects constructed
via ``numpy.random.SeedSequence``.

Construction (Step 1, trial-only)
---------------------------------
A trial SeedSequence is addressed directly from the master seed and the
trial index:

    trial_ss = SeedSequence(entropy=master_seed, spawn_key=(trial_index,))

Component streams (channel, pilots, reference, noise, data, solver) are then
spawned from that trial sequence. This avoids ``master.spawn(trial_index + 1)``,
which would make random access to trial ``t`` require spawning ``t+1``
children.

The ``data`` stream is a fifth child appended after the original four.
The ``solver`` stream is a sixth child for iterative-solver random
initialization (Step 9). ``SeedSequence.spawn`` numbers children
sequentially, so appending a stream does **not** change earlier
sequences for a given ``(master_seed, trial_index)``.

Operating-point worlds (Step 14)
--------------------------------
Monte Carlo operating points include SNR and RSR. Those worlds are **not**
addressed by :func:`get_trial_rngs`. They use :func:`get_operating_point_rngs`:

    spawn_key = (trial_index, snr_key, rsr_key)

where ``snr_key`` / ``rsr_key`` are millidB integers plus a fixed offset
(see :func:`db_to_key`). Python's built-in ``hash()`` is **never** used:
it is process-salted and not reproducible.

Policy: each ``(trial_index, snr_db, rsr_db)`` is an independent draw.
Looping SNR in reverse does not change the world attached to a given
tuple. Workers must call these constructors from the stable key; they
must never inherit or mutate a process-global RNG.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Fixed spawn order. Children are numbered sequentially, so appending a
# stream does not retune earlier ones. Step 2 uses channel; Step 4
# pilots uses pilots; Step 3 does not consume reference; QAM data may
# use data; Step 9 random GS initialization may use solver.
COMPONENT_STREAMS: tuple[str, ...] = (
    "channel",
    "pilots",
    "reference",
    "noise",
    "data",
    "solver",
)

# MillidB encoding for operating-point spawn keys. SNR/RSR may be negative,
# but SeedSequence spawn_key entries should be non-negative integers.
# key = round(db * 1000) + 1_000_000  maps -1000 dB .. +inf millidB.
DB_KEY_SCALE: int = 1000
DB_KEY_OFFSET: int = 1_000_000
# Residual allowed when checking that db * 1000 is an integer.
_DB_KEY_RESIDUAL: float = 1e-8


def db_to_key(value_db: float, name: str = "dB") -> int:
    """Map a dB operating point to a stable non-negative spawn-key integer.

    ``snr_db = -5`` → millidB ``-5000`` → key ``995000``.
    Values must be multiples of ``0.001`` dB. Does **not** use ``hash()``.
    """
    try:
        x = float(value_db)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number, got {value_db!r}") from exc
    if not np.isfinite(x):
        raise ValueError(f"{name} must be finite, got {value_db!r}")
    scaled = x * float(DB_KEY_SCALE)
    nearest = round(scaled)
    if abs(scaled - nearest) > _DB_KEY_RESIDUAL:
        raise ValueError(
            f"{name}={value_db!r} must be a multiple of "
            f"{1.0 / DB_KEY_SCALE} dB for millidB spawn-key encoding "
            f"(residual {scaled - nearest})"
        )
    key = int(nearest) + DB_KEY_OFFSET
    if key < 0:
        raise ValueError(
            f"{name}={value_db!r} is below the supported millidB range "
            f"(offset {DB_KEY_OFFSET})"
        )
    return key


def operating_point_spawn_key(
    trial_index: int, snr_db: float, rsr_db: float
) -> tuple[int, int, int]:
    """Stable ``spawn_key`` for one ``(trial, SNR, RSR)`` world."""
    if isinstance(trial_index, (bool, np.bool_)) or int(trial_index) != trial_index:
        raise TypeError(f"trial_index must be an integer, got {trial_index!r}")
    trial_index = int(trial_index)
    if trial_index < 0:
        raise ValueError(f"trial_index must be >= 0, got {trial_index}")
    return (
        trial_index,
        db_to_key(snr_db, "snr_db"),
        db_to_key(rsr_db, "rsr_db"),
    )


@dataclass(frozen=True, eq=False)
class TrialRNGs:
    """Independent ``Generator`` objects for one Monte Carlo trial."""

    channel: np.random.Generator
    pilots: np.random.Generator
    reference: np.random.Generator
    noise: np.random.Generator
    data: np.random.Generator
    solver: np.random.Generator


def get_trial_rngs(master_seed: int, trial_index: int) -> TrialRNGs:
    """Return independent RNGs for ``(master_seed, trial_index)``.

    Parameters
    ----------
    master_seed
        Simulation-wide seed from :class:`~rydberg_sim.config.SimulationConfig`.
    trial_index
        Non-negative Monte Carlo trial index. Trial 137 generated in
        isolation matches trial 137 generated after trials 0..136.

    Returns
    -------
    TrialRNGs
        Independent ``numpy.random.Generator`` objects, one per
        component stream, in the order ``channel``, ``pilots``,
        ``reference``, ``noise``, ``data``, ``solver``.
    """
    if isinstance(trial_index, (bool, np.bool_)) or int(trial_index) != trial_index:
        raise TypeError(f"trial_index must be an integer, got {trial_index!r}")
    trial_index = int(trial_index)
    if trial_index < 0:
        raise ValueError(f"trial_index must be >= 0, got {trial_index}")

    trial_ss = np.random.SeedSequence(
        entropy=int(master_seed),
        spawn_key=(trial_index,),
    )
    return _rngs_from_seed_sequence(trial_ss)


def _rngs_from_seed_sequence(trial_ss: np.random.SeedSequence) -> TrialRNGs:
    spawned = trial_ss.spawn(len(COMPONENT_STREAMS))
    channel_ss, pilots_ss, reference_ss, noise_ss, data_ss, solver_ss = spawned
    return TrialRNGs(
        channel=np.random.default_rng(channel_ss),
        pilots=np.random.default_rng(pilots_ss),
        reference=np.random.default_rng(reference_ss),
        noise=np.random.default_rng(noise_ss),
        data=np.random.default_rng(data_ss),
        solver=np.random.default_rng(solver_ss),
    )


def get_operating_point_rngs(
    master_seed: int,
    trial_index: int,
    snr_db: float,
    rsr_db: float,
) -> TrialRNGs:
    """Return independent RNGs for one ``(master_seed, trial, SNR, RSR)`` world.

    This is the Step-14 Monte Carlo constructor. It does **not** retune
    :func:`get_trial_rngs`, which remains ``spawn_key=(trial_index,)``.

    Stream order is the same as Step 1: channel, pilots, reference, noise,
    data, solver. Workers must call this from the stable key; they must
    not share or mutate a global RNG, and must not derive state from
    loop order.
    """
    spawn_key = operating_point_spawn_key(trial_index, snr_db, rsr_db)
    trial_ss = np.random.SeedSequence(
        entropy=int(master_seed),
        spawn_key=spawn_key,
    )
    return _rngs_from_seed_sequence(trial_ss)

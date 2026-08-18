"""Deterministic per-trial RNG policy (Step 1).

Every Monte Carlo trial is independently reproducible from
``(master_seed, trial_index)``. Trial ``t`` produces the same stream
whether it is drawn alone or after trials ``0, ..., t-1``.

This module never touches the global NumPy RNG
(``np.random.seed``, ``np.random.randn``, ``np.random.random``, ...).
All randomness comes from ``numpy.random.Generator`` objects constructed
via ``numpy.random.SeedSequence``.

Construction
------------
A trial SeedSequence is addressed directly from the master seed and the
trial index:

    trial_ss = SeedSequence(entropy=master_seed, spawn_key=(trial_index,))

Component streams (channel, pilots, reference, noise) are then spawned
from that trial sequence. This avoids ``master.spawn(trial_index + 1)``,
which would make random access to trial ``t`` require spawning ``t+1``
children.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Fixed spawn order. Later stages consume pilots/reference/noise; Step 2
# uses only the channel stream.
COMPONENT_STREAMS: tuple[str, ...] = ("channel", "pilots", "reference", "noise")


@dataclass(frozen=True, eq=False)
class TrialRNGs:
    """Independent ``Generator`` objects for one Monte Carlo trial."""

    channel: np.random.Generator
    pilots: np.random.Generator
    reference: np.random.Generator
    noise: np.random.Generator


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
        Four independent ``numpy.random.Generator`` objects, one per
        component stream, in the order ``channel``, ``pilots``,
        ``reference``, ``noise``.
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
    channel_ss, pilots_ss, reference_ss, noise_ss = trial_ss.spawn(len(COMPONENT_STREAMS))
    return TrialRNGs(
        channel=np.random.default_rng(channel_ss),
        pilots=np.random.default_rng(pilots_ss),
        reference=np.random.default_rng(reference_ss),
        noise=np.random.default_rng(noise_ss),
    )

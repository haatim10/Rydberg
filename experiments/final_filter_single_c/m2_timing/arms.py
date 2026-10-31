"""Arms A, C2 and C3 as complete online calls (post hoc follow-up).

A is the frozen gated() call. C2 and C3 run the same upstream calls (start,
EM tail, frozen recovery check), then the c = 4 noise-scaled filter:
C2 on the whole high-SNR route, C3 only where the routed EM tail is > 0.
"""
import numpy as np
import torch

from tcom_reliability_20261021 import inference as reliability
from tcom_pilot_em_20261018 import continuation
from ..common import RULE, BASE_POLICY, HIGH_ROUTE_DB, route, noise_filter

C = 4.0
ARMS = ('A', 'C2', 'C3')


def filters(arm, nominal, steps):
    """Whether this arm applies the final filter for a routed setting."""
    high = float(np.mean(nominal)) > HIGH_ROUTE_DB + 1e-10
    if arm == 'C2':
        return high
    if arm == 'C3':
        return high and steps > 0
    raise ValueError(arm)


@torch.no_grad()
def arm_call(obs, seed, arm, est):
    if arm == 'A':
        return reliability.gated(obs, seed, RULE, est, BASE_POLICY).numpy()
    family, steps, ordinary, nominal = route(obs)
    pre = est['starts'].start(family, obs, seed=seed)
    if not ordinary and steps:
        pre = continuation.estimate(obs, pre, steps)
    # Recovery fires only at reference key 5, where the table filter is the
    # identity, so the frozen check sees exactly this incumbent.
    selected = reliability.from_base(obs, pre, RULE)
    return noise_filter(obs, selected, C) if filters(arm, nominal, steps) else selected.numpy()


def offline(cache, i, arm, obs):
    """Arm output from a cached pre-filter estimate (fit index i)."""
    if arm == 'A':
        return cache['frozen'][i]
    g = cache['g_pre'][i]
    return noise_filter(obs, g, C) if filters(arm, cache['nominal_snr'], cache['steps']) else g

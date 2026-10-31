"""Shared pieces: data blocks, cached pre-filter estimates, arm filters, scores.

Everything upstream of the final filter is the frozen receiver's own code. The
pre-filter estimate is reconstructed from the same calls the receiver makes,
and arm A is checked against the frozen gated() output for exact equality.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

from tcom_followups_20261017.common import REPO
from tcom_hybrid_20261020 import inference as hybrid_inference
from tcom_reliability_20261021 import inference as reliability
from tcom_incumbent_guard_20261019 import risk_data, shrink
from tcom_incumbent_guard_20261019.risk_data import observed, fit_holdout
from tcom_start_budget_20261020 import study as finish_module
from tcom_pilot_em_20261018 import continuation
from tcom_runtime_repair_20261020.inference import receiver_keys
from sahure.math import hankel, unhankel, smooth_shrink
from . import uncertainty_vendored as U

HERE = Path(__file__).resolve().parent
CACHE = HERE/'cache'
RESULTS = HERE/'results'
PREREG = json.loads((HERE/'configs/preregistration.json').read_text())
SEEDS = tuple(PREREG['frozen_receiver']['seeds'])
REGIMES = PREREG['settings']['regimes']
GRID_SNRS = tuple(map(float, PREREG['settings']['grid_snrs_db']))
GRID_RSRS = tuple(map(float, PREREG['settings']['grid_rsrs_db']))
OFF_SNRS = tuple(map(float, PREREG['settings']['test_offgrid_snrs_db']))
OFF_RSRS = tuple(map(float, PREREG['settings']['test_offgrid_rsrs_db']))
WEAK_RSRS = tuple(map(float, PREREG['settings']['test_weak_rsr_db']))
COUNT = PREREG['settings']['channels_per_cell']
SEED = PREREG['seeds']
NOISE_C = tuple(map(float, PREREG['grid']['noise_c']))
POSTERIOR_C = tuple(sorted(set(NOISE_C) | {float(np.sqrt(2)*c) for c in NOISE_C}))
HIGH_ROUTE_DB = 30.
RULE = json.loads((REPO/'results/tcom_reliability_20261021/policy.json').read_text())['candidate_rule']
BASE_POLICY = reliability.BASE_POLICY
HANKEL_MAX = U.hankel_size(32)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def cellname(regime, rsr, snr):
    return f'{regime}_rsr{rsr:g}_snr{snr:g}'


def physical_seed(role_seed, rsr):
    return int(role_seed)*1000 + int(round(float(rsr)*10))


def make_block(role_seed, regime, rsr, snrs):
    """128 channels for one (regime, reference) block; truth kept for scoring only."""
    return risk_data.make_anchor(physical_seed(role_seed, rsr), range(COUNT), REGIMES[regime], float(rsr),
                                 truth=True, snrs=tuple(snrs), holdout_seed=SEED['heldout_pilots_and_noise'])


def engine():
    from tcom_start_ablation_20261020.starts import load_starts
    return dict(starts=load_starts())


def route(obs):
    plan = hybrid_inference.plans(obs, BASE_POLICY)
    keys = {(p['family'], p['steps'], p['ordinary']) for p in plan}
    if len(keys) != 1:
        raise ValueError('cell is not homogeneous in routing')
    family, steps, ordinary = keys.pop()
    nominal = np.array([s for _, _, s in receiver_keys(obs)])
    return family, steps, ordinary, nominal


@torch.no_grad()
def receiver_cache(obs, est):
    """Per fit: pre-filter estimate, frozen output, recovery flags; exact parity asserted."""
    family, steps, ordinary, nominal = route(obs)
    pre_all, frozen_all, accepted_all, triggered_all = [], [], [], []
    for seed in SEEDS:
        start = est['starts'].start(family, obs, seed=seed)
        pre = start if (ordinary or not steps) else continuation.estimate(obs, start, steps)
        incumbent = pre if ordinary else finish_module.finish(obs, pre)
        hybrid = hybrid_inference.hybrid(obs, seed, BASE_POLICY, est)
        if not torch.equal(hybrid, incumbent):
            raise AssertionError('reconstructed incumbent differs from frozen hybrid output')
        out, diag = reliability.from_base(obs, incumbent, RULE, diagnostics=True)
        frozen = reliability.gated(obs, seed, RULE, est, BASE_POLICY)
        if not torch.equal(frozen, out):
            raise AssertionError('from_base output differs from frozen gated output')
        accepted = torch.from_numpy(diag['accepted'])
        g_pre = torch.where(accepted[:, None, None], out, pre)
        pre_all.append(g_pre.numpy()); frozen_all.append(frozen.numpy())
        accepted_all.append(diag['accepted']); triggered_all.append(diag['triggered'])
    return dict(g_pre=np.stack(pre_all), frozen=np.stack(frozen_all), accepted=np.stack(accepted_all),
                triggered=np.stack(triggered_all), family=family, steps=steps, ordinary=ordinary,
                nominal_snr=nominal)


def table_c(obs, ordinary):
    if ordinary:
        return 0.
    cfg = finish_module.filter_for(obs)
    return float(cfg['c']) if cfg['beta'] > 0 else 0.


@torch.no_grad()
def noise_filter(obs, g, c):
    """Existing noise-scaled shrinkage, beta = 1 (identity at c = 0)."""
    return shrink.apply(obs, torch.as_tensor(g), float(c), 1.).numpy()


def posterior_tau(obs, g):
    """tau_k / c = sqrt(2 u_k max(m,q)), u_k at the filter input M = g S + B."""
    z, s, b, v = (x.numpy() for x in obs.args())
    u = U.user_uncertainty(z, np.asarray(g) @ s + b, s, v)
    return U.tau_uncertainty(u, z.shape[1], 1.)


@torch.no_grad()
def posterior_filter(obs, g, c, unit_tau=None):
    if c == 0:
        return np.asarray(g)
    unit_tau = posterior_tau(obs, g) if unit_tau is None else unit_tau
    estimate = torch.as_tensor(g)
    h = hankel(estimate.mT)
    tau = torch.as_tensor(c*unit_tau)
    scale = (h.abs().square().sum((-2, -1))/min(h.shape[-2:])).clamp_min(1e-24).sqrt()
    filtered, _ = smooth_shrink(h, tau/scale)
    return (unhankel(filtered).mT).numpy()


def heldout_nll(g, hold):
    z, s, b, v = (x.numpy() for x in hold.args())
    return continuation.nll(z, np.asarray(g) @ s + b, v).mean((-2, -1))


def squared_error(g, truth):
    return np.sum(np.abs(np.asarray(g) - truth)**2, axis=(-2, -1))

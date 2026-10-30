"""Common PyTorch backend; estimators receive observations, never channel truth."""
import numpy as np
import torch
from sahure.data import Observations
from sahure.math import spectral_initial, posterior_field, right_solve, hankel, unhankel, nll
from hybrid_optshrink.model import opt_filter

BUDGETS = (5, 10, 20, 50, 100)
PARAMS = [dict(rank=0, strength=0.)] + [
    dict(rank=r, strength=a) for r in range(1, 16) for a in (.25, .5, .75, 1.)]


def tensors(data):
    return Observations(*(torch.tensor(data[k], dtype=torch.complex128 if k in ('s', 'b') else torch.float64)
                          for k in ('z', 's', 'b', 'sigma2')))


def repeat(obs, count):
    if len(obs) != 1:
        raise ValueError('candidate expansion requires one observation')
    return Observations(*(x.expand(count, *x.shape[1:]) for x in obs.args()))


def correction(g, params, hard=False):
    if len(params) != len(g):
        raise ValueError('one configuration per batch member')
    active = [i for i, p in enumerate(params) if p['strength'] != 0 and (not hard or p['rank'] < 16)]
    if not active:
        return g.clone()
    ids = torch.tensor(active, device=g.device)
    cur = g[ids]
    h = hankel(cur.mT)
    ranks = torch.tensor([params[i]['rank'] for i in active], device=g.device)[:, None]
    if hard:
        u, sv, vh = torch.linalg.svd(h, full_matrices=False)
        weights = sv * (torch.arange(sv.shape[-1], device=g.device) < ranks[..., None])
        hh = (u * weights[..., None, :]) @ vh
    else:
        hh, _ = opt_filter(h, ranks)
    strength = torch.tensor([params[i]['strength'] for i in active], dtype=g.real.dtype,
                            device=g.device)[:, None, None]
    out = g.clone()
    out[ids] = cur + strength * (unhankel(hh).mT - cur)
    return out


@torch.no_grad()
def trajectory(obs, params, budgets=BUDGETS, hard=False, progress=None):
    if obs.z.shape[1] != 32 or obs.s.shape[1] != 3:
        raise ValueError('this fixed feasibility protocol requires N32 K3')
    g = spectral_initial(*obs.args()[:3])
    gram = obs.s @ obs.s.mH
    op = right_solve(obs.s.mH, gram)
    saved = {}
    for t in range(1, max(budgets) + 1):
        if progress is not None and t % 10 == 0:
            progress(t)
        q = posterior_field(obs.z, g @ obs.s + obs.b, obs.sigma2)
        g = correction((q - obs.b) @ op, params, hard)
        if not torch.isfinite(g).all():
            raise FloatingPointError(f'nonfinite classical estimate at update {t}')
        if t in budgets:
            saved[t] = g.clone()
    return saved


@torch.no_grad()
def development_scores(context, heldout, progress=None):
    """No hidden full-pilot initialization: candidate fits use context only."""
    a = repeat(context, len(PARAMS))
    path = trajectory(a, PARAMS, progress=progress)
    scores = []
    for placement in ('loop', 'final'):
        rows = []
        for t in BUDGETS:
            g = path[t] if placement == 'loop' else correction(path[t][:1].expand(len(PARAMS), -1, -1), PARAMS)
            values = nll(heldout.z, g @ heldout.s + heldout.b, heldout.sigma2).mean((-2, -1))
            rows.append(values.numpy())
        scores.append(rows)
    result = np.asarray(scores)
    if not np.isfinite(result).all():
        raise FloatingPointError('nonfinite candidate validation likelihood')
    return result


@torch.no_grad()
def estimate(obs, config):
    p = dict(rank=config['rank'], strength=config['strength'])
    t = config['iterations']
    if config['placement'] == 'loop':
        return trajectory(obs, [p] * len(obs), (t,))[t]
    if config['placement'] == 'final':
        g = trajectory(obs, [PARAMS[0]] * len(obs), (t,))[t]
        return correction(g, [p] * len(obs))
    raise ValueError('unknown filter placement')


@torch.no_grad()
def hard_one(obs):
    """Historical one-sweep HS-Rician; all selection and refit work is counted."""
    candidates = [dict(rank=r, strength=1.) for r in range(1, 17)]
    outputs, ranks = [], []
    for i in range(len(obs)):
        row = obs.batch(slice(i, i + 1))
        a = Observations(row.z[..., :21], row.s[..., :21], row.b[..., :21], row.sigma2)
        g = trajectory(repeat(a, 16), candidates, (25,), hard=True)[25]
        scores = nll(row.z[..., 21:], g @ row.s[..., 21:] + row.b[..., 21:], row.sigma2).mean((-2, -1))
        r = int(scores.argmin()) + 1
        outputs.append(trajectory(row, [dict(rank=r, strength=1.)], (100,), hard=True)[100])
        ranks.append(r)
    return torch.cat(outputs), ranks


def lookup_key(obs):
    """Known normalized variance and reference only; never true signal power/rank."""
    snr = 10 * torch.log10(3 / obs.sigma2)
    rsr = 10 * torch.log10(obs.b.abs().square().mean((-2, -1)))
    ss = torch.tensor(list(range(-10, 21, 5)), dtype=snr.dtype)
    rr = torch.tensor([5, 12, 20], dtype=rsr.dtype)
    si = (snr[:, None] - ss).abs().argmin(-1)
    ri = (rsr[:, None] - rr).abs().argmin(-1)
    return [f'{int(ss[s])}:{int(rr[r])}' for s, r in zip(si, ri)]

"""Complex Pi=I adaptation of Yin and Smith (2021), Algorithm 2.

The measurement model remains exact Rician EM. This is not a reproduction of
their real additive-noise experiment or a claim of transferred optimality.
"""
import torch
from sahure.data import Observations
from sahure.math import hankel, unhankel, spectral_initial, posterior_field, right_solve, nll
from hybrid_optshrink.model import opt_filter
from fair_optshrink.estimator import repeat


@torch.no_grad()
def denoise(g, ranks, *, tolerance=1e-5, max_sweeps=1000):
    """Per-user stopping, Frobenius structure discrepancy, zero-safe convention."""
    h = hankel(g.mT)
    shape = h.shape
    ranks = torch.broadcast_to(torch.as_tensor(ranks, device=g.device), shape[:-2]).reshape(-1)
    current = h.reshape(-1, *shape[-2:]).clone()
    active = torch.ones(len(current), dtype=torch.bool, device=g.device)
    sweeps = torch.zeros(len(current), dtype=torch.int64, device=g.device)
    residual = torch.full((len(current),), float('inf'), dtype=g.real.dtype, device=g.device)
    for t in range(1, max_sweeps+1):
        ids = active.nonzero().flatten()
        low, _ = opt_filter(current[ids], ranks[ids])
        projected = hankel(unhankel(low))
        distance = (projected-low).norm(dim=(-2, -1))
        scale = projected.norm(dim=(-2, -1))
        residual[ids] = torch.where(scale > 0, distance/scale.clamp_min(1e-300), 0.)
        current[ids] = projected
        sweeps[ids] = t
        active[ids] = (distance >= tolerance*scale) & (scale > 0)
        if not active.any():
            break
    if not torch.isfinite(current).all():
        raise FloatingPointError('nonfinite LRHD output')
    return unhankel(current.reshape(shape)).mT, dict(
        sweeps=sweeps.reshape(shape[:-2]), residual=residual.reshape(shape[:-2]),
        capped=active.reshape(shape[:-2]))


def empty_diagnostics():
    return dict(calls=0, user_denoisings=0, sweeps=0, max_sweeps=0,
                capped=0, max_residual=0.)


def merge_diagnostics(out, info):
    out['calls'] += 1
    out['user_denoisings'] += info['sweeps'].numel()
    out['sweeps'] += int(info['sweeps'].sum())
    out['max_sweeps'] = max(out['max_sweeps'], int(info['sweeps'].max()))
    out['capped'] += int(info['capped'].sum())
    out['max_residual'] = max(out['max_residual'], float(info['residual'].max()))


@torch.no_grad()
def correct(g, params, diagnostics, **options):
    ids = [i for i, p in enumerate(params) if p['rank'] > 0 and p['strength'] > 0]
    if not ids:
        return g.clone()
    cur = g[ids]
    rank = torch.tensor([params[i]['rank'] for i in ids], device=g.device)[:, None]
    filtered, info = denoise(cur, rank, **options)
    merge_diagnostics(diagnostics, info)
    beta = torch.tensor([params[i]['strength'] for i in ids], dtype=g.real.dtype, device=g.device)[:, None, None]
    out = g.clone()
    out[ids] = cur + beta*(filtered-cur)
    return out


@torch.no_grad()
def trajectory(obs, params, *, budgets, **options):
    if len(params) != len(obs):
        raise ValueError('one parameter choice per batch element')
    diagnostics = empty_diagnostics()
    g = spectral_initial(obs.z, obs.s, obs.b)
    op = right_solve(obs.s.mH, obs.s @ obs.s.mH)
    saved = {}
    for t in range(1, max(budgets)+1):
        g = (posterior_field(obs.z, g@obs.s+obs.b, obs.sigma2)-obs.b)@op
        g = correct(g, params, diagnostics, **options)
        if not torch.isfinite(g).all():
            raise FloatingPointError('nonfinite Rician-LRHD estimate')
        if t in budgets:
            saved[t] = g.clone()
    return saved, diagnostics


@torch.no_grad()
def estimate(obs, params, *, iterations, placement='loop', **options):
    if placement not in ('loop', 'final'):
        raise ValueError('unknown placement')
    inner = params if placement == 'loop' else [dict(rank=0, strength=0.)]*len(obs)
    saved, diagnostics = trajectory(obs, inner, budgets=(iterations,), **options)
    g = saved[iterations]
    if placement == 'final':
        g = correct(g, params, diagnostics, **options)
    return g, diagnostics


@torch.no_grad()
def select_and_estimate(obs, base, **options):
    """Same 21/9 rank search and all-30 refit as existing OptShrink control."""
    if len(obs) != 1:
        raise ValueError('rank selection requires one world')
    params = [dict(rank=0, strength=0.)]+[dict(rank=r, strength=base['strength']) for r in range(1,16)]
    context = Observations(obs.z[..., :21], obs.s[..., :21], obs.b[..., :21], obs.sigma2)
    common = dict(iterations=base['iterations'], placement=base['placement'], **options)
    gs, fit = estimate(repeat(context, 16), params, **common)
    scores = nll(obs.z[..., 21:], gs@obs.s[..., 21:]+obs.b[..., 21:], obs.sigma2).mean((-2,-1))
    if not torch.isfinite(scores).all():
        raise FloatingPointError('nonfinite LRHD selection scores')
    selected = int(scores.argmin())
    g, refit = estimate(obs, [params[selected]], **common)
    return g, dict(config={**base, **params[selected]}, scores=scores.tolist(), fit=fit, refit=refit)

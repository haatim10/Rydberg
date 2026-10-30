import torch
from sahure.data import Observations
from sahure.math import nll
from fair_optshrink.estimator import repeat, trajectory, correction, estimate, lookup_key


def parameters(strength):
    return [dict(rank=0, strength=0.)] + [dict(rank=r, strength=strength) for r in range(1, 16)]


@torch.no_grad()
def fit_candidates(obs, base):
    """One shared rank for all users. Candidate fits never see withheld pilots."""
    if len(obs) != 1:
        raise ValueError('per-observation selection expects one world')
    a = Observations(obs.z[..., :21], obs.s[..., :21], obs.b[..., :21], obs.sigma2)
    t = base['iterations']; params = parameters(base['strength'])
    if base['placement'] == 'loop':
        gs = trajectory(repeat(a, 16), params, (t,))[t]
    else:
        em = trajectory(a, [params[0]], (t,))[t]
        gs = correction(em.expand(16, -1, -1), params)
    scores = nll(obs.z[..., 21:], gs @ obs.s[..., 21:] + obs.b[..., 21:], obs.sigma2).mean((-2, -1))
    if not torch.isfinite(scores).all():
        raise FloatingPointError('rank-selection scores not finite')
    index = int(scores.argmin())
    cfg = dict(**params[index], iterations=t, placement=base['placement'])
    return cfg, scores


@torch.no_grad()
def select_and_estimate(obs, frozen, mode='joint'):
    """All rank-search, initialization and all-pilot refit costs belong to inference."""
    entry = frozen['choices'][lookup_key(obs)[0]]
    placements = ('loop', 'final') if mode == 'joint' else (mode,)
    candidates = [fit_candidates(obs, entry[p]) for p in placements]
    which = min(range(len(candidates)), key=lambda i: float(candidates[i][1].min()))
    cfg, _ = candidates[which]
    g = estimate(obs, cfg)
    return g, dict(config=cfg, scores={p: scores.numpy().tolist() for p, (_, scores) in zip(placements, candidates)})

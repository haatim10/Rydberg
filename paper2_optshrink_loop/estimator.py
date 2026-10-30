"""Observation-only Rician EM with in-loop or final Hankel OptShrink."""
import numpy as np
from codex_verification.solver import geometry, initial, ratio, fit
from sahure.classical import likelihood

BUDGETS = (10, 20, 50, 100)
CANDIDATES = [dict(rank=0, strength=0.)] + [
    dict(rank=r, strength=a) for r in range(1, 16) for a in (.25, .5, 1.)]


def weights(s, n, m, ranks):
    """Batched empirical D-transform, algebraically matching the prior primitive."""
    s = np.asarray(s, dtype=float)
    ranks = np.broadcast_to(np.asarray(ranks), s.shape[:-1])
    q = min(n, m)
    if s.shape[-1] != q or np.any((ranks <= 0) | (ranks >= q)):
        raise ValueError('positive rank and nonempty noise tail required')
    scale = s[..., :1]
    ss = np.divide(s, scale, out=np.zeros_like(s), where=scale > 0)
    z = ss[..., :, None]
    tail = np.arange(q) >= ranks[..., None]
    gap = z*z - ss[..., None, :]**2
    pole = np.any(tail[..., None, :] & (gap <= 1e-13*np.maximum(z*z, 1e-300)), -1)
    safe = np.where(tail[..., None, :] & (gap > 0), gap, 1.)
    trace = np.sum(np.where(tail[..., None, :], z/safe, 0.), -1)
    deriv = np.sum(np.where(tail[..., None, :], -(z*z+ss[..., None, :]**2)/safe**2, 0.), -1)
    zz = np.where(ss > 0, ss, 1.)
    n0, m0, ell = (n-ranks)[..., None], (m-ranks)[..., None], (q-ranks)[..., None]
    a, b = (trace+(n0-ell)/zz)/n0, (trace+(m0-ell)/zz)/m0
    ap, bp = (deriv-(n0-ell)/zz**2)/n0, (deriv-(m0-ell)/zz**2)/m0
    denom = ap*b + a*bp
    active = (np.arange(q) < ranks[..., None]) & ~pole & (ss > 0) & (denom != 0)
    out = np.divide(-2*a*b, denom, out=np.zeros_like(ss), where=active)*scale
    if not np.isfinite(out).all() or np.any(out < 0):
        raise FloatingPointError('invalid OptShrink weights')
    return out


def shrink(gs, configs):
    """One lift/filter/average, supporting a batch of different configurations."""
    out = np.asarray(gs).copy()
    active = np.array([c['strength'] != 0 for c in configs])
    if not active.any():
        return out
    idx, gather, count = geometry(gs.shape[-2])
    cur = gs[active].transpose(0, 2, 1)
    h = cur[..., idx]
    u, sv, vh = np.linalg.svd(h, full_matrices=False)
    ranks = np.array([c['rank'] for c in configs])[active, None]
    ww = weights(sv, h.shape[-2], h.shape[-1], ranks)
    hh = (u*ww[..., None, :])@vh
    flat = hh.reshape(len(cur), cur.shape[1], -1)
    flat = np.concatenate([flat, np.zeros((*flat.shape[:-1], 1), complex)], -1)
    filtered = (flat[..., gather].sum(-1)/count).transpose(0, 2, 1)
    strength = np.array([c['strength'] for c in configs])[active, None, None]
    out[active] += strength*(filtered-out[active])
    return out


def trajectory(z, s, b, variance, configs, budgets=BUDGETS):
    g0 = initial(s, z, b)
    gs = np.broadcast_to(g0, (len(configs), *g0.shape)).copy()
    sh = s.conj().T
    op = np.linalg.solve((s@sh).T, sh.T).T
    saved = {}
    for t in range(1, max(budgets)+1):
        mu = gs@s+b
        latent = z*np.exp(1j*np.angle(mu))*ratio(2*z*np.abs(mu)/variance)
        gs = shrink((latent-b)@op, configs)
        if not np.isfinite(gs).all():
            raise FloatingPointError(f'nonfinite estimate at iteration {t}')
        if t in budgets:
            saved[t] = gs.copy()
    return saved


def development_scores(z, s, b, variance):
    """No truth argument; score held-out observations only."""
    loop = trajectory(z[:, :21], s[:, :21], b[:, :21], variance, CANDIDATES)
    # Candidate zero is ordinary EM at each iteration budget.
    final = {t: shrink(np.repeat(gs[:1], len(CANDIDATES), axis=0), CANDIDATES)
             for t, gs in loop.items()}
    return {name: np.stack([likelihood(z[:, 21:], bank[t]@s[:, 21:]+b[:, 21:],
                       variance).mean((-2, -1)) for t in BUDGETS])
            for name, bank in [('loop', loop), ('final', final)]}


def estimate(z, s, b, variance, config):
    iterations = config['iterations']
    param = dict(rank=config['rank'], strength=config['strength'])
    placement = config.get('placement', 'loop')
    if placement == 'loop':
        return trajectory(z, s, b, variance, [param], (iterations,))[iterations][0]
    if placement == 'final':
        g = trajectory(z, s, b, variance, [CANDIDATES[0]], (iterations,))[iterations]
        return shrink(g, [param])[0]
    raise ValueError(placement)


def hs_rician(z, s, b, variance):
    ranks = np.arange(1, (z.shape[0]+1)//2+1)
    gs = fit(s[:, :21], z[:, :21], b[:, :21], variance,
             iterations=25, ranks=ranks, sweeps=4)
    scores = likelihood(z[:, 21:], gs@s[:, 21:]+b[:, 21:], variance).mean((-2, -1))
    rank = int(ranks[scores.argmin()])
    g = fit(s, z, b, variance, iterations=100, ranks=[rank], sweeps=4)[0]
    return g, rank


def noise_index(variance):
    levels = np.arange(-10, 21, 5)
    return int(np.argmin(abs(levels-10*np.log10(3/variance))))

"""Independent vectorized evaluation of the Paper 1 update equations.

No change to the production estimator. Double precision; same initializer,
noise convention, exhaustive validation ranks, and Cadzow sweeps. No histories
or repeated validation/allocation inside each receive-row update. Numerical
agreement with production is tested before the Monte Carlo campaign.
"""
from functools import lru_cache
import time
import numpy as np
from scipy.special import i0e, i1e
from rydberg_sim.spectral import spectral_initialize_channel_rows


@lru_cache(None)
def geometry(n):
    p = (n - 1) // 2  # first balanced pencil, as production best_pencil
    idx = np.arange(n - p)[:, None] + np.arange(p + 1)[None, :]
    # Gather only actual anti-diagonal entries, avoiding a dense mostly-zero
    # matrix multiplication whose work grows by an unnecessary factor N.
    count=np.bincount(idx.ravel(),minlength=n)
    gather=np.full((n,int(count.max())),idx.size,dtype=int)
    for i in range(n):
        positions=np.flatnonzero(idx.ravel()==i)
        gather[i,:len(positions)]=positions
    return idx, gather, count


def project(g, ranks, sweeps=4):
    """g: (batch,N,K), scalar or batch-vector ranks. Full rank is exact no-op."""
    g = np.asarray(g, complex)
    ranks = np.broadcast_to(np.asarray(ranks), (len(g),))
    n, k = g.shape[-2:]
    cap = (n + 1) // 2
    active = ranks < cap
    if not active.any():
        return g.copy()
    idx, gather, count = geometry(n)
    cur = g[active].transpose(0, 2, 1).copy()
    rr = ranks[active]
    for _ in range(sweeps):
        h = cur[..., idx]
        u, s, vh = np.linalg.svd(h, full_matrices=False)
        s *= np.arange(len(s[0, 0]))[None, None, :] < rr[:, None, None]
        h = (u * s[..., None, :]) @ vh
        flat=h.reshape(len(cur),k,-1)
        flat=np.concatenate([flat,np.zeros((*flat.shape[:-1],1),complex)],axis=-1)
        cur = flat[...,gather].sum(axis=-1) / count
    out = g.copy()
    out[active] = cur.transpose(0, 2, 1)
    return out


def initial(s, z, b):
    return spectral_initialize_channel_rows(s, z, b).G0


def ratio(x):
    out = np.empty_like(x)
    large = x > 1e4
    y = x[~large]
    out[~large] = i1e(y) / i0e(y)
    y = x[large]
    out[large] = 1 - 1 / (2*y) - 1 / (8*y*y)
    return np.clip(out, 0, 1)


def fit(s, z, b, sigma2, *, iterations, ranks=None, sweeps=4):
    """Rank candidates batch together; all share this world's measurements."""
    g0 = initial(s, z, b)
    ranks = np.array([(z.shape[0]+1)//2] if ranks is None else np.atleast_1d(ranks))
    g = np.broadcast_to(g0, (len(ranks), *g0.shape)).copy()
    # A right least-squares operator; use solve, never a ridge or pseudoinverse.
    sh = s.conj().T
    op = np.linalg.solve((s @ sh).T, sh.T).T
    for _ in range(iterations):
        lam = g @ s + b
        kap = (2.0/sigma2) * z * np.abs(lam)
        g = (z * np.exp(1j*np.angle(lam)) * ratio(kap) - b) @ op
        g = project(g, ranks, sweeps)
    return g


def select(s, z, b, sigma2, *, iterations=25, sweeps=4):
    p = s.shape[1]
    nv = max(1, int(round(.3*p)))
    ranks = np.arange(1, (z.shape[0]+1)//2+1)
    g = fit(s[:, :-nv], z[:, :-nv], b[:, :-nv], sigma2,
            iterations=iterations, ranks=ranks, sweeps=sweeps)
    residual = z[:, -nv:] - np.abs(g @ s[:, -nv:] + b[:, -nv:])
    scores = np.sum(residual**2, axis=(1, 2))
    return int(ranks[np.argmin(scores)]), scores


def estimate(w, *, iterations=100, select_iterations=25, placement=False,
             forced=False):
    t0 = time.perf_counter()
    rank, scores = select(w.S, w.Z, w.B, w.sigma2, iterations=select_iterations)
    t1 = time.perf_counter()
    cap = (w.Z.shape[0]+1)//2
    ranks = [cap, rank] + ([cap-1] if forced else [])
    gs = fit(w.S, w.Z, w.B, w.sigma2, iterations=iterations, ranks=ranks)
    out = {"em": gs[0], "hs": gs[1]}
    t2 = time.perf_counter()
    if forced:
        out["forced"] = gs[2]
    if placement:
        out["post1"] = project(gs[:1], rank, 1)[0]
        out["post4"] = project(gs[:1], rank, 4)[0]
        out["inter1_sharedrank"] = fit(w.S,w.Z,w.B,w.sigma2,
            iterations=iterations,ranks=[rank],sweeps=1)[0]
    return out, rank, scores, (t1-t0, t2-t1)


def effective_ranks(g):
    idx, _, _ = geometry(len(g))
    s = np.linalg.svd(g.T[:, idx], compute_uv=False)
    p = s / s.sum(axis=-1, keepdims=True)
    return np.exp(-np.sum(p*np.log(np.maximum(p, 1e-300)), axis=-1))

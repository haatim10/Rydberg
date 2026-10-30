"""Label-free classical controls for the Paper 1 reviewer revision.

All methods see Z = |G S + B + W| with total complex variance sigma2.
Candidates share the verified P1 spectral initializer and pilot split. A
single rank/threshold is selected per world and applied to every user column.
Rician-Hankel GD is an adapted control, not Xu's physical-2D PGD method.
Smooth filtering is an empirical rule, not OptShrink or a nuclear-norm prox.
"""
import time
import numpy as np
from scipy.special import i0e
from codex_verification.solver import geometry, initial, project, ratio
from sahure.classical import linearized_ls


THRESHOLD_FACTORS = (0., .05, .1, .2, .4, .8, 1.6)


def posterior_field(z, mu, sigma2):
    """Exact Rician conditional mean, evaluated with P1's stable Bessel ratio.

    At mu=0, ratio(0)=0 and the result is exactly zero. There is no division
    by amplitude or amplitude floor; the continuous small-field limit holds.
    """
    return z * np.exp(1j * np.angle(mu)) * ratio((2.0 / sigma2) * z * np.abs(mu))


def rician_nll(z, mu, sigma2):
    """Elementwise NLL up to observation-only terms, stable near zero."""
    a = np.abs(mu)
    kappa = 2 * z * a / sigma2
    out = (z - a)**2 / sigma2 - np.log(i0e(kappa))
    small = kappa < .01
    if np.any(small):
        x2 = kappa[small]**2
        # log I0(x) = x²/4 - x⁴/64 + x⁶/576 + O(x⁸).
        zz = np.broadcast_to(z, a.shape)
        out[small] = ((zz[small]**2 + a[small]**2) / sigma2
                      - x2/4 + x2**2/64 - x2**3/576)
    return out


def rician_gradient(g, z, s, b, sigma2):
    """Wirtinger dL/dG*: the packed real gradient corresponds to 2D."""
    mu = g @ s + b
    return ((mu - posterior_field(z, mu, sigma2)) @ s.conj().T) / sigma2


def smooth_project(g, factors, sweeps=4):
    """Per-user Hankel sigma -> sigma³/(sigma²+tau²); factor zero is exact.

    tau = factor * RMS(singular values), recomputed on each Cadzow-like
    sweep's current Hankel matrix. This is shrinkage/averaging, not a hard
    rank projection. It does not have OptShrink's optimality guarantee.
    """
    g = np.asarray(g, dtype=np.complex128)
    if g.ndim != 3:
        raise ValueError('g must have shape (candidate,N,K)')
    factors = np.broadcast_to(np.asarray(factors, dtype=float), (len(g),))
    if np.any(~np.isfinite(factors)) or np.any(factors < 0):
        raise ValueError('threshold factors must be finite and nonnegative')
    if sweeps < 1:
        raise ValueError('sweeps must be positive')
    active = factors > 0
    out = g.copy()
    if not np.any(active):
        return out
    n, k = g.shape[-2:]
    idx, gather, count = geometry(n)
    cur = g[active].transpose(0, 2, 1).copy()
    relative = factors[active, None, None]
    for _ in range(sweeps):
        h = cur[..., idx]
        u, singular, vh = np.linalg.svd(h, full_matrices=False)
        power = singular**2
        tau2 = relative**2 * np.mean(power, axis=-1, keepdims=True)
        weights = np.divide(power, power + tau2, out=np.zeros_like(power),
                            where=(power + tau2) > 0)
        filtered = (u * (singular * weights)[..., None, :]) @ vh
        flat = filtered.reshape(len(cur), k, -1)
        flat = np.concatenate([flat, np.zeros((*flat.shape[:-1], 1), complex)], axis=-1)
        cur = flat[..., gather].sum(axis=-1) / count
    out[active] = cur.transpose(0, 2, 1)
    return out


def fit_candidates(s, z, b, sigma2, *, core='em', iterations=100,
                   ranks=None, factors=None, sweeps=4):
    """Vectorize rank or threshold candidates for one observation world.

    GD: G <- G - (M-Q)S^H / ||S||_2². In Wirtinger notation this is
    G <- G - 2 eta D, eta=sigma2/(2||S||_2²), fixing gamma=1.
    EM: G <- (Q-B) S^H (SS^H)^-1, using a solve.
    Structural processing need not monotonically improve likelihood.
    """
    if core not in ('em', 'gd'):
        raise ValueError('core must be em or gd')
    if iterations < 0 or sweeps < 1:
        raise ValueError('iterations must be nonnegative and sweeps positive')
    if ranks is not None and factors is not None:
        raise ValueError('choose hard ranks or smooth factors, not both')
    cap = (z.shape[0] + 1) // 2
    if factors is None:
        ranks = np.atleast_1d(cap if ranks is None else ranks)
        if np.any(ranks != np.floor(ranks)) or np.any((ranks < 1) | (ranks > cap)):
            raise ValueError('rank outside Hankel capacity')
        count = len(ranks)
    else:
        factors = np.atleast_1d(factors)
        count = len(factors)
    g0 = initial(s, z, b)
    g = np.broadcast_to(g0, (count, *g0.shape)).copy()
    sh = s.conj().T
    if core == 'em':
        op = np.linalg.solve((s @ sh).T, sh.T).T
    else:
        norm2 = np.linalg.norm(s, 2)**2
        if norm2 <= 0:
            raise ValueError('pilot spectral norm must be positive')
    for _ in range(iterations):
        mu = g @ s + b
        q = posterior_field(z, mu, sigma2)
        g = (q - b) @ op if core == 'em' else g - ((mu - q) @ sh) / norm2
        g = (project(g, ranks, sweeps) if factors is None
             else smooth_project(g, factors, sweeps))
    if not np.all(np.isfinite(g)):
        raise FloatingPointError('nonfinite estimate')
    return g


def split_columns(p, k, val_fraction=.3):
    """Use the same trailing held-out columns as verified P1, no resampling."""
    if not np.isfinite(val_fraction) or not 0 < val_fraction < 1:
        raise ValueError('val_fraction must be strictly between zero and one')
    nv = max(1, int(round(val_fraction * p)))
    if p - nv < k:
        raise ValueError('context has fewer pilots than users')
    return np.arange(p - nv), np.arange(p - nv, p)


def select_candidates(s, z, b, sigma2, *, core='em', ranks=None, factors=None,
                      iterations=25, val_fraction=.3, sweeps=4):
    """Held-out amplitude residual, not NMSE or any channel-truth quantity."""
    context, heldout = split_columns(s.shape[1], s.shape[0], val_fraction)
    candidates = fit_candidates(s[:, context], z[:, context], b[:, context], sigma2,
                                core=core, iterations=iterations, ranks=ranks,
                                factors=factors, sweeps=sweeps)
    residual = z[:, heldout] - np.abs(candidates @ s[:, heldout] + b[:, heldout])
    scores = np.sum(residual**2, axis=(-2, -1))
    return int(np.argmin(scores)), scores


def _observations(w):
    # Only these four fields are read; G, path metadata and true rho are excluded.
    s, z, b = np.asarray(w.S, complex), np.asarray(w.Z, float), np.asarray(w.B, complex)
    sigma2 = float(w.sigma2)
    if s.ndim != 2 or z.ndim != 2 or b.shape != z.shape or s.shape[1] != z.shape[1]:
        raise ValueError('invalid observation dimensions')
    if (not np.isfinite(sigma2) or sigma2 <= 0 or np.any(z < 0)
            or not all(np.all(np.isfinite(a)) for a in (s, z, b))):
        raise ValueError('observations must be finite, amplitudes nonnegative, variance positive')
    return s, z, b, sigma2


def estimate_comparators(w, *, iterations=100, select_iterations=25,
                         val_fraction=.3, sweeps=4):
    """Return (method->G, JSON-serializable diagnostics), without reading truth.

    EM/HS are included as parity controls. Every final estimate is refitted
    from spectral initialization on ALL pilots. Selectors use the same split
    but choose their own parameters; rank is shared across users, not methods.
    """
    s, z, b, sigma2 = _observations(w)
    context, heldout = split_columns(s.shape[1], s.shape[0], val_fraction)
    cap = (z.shape[0] + 1) // 2
    rank_grid = np.arange(1, cap + 1)
    estimates, details = {}, {}
    for name, core, values, smooth in [
            ('hs', 'em', rank_grid, False),
            ('rician_hankel', 'gd', rank_grid, False),
            ('em_smooth', 'em', np.asarray(THRESHOLD_FACTORS), True)]:
        options = dict(factors=values) if smooth else dict(ranks=values)
        t0 = time.perf_counter()
        index, scores = select_candidates(s, z, b, sigma2, core=core,
            iterations=select_iterations, val_fraction=val_fraction, sweeps=sweeps, **options)
        t1 = time.perf_counter()
        selected = float(values[index]) if smooth else int(values[index])
        fit_options = dict(factors=[selected]) if smooth else dict(ranks=[selected])
        estimates[name] = fit_candidates(s, z, b, sigma2, core=core,
            iterations=iterations, sweeps=sweeps, **fit_options)[0]
        t2 = time.perf_counter()
        details[name] = dict(selected_threshold=selected if smooth else None,
                            selected_rank=None if smooth else selected,
                            candidates=values.tolist(), scores=scores.tolist(),
                            seconds_select=t1-t0, seconds_final_fit=t2-t1)
    for name, core in [('em', 'em'), ('rician_gd', 'gd')]:
        t0 = time.perf_counter()
        estimates[name] = fit_candidates(s, z, b, sigma2, core=core,
            iterations=iterations, ranks=[cap], sweeps=sweeps)[0]
        details[name] = dict(selected_rank=None, selected_threshold=None, candidates=[], scores=[],
                            seconds_select=0., seconds_final_fit=time.perf_counter()-t0)
    t0 = time.perf_counter()
    # Xu et al., IEEE WCL, DOI 10.1109/LWC.2025.3584040, 1D eq. (13).
    estimates['xu_linearized'] = linearized_ls(z, s, b)
    details['xu_linearized'] = dict(selected_rank=None, selected_threshold=None, candidates=[], scores=[],
                                   seconds_select=0., seconds_final_fit=time.perf_counter()-t0)
    diagnostics = dict(methods=details, context_columns=context.tolist(), heldout_columns=heldout.tolist(),
                       iterations=int(iterations), select_iterations=int(select_iterations),
                       sweeps=int(sweeps), val_fraction=float(val_fraction),
                       threshold_scale='current per-user Hankel RMS singular value, each sweep',
                       selection_objective='sum of held-out amplitude residual squares',
                       final_fit='spectral reinitialization and all pilots',
                       gd_step='(M-Q)S^H / ||S||_2^2; packed-real gradient = 2*dL/dG*',
                       rank_sharing='one rank across users; independent selector for each method')
    return estimates, diagnostics

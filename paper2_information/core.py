"""Complex noise convention: E|w|^2 = variance; each real part has variance/2.

No ground truth, channel-family label, true rank or true SNR enters this module.
The Fisher matrix uses coordinates [Re(h_n), Im(h_n)] for each antenna n.
"""
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter

import numpy as np
from scipy.special import i0e, i1e


def ratio(x):
    x = np.asarray(x, dtype=float)
    return i1e(x) / i0e(x)


def nll(z, a, variance):
    """Rician negative log likelihood, omitting only the z-only constant.

    log(variance) retained for unit checks. Valid at z=0 (density constant
    omitted); optimizing the channel does not depend on the omitted log(2z).
    """
    x = 2 * np.asarray(z) * np.asarray(a) / variance
    return np.log(variance) + (z - a)**2 / variance - np.log(i0e(x))


def amplitude_score(z, a, variance):
    return 2 * (z * ratio(2*z*a/variance) - a) / variance


def fisher_weight_quadrature(t, order=160):
    """Dimensionless amplitude information v*I_a from score-squared integral.

    Integrates x=z/sqrt(v) over a +/- 10 interval, truncated at zero.
    Unlike difference-of-large-moments expressions this is stable at high SNR.
    """
    t = np.asarray(t, dtype=float)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    low, high = np.maximum(0, t-10), t+10
    half = (high-low)/2
    x = (low+high)[..., None]/2 + half[..., None]*nodes
    u = 2*x*t[..., None]
    density = 2*x*np.exp(-(x-t[..., None])**2)*i0e(u)
    score = 2*(x*ratio(u)-t[..., None])
    return np.sum(density*score**2*weights, axis=-1)*half


@lru_cache(maxsize=1)
def fisher_table():
    t = np.geomspace(1e-5, 1e3, 1025)
    return np.log(t), fisher_weight_quadrature(t)


def amplitude_information(a, variance):
    if variance <= 0:
        raise ValueError("Positive complex noise variance required")
    t = np.asarray(a, dtype=float)/np.sqrt(variance)
    grid, weights = fisher_table()
    q = np.interp(np.log(np.maximum(t, 1e-300)), grid, weights)
    q = np.where(t < 1e-5, 4*t*t, q)
    q = np.where(t > 1e3, 2., q)
    return q/variance


def real_fisher(h, S, B, variance):
    field = h @ S + B
    c = np.exp(-1j*np.angle(field))[:, None, :] * S[None, :, :]
    gradient = np.concatenate((c.real, -c.imag), axis=1)
    weight = amplitude_information(np.abs(field), variance)
    return np.einsum('nip,np,njp->nij', gradient, weight, gradient)


def filter_correction(d, information, tau):
    if tau < 0:
        raise ValueError("tau must be nonnegative")
    if tau == 0:
        return d.copy()
    k = d.shape[1]
    vector = np.concatenate((d.real, d.imag), axis=1)
    result = np.linalg.solve(np.eye(2*k)[None]+tau*information, vector[..., None])[..., 0]
    return result[:, :k] + 1j*result[:, k:]


def _hankel(h):
    n = len(h)
    indices = np.arange(n//2)[:, None] + np.arange(n-n//2+1)[None, :]
    return h.T[:, indices], indices


def _unhankel(x, indices, n):
    result = np.zeros((n, len(x)), dtype=complex)
    count = np.bincount(indices.ravel(), minlength=n)
    for k, matrix in enumerate(x):
        np.add.at(result[:, k], indices.ravel(), matrix.ravel())
    return result/count[:, None]


def smooth_structure(h, lam):
    """s -> s^3/(s^2+lam^2), evaluated with batched matrix solves."""
    if lam < 0:
        raise ValueError("lambda must be nonnegative")
    if lam == 0:
        return h.copy()
    x, indices = _hankel(h)
    gram = x @ x.conj().swapaxes(-1, -2)
    shrunk = x-lam**2*np.linalg.solve(gram+lam**2*np.eye(x.shape[-2]), x)
    return _unhankel(shrunk, indices, len(h))


def hard_structure(h, rank=7):
    x, indices = _hankel(h)
    u, s, vh = np.linalg.svd(x, full_matrices=False)
    s[:, rank:] = 0
    return _unhankel((u*s[:, None, :])@vh, indices, len(h))


def smooth_structure_torch(h, lam):
    """Differentiable verification helper, not a trained model."""
    import torch
    if float(lam) == 0:
        return h.clone()
    n = len(h)
    indices = torch.arange(n//2, device=h.device)[:, None]+torch.arange(n-n//2+1, device=h.device)[None]
    x = h.T[:, indices]
    eye = torch.eye(x.shape[-2], dtype=h.dtype, device=h.device)
    y = x-lam**2*torch.linalg.solve(x@x.mH+lam**2*eye, x)
    result = torch.zeros((h.shape[1], n), dtype=h.dtype, device=h.device)
    result = result.scatter_add(1, indices.flatten()[None].expand(h.shape[1], -1), y.flatten(1))
    return (result/torch.bincount(indices.flatten(), minlength=n)).T


def initialize(z, S, B):
    """Receiver-only linearized least squares, with no true-channel scaling."""
    c = np.exp(-1j*np.angle(B))[:, None, :]*S[None]
    design = np.concatenate((c.real, -c.imag), axis=1).swapaxes(1, 2)
    y = z-np.abs(B)
    real = np.stack([np.linalg.lstsq(a, b, rcond=1e-10)[0] for a, b in zip(design, y)])
    k = S.shape[0]
    return real[:, :k]+1j*real[:, k:]


@dataclass(frozen=True)
class Config:
    method: str = 'none'  # none, hard, soft, information, isotropic_information
    iterations: int = 10
    lam: float = 1.
    alpha: float = 1.
    tau: float = 1.
    rank: int = 7
    scale: float = float(np.sqrt(32))  # declared prior RMS per channel coefficient
    ridge: float = 0.  # normalized latent least-squares penalty (not MAP precision)
    postprocess: bool = False


def estimate(Z, S, B, noise_variance, config):
    """Return channel and receiver-only diagnostics; never uses ground truth."""
    return _estimate(Z, S, B, noise_variance, config)


def _estimate(Z, S, B, noise_variance, config, *, initial=None, oracle_channel=None):
    """Private hook for numerical tests and explicitly oracle-only diagnostics."""
    start = perf_counter()
    cfg = config
    if cfg.method not in ('none', 'hard', 'soft', 'information', 'isotropic_information'):
        raise ValueError("Unknown correction method")
    if cfg.scale <= 0 or noise_variance <= 0 or cfg.iterations < 1 or cfg.ridge < 0 or not 0 <= cfg.alpha <= 1:
        raise ValueError("Invalid configuration")
    z, s, b = np.asarray(Z, float)/cfg.scale, np.asarray(S, complex), np.asarray(B, complex)/cfg.scale
    v = float(noise_variance)/cfg.scale**2
    if z.ndim != 2 or b.shape != z.shape or s.ndim != 2 or s.shape[1] != z.shape[1] or np.any(z < 0):
        raise ValueError("Invalid observation shapes or magnitudes")
    if not all(np.isfinite(x).all() for x in (z, s, b, v)):
        raise ValueError("Nonfinite input")
    h = initialize(z, s, b) if initial is None else np.asarray(initial)/cfg.scale
    init_norm = float(np.linalg.norm(h))
    gram = s@s.conj().T+cfg.ridge*np.eye(len(s))
    right = s.conj().T@np.linalg.solve(gram, np.eye(len(s)))
    norms, information_extremes = [], []
    for i in range(cfg.iterations):
        field = h@s+b
        target = z*ratio(2*z*np.abs(field)/v)*np.exp(1j*np.angle(field))-b
        h = target@right
        apply = not cfg.postprocess or i == cfg.iterations-1
        if apply and cfg.method != 'none' and cfg.alpha != 0:
            structured = hard_structure(h, cfg.rank) if cfg.method == 'hard' else smooth_structure(h, cfg.lam)
            d = structured-h
            proposed = float(np.linalg.norm(d))
            if cfg.method in ('information', 'isotropic_information'):
                point = h if oracle_channel is None else np.asarray(oracle_channel)/cfg.scale
                information = real_fisher(point, s, b, v)
                eigenvalues = np.linalg.eigvalsh(information)
                information_extremes.append([float(eigenvalues.min()), float(eigenvalues.max())])
                if cfg.method == 'isotropic_information':
                    d = d/(1+cfg.tau*np.trace(information, axis1=1, axis2=2)[:, None]/(2*len(s)))
                else:
                    d = filter_correction(d, information, cfg.tau)
            h = h+cfg.alpha*d
            norms.append([proposed, float(np.linalg.norm(cfg.alpha*d))])
        if not np.isfinite(h).all():
            raise FloatingPointError(f"Nonfinite iterate {i+1}")
    return h*cfg.scale, {
        'iterations': cfg.iterations, 'seconds': perf_counter()-start,
        'numerical_failures': 0, 'correction_norms_normalized': norms,
        'information_eigenvalue_extremes': information_extremes,
        'initial_norm_normalized': init_norm, 'oracle_only': oracle_channel is not None,
        'nll_sum': float(nll(z, np.abs(h@s+b), v).sum()),
        'observations': int(z.size), 'pilot_slots': int(z.shape[1]),
    }

"""Von Mises phase-posterior uncertainty and the derived Hankel-filter scale.

All functions use receiver observables only: magnitudes Z, the current mean
M = G S + B, pilots S and total complex noise variance v. See DERIVATION.md.

Given Z and M, the phase of Y = M + W is von Mises with concentration
kappa = 2 Z |M| / v, so the posterior mean is Q = Z R(kappa) M/|M| (the
existing EM field) and, because |Y| = Z exactly, the posterior variance is
Var = E|Y|^2 - |E Y|^2 = Z^2 (1 - R(kappa)^2).
"""
import numpy as np
from scipy.special import i0e, i1e

# Matches the switch used by the existing posterior helpers (x^2 < 1e-4).
SMALL_X2 = 1e-4
# Above this kappa the asymptotic series is used for 1 - R^2; the direct
# form loses about kappa * eps relative accuracy to cancellation.
ASYMPTOTIC_X = 1e3
# 1 - R(x)^2 = sum_j A[j] x^-(j+1), from the Hankel expansions of I0 and I1
# (derived in DERIVATION.md; checked against mpmath in tests). Truncation
# error at x = 1e3 is about (13/8) x^-6, i.e. 2e-15 relative.
ASYMPTOTIC = (1., 0., 1/8, 1/4, 75/128)


def bessel_ratio(x):
    """R(x) = I1(x)/I0(x) with the repository's small-argument series."""
    x = np.asarray(x, dtype=np.float64)
    xs = np.minimum(x*x, SMALL_X2)
    with np.errstate(invalid='ignore', divide='ignore'):
        regular = i1e(x)/i0e(x)
    return np.where(x*x < SMALL_X2, .5*x*(1 - xs/8 + xs*xs/48), regular)


def one_minus_r2(x):
    """1 - R(x)^2, accurate for all x >= 0 (1 at x=0, ~1/x as x -> inf)."""
    x = np.asarray(x, dtype=np.float64)
    r = bessel_ratio(np.minimum(x, ASYMPTOTIC_X))
    direct = (1 - r)*(1 + r)
    with np.errstate(divide='ignore'):
        inv = 1/np.maximum(x, ASYMPTOTIC_X)
    series = sum(a*inv**(j+1) for j, a in enumerate(ASYMPTOTIC))
    return np.where(x > ASYMPTOTIC_X, series, direct)


def concentration(z, mu, v):
    """kappa = 2 Z |M| / v; v has one entry per batch row."""
    return 2*z*np.abs(mu)/np.asarray(v, dtype=np.float64)[:, None, None]


def field_variance(z, mu, v):
    """Posterior Var[Y | Z, M] per entry; equals Z^2 when M = 0."""
    return z*z*one_minus_r2(concentration(z, mu, v))


def channel_covariance(var, s):
    """Cov(g_n) ~= A^-1 S diag(Var_n,:) S^H A^-1, A = S S^H.

    var: [batch, N, P] real; s: [batch, K, P] complex. Returns [batch, N, K, K].
    Entries are treated as independent; the pseudo-covariance is ignored.
    """
    gram = s @ s.conj().swapaxes(-1, -2)
    w = np.linalg.solve(gram, s)                                 # A^-1 S
    return np.einsum('bkp,bnp,bjp->bnkj', w, var, w.conj())


def user_uncertainty(z, mu, s, v):
    """u_k = mean_n Re[Cov(g_n)]_kk, the predicted per-entry channel MSE.

    Only the diagonal is needed, so this avoids forming the K x K blocks:
    [Cov(g_n)]_kk = sum_p |(A^-1 S)_kp|^2 Var_np.
    """
    var = field_variance(z, mu, v)
    gram = s @ s.conj().swapaxes(-1, -2)
    w = np.linalg.solve(gram, s)
    return np.einsum('bkp,bnp->bk', np.abs(w)**2, var)/z.shape[1]


def nominal_scale(s, v):
    """Existing f5 numerator: v tr[(S S^H)^-1] / K, one value per batch row."""
    gram = s @ s.conj().swapaxes(-1, -2)
    inverse = np.linalg.inv(gram)
    return np.asarray(v)*np.trace(inverse, axis1=-2, axis2=-1).real/s.shape[1]


def hankel_size(n):
    """max(m, q) for the repository's (N-p) x (p+1) Hankel lift."""
    cols = (n - 1)//2 + 1
    return max(n - cols + 1, cols)


def tau_uncertainty(u, n, c):
    """tau_unc,k = c sqrt(2 u_k max(m, q)) (absolute, before /s_H)."""
    return c*np.sqrt(2*np.maximum(u, 0)*hankel_size(n))


def tau_existing(s, v, n, c):
    """Existing final-filter scale, tcom_incumbent_guard_20261019/shrink.py."""
    gram = s @ s.conj().swapaxes(-1, -2)
    diag = np.linalg.inv(gram).diagonal(axis1=-2, axis2=-1).real
    return c*np.sqrt(2*np.asarray(v)[:, None]*diag*hankel_size(n))

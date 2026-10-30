"""Observable paired channel-risk score; no channel-truth inputs.

For context-only estimates A and C and *independent* held-out pilots
S[:,p] ~ proper CN(0,I), the conditional expectation of this score is
||A-G||_F**2 - ||C-G||_F**2. The known nonzero reference may be deterministic.
Noise must be independent, zero mean, with the supplied second moment.
Default normalization divides by the known dimension N*K, not by true channel
energy. Thus this is raw coefficient MSE per entry, not per-channel NMSE or dB.

The midpoint is a pair-specific control variate: finite-sample pair scores need
not be transitive. Selecting a candidate using these scores introduces selection
uncertainty; the unbiased identity is for fixed context-derived candidates.
"""
from __future__ import annotations
import numpy as np


def _array(value, dtype):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def paired_risk_numpy(z, s, b, sigma2, left, right, *, normalize=True):
    """Return one float64 score per anchor; negative favors ``left``.

    Inputs have shapes z,b:[batch,N,P], s:[batch,K,P], sigma2:[batch],
    left,right:[batch,N,K]. All computation is float64/complex128. No input is
    modified. This function does not fit either candidate or use held-out data
    to construct its midpoint; callers must enforce the context-only boundary.
    """
    z = _array(z, np.float64)
    s, b, left, right = (_array(v, np.complex128) for v in (s, b, left, right))
    sigma2 = _array(sigma2, np.float64)
    if z.ndim != 3 or s.ndim != 3 or b.shape != z.shape:
        raise ValueError('z,b:[batch,N,P] and s:[batch,K,P] required')
    batch, n, p = z.shape
    if min(batch, n, p, s.shape[1]) < 1 or s.shape[0] != batch or s.shape[2] != p:
        raise ValueError('positive aligned batch/pilot dimensions required')
    k = s.shape[1]
    if left.shape != (batch, n, k) or right.shape != left.shape or sigma2.shape != (batch,):
        raise ValueError('estimates:[batch,N,K] and sigma2:[batch] required')
    if not all(np.isfinite(v).all() for v in (z, s, b, sigma2, left, right)):
        raise ValueError('all inputs must be finite')
    if (z < 0).any() or (sigma2 < 0).any() or (np.abs(b) == 0).any():
        raise ValueError('nonnegative magnitudes/variance and nonzero reference required')
    delta_field = (left - right) @ s
    midpoint_field = (0.5 * left + 0.5 * right) @ s
    magnitude = np.abs(b + midpoint_field)
    # Difference of squares avoids subtracting two separately rounded squares.
    residual_intensity = (z - magnitude) * (z + magnitude) - sigma2[:, None, None]
    innovation = residual_intensity / b.conj()
    score = -2.0 * np.real(delta_field.conj() * innovation).sum(axis=(-2, -1)) / p
    if normalize:
        score = score / (n * k)
    if not np.isfinite(score).all():
        raise FloatingPointError('nonfinite paired risk score')
    return score


def paired_risk(obs, left, right, *, normalize=True):
    """Observations/tensor convenience interface returning NumPy float64."""
    return paired_risk_numpy(obs.z, obs.s, obs.b, obs.sigma2, left, right,
                             normalize=normalize)

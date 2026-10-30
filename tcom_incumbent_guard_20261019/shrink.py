"""Final noise-scaled Hankel shrinkage; all inputs are receiver observables.

The scale is a heuristic based on the fitting pilot Gram matrix, not a claim
that the nonlinear estimate has a known covariance. No rank is specified.
"""
import torch
from sahure.math import hankel, unhankel, smooth_shrink

CONFIGS = {f'shrink_c{c:g}_beta{beta:g}': dict(kind='noise_scaled_hankel', c=c, beta=beta)
           for c in (.5, 1., 2., 4.) for beta in (.5, 1.)}


@torch.no_grad()
def apply(obs, estimate, c, beta):
    if c < 0 or not 0 <= beta <= 1:
        raise ValueError('c must be nonnegative and beta between zero and one')
    if c == 0 or beta == 0:
        return estimate
    h = hankel(estimate.mT)
    gram = obs.s @ obs.s.mH
    inverse_diagonal = torch.linalg.inv(gram).diagonal(dim1=-2, dim2=-1).real
    variance = 2 * obs.sigma2[:, None] * inverse_diagonal
    tau = c * (variance.clamp_min(0) * max(h.shape[-2:])).sqrt()
    scale = (h.abs().square().sum((-2, -1)) / min(h.shape[-2:])).clamp_min(1e-24).sqrt()
    filtered, _ = smooth_shrink(h, tau / scale)
    return estimate + beta * (unhankel(filtered).mT - estimate)

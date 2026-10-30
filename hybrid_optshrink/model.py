"""Exact Rician EM plus OptShrink; no channel truth enters an estimator.

Native SVD derivatives are used, not an identity/straight-through backward.
They are checked away from singular-value ties. Ties/cutoff boundaries are
not claimed differentiable; nonfinite training derivatives abort the run.
"""
import json
from pathlib import Path
import torch
from torch import nn
from sahure.math import spectral_initial, posterior_field, right_solve, hankel, unhankel
from sahure.model import AdaptiveHankelEstimator

OLD_SELECTION = Path('results/paper2_optshrink_loop/frozen_selection.json')


def opt_weights(s, rows, cols, ranks):
    """Differentiable translation of the checked NumPy empirical D-transform."""
    q = min(rows, cols)
    ranks = torch.broadcast_to(ranks, s.shape[:-1])
    if torch.any((ranks < 1) | (ranks >= q)):
        raise ValueError('OptShrink requires a nonempty noise tail')
    scale = s[..., :1].clamp_min(1e-300)
    ss = s / scale
    z = ss[..., :, None]
    tail = torch.arange(q, device=s.device) >= ranks[..., None]
    gap = z.square() - ss[..., None, :].square()
    pole = (tail[..., None, :] & (gap <= 1e-13*z.square().clamp_min(1e-300))).any(-1)
    safe = torch.where(tail[..., None, :] & (gap > 0), gap, torch.ones_like(gap))
    trace = torch.where(tail[..., None, :], z/safe, 0.).sum(-1)
    deriv = torch.where(tail[..., None, :], -(z.square()+ss[..., None, :].square())/safe.square(), 0.).sum(-1)
    zz = torch.where(ss > 0, ss, torch.ones_like(ss))
    n0, m0, ell = (rows-ranks)[..., None], (cols-ranks)[..., None], (q-ranks)[..., None]
    a, b = (trace+(n0-ell)/zz)/n0, (trace+(m0-ell)/zz)/m0
    ap, bp = (deriv-(n0-ell)/zz.square())/n0, (deriv-(m0-ell)/zz.square())/m0
    denom = ap*b+a*bp
    active = (torch.arange(q, device=s.device) < ranks[..., None]) & ~pole & (ss > 0) & (denom != 0)
    safe_denom = torch.where(active, denom, torch.ones_like(denom))
    return torch.where(active, -2*a*b/safe_denom, 0.)*scale


def opt_filter(h, ranks):
    u, s, vh = torch.linalg.svd(h, full_matrices=False)
    w = opt_weights(s, h.shape[-2], h.shape[-1], ranks)
    return (u*w[..., None, :])@vh, s


class OptController(nn.Module):
    def __init__(self, mode='adaptive', stages=10, tables=None):
        super().__init__()
        if mode not in ('adaptive', 'global', 'fixed', 'residual'):
            raise ValueError(mode)
        self.mode, self.stages = mode, stages
        if tables is None:
            old = json.loads(OLD_SELECTION.read_text())['noise_aware']
            tables = [old[str(s)]['fixed']['10'] for s in range(-10, 21, 5)]
        self.register_buffer('ranks', torch.tensor([max(1, p['rank']) for p in tables]))
        self.register_buffer('strengths', torch.tensor([p['strength'] for p in tables], dtype=torch.float64))
        self.register_buffer('levels', torch.arange(-10, 21, 5, dtype=torch.float64))
        if mode == 'adaptive':
            self.net = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 1))
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        elif mode == 'global':
            self.offsets = nn.Parameter(torch.zeros(stages))

    def forward(self, z, s, b, sigma2, *, strength_override=None, residual_c=0., diagnostics=False):
        if torch.any(sigma2 <= 0):
            raise ValueError('positive total complex noise variance required')
        g = spectral_initial(z, s, b)
        gram = s@s.mH
        idx = (10*torch.log10(3/sigma2)[:, None]-self.levels).abs().argmin(-1)
        ranks = self.ranks[idx, None]
        base = self.strengths[idx, None].expand(-1, s.shape[1])
        if strength_override is not None:
            base = torch.broadcast_to(torch.as_tensor(strength_override, dtype=z.dtype, device=z.device), base.shape)
        eye = torch.eye(s.shape[1], dtype=s.dtype, device=s.device)
        proxy = sigma2*torch.linalg.solve(gram, eye.expand(len(z), -1, -1)).diagonal(dim1=-2, dim2=-1).real.sum(-1)/s.shape[1]
        reference_power = b.abs().square().mean((-2, -1))
        trace = []
        for t in range(self.stages):
            q = posterior_field(z, g@s+b, sigma2)
            dc = right_solve((q-b)@s.mH, gram)
            if self.mode in ('fixed','residual') and bool(torch.all(base == 0)):
                g = dc
                if diagnostics:
                    trace.append(torch.zeros(len(z),s.shape[1],2,dtype=z.dtype,device=z.device))
                continue
            filtered, sv = opt_filter(hankel(dc.mT), ranks)
            structured = unhankel(filtered).mT
            residual = (z-(dc@s+b).abs()).square().mean((-2, -1))/sigma2
            if self.mode == 'adaptive':
                weights = sv/sv.sum(-1, keepdim=True).clamp_min(1e-12)
                entropy = -(weights*weights.clamp_min(1e-12).log()).sum(-1)
                power = dc.abs().square().mean(-2).clamp_min(1e-12)
                const = lambda value: torch.full_like(entropy, value)
                features = torch.stack((entropy.exp()/sv.shape[-1], weights[..., 0],
                    weights[..., -((sv.shape[-1]+1)//2):].sum(-1),
                    residual.log1p()[:, None].expand_as(power),
                    (proxy[:, None]/power).log1p(),
                    (reference_power[:, None]/(s.shape[1]*power)).log1p(),
                    const(s.shape[-1]/(2*s.shape[1])), const((t+1)/self.stages)), -1)
                beta = (base+.5*torch.tanh(self.net(features).squeeze(-1))).clamp(0, 1)
            elif self.mode == 'global':
                beta = (base+.5*torch.tanh(self.offsets[t])).clamp(0, 1)
            elif self.mode == 'residual' and residual_c > 0:
                beta = base*(residual/residual_c).clamp(0, 1)[:, None]
            else:
                beta = base
            g = dc+beta[:, None, :]*(structured-dc)
            if diagnostics:
                trace.append(torch.stack((beta, (g-dc).norm(dim=-2)/dc.norm(dim=-2).clamp_min(1e-12)), -1).detach())
        if not torch.isfinite(g).all():
            raise FloatingPointError('nonfinite OptShrink estimate')
        return (g, torch.stack(trace, 1)) if diagnostics else g


def build(name, seed=4101):
    torch.manual_seed(seed)
    if name == 'smooth':
        return AdaptiveHankelEstimator().double()
    return OptController(mode=name).double()

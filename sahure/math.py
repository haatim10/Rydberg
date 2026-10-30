"""Exact Rician statistics with total complex noise variance.

Wirtinger D = dL/dG*. PyTorch's real-loss complex gradient equals 2D.
Small-argument series are numerical evaluations, not forward-model Taylor
linearizations. No strong-reference approximation is used.
"""
import torch


def variance(v, x):
    v = torch.as_tensor(v, dtype=x.real.dtype, device=x.device)
    if torch.any(v <= 0):
        raise ValueError("sigma2 must be positive")
    return v.reshape(-1, 1, 1) if v.ndim == 1 else v


def nll(z, mu, sigma2, full=False):
    """Elementwise NLL up to data-only log(2z/sigma2), unless full=True."""
    v = variance(sigma2, mu)
    u = mu.real.square() + mu.imag.square()
    a = torch.sqrt(u.clamp_min(1e-30))
    x = 2*z*a/v
    x2 = 4*z.square()*u/v.square()
    # log I0(x) = x²/4 - x⁴/64 + x⁶/576 + O(x⁸).
    xs = x2.clamp_max(1e-4)  # avoid overflow in the inactive series branch
    small = (z.square()+u)/v - (xs/4-xs.square()/64+xs.pow(3)/576)
    regular = (z-a).square()/v - torch.log(torch.special.i0e(x))
    out = torch.where(x2 < 1e-4, small, regular)
    if full:
        out = out - torch.log(2*z/v)
    return out


def posterior_field(z, mu, sigma2):
    v = variance(sigma2, mu)
    u = mu.real.square()+mu.imag.square()
    a = torch.sqrt(u.clamp_min(1e-30))
    x = 2*z*a/v
    x2 = 4*z.square()*u/v.square()
    r = torch.special.i1e(x)/torch.special.i0e(x)
    xs = x2.clamp_max(1e-4)
    small = z.square()/v * (1-xs/8+xs.square()/48)
    weight = torch.where(x2 < 1e-4, small, z*r/a)
    return mu*weight


def wirtinger(g, z, s, b, sigma2):
    mu = g@s+b
    return ((mu-posterior_field(z, mu, sigma2))/variance(sigma2, mu))@s.mH


def right_solve(x, gram):
    return torch.linalg.solve(gram.mT, x.mT).mT


@torch.no_grad()
def spectral_initial(z, s, b):
    """Repository spectral initializer, batched; uses ONLY given observations."""
    batch, n, p = z.shape
    a = torch.cat((s.mT[:, None].expand(-1,n,-1,-1), b[...,None]), dim=-1)
    spec = a.mH@(z[...,None]*a)
    spec = (spec+spec.mH)/2
    _, vec = torch.linalg.eigh(spec)
    vec = vec[..., -1]
    projection = (a@vec[...,None]).squeeze(-1).abs()
    scale = (projection*z).sum(-1)/projection.square().sum(-1).clamp_min(1e-30)
    anchored = vec*torch.exp(-1j*torch.angle(vec[...,-1]))[...,None]
    return (scale[...,None]*anchored)[...,:s.shape[1]]


def hankel(g):
    """(...,N) -> (...,N-p,p+1), p=floor((N-1)/2)."""
    return g.unfold(-1, (g.shape[-1]-1)//2+1, 1)


def unhankel(h):
    rows, cols = h.shape[-2:]
    n = rows+cols-1
    idx = (torch.arange(rows,device=h.device)[:,None]+torch.arange(cols,device=h.device)).flatten()
    out = torch.zeros(*h.shape[:-2],n,dtype=h.dtype,device=h.device)
    out.index_add_(-1, idx, h.flatten(-2))
    count = torch.bincount(idx,minlength=n).to(h.real.dtype)
    return out/count


def smooth_shrink(h, relative_tau):
    """sigma -> sigma³/(sigma²+tau²), without singular-vector derivatives."""
    q = min(h.shape[-2:])
    scale = (h.abs().square().sum((-2,-1))/q).clamp_min(1e-24).sqrt()
    hn = h/scale[...,None,None]
    t2 = relative_tau.square()[...,None,None]
    eye = torch.eye(h.shape[-1],dtype=h.dtype,device=h.device)
    gram = hn.mH@hn+t2*eye
    out = hn-t2*torch.linalg.solve(gram,hn.mH).mH
    return out*scale[...,None,None], relative_tau*scale

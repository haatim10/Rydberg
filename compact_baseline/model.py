"""A small recurrent adaptation inspired by NA-ALISTA, not its reproduction.

Keep the existing exact Rician EM core and smooth Hankel operator. A shared
LSTM uses two observable residual/update features to control shrinkage and
mixing. Its state resets for every received block, separately for each user.
"""
import math
import torch
from torch import nn
from sahure.math import posterior_field, right_solve, spectral_initial, hankel, unhankel, smooth_shrink, nll


class RecurrentHankelEstimator(nn.Module):
    def __init__(self, stages=10, hidden_size=4):
        super().__init__()
        self.stages, self.hidden_size = stages, hidden_size
        self.steps = nn.Parameter(torch.zeros(stages))
        self.cell = nn.LSTMCell(2, hidden_size)
        self.head = nn.Linear(hidden_size, 2)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([math.log(math.expm1(.099)), math.log(.1/.9)]))

    def forward(self, z, s, b, sigma2, *, diagnostics=False):
        if z.ndim != 3 or s.ndim != 3 or b.shape != z.shape or s.shape[-1] != z.shape[-1]:
            raise ValueError('expected batched Z/B(batch,N,P), S(batch,K,P)')
        if sigma2.ndim != 1 or len(sigma2) != len(z) or not torch.all(sigma2 > 0):
            raise ValueError('positive noise variance per block required')
        batch, users = s.shape[:2]
        g = spectral_initial(z, s, b)
        gram = s @ s.mH
        hidden = z.new_zeros(batch*users, self.hidden_size)
        memory = torch.zeros_like(hidden)
        trace = []
        for t in range(self.stages):
            gamma = 2*torch.sigmoid(self.steps[t])
            qfield = posterior_field(z, g@s+b, sigma2)
            update = right_solve((qfield-b)@s.mH, gram)
            dc = g+gamma*(update-g)
            residual = torch.log1p((z-(dc@s+b).abs()).square().mean((-2,-1))/sigma2)
            power = dc.abs().square().mean(-2).clamp_min(1e-12)
            backprojected = torch.log1p((update-g).abs().square().mean(-2)/power)
            features = torch.stack((residual[:,None].expand(-1,users), backprojected), -1)
            hidden, memory = self.cell(features.reshape(batch*users,2), (hidden,memory))
            raw = self.head(hidden).reshape(batch,users,2)
            relative_tau = .001+torch.nn.functional.softplus(raw[...,0])
            beta = torch.sigmoid(raw[...,1])
            hs, tau = smooth_shrink(hankel(dc.mT), relative_tau)
            structured = unhankel(hs).mT
            g = dc+beta[:,None,:]*(structured-dc)
            if diagnostics:
                trace.append(dict(gamma=gamma.detach(), beta=beta.detach(), tau=tau.detach(),
                    correction=((g-dc).norm(dim=-2)/dc.norm(dim=-2).clamp_min(1e-12)).detach(),
                    nll=nll(z,g@s+b,sigma2).mean((-2,-1)).detach()))
        return (g, trace) if diagnostics else g

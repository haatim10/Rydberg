"""Two architectures with exactly the same observable initialization/LS map.

The proposed forward is algebraically identical to the original adaptive core;
its common initializer and pilot-only solve are supplied at entry for both
architectures. This equivalence is checked before training.
"""
import math
import torch
from sahure.model import AdaptiveHankelEstimator,ModelConfig
from sahure.math import posterior_field,hankel,unhankel,smooth_shrink,spectral_initial,right_solve
from urformer_reproduction_20261022.model import URformer

ARCHES=('proposed','urformer')


class Proposed(AdaptiveHankelEstimator):
    def __init__(self):
        super().__init__(ModelConfig(core='em',structure='adaptive',stages=10))

    def forward(self,g,z,s,b,v,op):
        batch,n,p=z.shape;k=s.shape[-2]
        noise_proxy=v*op.abs().square().sum((-2,-1))/k
        for t in range(10):
            gamma=2*torch.sigmoid(self.steps[t])
            q=posterior_field(z,g@s+b,v)
            dc=g+gamma*((q-b)@op-g)
            h=hankel(dc.mT);sv=torch.linalg.svdvals(h);cap=sv.shape[-1]
            weights=sv/sv.sum(-1,keepdim=True).clamp_min(1e-12)
            ent=-(weights*weights.clamp_min(1e-12).log()).sum(-1)
            residual=((z-(dc@s+b).abs()).square().mean((-2,-1))/v).log1p()
            power=dc.abs().square().mean(-2).clamp_min(1e-12)
            const=lambda x:torch.full_like(ent,x)
            features=torch.stack((ent.exp()/cap,weights[...,0],weights[...,-((cap+1)//2):].sum(-1),
                residual[:,None].expand(-1,k),(noise_proxy[:,None]/power).log1p(),
                const(p/(2*k)),const(math.log(cap)),const((t+1)/10)),dim=-1)
            raw=self.net(features)
            rel=.001+torch.nn.functional.softplus(raw[...,0]);beta=torch.sigmoid(raw[...,1])
            hs,_=smooth_shrink(h,rel)
            structured=unhankel(hs).mT
            g=dc+beta[:,None,:]*(structured-dc)
        return g


def build(arch,seed,dtype=torch.float32):
    torch.manual_seed(int(seed))
    return (Proposed() if arch=='proposed' else URformer()).to(dtype=dtype)


def frontend(z,s,b,v):
    """Uses context observations only, and is identical for both models."""
    g=spectral_initial(z,s,b)
    op=right_solve(s.mH,s@s.mH)
    return g,z,s,b,v,op


def ratio(g,h):
    return (g-h).abs().square().sum((-2,-1))/h.abs().square().sum((-2,-1))

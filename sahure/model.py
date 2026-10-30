"""Two exact-Rician cores and three structural variants."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from .math import posterior_field, right_solve, spectral_initial, hankel, unhankel, smooth_shrink, nll


@dataclass(frozen=True)
class ModelConfig:
    core: str = "em"
    structure: str = "adaptive"
    stages: int = 10
    fixed_steps: bool = False
    gate_one: bool = False
    final_only: bool = False


class AdaptiveHankelEstimator(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        if config.core not in ("em", "gd") or config.structure not in ("none","global","adaptive"):
            raise ValueError("unknown model variant")
        self.config = config
        self.steps = nn.Parameter(torch.zeros(config.stages),requires_grad=not config.fixed_steps)
        # 0.001 floor avoids a singular solve; initial relative tau exactly .1.
        bias = torch.tensor([math.log(math.expm1(.099)),math.log(.1/.9)])
        if config.structure == "global":
            self.structural = nn.Parameter(bias.repeat(config.stages,1))
        elif config.structure == "adaptive":
            self.net = nn.Sequential(nn.Linear(8,16),nn.SiLU(),nn.Linear(16,2))
            nn.init.zeros_(self.net[-1].weight)
            with torch.no_grad(): self.net[-1].bias.copy_(bias)

    def forward(self,z,s,b,sigma2, *, diagnostics=False, gate_override=None):
        if z.ndim != 3 or s.ndim != 3 or b.shape != z.shape or s.shape[-1] != z.shape[-1]:
            raise ValueError("expected batched Z/B(b,N,P), S(b,K,P)")
        if not torch.all(sigma2 > 0): raise ValueError("sigma2 must be positive")
        g = spectral_initial(z,s,b)
        gram = s@s.mH
        norm2 = torch.linalg.eigvalsh(gram)[...,-1].real[:,None,None] if self.config.core=='gd' else None
        # This inverse trace is an observable LS noise proxy, not channel truth.
        if self.config.structure=='adaptive':
            eye = torch.eye(s.shape[1],dtype=s.dtype,device=s.device).expand(s.shape[0],-1,-1)
            noise_proxy = sigma2*torch.linalg.solve(gram,eye).diagonal(dim1=-2,dim2=-1).real.sum(-1)/s.shape[1]
        trace = []
        for t in range(self.config.stages):
            gamma = 2*torch.sigmoid(self.steps[t])
            mu = g@s+b
            qfield = posterior_field(z,mu,sigma2)
            if self.config.core == "em":
                dc = g+gamma*(right_solve((qfield-b)@s.mH,gram)-g)
            else:
                dc = g-gamma*((mu-qfield)@s.mH)/norm2
            active = self.config.structure != "none" and (not self.config.final_only or t==self.config.stages-1)
            if active:
                h = hankel(dc.mT)
                if self.config.structure=='adaptive':
                    sv = torch.linalg.svdvals(h)
                    cap = sv.shape[-1]
                    weights = sv/sv.sum(-1,keepdim=True).clamp_min(1e-12)
                    ent = -(weights*weights.clamp_min(1e-12).log()).sum(-1)
                    residual = ((z-(dc@s+b).abs()).square().mean((-2,-1))/sigma2).log1p()
                    power = dc.abs().square().mean(-2).clamp_min(1e-12)
                    expand = lambda x: x[:,None].expand(-1,s.shape[1])
                    const = lambda v: torch.full_like(ent,v)
                    features = torch.stack((ent.exp()/cap,weights[...,0],weights[...,-((cap+1)//2):].sum(-1),
                        expand(residual),(noise_proxy[:,None]/power).log1p(),const(s.shape[-1]/(2*s.shape[1])),
                        const(math.log(cap)),const((t+1)/self.config.stages)),dim=-1)
                    raw = self.net(features)
                else:
                    # Global controls do not pay for unused spectral features.
                    raw = self.structural[t].expand(s.shape[0],s.shape[1],2)
                relative_tau = .001+torch.nn.functional.softplus(raw[...,0])
                beta = torch.sigmoid(raw[...,1])
                if self.config.gate_one: beta = torch.ones_like(beta)
                if gate_override is not None: beta = torch.full_like(beta,gate_override)
                hs,tau = smooth_shrink(h,relative_tau)
                structured = unhankel(hs).mT
                g = dc+beta[:,None,:]*(structured-dc)
                if diagnostics:
                    correction = (g-dc).norm(dim=-2)/dc.norm(dim=-2).clamp_min(1e-12)
            else:
                g = dc
                beta=tau=correction=torch.zeros_like(g.real[:,0,:])
            if diagnostics:
                trace.append(dict(beta=beta.detach(),tau=tau.detach(),correction=correction.detach(),
                    gamma=gamma.detach(),nll=nll(z,g@s+b,sigma2).mean((-2,-1)).detach(),g=g.detach()))
        return (g,trace) if diagnostics else g

    def estimate(self,Z,S,B,sigma2,**kwargs):
        """Single-world or batched observation interface; returns torch tensors."""
        dtype=self.steps.dtype;device=self.steps.device
        cdtype=torch.complex128 if dtype==torch.float64 else torch.complex64
        z=torch.as_tensor(Z,dtype=dtype,device=device)
        s=torch.as_tensor(S,dtype=cdtype,device=device)
        b=torch.as_tensor(B,dtype=cdtype,device=device)
        single=z.ndim==2
        if single: z,s,b=z[None],s[None],b[None]
        v=torch.as_tensor(sigma2,dtype=dtype,device=device)
        if v.ndim==0: v=v.expand(z.shape[0])
        result=self(z,s,b,v,**kwargs)
        if not single:return result
        if isinstance(result,tuple):
            g,trace=result
            return g[0],[{k:(v[0] if v.ndim>0 else v) for k,v in stage.items()} for stage in trace]
        return result[0]

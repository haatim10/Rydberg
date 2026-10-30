"""Full-range smooth gate with exactly the strong OptShrink rank/update lookup.

Each case executes only its prescribed number of updates, even in mixed-SNR
batches. Native, checked OptShrink derivatives are inherited unchanged.
"""
import json
import torch
from torch import nn
from sahure.math import spectral_initial, posterior_field, right_solve, hankel, unhankel
from hybrid_optshrink.model import OLD_SELECTION, opt_filter, opt_weights


def smooth_gate(raw, base):
    # These fixed initialization anchors are not clipping learned outputs.
    anchor=base.clamp(.05,.95)
    beta=torch.sigmoid(torch.logit(anchor)+raw)
    return beta,beta*(1-beta)


class OptController(nn.Module):
    def __init__(self,mode='adaptive',stages=None,tables=None):
        super().__init__()
        if mode not in ('adaptive','global','clipped','fixed','residual'):
            raise ValueError(mode)
        self.mode=mode
        if tables is None:
            old=json.loads(OLD_SELECTION.read_text())['noise_aware']
            tables=[old[str(s)]['loop'] for s in range(-10,21,5)]
        self.register_buffer('ranks',torch.tensor([max(1,p['rank']) for p in tables]))
        self.register_buffer('strengths',torch.tensor([p['strength'] for p in tables],dtype=torch.float64))
        self.register_buffer('levels',torch.arange(-10,21,5,dtype=torch.float64))
        self.register_buffer('budgets',torch.tensor([p['iterations'] if stages is None else stages for p in tables]))
        self.stages=int(self.budgets.max())
        if mode in ('adaptive','clipped'):
            self.net=nn.Sequential(nn.Linear(8,16),nn.SiLU(),nn.Linear(16,1))
            nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        elif mode=='global':
            self.offsets=nn.Parameter(torch.zeros(self.stages))

    def forward(self,z,s,b,sigma2,*,strength_override=None,residual_c=0.,diagnostics=False):
        if torch.any(sigma2<=0):raise ValueError('positive variance required')
        g=spectral_initial(z,s,b);gram=s@s.mH
        index=(10*torch.log10(3/sigma2)[:,None]-self.levels).abs().argmin(-1)
        ranks=self.ranks[index,None];budgets=self.budgets[index]
        base=self.strengths[index,None].expand(-1,s.shape[1])
        if strength_override is not None:
            base=torch.broadcast_to(torch.as_tensor(strength_override,dtype=z.dtype,device=z.device),base.shape)
        eye=torch.eye(s.shape[1],dtype=s.dtype,device=s.device)
        proxy=sigma2*torch.linalg.solve(gram,eye.expand(len(z),-1,-1)).diagonal(dim1=-2,dim2=-1).real.sum(-1)/s.shape[1]
        reference=b.abs().square().mean((-2,-1));trace=[]
        for t in range(int(budgets.max())):
            active=torch.nonzero(budgets>t,as_tuple=True)[0]
            zz,ss,bb,vv=z[active],s[active],b[active],sigma2[active]
            q=posterior_field(zz,g[active]@ss+bb,vv)
            dc=right_solve((q-bb)@ss.mH,gram[active]);ba=base[active]
            if self.mode in ('fixed','residual') and bool(torch.all(ba==0)):
                updated=dc;beta=ba;slope=torch.zeros_like(ba)
            else:
                filtered,sv=opt_filter(hankel(dc.mT),ranks[active])
                structured=unhankel(filtered).mT
                residual=(zz-(dc@ss+bb).abs()).square().mean((-2,-1))/vv
                if self.mode in ('adaptive','clipped'):
                    w=sv/sv.sum(-1,keepdim=True).clamp_min(1e-12)
                    entropy=-(w*w.clamp_min(1e-12).log()).sum(-1)
                    power=dc.abs().square().mean(-2).clamp_min(1e-12)
                    features=torch.stack((entropy.exp()/sv.shape[-1],w[...,0],
                        w[...,-((sv.shape[-1]+1)//2):].sum(-1),
                        residual.log1p()[:,None].expand_as(power),(proxy[active,None]/power).log1p(),
                        (reference[active,None]/(s.shape[1]*power)).log1p(),
                        torch.full_like(power,s.shape[-1]/(2*s.shape[1])),
                        ((t+1)/budgets[active].to(z.dtype))[:,None].expand_as(power)),-1)
                    raw=self.net(features).squeeze(-1)
                    if self.mode=='adaptive':beta,slope=smooth_gate(raw,ba)
                    else:
                        unbounded=ba+.5*torch.tanh(raw)
                        beta=unbounded.clamp(0,1)
                        slope=.5*(1-torch.tanh(raw).square())*((unbounded>0)&(unbounded<1))
                elif self.mode=='global':beta,slope=smooth_gate(self.offsets[t].expand_as(ba),ba)
                elif self.mode=='residual' and residual_c>0:
                    beta=ba*(residual/residual_c).clamp(0,1)[:,None];slope=torch.zeros_like(ba)
                else:beta=ba;slope=torch.zeros_like(ba)
                updated=dc+beta[:,None,:]*(structured-dc)
            g=g.index_copy(0,active,updated)
            if diagnostics:
                info=torch.stack((beta,(updated-dc).norm(dim=-2)/dc.norm(dim=-2).clamp_min(1e-12),
                                  torch.ones_like(beta),slope),-1).detach()
                row=torch.zeros(len(z),s.shape[1],4,dtype=z.dtype,device=z.device)
                trace.append(row.index_copy(0,active,info))
        if not torch.isfinite(g).all():raise FloatingPointError('nonfinite corrected estimator')
        if not diagnostics:return g
        while len(trace)<self.stages:
            trace.append(torch.zeros(len(z),s.shape[1],4,dtype=z.dtype,device=z.device))
        return g,torch.stack(trace,1)


def build(name,seed=4201):
    torch.manual_seed(seed)
    return OptController(mode=name).double()
